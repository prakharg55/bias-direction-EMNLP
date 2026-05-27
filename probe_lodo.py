"""
Leave-One-Dataset-Out (LODO) probing transfer test.

For each bias type B with >= 2 source datasets, and each held-out dataset H in B:
  1. d_B,-H = normalize(mean of per-source unit directions in B, excluding H)
  2. Take held-out source (B, H) filtered pairs (flipped + resisted)
  3. Project each pair's X_biased onto d_B,-H -> scalar score
  4. AUROC for separating flipped from resisted
  5. 95% bootstrap CI from n_bootstrap pair resamples
  6. Random-direction baseline: same procedure, n_random_seeds unit random directions

Skipped: bias types with only 1 source dataset (can't LODO).
  - spurious_few_shot_hindsight (only hindsight_neglect dataset)

Statistical tests:
  - Per-source: AUROC vs random baseline mean (sign)
  - Pooled: paired Wilcoxon LODO AUROC > random AUROC (per source)
  - Per-bias: mean LODO AUROC, mean random AUROC

Outputs:
  paper/cache/{model_tag}/lodo_results.csv     per-source numbers
  paper/cache/{model_tag}/lodo_summary.json    per-bias + overall stats
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

CACHE = Path(__file__).resolve().parent / "cache"
N_BOOTSTRAP = 1000
N_RANDOM_SEEDS = 50
MIN_TEST_PAIRS_PER_CLASS = 5
GLOBAL_SEED = 42


def normalize(v, eps=1e-8):
    n = np.linalg.norm(v)
    return v / (n + eps)


def bootstrap_auroc_ci(y_true, scores, n_bootstrap=1000, alpha=0.05, rng=None):
    """Percentile bootstrap CI for AUROC."""
    if rng is None:
        rng = np.random.default_rng(GLOBAL_SEED)
    n = len(y_true)
    aurocs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aurocs.append(roc_auc_score(y_true[idx], scores[idx]))
    aurocs = np.asarray(aurocs)
    if len(aurocs) == 0:
        return float("nan"), float("nan")
    lo = float(np.percentile(aurocs, 100 * alpha / 2))
    hi = float(np.percentile(aurocs, 100 * (1 - alpha / 2)))
    return lo, hi


def load_source_X_and_labels(model_tag, source_name):
    """Return X (filtered pairs, fp32) and binary label y (1=flipped, 0=resisted)."""
    src_dir = CACHE / model_tag / source_name
    X = np.load(src_dir / "X_biased.npy").astype(np.float32)
    pairs = json.load(open(src_dir / "pairs_meta.json"))
    assert X.shape[0] == len(pairs), \
        f"{source_name}: X rows ({X.shape[0]}) != pairs ({len(pairs)})"

    flip_mask = np.array([p["category"] == "flipped" for p in pairs])
    rsst_mask = np.array([p["category"] == "resisted" for p in pairs])
    keep_mask = flip_mask | rsst_mask
    X = X[keep_mask]
    y = flip_mask[keep_mask].astype(np.int64)
    return X, y, int(flip_mask.sum()), int(rsst_mask.sum())


def analyze_model(model_tag):
    print(f"\n{'=' * 100}")
    print(f"LODO probing analysis: {model_tag}")
    print(f"{'=' * 100}")

    d = np.load(CACHE / model_tag / "directions.npz", allow_pickle=True)
    directions = d["directions"]  # (n_src, n_dim)
    source_names = [str(s) for s in d["source_names"]]
    bias_names = [str(b) for b in d["bias_names"]]
    dataset_names = [str(ds) for ds in d["dataset_names"]]
    n_dim = int(d["n_dim"])

    bias_to_indices = defaultdict(list)
    for i, b in enumerate(bias_names):
        bias_to_indices[b].append(i)

    rng = np.random.default_rng(GLOBAL_SEED)
    results = []

    print(f"\n{'source':<55s} {'n_flip':>6s} {'n_rsst':>6s} {'AUROC':>7s} {'95% CI':>16s} "
          f"{'rand':>6s} {'lift':>6s}")
    print("-" * 110)

    for bias_type, indices in bias_to_indices.items():
        if len(indices) < 2:
            print(f"\n  [skip bias '{bias_type}': only {len(indices)} source, cannot LODO]")
            continue

        for held_out_i in indices:
            other_indices = [i for i in indices if i != held_out_i]
            avg = np.mean(directions[other_indices], axis=0)
            d_lodo = normalize(avg)

            held_src = source_names[held_out_i]
            X_test, y_test, n_flip, n_rsst = load_source_X_and_labels(model_tag, held_src)
            if n_flip < MIN_TEST_PAIRS_PER_CLASS or n_rsst < MIN_TEST_PAIRS_PER_CLASS:
                print(f"  [skip {held_src}: insufficient test pairs ({n_flip} flip / {n_rsst} rsst)]")
                continue

            scores = X_test @ d_lodo
            auroc = float(roc_auc_score(y_test, scores))
            ci_lo, ci_hi = bootstrap_auroc_ci(
                y_test, scores, n_bootstrap=N_BOOTSTRAP, rng=rng,
            )

            # Random baseline: n_random_seeds unit random directions
            random_aurocs = []
            for seed in range(N_RANDOM_SEEDS):
                local_rng = np.random.default_rng(GLOBAL_SEED + 1 + seed)
                r = local_rng.standard_normal(n_dim).astype(np.float32)
                r = normalize(r)
                rand_scores = X_test @ r
                random_aurocs.append(float(roc_auc_score(y_test, rand_scores)))
            random_aurocs = np.array(random_aurocs)
            rand_mean = float(random_aurocs.mean())
            rand_lo = float(np.percentile(random_aurocs, 2.5))
            rand_hi = float(np.percentile(random_aurocs, 97.5))

            results.append({
                "model": model_tag,
                "bias": bias_type,
                "held_out_dataset": dataset_names[held_out_i],
                "source": held_src,
                "n_flip": n_flip,
                "n_rsst": n_rsst,
                "lodo_auroc": auroc,
                "lodo_ci_lo": ci_lo,
                "lodo_ci_hi": ci_hi,
                "random_auroc_mean": rand_mean,
                "random_auroc_std": float(random_aurocs.std()),
                "random_ci_lo": rand_lo,
                "random_ci_hi": rand_hi,
                "lift": auroc - rand_mean,
            })

            ci_str = f"[{ci_lo:.3f},{ci_hi:.3f}]"
            print(f"  {held_src:<53s} {n_flip:>6d} {n_rsst:>6d} {auroc:>7.3f} {ci_str:>16s} "
                  f"{rand_mean:>6.3f} {auroc - rand_mean:>+6.3f}")

    return results, source_names, bias_names


def summarize(results):
    """Print and return per-bias and overall statistics."""
    summary = {"per_bias": {}, "overall": {}}
    if not results:
        return summary

    # Per-bias
    by_bias = defaultdict(list)
    for r in results:
        by_bias[r["bias"]].append(r)
    print(f"\n{'-' * 100}\nPer-bias summary\n{'-' * 100}")
    print(f"{'bias':<35s} {'n':>3s} {'mean LODO':>10s} {'mean rand':>10s} {'mean lift':>10s}")
    for b, rs in by_bias.items():
        a = np.array([r["lodo_auroc"] for r in rs])
        rand = np.array([r["random_auroc_mean"] for r in rs])
        summary["per_bias"][b] = {
            "n_held_out_sources": len(rs),
            "mean_lodo_auroc": float(a.mean()),
            "mean_random_auroc": float(rand.mean()),
            "mean_lift": float((a - rand).mean()),
        }
        print(f"  {b:<33s} {len(rs):>3d} {a.mean():>10.3f} {rand.mean():>10.3f} {(a - rand).mean():>+10.3f}")

    # Overall
    a_all = np.array([r["lodo_auroc"] for r in results])
    rand_all = np.array([r["random_auroc_mean"] for r in results])
    lifts = a_all - rand_all

    # Wilcoxon paired test: lodo_auroc > random_auroc (one-sided)
    try:
        stat, p = wilcoxon(lifts, alternative="greater")
        wilcox_stat, wilcox_p = float(stat), float(p)
    except ValueError:
        wilcox_stat, wilcox_p = float("nan"), float("nan")

    # Wilcoxon one-sample: lodo_auroc > 0.5 (one-sided)
    try:
        stat2, p2 = wilcoxon(a_all - 0.5, alternative="greater")
        wilcox_stat_chance, wilcox_p_chance = float(stat2), float(p2)
    except ValueError:
        wilcox_stat_chance, wilcox_p_chance = float("nan"), float("nan")

    summary["overall"] = {
        "n_sources": len(results),
        "mean_lodo_auroc": float(a_all.mean()),
        "median_lodo_auroc": float(np.median(a_all)),
        "mean_random_auroc": float(rand_all.mean()),
        "mean_lift": float(lifts.mean()),
        "wilcoxon_lodo_vs_random_p_one_sided": wilcox_p,
        "wilcoxon_lodo_vs_random_stat": wilcox_stat,
        "wilcoxon_lodo_vs_0.5_p_one_sided": wilcox_p_chance,
        "wilcoxon_lodo_vs_0.5_stat": wilcox_stat_chance,
        "n_sources_above_random": int((a_all > rand_all).sum()),
        "n_sources_above_chance": int((a_all > 0.5).sum()),
    }

    print(f"\n{'-' * 100}\nOverall summary\n{'-' * 100}")
    print(f"  n_sources tested:                  {summary['overall']['n_sources']}")
    print(f"  mean LODO AUROC:                   {a_all.mean():.3f}")
    print(f"  median LODO AUROC:                 {np.median(a_all):.3f}")
    print(f"  mean random-direction AUROC:       {rand_all.mean():.3f}")
    print(f"  mean lift (LODO - random):         {lifts.mean():+.3f}")
    print(f"  sources beating random:            {(a_all > rand_all).sum()}/{len(results)}")
    print(f"  sources beating chance (>0.5):     {(a_all > 0.5).sum()}/{len(results)}")
    print(f"  Wilcoxon LODO > random (paired):   p = {wilcox_p:.4g}")
    print(f"  Wilcoxon LODO > 0.5 (one-sample):  p = {wilcox_p_chance:.4g}")

    return summary


def save_results(model_tag, results, summary):
    out_dir = CACHE / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = out_dir / "lodo_results.csv"
    if results:
        keys = list(results[0].keys())
        with open(csv_path, "w") as f:
            f.write(",".join(keys) + "\n")
            for r in results:
                f.write(",".join(str(r[k]) for k in keys) + "\n")
        print(f"\n  Saved per-source results: {csv_path}")

    # JSON summary
    json_path = out_dir / "lodo_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary: {json_path}")


def main():
    for model_tag in ("llama31_8b", "qwen25_7b", "gemma2_9b", "mistral7b", "olmo2_7b"):
        results, _, _ = analyze_model(model_tag)
        summary = summarize(results)
        save_results(model_tag, results, summary)


if __name__ == "__main__":
    main()
