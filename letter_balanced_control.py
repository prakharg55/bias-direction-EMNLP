"""
Letter-balanced control ablation.

Tests whether the bias direction is genuinely encoding "follow the bias" or partially
a letter-prediction artifact.

For each source:
  1. Group flipped pairs by their bias_target letter
  2. Group resisted pairs by their correct_letter
  3. Subsample each group to min_count per letter (strict balancing)
  4. Recompute the direction from the balanced subsets: mean(X[bal_flip]) - mean(X[bal_rsst])

Then run LODO probing with:
  - Direction averaged from balanced per-source directions (other datasets, same bias)
  - Held-out test pairs taken from the balanced subset of the held-out source

Compare balanced AUROC to the original (unbalanced) LODO AUROC from Part 1.

If balanced AUROC ~ original AUROC: direction encodes real bias-following signal.
If balanced AUROC drops substantially: some of the original signal was letter-prediction.

Sample-size thresholds:
  MIN_PER_LETTER_TRAIN = 3 per letter (used to decide balanceability)
  MIN_TEST_PAIRS = 5 minimum flipped and 5 resisted in held-out balanced test pairs

Outputs:
  cache/{model_tag}/letter_balanced/comparison.csv
  cache/{model_tag}/letter_balanced/summary.json
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

CACHE = Path(__file__).resolve().parent / "cache"
SEED = 42
MIN_PER_LETTER_TRAIN = 3
MIN_TEST_PAIRS = 5


def normalize(v, eps=1e-8):
    return v / (np.linalg.norm(v) + eps)


def balanced_indices(pairs, indices, key_field, min_per_letter=MIN_PER_LETTER_TRAIN, seed=42):
    """Group pairs[indices] by key_field, subsample to min count per letter.

    Returns (sorted balanced indices, dict letter->count) or (None, None) if insufficient.
    """
    rng = np.random.default_rng(seed)
    by_letter = defaultdict(list)
    for i in indices:
        by_letter[pairs[i][key_field]].append(i)
    if not by_letter:
        return None, None
    min_count = min(len(v) for v in by_letter.values())
    if min_count < min_per_letter:
        return None, None
    balanced = []
    for letter, idxs in sorted(by_letter.items()):
        sampled = rng.choice(idxs, size=min_count, replace=False)
        balanced.extend(sampled.tolist())
    return sorted(balanced), {l: min_count for l in by_letter}


def compute_balanced_for_source(model_tag, src_name):
    """Compute the balanced direction for one source. Returns dict."""
    src_dir = CACHE / model_tag / src_name
    X = np.load(src_dir / "X_biased.npy").astype(np.float32)
    pairs = json.load(open(src_dir / "pairs_meta.json"))

    flip_idx = [i for i, p in enumerate(pairs) if p["category"] == "flipped"]
    rsst_idx = [i for i, p in enumerate(pairs) if p["category"] == "resisted"]

    orig_flip_letter_dist = dict(Counter(pairs[i]["bias_target"] for i in flip_idx))
    orig_rsst_letter_dist = dict(Counter(pairs[i]["correct_letter"] for i in rsst_idx))

    info = {
        "source": src_name,
        "n_orig_flipped": len(flip_idx),
        "n_orig_resisted": len(rsst_idx),
        "orig_flip_letter_dist": orig_flip_letter_dist,
        "orig_rsst_letter_dist": orig_rsst_letter_dist,
    }

    bal_flip_idx, flip_letter_counts = balanced_indices(
        pairs, flip_idx, "bias_target", seed=SEED,
    )
    bal_rsst_idx, rsst_letter_counts = balanced_indices(
        pairs, rsst_idx, "correct_letter", seed=SEED + 1,
    )

    if bal_flip_idx is None or bal_rsst_idx is None:
        info["balanceable"] = False
        info["n_balanced_flipped"] = 0 if bal_flip_idx is None else len(bal_flip_idx)
        info["n_balanced_resisted"] = 0 if bal_rsst_idx is None else len(bal_rsst_idx)
        info["direction"] = None
        return info

    mu_flip = X[bal_flip_idx].mean(axis=0)
    mu_rsst = X[bal_rsst_idx].mean(axis=0)
    d_balanced = normalize(mu_flip - mu_rsst).astype(np.float32)

    info["balanceable"] = True
    info["n_balanced_flipped"] = len(bal_flip_idx)
    info["n_balanced_resisted"] = len(bal_rsst_idx)
    info["balanced_flip_idx"] = list(bal_flip_idx)
    info["balanced_rsst_idx"] = list(bal_rsst_idx)
    info["flip_letter_counts"] = flip_letter_counts
    info["rsst_letter_counts"] = rsst_letter_counts
    info["direction"] = d_balanced
    return info


def analyze_model(model_tag):
    print(f"\n{'=' * 90}\n{model_tag}\n{'=' * 90}")

    # Load source labels
    d_npz = np.load(CACHE / model_tag / "directions.npz", allow_pickle=True)
    source_names = [str(s) for s in d_npz["source_names"]]
    bias_names = [str(b) for b in d_npz["bias_names"]]
    dataset_names = [str(d) for d in d_npz["dataset_names"]]
    per_source_dirs_orig = d_npz["directions"].astype(np.float32)

    # Phase 1: balanced directions per source
    print("\n[phase 1] Building balanced per-source directions")
    source_info = {}
    for s_name in source_names:
        info = compute_balanced_for_source(model_tag, s_name)
        source_info[s_name] = info
        flip_letters = sorted(info["orig_flip_letter_dist"].keys())
        if info["balanceable"]:
            print(f"  {s_name:<55s}  "
                  f"orig f={info['n_orig_flipped']:>4d} r={info['n_orig_resisted']:>4d}  "
                  f"bal f={info['n_balanced_flipped']:>3d} r={info['n_balanced_resisted']:>3d}  "
                  f"letters={','.join(flip_letters)}")
        else:
            print(f"  {s_name:<55s}  SKIP (orig f={info['n_orig_flipped']} r={info['n_orig_resisted']}; "
                  f"insufficient letters with >= {MIN_PER_LETTER_TRAIN})")

    n_balanceable = sum(1 for v in source_info.values() if v["balanceable"])

    # Cosine: original direction vs balanced direction per source
    print(f"\n[phase 1b] Cosine similarity between original and balanced direction per source:")
    cos_orig_vs_balanced = {}
    for i, s_name in enumerate(source_names):
        info = source_info[s_name]
        if not info["balanceable"]:
            cos_orig_vs_balanced[s_name] = None
            continue
        cos = float(np.dot(per_source_dirs_orig[i], info["direction"]))
        cos_orig_vs_balanced[s_name] = cos
        print(f"  {s_name:<55s}  cos = {cos:+.4f}")

    # Phase 2: LODO probing using balanced directions + balanced test pairs
    print(f"\n[phase 2] LODO probing on balanced setup")
    bias_to_indices = defaultdict(list)
    for i, b in enumerate(bias_names):
        bias_to_indices[b].append(i)

    lodo_balanced_rows = []
    for bias_type, indices in bias_to_indices.items():
        if len(indices) < 2:
            continue  # singleton biases excluded
        for held_out_i in indices:
            held_src = source_names[held_out_i]
            info = source_info[held_src]
            if not info["balanceable"]:
                print(f"  [skip {held_src}: not balanceable]")
                continue
            if info["n_balanced_flipped"] < MIN_TEST_PAIRS or info["n_balanced_resisted"] < MIN_TEST_PAIRS:
                print(f"  [skip {held_src}: too few balanced test pairs "
                      f"({info['n_balanced_flipped']} flip, {info['n_balanced_resisted']} rsst)]")
                continue

            # Build LODO direction from OTHER balanced per-source directions in same bias
            other_dirs = []
            for j in indices:
                if j == held_out_i:
                    continue
                other_info = source_info[source_names[j]]
                if other_info["balanceable"]:
                    other_dirs.append(other_info["direction"])
            if not other_dirs:
                print(f"  [skip {held_src}: no other balanceable sources in bias]")
                continue
            d_lodo = normalize(np.mean(other_dirs, axis=0).astype(np.float32))

            # Test on held-out source's balanced pairs
            src_dir = CACHE / model_tag / held_src
            X = np.load(src_dir / "X_biased.npy").astype(np.float32)

            keep = list(info["balanced_flip_idx"]) + list(info["balanced_rsst_idx"])
            y = np.array([1] * len(info["balanced_flip_idx"]) + [0] * len(info["balanced_rsst_idx"]))
            scores = X[keep] @ d_lodo
            auroc = float(roc_auc_score(y, scores))

            lodo_balanced_rows.append({
                "model": model_tag,
                "bias": bias_type,
                "held_out_source": held_src,
                "n_balanced_flipped": int(info["n_balanced_flipped"]),
                "n_balanced_resisted": int(info["n_balanced_resisted"]),
                "balanced_lodo_auroc": auroc,
                "n_other_sources_in_avg": len(other_dirs),
            })
            print(f"  {held_src:<55s}  balanced LODO AUROC = {auroc:.3f}  "
                  f"(n_test={len(keep)})")

    # Phase 3: compare to original LODO AUROCs
    print(f"\n[phase 3] Comparison to original LODO results")
    orig_rows = []
    with open(CACHE / model_tag / "lodo_results.csv") as f:
        header = next(f).strip().split(",")
        for line in f:
            orig_rows.append(dict(zip(header, line.strip().split(","))))
    orig_lodo_by_src = {r["source"]: float(r["lodo_auroc"]) for r in orig_rows}

    comparisons = []
    for r in lodo_balanced_rows:
        src = r["held_out_source"]
        if src in orig_lodo_by_src:
            r["original_lodo_auroc"] = orig_lodo_by_src[src]
            r["delta"] = r["balanced_lodo_auroc"] - r["original_lodo_auroc"]
            r["cos_orig_vs_balanced_direction"] = cos_orig_vs_balanced[src]
            comparisons.append(r)

    print(f"\n  {'source':<55s} {'orig':>7s} {'balanced':>9s} {'delta':>7s} {'cos':>7s}")
    for c in comparisons:
        print(f"  {c['held_out_source']:<55s} {c['original_lodo_auroc']:>7.3f} "
              f"{c['balanced_lodo_auroc']:>9.3f} {c['delta']:>+7.3f} "
              f"{c['cos_orig_vs_balanced_direction']:>+7.3f}")

    # Summary
    if comparisons:
        deltas = np.array([c["delta"] for c in comparisons])
        orig_aurocs = np.array([c["original_lodo_auroc"] for c in comparisons])
        bal_aurocs = np.array([c["balanced_lodo_auroc"] for c in comparisons])
        cosines = np.array([c["cos_orig_vs_balanced_direction"] for c in comparisons])

        print(f"\n  ==== Summary ====")
        print(f"  n_test_points:                {len(comparisons)}")
        print(f"  mean original AUROC:          {orig_aurocs.mean():.3f}")
        print(f"  mean balanced AUROC:          {bal_aurocs.mean():.3f}")
        print(f"  mean delta:                   {deltas.mean():+.3f}")
        print(f"  median delta:                 {np.median(deltas):+.3f}")
        print(f"  mean cos(orig, balanced):     {cosines.mean():+.4f}")
        print(f"  n sources where balanced > orig: {(deltas > 0).sum()}/{len(deltas)}")
        try:
            stat, p_two = wilcoxon(deltas)
            print(f"  Wilcoxon (delta != 0):        p = {float(p_two):.4g}")
        except ValueError as e:
            print(f"  Wilcoxon failed: {e}")

        # Per-bias breakdown
        print(f"\n  ==== Per-bias mean delta ====")
        by_bias = defaultdict(list)
        for c in comparisons:
            by_bias[c["bias"]].append(c)
        for bias, rs in sorted(by_bias.items()):
            d = np.mean([r["delta"] for r in rs])
            o = np.mean([r["original_lodo_auroc"] for r in rs])
            b = np.mean([r["balanced_lodo_auroc"] for r in rs])
            print(f"  {bias:<35s}  n={len(rs):>2d}  orig={o:.3f}  bal={b:.3f}  delta={d:+.3f}")

    # === Save ===
    out_dir = CACHE / model_tag / "letter_balanced"
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    if comparisons:
        keys = ["model", "bias", "held_out_source", "n_balanced_flipped", "n_balanced_resisted",
                "n_other_sources_in_avg", "original_lodo_auroc", "balanced_lodo_auroc", "delta",
                "cos_orig_vs_balanced_direction"]
        with open(out_dir / "comparison.csv", "w") as f:
            f.write(",".join(keys) + "\n")
            for c in comparisons:
                f.write(",".join(str(c[k]) for k in keys) + "\n")
        print(f"\n  Saved: {out_dir / 'comparison.csv'}")

    # JSON summary
    summary = {
        "model": model_tag,
        "config": {
            "min_per_letter_train": MIN_PER_LETTER_TRAIN,
            "min_test_pairs": MIN_TEST_PAIRS,
            "seed": SEED,
        },
        "n_total_sources": len(source_info),
        "n_balanceable_sources": n_balanceable,
        "n_lodo_test_points": len(comparisons),
    }
    if comparisons:
        summary["overall"] = {
            "mean_original_auroc": float(orig_aurocs.mean()),
            "mean_balanced_auroc": float(bal_aurocs.mean()),
            "mean_delta": float(deltas.mean()),
            "median_delta": float(np.median(deltas)),
            "mean_cos_orig_balanced": float(cosines.mean()),
            "n_sources_balanced_above_orig": int((deltas > 0).sum()),
            "wilcoxon_p_two_sided": float(wilcoxon(deltas)[1]) if len(deltas) >= 1 else None,
        }
        summary["per_bias"] = {
            bias: {
                "n_held_out_sources": len(rs),
                "mean_original_auroc": float(np.mean([r["original_lodo_auroc"] for r in rs])),
                "mean_balanced_auroc": float(np.mean([r["balanced_lodo_auroc"] for r in rs])),
                "mean_delta": float(np.mean([r["delta"] for r in rs])),
            }
            for bias, rs in by_bias.items()
        }
    summary["per_source_letter_distributions"] = {
        s: {"orig_flip": info["orig_flip_letter_dist"], "orig_rsst": info["orig_rsst_letter_dist"]}
        for s, info in source_info.items()
    }
    summary["balanceable_status"] = {
        s: info["balanceable"] for s, info in source_info.items()
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {out_dir / 'summary.json'}")


def main():
    for model_tag in ("llama31_8b", "qwen25_7b", "gemma2_9b", "mistral7b", "olmo2_7b"):
        analyze_model(model_tag)


if __name__ == "__main__":
    main()
