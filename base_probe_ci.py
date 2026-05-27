"""
Confidence intervals and Wilcoxon test on the base-probe cue-attributable gap.

A reviewer will ask: "is the +0.0 to +0.09 cue-attributable signal really above
zero?" To answer, per xlabel model:

  - per-source LODO AUROC on biased activations  (X_biased)
  - per-source LODO AUROC on unbiased activations (X_unbiased)  -- paired
  - per-source paired difference                                 (biased - unbiased)
  - mean of differences + 95% bootstrap CI (2000 resamples)
  - Wilcoxon signed-rank test (paired, one-sided: biased > unbiased)

The diff is the cue-attributable signal (the part of base decodability NOT
explained by question-difficulty content), per source.

Output: cache/base_probe_ci.json + console summary.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

CACHE = Path(__file__).resolve().parent / "cache"
MIN_PAIRS = 5
N_BOOTSTRAP = 2000
SEED = 42

XLABEL_MODELS = ["llama31_8b_base_xlabel", "qwen25_7b_base_xlabel",
                 "gemma2_9b_base_xlabel", "mistral7b_base_xlabel",
                 "olmo2_7b_base_xlabel"]


def normalize(v, eps=1e-8):
    return v / (np.linalg.norm(v) + eps)


def per_source_aurocs(cache_dir, x_filename):
    """Per-source LODO AUROC computed from x_filename activations.

    Two-pass to bound peak memory to ONE X array at a time (~600 MB), so this
    can run on a login node without OOM.

      pass 1: for each source, load X, compute direction (or None if too few
              flipped/resisted), drop X. Cache only direction (~500 kB) + masks.
      pass 2: for each held-out source, build d_lodo from the OTHER sources'
              cached directions, reload that source's X, project, AUROC, drop X.
    """
    meta = {}
    for d in sorted(p for p in cache_dir.iterdir() if p.is_dir()):
        xp = d / x_filename
        pm = d / "pairs_meta.json"
        if not xp.exists() or not pm.exists():
            continue
        pairs = json.load(open(pm))
        flip = np.array([p["category"] == "flipped" for p in pairs])
        rsst = np.array([p["category"] == "resisted" for p in pairs])
        dvec = None
        if flip.sum() >= MIN_PAIRS and rsst.sum() >= MIN_PAIRS:
            X = np.load(xp).astype(np.float32)
            if X.shape[0] != len(pairs):
                del X
                continue
            dvec = normalize(X[flip].mean(0) - X[rsst].mean(0))
            del X
        meta[d.name] = {"bias": "__".join(d.name.split("__")[:-1]),
                        "flip": flip, "rsst": rsst, "dir": dvec, "xp": xp}

    by_bias = defaultdict(list)
    for n, s in meta.items():
        if s["dir"] is not None:
            by_bias[s["bias"]].append(n)

    out = {}
    for bias, names in by_bias.items():
        if len(names) < 2:
            continue
        for held in names:
            s = meta[held]
            if s["flip"].sum() < MIN_PAIRS or s["rsst"].sum() < MIN_PAIRS:
                continue
            others = [n for n in names if n != held]
            d_lodo = normalize(np.mean([meta[n]["dir"] for n in others], axis=0))
            X = np.load(s["xp"]).astype(np.float32)
            keep = s["flip"] | s["rsst"]
            X = X[keep]
            y = s["flip"][keep].astype(np.int64)
            out[held] = {"bias": bias, "auroc": float(roc_auc_score(y, X @ d_lodo))}
            del X
    return out


def bootstrap_mean_ci(values, n_boot=N_BOOTSTRAP, alpha=0.05, seed=SEED):
    rng = np.random.default_rng(seed)
    vals = np.asarray(values, dtype=np.float32)
    means = np.empty(n_boot, dtype=np.float32)
    for i in range(n_boot):
        means[i] = vals[rng.choice(len(vals), size=len(vals), replace=True)].mean()
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def main():
    summary = {}
    print(f"{'model':<30s} {'biased':>8s} {'unbiased':>9s} {'gap':>8s} "
          f"{'95% CI':>22s} {'Wilcoxon p':>12s} {'n':>4s}")
    print("-" * 100)
    for tag in XLABEL_MODELS:
        d = CACHE / tag
        if not d.exists():
            continue
        biased = per_source_aurocs(d, "X_biased.npy")
        unbias = per_source_aurocs(d, "X_unbiased.npy")
        common = sorted(set(biased) & set(unbias))
        if not common:
            print(f"  [{tag}] no paired sources")
            continue
        pairs = [(biased[s]["auroc"], unbias[s]["auroc"]) for s in common]
        diffs = np.array([b - u for b, u in pairs], dtype=np.float32)
        biased_arr = np.array([b for b, _ in pairs], dtype=np.float32)
        unbias_arr = np.array([u for _, u in pairs], dtype=np.float32)
        mean_diff = float(diffs.mean())
        ci = bootstrap_mean_ci(diffs)
        try:
            wstat, wp = wilcoxon(diffs, alternative="greater")
            wstat, wp = float(wstat), float(wp)
        except ValueError:
            wstat, wp = float("nan"), float("nan")
        summary[tag] = {
            "n_sources": len(diffs),
            "biased_mean": float(biased_arr.mean()),
            "unbiased_mean": float(unbias_arr.mean()),
            "cue_attributable_gap_mean": mean_diff,
            "cue_attributable_gap_ci95": ci,
            "wilcoxon_paired_p_one_sided": wp,
            "wilcoxon_stat": wstat,
        }
        print(f"{tag:<30s} {biased_arr.mean():>8.3f} {unbias_arr.mean():>9.3f} "
              f"{mean_diff:>+8.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]    {wp:>12.3g} "
              f"{len(diffs):>4d}")

    out = CACHE / "base_probe_ci.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
