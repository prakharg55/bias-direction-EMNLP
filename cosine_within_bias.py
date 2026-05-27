"""
Cosine similarity analysis: within-bias vs across-bias for per-source directions.

Demonstrates the geometric counterpart to the LODO probing test:
  - For each pair (i, j) of sources, compute cos(d_i, d_j).
  - Partition pairs into "within-bias" (same bias_name) and "across-bias" (different).
  - Compare distributions.

Statistical tests:
  - Mann-Whitney U: within > across (one-sided)
  - Per-bias mean within cosine vs random pair expectation

Outputs:
  paper/cache/{model_tag}/cosine_within_bias.json   summary stats
  paper/cache/{model_tag}/cosine_matrix.npz         (21, 21) cosine matrix + source names
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

CACHE = Path(__file__).resolve().parent / "cache"


def analyze_model(model_tag):
    print(f"\n{'=' * 90}")
    print(f"Cosine within-bias analysis: {model_tag}")
    print(f"{'=' * 90}")

    d = np.load(CACHE / model_tag / "directions.npz", allow_pickle=True)
    directions = d["directions"]
    source_names = [str(s) for s in d["source_names"]]
    bias_names = [str(b) for b in d["bias_names"]]
    n = len(source_names)

    # Pairwise cosine (directions are already unit-norm)
    cos_mat = directions @ directions.T

    # Partition off-diagonal pairs
    within, across = [], []
    within_by_bias = defaultdict(list)
    across_pairs_by_bias = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            c = float(cos_mat[i, j])
            if bias_names[i] == bias_names[j]:
                within.append(c)
                within_by_bias[bias_names[i]].append(c)
            else:
                across.append(c)
    within = np.array(within); across = np.array(across)

    # Stats
    print(f"\n  pairs:  within-bias n={len(within)}   across-bias n={len(across)}")
    print(f"\n  within mean: {within.mean():+.4f}  median: {np.median(within):+.4f}  std: {within.std():.4f}")
    print(f"  across mean: {across.mean():+.4f}  median: {np.median(across):+.4f}  std: {across.std():.4f}")
    print(f"  diff (within - across): {within.mean() - across.mean():+.4f}")

    try:
        u, p = mannwhitneyu(within, across, alternative="greater")
        u, p = float(u), float(p)
    except ValueError:
        u, p = float("nan"), float("nan")
    print(f"  Mann-Whitney U (within > across): U={u:.1f}  p={p:.4g}")

    # Per-bias breakdown
    print(f"\n  Per-bias within-cosines (mean):")
    per_bias = {}
    for b, vals in within_by_bias.items():
        a = np.array(vals)
        per_bias[b] = {
            "n_pairs": len(a),
            "mean": float(a.mean()),
            "median": float(np.median(a)),
            "min": float(a.min()),
            "max": float(a.max()),
        }
        print(f"    {b:<35s}  n={len(a):>2d}  mean={a.mean():+.3f}  range=[{a.min():+.3f}, {a.max():+.3f}]")

    summary = {
        "n_sources": n,
        "within": {
            "n_pairs": int(len(within)),
            "mean": float(within.mean()),
            "median": float(np.median(within)),
            "std": float(within.std()),
        },
        "across": {
            "n_pairs": int(len(across)),
            "mean": float(across.mean()),
            "median": float(np.median(across)),
            "std": float(across.std()),
        },
        "diff_within_minus_across": float(within.mean() - across.mean()),
        "mannwhitneyu_within_greater_across": {"U": u, "p_one_sided": p},
        "per_bias": per_bias,
    }

    # Save
    out_dir = CACHE / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "cosine_within_bias.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    mat_path = out_dir / "cosine_matrix.npz"
    np.savez(mat_path,
             cosine=cos_mat.astype(np.float32),
             source_names=np.array(source_names),
             bias_names=np.array(bias_names))
    print(f"\n  Saved: {json_path}")
    print(f"  Saved: {mat_path}")
    return summary


def main():
    for model_tag in ("llama31_8b", "qwen25_7b", "gemma2_9b", "mistral7b", "olmo2_7b"):
        analyze_model(model_tag)


if __name__ == "__main__":
    main()
