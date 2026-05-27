"""
Cluster significance test for the cross-bias direction cluster (Section 6).

Section 6 identifies, via hierarchical clustering on the per-source cosine
matrix, a 4-bias cluster on Qwen:
    suggested_answer, distractor_fact, wrong_few_shot, spurious_few_shot_squares
This script tests whether that cluster is statistically distinguishable from a
flat (no-cluster) structure, and whether the same structure is absent on Llama.

Test 1 -- Mann-Whitney U (one-sided), consistent with cosine_within_bias.py:
  Partition the CROSS-bias source-pair cosines into
    within-cluster : both sources' biases in the cluster (the biases differ)
    outside        : >= 1 source's bias outside the cluster
  H1: within-cluster > outside.
  Run on both models with the SAME 4-bias set. Qwen is expected to be
  significant; Llama is the negative control -- on Llama the 4-bias set is not
  derived from its data, so a null result there is genuinely informative.

  Caveat: the cluster was identified from this same cosine matrix, so the Qwen
  p-value is descriptive / anti-conservative (post-hoc selection). The honest
  independent confirmations are the cross-bias probing transfer matrix (a
  different metric) and the cross-bias causal intervention (a different
  modality). The Llama negative control is NOT post-hoc.

Test 2 -- subset-rank check:
  Among all C(n_biases, 4) four-bias subsets, rank the actual cluster by its
  mean within-subset cross-bias cosine. Reports the rank and the fraction of
  subsets at least as tight (a descriptive permutation-style p-value). With
  only 6 biases there are C(6,4)=15 subsets, so this is a coarse check; the
  rank itself is the informative quantity.

Inputs : cache/{model_tag}/cosine_matrix.npz   (from cosine_within_bias.py)
Output : cache/{model_tag}/cluster_significance.json
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

CACHE = Path(__file__).resolve().parent / "cache"

# Qwen's 4-bias cluster from the Section 6 hierarchical clustering.
CLUSTER = {"suggested_answer", "distractor_fact", "wrong_few_shot",
           "spurious_few_shot_squares"}


def cross_bias_pairs(cos, source_bias):
    """Yield (i, j, cos[i,j]) for every cross-bias source pair (biases differ)."""
    n = len(source_bias)
    for i in range(n):
        for j in range(i + 1, n):
            if source_bias[i] != source_bias[j]:
                yield i, j, float(cos[i, j])


def subset_mean_cosine(cos, source_bias, subset):
    """Mean cross-bias source-pair cosine restricted to biases in `subset`."""
    vals = [c for i, j, c in cross_bias_pairs(cos, source_bias)
            if source_bias[i] in subset and source_bias[j] in subset]
    return float(np.mean(vals)) if vals else float("nan")


def analyze_model(model_tag):
    npz_path = CACHE / model_tag / "cosine_matrix.npz"
    if not npz_path.exists():
        print(f"[skip {model_tag}: no cosine_matrix.npz -- run cosine_within_bias.py first]")
        return None

    d = np.load(npz_path, allow_pickle=True)
    cos = d["cosine"].astype(np.float64)
    source_bias = [str(b) for b in d["bias_names"]]
    biases = sorted(set(source_bias))

    print(f"\n{'=' * 84}")
    print(f"Cluster significance: {model_tag}   ({len(source_bias)} sources, {len(biases)} biases)")
    print(f"{'=' * 84}")
    cluster_biases = sorted(b for b in biases if b in CLUSTER)
    outside_biases = sorted(b for b in biases if b not in CLUSTER)
    print(f"  cluster  ({len(cluster_biases)}): {cluster_biases}")
    print(f"  outside  ({len(outside_biases)}): {outside_biases}")

    # --- Test 1: Mann-Whitney U on cross-bias source-pair cosines -----------
    within_cluster, outside = [], []
    for i, j, c in cross_bias_pairs(cos, source_bias):
        if source_bias[i] in CLUSTER and source_bias[j] in CLUSTER:
            within_cluster.append(c)
        else:
            outside.append(c)
    within_cluster = np.array(within_cluster)
    outside = np.array(outside)

    try:
        u, p = mannwhitneyu(within_cluster, outside, alternative="greater")
        u, p = float(u), float(p)
    except ValueError:
        u, p = float("nan"), float("nan")

    print(f"\n  Test 1 -- Mann-Whitney U (within-cluster > outside, one-sided)")
    print(f"    within-cluster pairs : n={len(within_cluster):>3d}  "
          f"mean={within_cluster.mean():+.4f}  median={np.median(within_cluster):+.4f}")
    print(f"    outside pairs        : n={len(outside):>3d}  "
          f"mean={outside.mean():+.4f}  median={np.median(outside):+.4f}")
    print(f"    diff (within - outside): {within_cluster.mean() - outside.mean():+.4f}")
    print(f"    U={u:.1f}  p={p:.4g}  {'(significant)' if p < 0.05 else '(n.s.)'}")

    # --- Test 2: subset-rank check ------------------------------------------
    observed = subset_mean_cosine(cos, source_bias, CLUSTER)
    all_subsets = [set(s) for s in combinations(biases, len(CLUSTER))]
    subset_means = [(s, subset_mean_cosine(cos, source_bias, s)) for s in all_subsets]
    subset_means.sort(key=lambda x: -x[1])
    rank = next(r for r, (s, m) in enumerate(subset_means, 1) if s == CLUSTER)
    n_ge = sum(1 for s, m in subset_means if m >= observed - 1e-12)
    perm_p = n_ge / len(subset_means)
    runner_up = subset_means[1][1] if rank == 1 else subset_means[0][1]

    print(f"\n  Test 2 -- subset-rank ({len(CLUSTER)}-bias subsets of {len(biases)} biases)")
    print(f"    observed cluster mean cross-bias cosine : {observed:+.4f}")
    print(f"    rank among all {len(all_subsets)} subsets             : "
          f"{rank} / {len(all_subsets)}")
    print(f"    {'tightest subset' if rank == 1 else 'rank-1 subset'} "
          f"cosine                  : {subset_means[0][1]:+.4f}")
    print(f"    descriptive perm-p (frac >= observed)   : {perm_p:.4g}")

    summary = {
        "model": model_tag,
        "cluster": cluster_biases,
        "outside_biases": outside_biases,
        "mann_whitney": {
            "within_cluster_n": int(len(within_cluster)),
            "within_cluster_mean": float(within_cluster.mean()),
            "within_cluster_median": float(np.median(within_cluster)),
            "outside_n": int(len(outside)),
            "outside_mean": float(outside.mean()),
            "outside_median": float(np.median(outside)),
            "diff_within_minus_outside": float(within_cluster.mean() - outside.mean()),
            "U": u,
            "p_one_sided": p,
        },
        "subset_rank": {
            "observed_mean_cosine": observed,
            "n_subsets": len(all_subsets),
            "rank": rank,
            "tightest_subset": sorted(subset_means[0][0]),
            "tightest_mean_cosine": float(subset_means[0][1]),
            "runner_up_or_top_mean_cosine": float(runner_up),
            "perm_p_descriptive": perm_p,
        },
    }
    out_path = CACHE / model_tag / "cluster_significance.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {out_path}")
    return summary


def main():
    results = {}
    for model_tag in ("llama31_8b", "qwen25_7b", "gemma2_9b", "mistral7b", "olmo2_7b"):
        r = analyze_model(model_tag)
        if r is not None:
            results[model_tag] = r

    if len(results) == 2:
        print(f"\n{'=' * 84}")
        print("Cross-model contrast")
        print(f"{'=' * 84}")
        for m, r in results.items():
            mw = r["mann_whitney"]
            print(f"  {m:<12s}  within-cluster {mw['within_cluster_mean']:+.4f}  vs  "
                  f"outside {mw['outside_mean']:+.4f}   "
                  f"(diff {mw['diff_within_minus_outside']:+.4f}, p={mw['p_one_sided']:.4g})")
        print("\n  Expectation: Qwen significant (real cluster); Llama n.s. "
              "(same 4-bias set is\n  not derived from Llama's data -- genuine negative control).")


if __name__ == "__main__":
    main()
