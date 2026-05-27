"""
Cross-bias cosine similarity analysis with bootstrap CIs and hierarchical clustering.

For the 6 per-bias averaged directions (computed from per-source unit directions):
    1. Compute point-estimate 6x6 cosine matrix
    2. Bootstrap 95% CI for each cell by resampling source datasets within each bias
    3. Hierarchical clustering (average linkage, distance = 1 - cosine)
    4. Reference: random direction expected cosine (theoretical: 0 +/- 1/sqrt(dim))
    5. Reference: within-bias mean cosine (from cosine_within_bias.json)

The singleton bias spurious_few_shot_hindsight gets the same direction every bootstrap
iteration (only 1 source to resample), so its CI width is 0. This is noted as a
limitation in the saved output.

Outputs:
    paper/cache/{model_tag}/cross_bias_cosine.npz
        cos_matrix   (n_biases, n_biases) point estimate
        ci_lower     (n_biases, n_biases) bootstrap 2.5%
        ci_upper     (n_biases, n_biases) bootstrap 97.5%
        bias_names   (n_biases,)
        linkage      (n_biases-1, 4) scipy linkage matrix for dendrogram

    paper/cache/{model_tag}/cross_bias_cosine.json   formatted summary
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

CACHE = Path(__file__).resolve().parent / "cache"
N_BOOTSTRAP = 1000
SEED = 42


def normalize_rows(M, eps=1e-8):
    """Normalize each row of M to unit L2 norm."""
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    return M / (norms + eps)


def compute_bias_directions(per_source_directions, bias_to_indices, bias_order, rng=None,
                            bootstrap=False):
    """Compute averaged per-bias directions.

    If bootstrap=True, sample source indices for each bias with replacement before averaging.
    """
    out = []
    for bias in bias_order:
        indices = bias_to_indices[bias]
        if bootstrap and len(indices) > 1:
            sampled = rng.choice(indices, size=len(indices), replace=True)
        else:
            sampled = np.array(indices)
        avg = per_source_directions[sampled].mean(axis=0)
        norm = np.linalg.norm(avg) + 1e-8
        out.append(avg / norm)
    return np.stack(out)


def analyze_model(model_tag):
    src_npz = CACHE / model_tag / "directions.npz"
    d = np.load(src_npz, allow_pickle=True)
    per_source = d["directions"].astype(np.float32)  # (n_src, n_dim)
    bias_names = [str(b) for b in d["bias_names"]]
    source_names = [str(s) for s in d["source_names"]]

    bias_to_indices = defaultdict(list)
    for i, b in enumerate(bias_names):
        bias_to_indices[b].append(i)
    bias_order = sorted(bias_to_indices.keys())

    print(f"\n{'=' * 100}")
    print(f"Cross-bias cosine analysis: {model_tag}")
    print(f"{'=' * 100}")
    print(f"\nBiases ({len(bias_order)}):")
    for b in bias_order:
        n = len(bias_to_indices[b])
        marker = "  (singleton — CI undefined)" if n == 1 else ""
        print(f"  {b:<35s}  N_datasets = {n}{marker}")

    n_biases = len(bias_order)

    # === Point estimate ===
    point_dirs = compute_bias_directions(per_source, bias_to_indices, bias_order)
    cos_mat = point_dirs @ point_dirs.T  # (n_biases, n_biases)

    # === Bootstrap CIs ===
    rng = np.random.default_rng(SEED)
    boot_cosines = np.zeros((N_BOOTSTRAP, n_biases, n_biases), dtype=np.float32)
    for k in range(N_BOOTSTRAP):
        sampled_dirs = compute_bias_directions(
            per_source, bias_to_indices, bias_order, rng=rng, bootstrap=True,
        )
        boot_cosines[k] = sampled_dirs @ sampled_dirs.T

    ci_lower = np.percentile(boot_cosines, 2.5, axis=0)
    ci_upper = np.percentile(boot_cosines, 97.5, axis=0)

    # === Print full matrix with CIs ===
    print(f"\nPoint-estimate cosine matrix with bootstrap 95% CI [lo, hi]:")
    print(f"{'':40s}", end="")
    for b in bias_order:
        print(f"  {b[:14]:>14s}", end="")
    print()
    for i, bi in enumerate(bias_order):
        print(f"  {bi[:38]:<38s}", end="")
        for j, bj in enumerate(bias_order):
            if i == j:
                print(f"  {'  1.000':>14s}", end="")
            else:
                c = cos_mat[i, j]
                lo = ci_lower[i, j]
                hi = ci_upper[i, j]
                # Compact format
                print(f"  {c:+.2f}[{lo:+.2f},{hi:+.2f}]"[:15].rjust(15), end="")
        print()

    # === Hierarchical clustering ===
    # distance = 1 - cosine (only off-diagonal, condensed form)
    dist = 1.0 - cos_mat
    # Make sure diagonal is exactly 0 and matrix is symmetric (numerical cleanup)
    np.fill_diagonal(dist, 0.0)
    dist = 0.5 * (dist + dist.T)
    # Clip tiny negatives that can arise from cosines > 1 due to fp drift
    dist = np.clip(dist, 0.0, 2.0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    print(f"\nHierarchical clustering (average linkage on 1 - cos):")
    print(f"  linkage matrix shape: {Z.shape}")
    # Print merge sequence
    print(f"  Merge sequence (cluster indices: 0..{n_biases-1} = bias_order; higher = merged clusters):")
    for k, (a, b, dist, n_in) in enumerate(Z):
        def desc(x):
            x = int(x)
            return bias_order[x] if x < n_biases else f"cluster_{x}"
        print(f"    step {k+1}: merge [{desc(a)}] + [{desc(b)}] at dist={dist:.3f} (n={int(n_in)})")

    # === Highest and lowest off-diagonal cosines ===
    pairs = []
    for i in range(n_biases):
        for j in range(i + 1, n_biases):
            pairs.append((bias_order[i], bias_order[j],
                          float(cos_mat[i, j]), float(ci_lower[i, j]), float(ci_upper[i, j])))
    pairs_sorted = sorted(pairs, key=lambda x: -x[2])
    print(f"\nMost similar bias pairs (top 3):")
    for bi, bj, c, lo, hi in pairs_sorted[:3]:
        print(f"  {bi:<32s} <-> {bj:<32s}  cos = {c:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
    print(f"Most dissimilar bias pairs (bottom 3):")
    for bi, bj, c, lo, hi in pairs_sorted[-3:]:
        print(f"  {bi:<32s} <-> {bj:<32s}  cos = {c:+.3f}  [{lo:+.3f}, {hi:+.3f}]")

    # === Compare to within-bias mean cosine (from earlier analysis) ===
    within_summary_path = CACHE / model_tag / "cosine_within_bias.json"
    within_mean = None
    if within_summary_path.exists():
        with open(within_summary_path) as f:
            within_summary = json.load(f)
        within_mean = within_summary["within"]["mean"]
        across_mean = within_summary["across"]["mean"]
        print(f"\nReference baselines (from earlier within-bias cosine analysis):")
        print(f"  source-level within-bias  mean cosine: {within_mean:+.3f}")
        print(f"  source-level across-bias  mean cosine: {across_mean:+.3f}")

    # Off-diagonal cross-bias cosines summary
    off_diag = np.array([cos_mat[i, j] for i in range(n_biases) for j in range(i + 1, n_biases)])
    print(f"\nCross-bias (per-bias-direction) cosine summary:")
    print(f"  off-diagonal mean:  {off_diag.mean():+.3f}")
    print(f"  off-diagonal median: {np.median(off_diag):+.3f}")
    print(f"  off-diagonal range: [{off_diag.min():+.3f}, {off_diag.max():+.3f}]")

    # === Save ===
    out_dir = CACHE / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "cross_bias_cosine.npz"
    np.savez(npz_path,
             cos_matrix=cos_mat.astype(np.float32),
             ci_lower=ci_lower.astype(np.float32),
             ci_upper=ci_upper.astype(np.float32),
             bias_names=np.array(bias_order),
             linkage=Z.astype(np.float32),
             n_bootstrap=np.int32(N_BOOTSTRAP))

    json_summary = {
        "model": model_tag,
        "n_biases": n_biases,
        "bias_names": bias_order,
        "n_sources_per_bias": {b: len(bias_to_indices[b]) for b in bias_order},
        "cos_matrix_point_estimate": {
            f"{bi}__vs__{bj}": float(cos_mat[i, j])
            for i, bi in enumerate(bias_order) for j, bj in enumerate(bias_order) if i < j
        },
        "ci_95_percent": {
            f"{bi}__vs__{bj}": [float(ci_lower[i, j]), float(ci_upper[i, j])]
            for i, bi in enumerate(bias_order) for j, bj in enumerate(bias_order) if i < j
        },
        "off_diagonal_summary": {
            "mean": float(off_diag.mean()),
            "median": float(np.median(off_diag)),
            "min": float(off_diag.min()),
            "max": float(off_diag.max()),
        },
        "most_similar_pairs_top3": [
            {"pair": f"{bi}-vs-{bj}", "cos": c, "ci": [lo, hi]}
            for bi, bj, c, lo, hi in pairs_sorted[:3]
        ],
        "most_dissimilar_pairs_bottom3": [
            {"pair": f"{bi}-vs-{bj}", "cos": c, "ci": [lo, hi]}
            for bi, bj, c, lo, hi in pairs_sorted[-3:]
        ],
        "linkage_merges": [
            {"step": k + 1,
             "a_idx": int(Z[k, 0]),
             "b_idx": int(Z[k, 1]),
             "dist": float(Z[k, 2]),
             "n_in_cluster": int(Z[k, 3])}
            for k in range(len(Z))
        ],
    }
    if within_mean is not None:
        json_summary["reference_baselines"] = {
            "source_level_within_bias_mean_cos": within_mean,
            "source_level_across_bias_mean_cos": across_mean,
        }
    with open(out_dir / "cross_bias_cosine.json", "w") as f:
        json.dump(json_summary, f, indent=2)
    print(f"\n  Saved: {npz_path}")
    print(f"  Saved: {out_dir / 'cross_bias_cosine.json'}")


def main():
    for model_tag in ("llama31_8b", "qwen25_7b", "gemma2_9b", "mistral7b", "olmo2_7b"):
        analyze_model(model_tag)


if __name__ == "__main__":
    main()
