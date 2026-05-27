"""
Cross-bias probing transfer matrix.

For each pair (B_train, B_test) of bias types:
  For each held-out source H in datasets(B_test):
    1. Compute training direction d_train:
       - If H in datasets(B_train): exclude H -> LODO direction (matches Part 1 LODO)
       - Else: full averaged direction from all sources of B_train
    2. Project H's filtered pairs (flipped + resisted) onto d_train
    3. Compute AUROC for separating flipped from resisted
  4. Cell value = mean AUROC over the held-out sources of B_test
  5. CI = 1.96 * (std / sqrt(n_held_out)) [SE-based; falls back to point estimate if n=1]

Diagonal cells (B_train == B_test) reproduce the within-bias LODO from Part 1
(consistency check).

Singleton bias (spurious_few_shot_hindsight, n=1):
  - Diagonal cell skipped (no LODO possible with 1 source)
  - Off-diagonal cells valid (it's the test source, train uses different bias)

Random direction baseline: per-cell mean AUROC over N_RANDOM_SEEDS random unit
directions, same projection + AUROC procedure.

Outputs:
  paper/cache/{model_tag}/cross_bias_probing.npz
    matrix          (n_biases, n_biases)  mean cell AUROC (NaN if undefined)
    ci_lower/upper  (n_biases, n_biases)  SE-based 95% CI
    random_matrix   (n_biases, n_biases)  random direction baseline AUROC
    n_held_out      (n_biases, n_biases)  number of held-out sources averaged
    bias_names      (n_biases,)
  paper/cache/{model_tag}/cross_bias_probing.json   formatted summary
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

CACHE = Path(__file__).resolve().parent / "cache"
N_RANDOM_SEEDS = 50
SEED = 42
MIN_PAIRS_PER_CLASS = 5


def normalize(v, eps=1e-8):
    return v / (np.linalg.norm(v) + eps)


def load_source(model_tag, source_name):
    """Load a source's X_biased (filtered pairs only, kept as fp16) and binary y label."""
    src_dir = CACHE / model_tag / source_name
    X = np.load(src_dir / "X_biased.npy")  # fp16 from extraction
    pairs = json.load(open(src_dir / "pairs_meta.json"))
    assert X.shape[0] == len(pairs), f"{source_name}: X rows != pairs"
    flip_mask = np.array([p["category"] == "flipped" for p in pairs])
    rsst_mask = np.array([p["category"] == "resisted" for p in pairs])
    keep = flip_mask | rsst_mask
    return X[keep], flip_mask[keep].astype(np.int64), int(flip_mask.sum()), int(rsst_mask.sum())


def project(X_fp16, d_fp32):
    """Project an fp16 hidden-state matrix onto an fp32 unit direction; returns fp32 scores."""
    return (X_fp16.astype(np.float32) @ d_fp32)


def analyze_model(model_tag):
    print(f"\n{'=' * 100}")
    print(f"Cross-bias probing transfer: {model_tag}")
    print(f"{'=' * 100}")

    d = np.load(CACHE / model_tag / "directions.npz", allow_pickle=True)
    per_source = d["directions"].astype(np.float32)
    bias_names = [str(b) for b in d["bias_names"]]
    source_names = [str(s) for s in d["source_names"]]
    n_dim = int(d["n_dim"])

    bias_to_indices = defaultdict(list)
    for i, b in enumerate(bias_names):
        bias_to_indices[b].append(i)
    bias_order = sorted(bias_to_indices.keys())
    n_b = len(bias_order)

    # Lazy-load source data: keep fp16 in memory to halve footprint.
    # Random AUROCs and labels don't depend on B_train, so precompute them per source first
    # (load each source once, compute, free the X array).
    print("\nLoading sources + precomputing labels and random baseline AUROC per source...")
    source_meta = {}        # name -> {"y", "n_flip", "n_rsst"}
    source_X = {}           # name -> X (fp16)
    random_auroc_per_source = {}

    rng = np.random.default_rng(SEED)
    random_dirs = []
    for k in range(N_RANDOM_SEEDS):
        local_rng = np.random.default_rng(SEED + 1 + k)
        r = normalize(local_rng.standard_normal(n_dim).astype(np.float32))
        random_dirs.append(r)
    random_dirs = np.stack(random_dirs)  # (N_RANDOM_SEEDS, n_dim) fp32

    for s_name in source_names:
        X, y, n_flip, n_rsst = load_source(model_tag, s_name)
        source_meta[s_name] = {"y": y, "n_flip": n_flip, "n_rsst": n_rsst}
        source_X[s_name] = X  # fp16
        # Skip if can't compute AUROC
        if len(np.unique(y)) < 2 or min(np.bincount(y)) < MIN_PAIRS_PER_CLASS:
            random_auroc_per_source[s_name] = float("nan")
            continue
        # Batched random AUROC: only convert to fp32 within the matmul scope
        X_fp32 = X.astype(np.float32)
        score_mat = X_fp32 @ random_dirs.T  # (n_pairs, N_RANDOM_SEEDS)
        del X_fp32
        rand_aurocs = [float(roc_auc_score(y, score_mat[:, k]))
                       for k in range(N_RANDOM_SEEDS)]
        random_auroc_per_source[s_name] = float(np.mean(rand_aurocs))
    print(f"  Loaded {len(source_X)} sources (kept as fp16 in memory)")

    # Compute matrices
    matrix = np.full((n_b, n_b), np.nan, dtype=np.float32)
    ci_lo = np.full((n_b, n_b), np.nan, dtype=np.float32)
    ci_hi = np.full((n_b, n_b), np.nan, dtype=np.float32)
    random_matrix = np.full((n_b, n_b), np.nan, dtype=np.float32)
    n_held_out = np.zeros((n_b, n_b), dtype=np.int32)

    # Per-cell detailed records
    detail_records = []

    for i, B_train in enumerate(bias_order):
        train_indices = bias_to_indices[B_train]
        for j, B_test in enumerate(bias_order):
            test_indices = bias_to_indices[B_test]
            cell_aurocs = []
            cell_random_aurocs = []
            cell_per_source = []
            for H_test_idx in test_indices:
                H_test = source_names[H_test_idx]
                # Build training indices
                used_train = [k for k in train_indices if k != H_test_idx]
                if not used_train:
                    # Singleton both-sides: skip
                    continue
                d_train = normalize(np.mean(per_source[used_train], axis=0))

                X = source_X[H_test]
                y = source_meta[H_test]["y"]
                if len(np.unique(y)) < 2 or min(np.bincount(y)) < MIN_PAIRS_PER_CLASS:
                    continue

                scores = project(X, d_train)
                auroc = float(roc_auc_score(y, scores))
                cell_aurocs.append(auroc)
                cell_per_source.append({"held_out_source": H_test, "auroc": auroc})

                # Random baseline (precomputed per source, doesn't depend on B_train)
                cell_random_aurocs.append(random_auroc_per_source[H_test])

            if cell_aurocs:
                a = np.array(cell_aurocs)
                matrix[i, j] = float(a.mean())
                n_held_out[i, j] = len(a)
                if len(a) >= 2:
                    se = float(a.std(ddof=1) / np.sqrt(len(a)))
                    ci_lo[i, j] = float(a.mean() - 1.96 * se)
                    ci_hi[i, j] = float(a.mean() + 1.96 * se)
                else:
                    ci_lo[i, j] = matrix[i, j]
                    ci_hi[i, j] = matrix[i, j]
                random_matrix[i, j] = float(np.mean(cell_random_aurocs))
                detail_records.append({
                    "B_train": B_train,
                    "B_test": B_test,
                    "n_held_out": int(len(a)),
                    "mean_auroc": float(matrix[i, j]),
                    "random_auroc": float(random_matrix[i, j]),
                    "lift": float(matrix[i, j] - random_matrix[i, j]),
                    "per_source": cell_per_source,
                })

    # === Print matrix ===
    print(f"\nCross-bias probing AUROC matrix")
    print(f"  rows = B_train (direction source); cols = B_test (held-out flip prediction)")
    print(f"  diagonal = within-bias LODO (consistency check with Part 1)")
    print(f"  random baseline ~ 0.50\n")

    print(f"{'B_train \\ B_test':<35s}", end="")
    for j, b in enumerate(bias_order):
        print(f"  {b[:14]:>14s}", end="")
    print(f"  {'row mean off-diag':>20s}")

    for i, bi in enumerate(bias_order):
        print(f"  {bi:<33s}", end="")
        offdiag = []
        for j, bj in enumerate(bias_order):
            v = matrix[i, j]
            if np.isnan(v):
                print(f"  {'nan':>14s}", end="")
            else:
                marker = "*" if i == j else " "
                print(f"  {marker}{v:>13.3f}", end="")
                if i != j:
                    offdiag.append(v)
        if offdiag:
            print(f"  {np.mean(offdiag):>20.3f}")
        else:
            print(f"  {'(no off-diag)':>20s}")

    print(f"\n  * = diagonal (within-bias LODO)")

    # === Diagonal consistency check vs Part 1 LODO ===
    lodo_path = CACHE / model_tag / "lodo_results.csv"
    if lodo_path.exists():
        # Parse Part 1 results for comparison
        lodo_rows = {}
        with open(lodo_path) as f:
            header = next(f).strip().split(",")
            for line in f:
                parts = line.strip().split(",")
                row = dict(zip(header, parts))
                key = (row["bias"], row["held_out_dataset"])
                lodo_rows[key] = float(row["lodo_auroc"])

        print(f"\nDiagonal consistency vs Part 1 LODO (per-source AUROC):")
        max_diff = 0.0
        for i, b in enumerate(bias_order):
            for rec in detail_records:
                if rec["B_train"] == b and rec["B_test"] == b:
                    for ps in rec["per_source"]:
                        H = ps["held_out_source"].split("__")[-1]
                        cross_auroc = ps["auroc"]
                        part1_auroc = lodo_rows.get((b, H))
                        if part1_auroc is not None:
                            diff = abs(cross_auroc - part1_auroc)
                            max_diff = max(max_diff, diff)
        print(f"  Max diff between cross-bias diagonal and Part 1 LODO AUROC: {max_diff:.6f}")
        print(f"  (should be ~0; both should compute identical LODO AUROCs)")

    # === Off-diagonal summary ===
    off_mask = ~np.eye(n_b, dtype=bool) & ~np.isnan(matrix)
    diag_mask = np.eye(n_b, dtype=bool) & ~np.isnan(matrix)
    print(f"\nOff-diagonal AUROC summary (cross-bias transfer):")
    if off_mask.any():
        off_vals = matrix[off_mask]
        print(f"  n cells: {off_mask.sum()}")
        print(f"  mean: {off_vals.mean():+.3f}")
        print(f"  median: {np.median(off_vals):+.3f}")
        print(f"  range: [{off_vals.min():+.3f}, {off_vals.max():+.3f}]")
    print(f"\nDiagonal AUROC summary (within-bias LODO):")
    if diag_mask.any():
        diag_vals = matrix[diag_mask]
        print(f"  n cells: {diag_mask.sum()}")
        print(f"  mean: {diag_vals.mean():+.3f}")
        print(f"  median: {np.median(diag_vals):+.3f}")

    # === Save ===
    out_dir = CACHE / model_tag
    npz_path = out_dir / "cross_bias_probing.npz"
    np.savez(npz_path,
             matrix=matrix,
             ci_lower=ci_lo,
             ci_upper=ci_hi,
             random_matrix=random_matrix,
             n_held_out=n_held_out,
             bias_names=np.array(bias_order))

    json_summary = {
        "model": model_tag,
        "n_biases": n_b,
        "bias_names": bias_order,
        "matrix": {
            f"{bi}__to__{bj}": {
                "mean_auroc": float(matrix[i, j]) if not np.isnan(matrix[i, j]) else None,
                "ci_95": [float(ci_lo[i, j]), float(ci_hi[i, j])] if not np.isnan(matrix[i, j]) else None,
                "random_auroc": float(random_matrix[i, j]) if not np.isnan(random_matrix[i, j]) else None,
                "n_held_out_sources": int(n_held_out[i, j]),
            }
            for i, bi in enumerate(bias_order)
            for j, bj in enumerate(bias_order)
        },
        "details": detail_records,
    }
    with open(out_dir / "cross_bias_probing.json", "w") as f:
        json.dump(json_summary, f, indent=2)

    print(f"\n  Saved: {npz_path}")
    print(f"  Saved: {out_dir / 'cross_bias_probing.json'}")


def main():
    for model_tag in ("llama31_8b", "qwen25_7b", "gemma2_9b", "mistral7b", "olmo2_7b"):
        analyze_model(model_tag)


if __name__ == "__main__":
    main()
