"""
Analyze cross-bias causal intervention results.

Builds the 5x5 cross-bias intervention recovery matrix per model:
  rows = source bias (whose direction is used)
  cols = target bias (whose flipped pairs are intervened on)
  cell = mean flip-recovery rate across the target bias's 4 held-out sources

Checks:
  - The diagonal should reproduce Section 7's within-bias recovery.
  - Off-diagonal cells measure causal cross-bias transfer.
  - Random-direction control gives the baseline.

Output: cache/{model_tag}/cross_bias_intervention_summary.json
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parent / "cache"

# Qwen's 4-bias cluster from Section 6.
QWEN_CLUSTER = {"suggested_answer", "distractor_fact", "wrong_few_shot",
                "spurious_few_shot_squares"}


def main():
    for model_tag in ("llama31_8b", "qwen25_7b"):
        csv_path = CACHE / model_tag / "cross_bias_intervention.csv"
        if not csv_path.exists():
            print(f"[skip {model_tag}: no CSV]")
            continue
        df = pd.read_csv(csv_path)
        real = df[df.direction_type == "real"]
        rand = df[df.direction_type == "random"]

        biases = sorted(set(real.source_bias) | set(real.target_bias))
        print(f"\n{'=' * 70}\n{model_tag}  ({len(biases)} biases)\n{'=' * 70}")

        # Build recovery matrix: matrix[source][target] = mean recovery
        matrix = {}
        for sb in biases:
            matrix[sb] = {}
            for tb in biases:
                cell = real[(real.source_bias == sb) & (real.target_bias == tb)]
                matrix[sb][tb] = float(cell.flip_recovery.mean()) if len(cell) else float("nan")

        # Print matrix
        print(f"\nCross-bias recovery matrix (rows=source direction, cols=target pairs):")
        hdr = "  " + "src\\tgt".ljust(26) + "".join(f"{b[:10]:>12s}" for b in biases)
        print(hdr)
        for sb in biases:
            line = "  " + sb.ljust(26)
            for tb in biases:
                v = matrix[sb][tb]
                mark = "*" if sb == tb else " "
                line += f"{v:>11.3f}{mark}"
            print(line)

        # Random baseline (per target bias)
        rand_by_tgt = {}
        for tb in biases:
            cell = rand[rand.target_bias == tb]
            rand_by_tgt[tb] = float(cell.flip_recovery.mean()) if len(cell) else float("nan")
        print(f"\nRandom-direction recovery per target bias:")
        for tb in biases:
            print(f"  {tb:<28s} {rand_by_tgt[tb]:.3f}")

        # Diagonal (within-bias) vs off-diagonal means
        diag = [matrix[b][b] for b in biases if not np.isnan(matrix[b][b])]
        offdiag = [matrix[sb][tb] for sb in biases for tb in biases
                   if sb != tb and not np.isnan(matrix[sb][tb])]
        print(f"\nDiagonal (within-bias) mean recovery:     {np.mean(diag):.3f}")
        print(f"Off-diagonal (cross-bias) mean recovery:  {np.mean(offdiag):.3f}")
        print(f"Random-direction mean recovery:           {np.mean(list(rand_by_tgt.values())):.3f}")

        # Cluster analysis (Qwen): within-cluster off-diagonal transfer
        cluster_biases = [b for b in biases if b in QWEN_CLUSTER]
        if len(cluster_biases) >= 2:
            within_cluster = [matrix[sb][tb] for sb in cluster_biases
                              for tb in cluster_biases
                              if sb != tb and not np.isnan(matrix[sb][tb])]
            non_cluster = [b for b in biases if b not in QWEN_CLUSTER]
            cluster_to_noncluster = [matrix[sb][tb] for sb in cluster_biases
                                     for tb in non_cluster
                                     if not np.isnan(matrix[sb][tb])]
            print(f"\n4-bias cluster analysis:")
            print(f"  within-cluster cross-transfer recovery (mean): {np.mean(within_cluster):.3f}")
            if cluster_to_noncluster:
                print(f"  cluster->non-cluster recovery (mean):          {np.mean(cluster_to_noncluster):.3f}")

        summary = {
            "model": model_tag,
            "biases": biases,
            "recovery_matrix": matrix,
            "random_recovery_per_target": rand_by_tgt,
            "diagonal_mean": float(np.mean(diag)),
            "offdiagonal_mean": float(np.mean(offdiag)),
            "random_mean": float(np.mean(list(rand_by_tgt.values()))),
        }
        out = CACHE / model_tag / "cross_bias_intervention_summary.json"
        with open(out, "w") as f:
            json.dump(summary, f, indent=2, default=float)
        print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
