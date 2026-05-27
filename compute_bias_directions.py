"""
Compute the per-bias averaged direction (unweighted mean of per-source unit directions,
renormalized).

For each bias type B with N_B source datasets:
    d_B = normalize( (1/N_B) * sum of [d_B,d for d in datasets(B)] )

This treats each dataset equally regardless of how many flipped/resisted pairs it has.
The singleton bias (spurious_few_shot_hindsight) ends up with d_B = its single dataset's
direction.

Output: cache/{model_tag}/bias_directions.npz containing:
    directions     (n_biases, n_dim)   fp32, unit L2 norm
    bias_names     (n_biases,)         e.g. 'suggested_answer'
    n_sources      (n_biases,)         number of datasets averaged per bias
    sources_per_bias  list[list[str]]  source names contributing to each bias
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parent / "cache"


def normalize(v, eps=1e-8):
    return v / (np.linalg.norm(v) + eps)


def compute_for_model(model_tag):
    src = CACHE / model_tag / "directions.npz"
    if not src.exists():
        print(f"[{model_tag}] missing directions.npz; skipping")
        return

    d = np.load(src, allow_pickle=True)
    directions = d["directions"]  # (n_src, n_dim)
    bias_names = [str(b) for b in d["bias_names"]]
    source_names = [str(s) for s in d["source_names"]]

    bias_to_indices = defaultdict(list)
    for i, b in enumerate(bias_names):
        bias_to_indices[b].append(i)

    unique_biases = sorted(bias_to_indices.keys())
    bias_directions = []
    n_sources_per_bias = []
    sources_per_bias = []

    print(f"\n{'=' * 80}\n{model_tag}\n{'=' * 80}")
    for bias in unique_biases:
        indices = bias_to_indices[bias]
        avg = np.mean(directions[indices], axis=0)
        bias_d = normalize(avg).astype(np.float32)
        bias_directions.append(bias_d)
        n_sources_per_bias.append(len(indices))
        sources_per_bias.append([source_names[i] for i in indices])
        unnorm = float(np.linalg.norm(np.mean(directions[indices], axis=0)))
        print(f"  {bias:<35s}  n_sources={len(indices)}  "
              f"avg-direction L2 (pre-norm) = {unnorm:.4f}")

    out_path = CACHE / model_tag / "bias_directions.npz"
    np.savez(
        out_path,
        directions=np.stack(bias_directions),
        bias_names=np.array(unique_biases),
        n_sources=np.array(n_sources_per_bias, dtype=np.int32),
        sources_per_bias=np.array(sources_per_bias, dtype=object),
    )
    print(f"\n  Saved {len(bias_directions)} per-bias directions (dim={bias_directions[0].shape[0]}) to {out_path}")


def main():
    for model_tag in ("llama31_8b", "qwen25_7b", "gemma2_9b", "mistral7b", "olmo2_7b"):
        compute_for_model(model_tag)


if __name__ == "__main__":
    main()
