"""
Compute the per-source bias direction using the strict-flipped contrast.

For each (bias, dataset) source in cache/{model_tag}/:
  flipped indices: pairs where unbiased predicted correct AND biased predicted bias_target
  resisted indices: pairs where unbiased and biased both predicted correct
  (Pairs with bias_target == correct were excluded at extraction time.)

  direction = normalize( mean(X_biased[flipped]) - mean(X_biased[resisted]) )

where X_biased is the per-layer last-token hidden state, concatenated across all
transformer layers, fp16, shape (n_pairs, n_layers * hidden_size).

The direction is one unit-norm vector of dimension n_layers * hidden_size per source.

Output: cache/{model_tag}/directions.npz with arrays:
    directions     (n_sources, n_dim)   fp32, unit L2 norm
    unnorm_norms   (n_sources,)         fp32, original L2 norm before normalization
    source_names   (n_sources,)         e.g. "suggested_answer__mmlu"
    bias_names     (n_sources,)
    dataset_names  (n_sources,)
    n_flipped      (n_sources,)         int32
    n_resisted     (n_sources,)         int32
    n_dim          ()                   int32

Sources with fewer than MIN_PAIRS_PER_CLASS flipped or resisted pairs are skipped
and logged.
"""
import json
import sys
from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parent / "cache"
MIN_PAIRS_PER_CLASS = 5


def compute_source_direction(src_dir):
    """Returns dict with direction (or None) and sample counts."""
    X = np.load(src_dir / "X_biased.npy")
    pairs = json.load(open(src_dir / "pairs_meta.json"))
    if len(pairs) != X.shape[0]:
        raise RuntimeError(
            f"{src_dir.name}: X_biased rows ({X.shape[0]}) != pairs_meta length ({len(pairs)})"
        )

    flipped_idx = [i for i, p in enumerate(pairs) if p["category"] == "flipped"]
    resisted_idx = [i for i, p in enumerate(pairs) if p["category"] == "resisted"]

    if len(flipped_idx) < MIN_PAIRS_PER_CLASS or len(resisted_idx) < MIN_PAIRS_PER_CLASS:
        return {
            "direction": None,
            "n_flipped": len(flipped_idx),
            "n_resisted": len(resisted_idx),
            "unnorm_norm": 0.0,
        }

    # Compute in fp32 for numerical stability; X is fp16 from extraction.
    flipped_mean = X[flipped_idx].astype(np.float32).mean(axis=0)
    resisted_mean = X[resisted_idx].astype(np.float32).mean(axis=0)
    raw = flipped_mean - resisted_mean
    norm = float(np.linalg.norm(raw))
    direction = raw / (norm + 1e-8)
    return {
        "direction": direction,
        "n_flipped": len(flipped_idx),
        "n_resisted": len(resisted_idx),
        "unnorm_norm": norm,
    }


def compute_for_model(model_tag):
    root = CACHE / model_tag
    if not root.exists():
        print(f"[{model_tag}] cache root does not exist, skipping")
        return

    # A directory is a real (bias x dataset) source only if it carries the
    # extracted arrays. This skips auxiliary dirs (cot_pilot, letter_balanced)
    # that other scripts write into the same cache root.
    sources = sorted(d for d in root.iterdir()
                     if d.is_dir() and (d / "X_biased.npy").exists()
                     and (d / "pairs_meta.json").exists())
    print(f"\n{'=' * 90}")
    print(f"Model: {model_tag}  ({len(sources)} sources)")
    print(f"{'=' * 90}")

    directions = []
    unnorm_norms = []
    source_names = []
    bias_names = []
    dataset_names = []
    n_flipped_list = []
    n_resisted_list = []

    n_dim = None
    print(f"{'source':<55s} {'flip':>6s} {'rsst':>6s} {'norm':>10s}")
    for src_dir in sources:
        bias_name = "__".join(src_dir.name.split("__")[:-1])
        dataset_name = src_dir.name.split("__")[-1]
        r = compute_source_direction(src_dir)
        if r["direction"] is None:
            print(f"  [skip] {src_dir.name:<53s} {r['n_flipped']:>6d} {r['n_resisted']:>6d} {'(insufficient samples)':>22s}")
            continue
        if n_dim is None:
            n_dim = r["direction"].shape[0]
        directions.append(r["direction"])
        unnorm_norms.append(r["unnorm_norm"])
        source_names.append(src_dir.name)
        bias_names.append(bias_name)
        dataset_names.append(dataset_name)
        n_flipped_list.append(r["n_flipped"])
        n_resisted_list.append(r["n_resisted"])
        print(f"  {src_dir.name:<53s} {r['n_flipped']:>6d} {r['n_resisted']:>6d} {r['unnorm_norm']:>10.3f}")

    if not directions:
        print(f"\n[{model_tag}] no valid sources; nothing saved")
        return

    out_path = root / "directions.npz"
    np.savez(
        out_path,
        directions=np.stack(directions).astype(np.float32),
        unnorm_norms=np.array(unnorm_norms, dtype=np.float32),
        source_names=np.array(source_names),
        bias_names=np.array(bias_names),
        dataset_names=np.array(dataset_names),
        n_flipped=np.array(n_flipped_list, dtype=np.int32),
        n_resisted=np.array(n_resisted_list, dtype=np.int32),
        n_dim=np.int32(n_dim),
    )
    print(f"\n[{model_tag}] saved {len(directions)} directions (dim={n_dim}) to {out_path}")


def main():
    for model_tag in ("llama31_8b", "qwen25_7b", "gemma2_9b", "mistral7b", "olmo2_7b"):
        compute_for_model(model_tag)


if __name__ == "__main__":
    main()
