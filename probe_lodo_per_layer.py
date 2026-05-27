"""
Layer-wise LODO probing analysis.

For each transformer layer l, compute the LODO probing AUROC using only that layer's
hidden-state slice. Tells us where in the model the bias signal is concentrated.

Methodology:
  For each source s, for each layer l:
    d_{s,l} = normalize(d_global[s, l*H : (l+1)*H])
    where d_global[s] is the cached per-source unit direction from directions.npz.
    (Mathematically equivalent to recomputing from X_biased per layer.)

  For each LODO test point (bias B, held-out dataset H), for each layer l:
    d_{B,-H,l} = normalize(mean of [d_{s,l} for s in datasets(B), s != H])
    Project held-out source's X_biased[:, l*H:(l+1)*H] (filtered pairs) onto d_{B,-H,l}
    AUROC(flipped vs resisted)

  Per-layer random baseline: 20 random unit vectors in R^H.

Outputs:
  cache/{model_tag}/layer_wise_lodo.csv      per-(source, layer) AUROC + random baseline
  cache/{model_tag}/layer_wise_summary.json  per-layer aggregated stats + per-bias breakdown
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

CACHE = Path(__file__).resolve().parent / "cache"

MODEL_ARCH = {
    "llama31_8b": {"n_layers": 32, "hidden_size": 4096},
    "qwen25_7b":  {"n_layers": 28, "hidden_size": 3584},
    "gemma2_9b":  {"n_layers": 42, "hidden_size": 3584},
    "qwen25_7b_base": {"n_layers": 28, "hidden_size": 3584},
    "mistral7b":  {"n_layers": 32, "hidden_size": 4096},
    "olmo2_7b":   {"n_layers": 32, "hidden_size": 4096},
    # xlabel base-probe caches: base-model activations, instruct labels. Same
    # architecture as the instruct counterpart (instruction-tuning does not
    # change the architecture).
    "llama31_8b_base_xlabel": {"n_layers": 32, "hidden_size": 4096},
    "qwen25_7b_base_xlabel":  {"n_layers": 28, "hidden_size": 3584},
    "gemma2_9b_base_xlabel":  {"n_layers": 42, "hidden_size": 3584},
    "mistral7b_base_xlabel":  {"n_layers": 32, "hidden_size": 4096},
    "olmo2_7b_base_xlabel":   {"n_layers": 32, "hidden_size": 4096},
}

SEED = 42
N_RANDOM_SEEDS = 20
MIN_TEST_PAIRS_PER_CLASS = 5


def normalize_rows(M, eps=1e-8):
    norms = np.linalg.norm(M, axis=-1, keepdims=True)
    return M / (norms + eps)


def normalize(v, eps=1e-8):
    return v / (np.linalg.norm(v) + eps)


def per_source_per_layer_from_cached(model_tag):
    """Slice cached per-source directions into per-layer chunks and re-normalize each.

    This is mathematically equivalent to recomputing per-layer directions from X_biased.
    Proof:
      d_s_global = normalize(mu_flip_full - mu_rsst_full)
      d_s_global[l*H:(l+1)*H] = (mu_flip_layer - mu_rsst_layer) / |mu_flip_full - mu_rsst_full|
      normalize(d_s_global[l*H:(l+1)*H])
        = (mu_flip_layer - mu_rsst_layer) / |mu_flip_layer - mu_rsst_layer|
        = the per-layer direction computed directly from X_biased

    Returns (n_sources, n_layers, H) array.
    """
    arch = MODEL_ARCH[model_tag]
    n_layers, H = arch["n_layers"], arch["hidden_size"]
    d_npz = np.load(CACHE / model_tag / "directions.npz", allow_pickle=True)
    d_global = d_npz["directions"].astype(np.float32)  # (n_sources, n_layers*H)
    source_names = [str(s) for s in d_npz["source_names"]]

    n_sources = d_global.shape[0]
    out = d_global.reshape(n_sources, n_layers, H)
    # Re-normalize each (source, layer) slice
    out = normalize_rows(out)
    return out, source_names


def generate_random_directions_per_layer(n_layers, hidden_size, n_seeds=N_RANDOM_SEEDS, seed=SEED):
    out = np.zeros((n_seeds, n_layers, hidden_size), dtype=np.float32)
    for seed_idx in range(n_seeds):
        rng = np.random.default_rng(seed + seed_idx)
        for ell in range(n_layers):
            out[seed_idx, ell] = normalize(rng.standard_normal(hidden_size).astype(np.float32))
    return out


def lodo_per_layer(model_tag):
    arch = MODEL_ARCH[model_tag]
    n_layers, H = arch["n_layers"], arch["hidden_size"]

    print(f"\n{'=' * 100}\n{model_tag}\n{'=' * 100}", flush=True)

    print(f"  Loading cached per-source directions and slicing per layer...", flush=True)
    directions, source_names = per_source_per_layer_from_cached(model_tag)
    n_sources = directions.shape[0]
    print(f"    {n_sources} sources, each split into {n_layers} layer chunks of {H} dims", flush=True)

    d_npz = np.load(CACHE / model_tag / "directions.npz", allow_pickle=True)
    bias_names = [str(b) for b in d_npz["bias_names"]]
    bias_to_indices = defaultdict(list)
    for i, b in enumerate(bias_names):
        bias_to_indices[b].append(i)

    print(f"  Generating random baseline directions ({N_RANDOM_SEEDS} seeds x {n_layers} layers)...", flush=True)
    random_directions = generate_random_directions_per_layer(n_layers, H)

    print(f"  Running LODO probing per layer...", flush=True)
    rows = []
    for bias_type, indices in bias_to_indices.items():
        if len(indices) < 2:
            continue
        for held_out_i in indices:
            held_src = source_names[held_out_i]
            other_indices = [i for i in indices if i != held_out_i]

            # Load held-out source data ONCE (not for each layer)
            src_dir = CACHE / model_tag / held_src
            X = np.load(src_dir / "X_biased.npy").astype(np.float32)
            pairs = json.load(open(src_dir / "pairs_meta.json"))
            flip_idx = np.array([i for i, p in enumerate(pairs) if p["category"] == "flipped"])
            rsst_idx = np.array([i for i, p in enumerate(pairs) if p["category"] == "resisted"])
            if len(flip_idx) < MIN_TEST_PAIRS_PER_CLASS or len(rsst_idx) < MIN_TEST_PAIRS_PER_CLASS:
                continue
            keep = np.concatenate([flip_idx, rsst_idx])
            y = np.concatenate([np.ones(len(flip_idx)), np.zeros(len(rsst_idx))]).astype(np.int64)
            X_keep = X[keep]

            for ell in range(n_layers):
                d_lodo = normalize(directions[other_indices, ell].mean(axis=0))

                X_layer = X_keep[:, ell * H : (ell + 1) * H]
                scores = X_layer @ d_lodo
                auroc = float(roc_auc_score(y, scores))

                rand_aurocs = []
                for seed_idx in range(N_RANDOM_SEEDS):
                    rand_scores = X_layer @ random_directions[seed_idx, ell]
                    rand_aurocs.append(float(roc_auc_score(y, rand_scores)))
                rand_mean = float(np.mean(rand_aurocs))

                rows.append({
                    "model": model_tag,
                    "bias": bias_type,
                    "held_out_source": held_src,
                    "layer": ell,
                    "auroc": auroc,
                    "random_auroc": rand_mean,
                    "n_flipped": int(len(flip_idx)),
                    "n_resisted": int(len(rsst_idx)),
                })
            print(f"    {held_src}: 32 layers done", flush=True)

    print(f"  Computed {len(rows)} (source, layer) rows", flush=True)
    return rows


def summarize(rows, model_tag):
    if not rows:
        return None
    n_layers = MODEL_ARCH[model_tag]["n_layers"]
    print(f"\n  Per-layer aggregated stats:", flush=True)
    print(f"  {'layer':>5s} {'mean AUROC':>11s} {'random':>8s} {'lift':>7s} {'min':>7s} {'max':>7s}  {'n':>3s}", flush=True)

    per_layer = {}
    for ell in range(n_layers):
        layer_rows = [r for r in rows if r["layer"] == ell]
        if not layer_rows:
            continue
        a = np.array([r["auroc"] for r in layer_rows])
        rand = np.array([r["random_auroc"] for r in layer_rows])
        per_layer[ell] = {
            "n_test_points": len(layer_rows),
            "mean_auroc": float(a.mean()),
            "median_auroc": float(np.median(a)),
            "std_auroc": float(a.std(ddof=1)) if len(a) >= 2 else 0.0,
            "min_auroc": float(a.min()),
            "max_auroc": float(a.max()),
            "mean_random_auroc": float(rand.mean()),
            "mean_lift": float((a - rand).mean()),
        }
        print(f"  {ell:>5d} {a.mean():>11.3f} {rand.mean():>8.3f} "
              f"{(a-rand).mean():>+7.3f} {a.min():>7.3f} {a.max():>7.3f}  {len(a):>3d}", flush=True)

    best_layer = max(per_layer.keys(), key=lambda l: per_layer[l]["mean_auroc"])
    print(f"\n  Best layer: {best_layer} (mean AUROC = {per_layer[best_layer]['mean_auroc']:.3f}, "
          f"lift = {per_layer[best_layer]['mean_lift']:+.3f})", flush=True)

    bias_breakdown = defaultdict(list)
    for r in rows:
        if r["layer"] == best_layer:
            bias_breakdown[r["bias"]].append(r["auroc"])
    for bias, vals in sorted(bias_breakdown.items()):
        print(f"    {bias:<35s} mean={np.mean(vals):.3f}", flush=True)

    return {
        "model": model_tag,
        "n_layers": n_layers,
        "n_test_points": len(set(r["held_out_source"] for r in rows)),
        "best_layer": best_layer,
        "best_layer_mean_auroc": per_layer[best_layer]["mean_auroc"],
        "best_layer_lift": per_layer[best_layer]["mean_lift"],
        "per_layer": per_layer,
        "per_bias_at_best_layer": {b: float(np.mean(v)) for b, v in bias_breakdown.items()},
    }


def main():
    for model_tag in ("llama31_8b", "qwen25_7b", "gemma2_9b", "mistral7b", "olmo2_7b"):
        rows = lodo_per_layer(model_tag)
        summary = summarize(rows, model_tag)

        out_dir = CACHE / model_tag
        out_csv = out_dir / "layer_wise_lodo.csv"
        keys = ["model", "bias", "held_out_source", "layer", "auroc", "random_auroc",
                "n_flipped", "n_resisted"]
        with open(out_csv, "w") as f:
            f.write(",".join(keys) + "\n")
            for r in rows:
                f.write(",".join(str(r[k]) for k in keys) + "\n")
        print(f"\n  Saved: {out_csv}", flush=True)

        with open(out_dir / "layer_wise_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved: {out_dir / 'layer_wise_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
