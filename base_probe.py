"""
Base-model probe analysis.

Closes the gap between the *behavioral* base-vs-instruct finding (base models
rarely follow prompt-induced biases) and a *mechanistic* one: do base models
internally encode the bias distinction even though they do not act on it?

Inputs: the xlabel caches produced by `extract.py --hidden_pairs_from <instruct>`.
Each xlabel cache holds the BASE model's hidden states for exactly the pairs the
INSTRUCT model flipped/resisted on, with the INSTRUCT label in the 'category'
field. So it has the standard source layout and the standard pipeline runs on it.

For each base model this script:
  1. computes per-source bias directions from base activations (compute_directions)
  2. runs LODO probing on the biased-prompt activations (probe_lodo) -- the headline:
     "is the instruct flipped/resisted distinction decodable in base activations?"
  3. runs the layer-wise LODO scan (probe_lodo_per_layer)
  4. CONTROL: repeats the LODO probe on the UNBIASED-prompt activations. If the
     distinction is decodable from X_biased but NOT from X_unbiased, the signal is
     driven by the bias cue, not by question content/difficulty.
  5. cosine-compares the base direction with the instruct direction per bias.

Output: cache/base_probe_summary.json + a console comparison table.

Interpretation:
  base biased-AUROC ~ 0.5            -> base does NOT encode the distinction
                                        => instruction-tuning CREATES the representation
  base biased-AUROC high, >> control -> base encodes it (cue-driven), but does not act
                                        => instruction-tuning ACTIVATES a dormant representation
  base biased-AUROC high ~ control   -> decoding question content, not the bias (caveat)
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compute_directions
import probe_lodo
import probe_lodo_per_layer

CACHE = Path(__file__).resolve().parent / "cache"
MIN_PAIRS = 5            # min flipped/resisted per source to define a direction
MIN_TEST = 5             # min flipped/resisted per held-out source to score
N_RANDOM = 20
GLOBAL_SEED = 42

# (xlabel cache tag, instruct cache tag whose labels it borrowed)
XLABEL_MODELS = [
    ("llama31_8b_base_xlabel", "llama31_8b"),
    ("gemma2_9b_base_xlabel",  "gemma2_9b"),
    ("mistral7b_base_xlabel",  "mistral7b"),
    ("olmo2_7b_base_xlabel",   "olmo2_7b"),
    ("qwen25_7b_base_xlabel",  "qwen25_7b"),   # positive control: this base DOES cave
]


def normalize(v, eps=1e-8):
    return v / (np.linalg.norm(v) + eps)


def load_sources(cache_dir, x_filename):
    """Load every source's X (from x_filename) + flipped/resisted masks.

    Returns {source_name: {"bias", "X", "flip", "rsst"}}.
    """
    out = {}
    for src_dir in sorted(d for d in cache_dir.iterdir() if d.is_dir()):
        xp = src_dir / x_filename
        pm = src_dir / "pairs_meta.json"
        if not xp.exists() or not pm.exists():
            continue
        X = np.load(xp).astype(np.float32)
        pairs = json.load(open(pm))
        if X.shape[0] != len(pairs):
            print(f"  [warn] {src_dir.name}: X rows {X.shape[0]} != pairs {len(pairs)}; skipping")
            continue
        flip = np.array([p["category"] == "flipped" for p in pairs])
        rsst = np.array([p["category"] == "resisted" for p in pairs])
        out[src_dir.name] = {
            "bias": "__".join(src_dir.name.split("__")[:-1]),
            "X": X, "flip": flip, "rsst": rsst,
        }
    return out


def lodo_per_bias(sources):
    """Self-contained LODO probing on whatever X the sources carry.

    Per source: d = normalize(mean(X[flip]) - mean(X[rsst])).
    Per bias with >=2 usable sources: hold out each, project held-out X onto the
    mean of the others' directions, AUROC(flipped vs resisted). Random baseline:
    N_RANDOM unit directions. Returns {bias: {"auroc", "random", "n"}}.
    """
    # per-source directions
    for s in sources.values():
        if s["flip"].sum() >= MIN_PAIRS and s["rsst"].sum() >= MIN_PAIRS:
            d = s["X"][s["flip"]].mean(0) - s["X"][s["rsst"]].mean(0)
            s["dir"] = normalize(d)
        else:
            s["dir"] = None

    by_bias = defaultdict(list)
    for name, s in sources.items():
        by_bias[s["bias"]].append(name)

    rng = np.random.default_rng(GLOBAL_SEED)
    out = {}
    for bias, names in by_bias.items():
        valid = [n for n in names if sources[n]["dir"] is not None]
        if len(valid) < 2:
            continue
        aurocs, rand_aurocs = [], []
        for held in valid:
            s = sources[held]
            if s["flip"].sum() < MIN_TEST or s["rsst"].sum() < MIN_TEST:
                continue
            others = [n for n in valid if n != held]
            d_lodo = normalize(np.mean([sources[n]["dir"] for n in others], axis=0))
            keep = s["flip"] | s["rsst"]
            X = s["X"][keep]
            y = s["flip"][keep].astype(np.int64)
            aurocs.append(float(roc_auc_score(y, X @ d_lodo)))
            for _ in range(N_RANDOM):
                r = normalize(rng.standard_normal(X.shape[1]).astype(np.float32))
                rand_aurocs.append(float(roc_auc_score(y, X @ r)))
        if aurocs:
            out[bias] = {
                "auroc": float(np.mean(aurocs)),
                "random": float(np.mean(rand_aurocs)) if rand_aurocs else float("nan"),
                "n": len(aurocs),
            }
    return out


def per_bias_direction(npz_path):
    """Mean per-bias unit direction from a directions.npz. Returns {bias: vec}."""
    d = np.load(npz_path, allow_pickle=True)
    dirs = d["directions"].astype(np.float32)
    bias_names = [str(b) for b in d["bias_names"]]
    by_bias = defaultdict(list)
    for i, b in enumerate(bias_names):
        by_bias[b].append(i)
    return {b: normalize(dirs[idx].mean(axis=0)) for b, idx in by_bias.items()}


def main():
    summary = {}
    table_rows = []

    for xlabel_tag, instruct_tag in XLABEL_MODELS:
        cache_dir = CACHE / xlabel_tag
        if not cache_dir.exists():
            print(f"\n[{xlabel_tag}] cache not found, skipping")
            continue

        print(f"\n{'#' * 100}\n# {xlabel_tag}   (base activations, labels from {instruct_tag})\n{'#' * 100}")

        # --- 1. directions from base activations (biased prompts) ---
        compute_directions.compute_for_model(xlabel_tag)
        if not (cache_dir / "directions.npz").exists():
            print(f"  [skip {xlabel_tag}] compute_directions produced no directions.npz "
                  f"(no valid sources) -- check the extraction job")
            continue

        # --- 2. headline: LODO probing on biased-prompt activations ---
        results, _, _ = probe_lodo.analyze_model(xlabel_tag)
        lodo_summary = probe_lodo.summarize(results)
        probe_lodo.save_results(xlabel_tag, results, lodo_summary)

        # --- 3. layer-wise scan ---
        try:
            rows = probe_lodo_per_layer.lodo_per_layer(xlabel_tag)
            pl_summary = probe_lodo_per_layer.summarize(rows, xlabel_tag)
            with open(cache_dir / "layer_wise_summary.json", "w") as f:
                json.dump(pl_summary, f, indent=2)
        except Exception as e:
            print(f"  [warn] per-layer scan failed: {e}")

        # --- 4. CONTROL: same LODO probe on unbiased-prompt activations ---
        print(f"\n  [control] LODO on UNBIASED-prompt activations (X_unbiased)...")
        ctrl = lodo_per_bias(load_sources(cache_dir, "X_unbiased.npy"))

        # --- 5. cosine vs instruct direction, per bias ---
        base_dirs = per_bias_direction(cache_dir / "directions.npz")
        instr_dirs = per_bias_direction(CACHE / instruct_tag / "directions.npz")
        cos = {b: float(np.dot(base_dirs[b], instr_dirs[b]))
               for b in base_dirs if b in instr_dirs}

        # instruct LODO numbers for side-by-side
        instr_lodo = json.load(open(CACHE / instruct_tag / "lodo_summary.json"))

        per_bias = {}
        for bias, e in lodo_summary["per_bias"].items():
            instr_e = instr_lodo["per_bias"].get(bias, {})
            row = {
                "bias": bias,
                "instruct_lodo": instr_e.get("mean_lodo_auroc"),
                "base_lodo_biased": e["mean_lodo_auroc"],
                "base_lodo_random": e["mean_random_auroc"],
                "base_lodo_unbiased_control": ctrl.get(bias, {}).get("auroc"),
                "cosine_base_vs_instruct": cos.get(bias),
                "n": e["n_held_out_sources"],
            }
            per_bias[bias] = row
            table_rows.append({"model": xlabel_tag, **row})

        summary[xlabel_tag] = {
            "instruct_tag": instruct_tag,
            "overall_base_lodo_biased": lodo_summary["overall"].get("mean_lodo_auroc"),
            "overall_base_lodo_random": lodo_summary["overall"].get("mean_random_auroc"),
            "overall_instruct_lodo": instr_lodo["overall"].get("mean_lodo_auroc"),
            "per_bias": per_bias,
        }

    # --- comparison table ---
    print(f"\n\n{'=' * 104}")
    print("BASE-PROBE COMPARISON  (does the base model encode the instruct flipped/resisted distinction?)")
    print(f"{'=' * 104}")
    print(f"{'model':<26s} {'bias':<26s} {'instruct':>9s} {'base':>8s} {'base_rand':>10s} "
          f"{'unbiased':>9s} {'cos':>7s}")
    print(f"{'':<26s} {'':<26s} {'LODO':>9s} {'LODO':>8s} {'baseline':>10s} {'control':>9s} {'b/instr':>7s}")
    print("-" * 104)

    def fmt(x):
        return f"{x:.3f}" if isinstance(x, (int, float)) and x is not None else "  -- "

    last = None
    for r in table_rows:
        m = r["model"] if r["model"] != last else ""
        last = r["model"]
        print(f"{m:<26s} {r['bias']:<26s} {fmt(r['instruct_lodo']):>9s} "
              f"{fmt(r['base_lodo_biased']):>8s} {fmt(r['base_lodo_random']):>10s} "
              f"{fmt(r['base_lodo_unbiased_control']):>9s} {fmt(r['cosine_base_vs_instruct']):>7s}")

    print(f"\n{'-' * 104}\nPer-model overall:")
    for tag, s in summary.items():
        print(f"  {tag:<26s} base LODO (biased) = {fmt(s['overall_base_lodo_biased'])}  "
              f"random = {fmt(s['overall_base_lodo_random'])}  "
              f"[instruct LODO = {fmt(s['overall_instruct_lodo'])}]")

    out_path = CACHE / "base_probe_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
