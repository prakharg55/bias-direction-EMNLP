"""
Cross-bias causal intervention.

The causal analog of the cross-bias probing transfer matrix (Section 6.2).
Probing transfer asks: does bias A's direction *predict* bias B's flipped vs
resisted? Cross-bias intervention asks the stronger causal question: does
subtracting bias A's direction *causally recover* bias B's flipped pairs?

For each (source bias A, target bias B) and each held-out source H of B:
  - Form direction d_A:
      A == B : LODO direction = mean of A's per-source directions excluding H
               (this is exactly the within-bias direction from intervene.py,
               so the diagonal reproduces Section 7's within-bias recovery)
      A != B : mean of ALL of A's per-source directions
  - Split d_A into per-layer chunks, renormalize each to unit norm.
  - Hook every transformer layer, subtract alpha * d_layer at the last-token
    position (identical protocol to intervene.py).
  - Forward-pass H's filtered pairs (flipped + resisted), measure recovery on
    the flipped subset.
  - Random-direction control: one random unit vector per held-out source.

alpha is the per-model sweet spot used throughout Section 7 (Llama 0.5, Qwen 4).
Only the 5 non-singleton biases are used (matches Section 7's 20 LODO test
points); the spurious_few_shot_hindsight singleton is excluded.

Output: cache/{model_tag}/cross_bias_intervention.csv
  one row per (source_bias, target_bias, held_out_source, direction_type)

Usage:
    python intervene_cross_bias.py --model llama31_8b --alpha 0.5
    python intervene_cross_bias.py --model qwen25_7b  --alpha 4.0
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_letter_token_ids, load_bct_source
from intervene import (
    MODEL_NAMES, SEED,
    evaluate_setup, get_transformer_layers, split_direction_per_layer,
)

REPO = Path(__file__).resolve().parents[1]
DUMPS = REPO / "dataset_dumps" / "test"
CACHE = Path(__file__).resolve().parent / "cache"


def normalize(v, eps=1e-8):
    return v / (np.linalg.norm(v) + eps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_NAMES.keys()))
    parser.add_argument("--alpha", type=float, required=True,
                        help="Sweet-spot alpha (Llama 0.5, Qwen 4.0).")
    parser.add_argument("--max_targets", type=int, default=None,
                        help="If set, only process first N target sources (testing).")
    args = parser.parse_args()

    model_tag = args.model
    model_name = MODEL_NAMES[model_tag]
    alpha = args.alpha

    print(f"[load] {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    print(f"[arch] n_layers={n_layers}, hidden_size={hidden_size}, alpha={alpha}", flush=True)

    letter_token_ids = get_letter_token_ids(tokenizer)

    # Per-source directions
    d_npz = np.load(CACHE / model_tag / "directions.npz", allow_pickle=True)
    per_source_dirs = d_npz["directions"].astype(np.float32)
    source_names = [str(s) for s in d_npz["source_names"]]
    bias_names = [str(b) for b in d_npz["bias_names"]]
    bias_to_indices = defaultdict(list)
    for i, b in enumerate(bias_names):
        bias_to_indices[b].append(i)

    # Non-singleton biases only (matches Section 7's 20 LODO test points)
    non_singleton = sorted(b for b, idx in bias_to_indices.items() if len(idx) >= 2)
    print(f"[plan] {len(non_singleton)} non-singleton biases: {non_singleton}", flush=True)

    # Random direction (single seed, same as intervene.py)
    rng = np.random.default_rng(SEED)
    random_full = rng.standard_normal(per_source_dirs.shape[1]).astype(np.float32)
    random_full = normalize(random_full)
    random_layers = split_direction_per_layer(random_full, n_layers, hidden_size)

    out_dir = CACHE / model_tag
    out_csv = out_dir / "cross_bias_intervention.csv"
    fields = [
        "model", "source_bias", "target_bias", "held_out_source",
        "alpha", "direction_type",
        "n_pairs", "n_flipped", "n_resisted",
        "overall_accuracy", "overall_target_rate",
        "flip_recovery", "flip_still_target",
        "rsst_preservation", "rsst_drift_to_target",
    ]
    if not out_csv.exists():
        out_csv.write_text(",".join(fields) + "\n")

    done = set()
    with open(out_csv) as f:
        header = next(f).strip().split(",")
        for line in f:
            parts = line.strip().split(",")
            row = dict(zip(header, parts))
            done.add((row["source_bias"], row["target_bias"],
                      row["held_out_source"], row["direction_type"]))
    print(f"[resume] {len(done)} rows already saved", flush=True)

    # Build target list: (target_bias, held_out_idx)
    target_list = []
    for tb in non_singleton:
        for held_out_idx in bias_to_indices[tb]:
            target_list.append((tb, held_out_idx))
    if args.max_targets is not None:
        target_list = target_list[:args.max_targets]
    print(f"[plan] {len(target_list)} target sources x "
          f"{len(non_singleton)} source biases + 1 random", flush=True)

    t0 = time.time()
    layers_all = list(range(n_layers))
    for tgt_i, (target_bias, held_out_idx) in enumerate(target_list):
        held_out_src = source_names[held_out_idx]
        print(f"\n[{tgt_i + 1}/{len(target_list)}] target={target_bias} "
              f"held-out={held_out_src} | elapsed={time.time() - t0:.0f}s", flush=True)

        # Load held-out source's filtered pairs
        parts = held_out_src.split("__")
        bias_name = "__".join(parts[:-1])
        dataset_name = parts[-1]
        jsonl_path = DUMPS / bias_name / f"{dataset_name}_{bias_name}.jsonl"
        if not jsonl_path.exists():
            jsonl_path = next(DUMPS.glob(f"{bias_name}/*{dataset_name}*"), None)
        rows, _ = load_bct_source(jsonl_path)
        pairs_meta_all = json.load(open(CACHE / model_tag / held_out_src / "pairs_meta_all.json"))
        if len(rows) != len(pairs_meta_all):
            print(f"  WARN: rows ({len(rows)}) != meta ({len(pairs_meta_all)}); skipping", flush=True)
            continue
        keep = [i for i, m in enumerate(pairs_meta_all)
                if m["category"] in ("flipped", "resisted")]
        kept_rows = [rows[i] for i in keep]
        kept_meta = [pairs_meta_all[i] for i in keep]
        n_flip = sum(1 for m in kept_meta if m["category"] == "flipped")
        n_rsst = sum(1 for m in kept_meta if m["category"] == "resisted")
        print(f"  pairs: {len(kept_rows)} (flipped={n_flip}, resisted={n_rsst})", flush=True)

        # Tokenize biased prompts once
        input_ids_list = []
        for r in kept_rows:
            inputs = tokenizer.apply_chat_template(
                r["biased_messages"], add_generation_prompt=True, return_tensors="pt",
            )
            ids = inputs[0] if isinstance(inputs, torch.Tensor) else inputs["input_ids"][0]
            input_ids_list.append(ids)
        correct_letters = [m["correct_letter"] for m in kept_meta]
        bias_targets = [m["bias_target"] for m in kept_meta]
        categories = [m["category"] for m in kept_meta]

        with open(out_csv, "a") as fout:
            # Real directions: one per source bias
            for source_bias in non_singleton:
                key = (source_bias, target_bias, held_out_src, "real")
                if key in done:
                    continue
                if source_bias == target_bias:
                    # Within-bias LODO direction (reproduces Section 7)
                    other = [i for i in bias_to_indices[source_bias] if i != held_out_idx]
                else:
                    # All of source bias's per-source directions
                    other = list(bias_to_indices[source_bias])
                d_avg = normalize(per_source_dirs[other].mean(axis=0))
                d_layers = split_direction_per_layer(d_avg, n_layers, hidden_size)

                t_run = time.time()
                metrics = evaluate_setup(
                    model, tokenizer, input_ids_list, correct_letters, bias_targets,
                    categories, d_layers, alpha, layers_all, letter_token_ids,
                )
                row = {
                    "model": model_tag,
                    "source_bias": source_bias,
                    "target_bias": target_bias,
                    "held_out_source": held_out_src,
                    "alpha": alpha,
                    "direction_type": "real",
                    **metrics,
                }
                fout.write(",".join(str(row[k]) for k in fields) + "\n")
                fout.flush()
                done.add(key)
                print(f"    src={source_bias:<26s} recovery={metrics['flip_recovery']:.3f} "
                      f"({time.time() - t_run:.0f}s)", flush=True)

            # Random-direction control (one per held-out source)
            key = ("(random)", target_bias, held_out_src, "random")
            if key not in done:
                metrics = evaluate_setup(
                    model, tokenizer, input_ids_list, correct_letters, bias_targets,
                    categories, random_layers, alpha, layers_all, letter_token_ids,
                )
                row = {
                    "model": model_tag,
                    "source_bias": "(random)",
                    "target_bias": target_bias,
                    "held_out_source": held_out_src,
                    "alpha": alpha,
                    "direction_type": "random",
                    **metrics,
                }
                fout.write(",".join(str(row[k]) for k in fields) + "\n")
                fout.flush()
                done.add(key)
                print(f"    src=(random)                 recovery={metrics['flip_recovery']:.3f}", flush=True)

    print(f"\n[done] total elapsed: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
