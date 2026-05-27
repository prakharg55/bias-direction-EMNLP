"""
Selectivity check: causal intervention on UNBIASED prompts.

Mirrors `intervene.py` but tokenizes the unbiased version of each pair instead of
the biased version. For each LODO test point (B, H) and each held-out filtered
pair (flipped + resisted), we:

  1. Take the unbiased prompt for that pair (the prompt with no bias framing).
  2. Forward-pass through the model with the same per-layer LODO bias direction
     hooked into every transformer layer's last-token hidden state, subtracting
     alpha * d_ell.
  3. Compare the resulting argmax letter to the correct letter.

By construction of the filtered set, P(correct on unbiased prompt, no intervention)
= 1.0. So the metric we care about here is:

  P(correct on unbiased prompt WITH intervention)

which should stay near 1.0 if intervention is bias-specific (selective), and will
drop substantially if intervention damages general capability.

Output: cache/{model_tag}/selectivity_results.csv  one row per (source, alpha, dir_type)

Usage:
    python intervene_selectivity.py --model llama31_8b --alpha 0.5
    python intervene_selectivity.py --model qwen25_7b --alpha 4.0
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.special import logsumexp
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import LETTERS, get_letter_token_ids, load_bct_source
from intervene import (
    MODEL_NAMES, MODEL_LOAD, BATCH_SIZE, SEED,
    make_hook, get_transformer_layers, batch_forward_logits,
    letter_logits_from_batch, split_direction_per_layer,
)

REPO = Path(__file__).resolve().parents[1]
DUMPS = REPO / "dataset_dumps" / "test"
CACHE = Path(__file__).resolve().parent / "cache"


@torch.no_grad()
def evaluate_on_unbiased(model, tokenizer, input_ids_list, correct_letters, bias_targets,
                         categories, direction_layers, alpha, letter_token_ids,
                         layers_to_hook=None, all_tokens=False):
    """Forward pass on unbiased prompts with intervention hook.

    layers_to_hook: iterable of layer indices to register hooks at. If None,
    hooks every transformer layer (the all-layers protocol).

    all_tokens=True selects the LiReF protocol (modify every residual stream
    position at the hooked layer(s)). Default modifies only the last token.
    """
    n_layers = len(direction_layers)
    if layers_to_hook is None:
        layers_to_hook = range(n_layers)
    handles = []
    if alpha != 0.0:
        for ell in layers_to_hook:
            d_layer = direction_layers[ell]
            handles.append(get_transformer_layers(model)[ell].register_forward_hook(
                make_hook(d_layer, alpha, all_tokens=all_tokens)))
    try:
        full_logits = batch_forward_logits(model, tokenizer, input_ids_list)
    finally:
        for h in handles:
            h.remove()
    letter_log = letter_logits_from_batch(full_logits, letter_token_ids)
    pred_letters = [LETTERS[i] for i in letter_log.argmax(axis=1)]
    n = len(pred_letters)

    flipped_idx = [i for i, c in enumerate(categories) if c == "flipped"]
    resisted_idx = [i for i, c in enumerate(categories) if c == "resisted"]

    overall_correct = sum(1 for i in range(n) if pred_letters[i] == correct_letters[i])
    overall_target = sum(1 for i in range(n) if pred_letters[i] == bias_targets[i])

    flip_correct = sum(1 for i in flipped_idx if pred_letters[i] == correct_letters[i])
    rsst_correct = sum(1 for i in resisted_idx if pred_letters[i] == correct_letters[i])

    return {
        "n_pairs": n,
        "n_flipped": len(flipped_idx),
        "n_resisted": len(resisted_idx),
        "unbiased_acc": overall_correct / n,
        "unbiased_target_rate": overall_target / n,
        "unbiased_acc_flipped": flip_correct / len(flipped_idx) if flipped_idx else float("nan"),
        "unbiased_acc_resisted": rsst_correct / len(resisted_idx) if resisted_idx else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_NAMES.keys()))
    parser.add_argument("--alpha", type=float, required=True,
                        help="Single alpha value to evaluate (sweet spot per model).")
    parser.add_argument("--max_sources", type=int, default=None)
    parser.add_argument("--single_layer", type=int, default=None,
                        help="If set, hook ONLY this layer. Otherwise hook all layers.")
    parser.add_argument("--all_tokens", action="store_true",
                        help="Modify ALL residual stream positions at each hooked layer "
                             "(LiReF protocol). Default modifies only the last token.")
    args = parser.parse_args()

    model_tag = args.model
    alpha = args.alpha
    model_name = MODEL_NAMES[model_tag]

    print(f"[load] {model_name}", flush=True)
    # Mirror intervene.py: apply MODEL_LOAD overrides so gemma gets bf16+eager
    # (fp16 overflows; sdpa skips soft-capping), olmo gets bf16, and base models
    # reuse their instruct tokenizer for byte-identical prompts.
    load_cfg = MODEL_LOAD.get(model_tag, {})
    tokenizer = AutoTokenizer.from_pretrained(load_cfg.get("tokenizer", model_name))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs = dict(torch_dtype=load_cfg.get("dtype", torch.float16), device_map="auto")
    if "attn" in load_cfg:
        model_kwargs["attn_implementation"] = load_cfg["attn"]
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()

    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    print(f"[arch] n_layers={n_layers}, hidden_size={hidden_size}", flush=True)
    print(f"[plan] alpha={alpha}, evaluating intervention on UNBIASED prompts", flush=True)

    letter_token_ids = get_letter_token_ids(tokenizer)

    d_npz = np.load(CACHE / model_tag / "directions.npz", allow_pickle=True)
    per_source_dirs = d_npz["directions"].astype(np.float32)
    source_names = [str(s) for s in d_npz["source_names"]]
    bias_names = [str(b) for b in d_npz["bias_names"]]
    bias_to_indices = defaultdict(list)
    for i, b in enumerate(bias_names):
        bias_to_indices[b].append(i)

    rng = np.random.default_rng(SEED)
    random_full = rng.standard_normal(per_source_dirs.shape[1]).astype(np.float32)
    random_full /= np.linalg.norm(random_full)
    random_layers = split_direction_per_layer(random_full, n_layers, hidden_size)

    test_list = []
    for bias_type, indices in bias_to_indices.items():
        if len(indices) < 2:
            continue
        for held_out_idx in indices:
            test_list.append((bias_type, held_out_idx))
    if args.max_sources is not None:
        test_list = test_list[:args.max_sources]
    print(f"[plan] {len(test_list)} LODO test points  x  2 directions (real, random)", flush=True)

    out_dir = CACHE / model_tag
    out_csv = out_dir / "selectivity_results.csv"
    tok_suffix = "_all_tokens" if args.all_tokens else ""
    mode_str = "all_layers" if args.single_layer is None else f"layer_{args.single_layer}{tok_suffix}"
    fields = [
        "model", "bias", "held_out_source", "mode", "alpha", "direction_type",
        "n_pairs", "n_flipped", "n_resisted",
        "unbiased_acc", "unbiased_target_rate",
        "unbiased_acc_flipped", "unbiased_acc_resisted",
    ]
    if not out_csv.exists():
        out_csv.write_text(",".join(fields) + "\n")
    else:
        # Migrate legacy header (no 'mode' column) by inserting mode=all_layers
        with open(out_csv) as f:
            first_line = f.readline().strip()
        legacy_header = first_line.split(",")
        if "mode" not in legacy_header:
            print(f"[migrate] adding mode column to legacy CSV (assigning 'all_layers' to existing rows)", flush=True)
            with open(out_csv) as f:
                lines = f.readlines()
            new_lines = [",".join(fields) + "\n"]
            for line in lines[1:]:
                parts = line.strip().split(",")
                legacy_row = dict(zip(legacy_header, parts))
                legacy_row["mode"] = "all_layers"
                new_lines.append(",".join(str(legacy_row[k]) for k in fields) + "\n")
            out_csv.write_text("".join(new_lines))

    done = set()
    with open(out_csv) as f:
        header = next(f).strip().split(",")
        for line in f:
            parts = line.strip().split(",")
            row = dict(zip(header, parts))
            done.add((row["held_out_source"], row["mode"], row["alpha"], row["direction_type"]))
    print(f"[resume] {len(done)} rows already saved", flush=True)

    t0 = time.time()
    for src_i, (bias_type, held_out_idx) in enumerate(test_list):
        held_out_src = source_names[held_out_idx]
        print(f"\n[{src_i + 1}/{len(test_list)}] {bias_type} | held-out={held_out_src} | elapsed={time.time() - t0:.0f}s", flush=True)

        other = [i for i in bias_to_indices[bias_type] if i != held_out_idx]
        d_avg = per_source_dirs[other].mean(axis=0)
        d_avg = d_avg / (np.linalg.norm(d_avg) + 1e-8)
        d_layers_real = split_direction_per_layer(d_avg, n_layers, hidden_size)

        parts = held_out_src.split("__")
        bias_name = "__".join(parts[:-1])
        dataset_name = parts[-1]
        jsonl_path = DUMPS / bias_name / f"{dataset_name}_{bias_name}.jsonl"
        if not jsonl_path.exists():
            jsonl_path = next(DUMPS.glob(f"{bias_name}/*{dataset_name}*"), None)
        rows, _n_skipped = load_bct_source(jsonl_path)

        pairs_meta_all = json.load(open(CACHE / model_tag / held_out_src / "pairs_meta_all.json"))
        if len(rows) != len(pairs_meta_all):
            print(f"  WARN: rows ({len(rows)}) != meta ({len(pairs_meta_all)}); skipping", flush=True)
            continue

        keep = [i for i, m in enumerate(pairs_meta_all) if m["category"] in ("flipped", "resisted")]
        kept_rows = [rows[i] for i in keep]
        kept_meta = [pairs_meta_all[i] for i in keep]
        print(f"  pairs to evaluate: {len(kept_rows)} (flipped={sum(1 for m in kept_meta if m['category']=='flipped')}, resisted={sum(1 for m in kept_meta if m['category']=='resisted')})", flush=True)

        # Tokenize UNBIASED messages (key difference vs intervene.py)
        input_ids_list = []
        for r in kept_rows:
            inputs = tokenizer.apply_chat_template(
                r["unbiased_messages"], add_generation_prompt=True, return_tensors="pt",
            )
            ids = inputs[0] if isinstance(inputs, torch.Tensor) else inputs["input_ids"][0]
            input_ids_list.append(ids)
        correct_letters = [m["correct_letter"] for m in kept_meta]
        bias_targets = [m["bias_target"] for m in kept_meta]
        categories = [m["category"] for m in kept_meta]

        layers_to_hook = [args.single_layer] if args.single_layer is not None else None
        with open(out_csv, "a") as fout:
            for direction_type, d_layers in [("real", d_layers_real), ("random", random_layers)]:
                key = (held_out_src, mode_str, f"{alpha}", direction_type)
                if key in done:
                    print(f"  [skip {direction_type}: already done]", flush=True)
                    continue
                t_run = time.time()
                metrics = evaluate_on_unbiased(
                    model, tokenizer, input_ids_list, correct_letters, bias_targets,
                    categories, d_layers, alpha, letter_token_ids,
                    layers_to_hook=layers_to_hook,
                    all_tokens=args.all_tokens,
                )
                row = {
                    "model": model_tag,
                    "bias": bias_type,
                    "held_out_source": held_out_src,
                    "mode": mode_str,
                    "alpha": alpha,
                    "direction_type": direction_type,
                    **metrics,
                }
                fout.write(",".join(str(row[k]) for k in fields) + "\n")
                fout.flush()
                done.add(key)
                print(f"  {direction_type}: {time.time() - t_run:.1f}s  unbiased_acc={metrics['unbiased_acc']:.3f}", flush=True)
        print(f"  finished {held_out_src} (cumulative elapsed {time.time() - t0:.0f}s)", flush=True)

    print(f"\n[done] total elapsed: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
