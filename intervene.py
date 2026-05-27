"""
Causal intervention via bias-direction subtraction at the last-token position.

For each LODO test point (B, H):
  1. d_LODO = normalize(mean of per-source unit directions for B's datasets != H)
  2. Split d_LODO into per-layer chunks, renormalize each chunk to unit norm
  3. For each layer mode (all_layers or single layer ℓ) and α value:
     - Hook: subtract α · d_ℓ from output[0][:, -1, :] at hooked layer(s)
     - Forward pass on held-out source's biased prompts (flipped + resisted pairs only)
     - Read 16-letter logits via logsumexp over single-token variants
     - argmax per pair, compute metrics
  4. Random direction baseline (1 seed) using same procedure

Primary metric: recovery rate on flipped pairs (= fraction now producing the correct letter).
Selectivity metric: preservation rate on resisted pairs (= fraction still producing correct).

Output: cache/{model_tag}/intervention_results.csv with one row per
(source, mode, alpha, direction_type) combination.

Usage:
    python intervene.py --model llama31_8b
    python intervene.py --model qwen25_7b
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

REPO = Path(__file__).resolve().parents[1]
DUMPS = REPO / "dataset_dumps" / "test"
CACHE = Path(__file__).resolve().parent / "cache"

MODEL_NAMES = {
    "llama31_8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen25_7b":  "Qwen/Qwen2.5-7B-Instruct",
    "gemma2_9b":  "google/gemma-2-9b-it",
    "qwen25_7b_base": "Qwen/Qwen2.5-7B",
    "mistral7b":  "mistralai/Mistral-7B-Instruct-v0.3",
    "olmo2_7b":   "allenai/OLMo-2-1124-7B-Instruct",
}

# Per-model load overrides keyed by model tag (default: fp16, default
# attention, own tokenizer). Gemma-2 needs bf16 + eager attention (fp16
# overflows; flash/sdpa skip its attention soft-capping). The Qwen base model
# reuses the Instruct tokenizer so its prompts match the Instruct run exactly.
MODEL_LOAD = {
    "gemma2_9b":      {"dtype": torch.bfloat16, "attn": "eager"},
    "qwen25_7b_base": {"dtype": torch.float16, "tokenizer": "Qwen/Qwen2.5-7B-Instruct"},
    "olmo2_7b":       {"dtype": torch.bfloat16},
}

# Full bidirectional sweep (paper Section 7). Set as the default so it does not
# have to be passed as a comma-separated CLI/env value (SLURM --export splits on
# commas, which silently truncates a comma-separated alpha list).
ALPHAS = [-8.0, -4.0, -2.0, -1.0, -0.5, -0.25, -0.1, 0.0,
          0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
# Batch size 8: Gemma-2-9B (42 layers, eager attention) OOMs at 16 on a 44 GB
# GPU. Batch size only chunks the forward passes; it does not affect results.
BATCH_SIZE = 8
SEED = 42


def make_hook(d_layer_tensor, alpha, all_tokens=False):
    """Hook subtracts (alpha * d_layer) from the layer output residual stream.

    By default modifies only the last-token position. With all_tokens=True,
    modifies every position in the residual stream (the LiReF protocol from
    Hong et al. 2025: single specific layer, all token positions). Pad positions
    are still touched but are masked out by attention downstream so the effect
    is inert there.
    """
    delta_fp32 = (alpha * d_layer_tensor).float()
    def hook(module, input, output):
        if isinstance(output, tuple):
            hs = output[0]
        else:
            hs = output
        orig_dtype = hs.dtype
        delta = delta_fp32.to(hs.device).to(orig_dtype)
        if all_tokens:
            # Broadcast delta of shape (H,) across (B, T, H)
            hs[:, :, :] = hs[:, :, :] - delta
        else:
            hs[:, -1, :] = hs[:, -1, :] - delta
        if isinstance(output, tuple):
            return (hs,) + output[1:]
        return hs
    return hook


def get_transformer_layers(model):
    """Locate the transformer layer list (works for Llama, Qwen)."""
    # Both Llama and Qwen use model.model.layers
    return model.model.layers


@torch.no_grad()
def batch_forward_logits(model, tokenizer, input_ids_list, batch_size=BATCH_SIZE):
    """Run forward pass in batches with left-padding. Returns (n_pairs, vocab_size) at last position."""
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id
    n = len(input_ids_list)
    vocab_size = model.config.vocab_size
    out_logits = np.empty((n, vocab_size), dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = input_ids_list[start:end]
        max_len = max(ids.shape[0] for ids in batch)
        padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, ids in enumerate(batch):
            L = ids.shape[0]
            padded[i, max_len - L:] = ids
            attn[i, max_len - L:] = 1
        padded = padded.to(device)
        attn = attn.to(device)
        out = model(input_ids=padded, attention_mask=attn, use_cache=False)
        out_logits[start:end] = out.logits[:, -1, :].float().cpu().numpy()
    return out_logits


def letter_logits_from_batch(full_logits, letter_token_ids):
    """For each pair, logsumexp over the letter's token variants. Returns (n, 16)."""
    n = full_logits.shape[0]
    out = np.empty((n, len(LETTERS)), dtype=np.float32)
    for i in range(n):
        for j, L in enumerate(LETTERS):
            ids = letter_token_ids[L]
            out[i, j] = logsumexp([full_logits[i, k] for k in ids])
    return out


def evaluate_setup(model, tokenizer, input_ids_list, correct_letters, bias_targets, categories,
                   direction_layers, alpha, layers_to_hook, letter_token_ids, all_tokens=False):
    """Apply hook, run forward pass, compute metrics. Returns dict.

    all_tokens=True selects the LiReF protocol (modify every residual stream
    position at the hooked layer(s)). Default modifies only the last token.
    """
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

    if flipped_idx:
        flip_recovery = sum(1 for i in flipped_idx if pred_letters[i] == correct_letters[i]) / len(flipped_idx)
        flip_still_target = sum(1 for i in flipped_idx if pred_letters[i] == bias_targets[i]) / len(flipped_idx)
    else:
        flip_recovery = float("nan")
        flip_still_target = float("nan")

    if resisted_idx:
        rsst_preserve = sum(1 for i in resisted_idx if pred_letters[i] == correct_letters[i]) / len(resisted_idx)
        rsst_drift_to_target = sum(1 for i in resisted_idx if pred_letters[i] == bias_targets[i]) / len(resisted_idx)
    else:
        rsst_preserve = float("nan")
        rsst_drift_to_target = float("nan")

    return {
        "n_pairs": n,
        "n_flipped": len(flipped_idx),
        "n_resisted": len(resisted_idx),
        "overall_accuracy": overall_correct / n,
        "overall_target_rate": overall_target / n,
        "flip_recovery": flip_recovery,
        "flip_still_target": flip_still_target,
        "rsst_preservation": rsst_preserve,
        "rsst_drift_to_target": rsst_drift_to_target,
    }


def split_direction_per_layer(d_full, n_layers, hidden_size, renormalize=True):
    """Split a (n_layers * hidden_size,) vector into per-layer unit-norm chunks."""
    out = []
    for ell in range(n_layers):
        chunk = d_full[ell * hidden_size : (ell + 1) * hidden_size].astype(np.float32)
        if renormalize:
            chunk = chunk / (np.linalg.norm(chunk) + 1e-8)
        out.append(torch.tensor(chunk, dtype=torch.float32))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_NAMES.keys()))
    parser.add_argument("--max_sources", type=int, default=None,
                        help="If set, only process first N held-out sources (for testing).")
    parser.add_argument("--alphas", type=str, default=None,
                        help="Comma-separated alphas. Defaults to the full sweep.")
    parser.add_argument("--out_suffix", type=str, default="",
                        help="Suffix for the output CSV (intervention_results<suffix>.csv). "
                             "Used to run disjoint alpha subsets as parallel jobs without "
                             "writing to the same file; merge the pieces afterward.")
    parser.add_argument("--modes", type=str, default="all_and_subset",
                        choices=("all_only", "all_and_subset", "all_and_full_scan"),
                        help="Which intervention modes to run. "
                             "'all_only' = only all_layers (fastest), "
                             "'all_and_subset' = all_layers + every 4th layer, "
                             "'all_and_full_scan' = all_layers + every layer.")
    parser.add_argument("--scan_alpha", type=float, default=2.0,
                        help="Fixed alpha used for the per-layer scan modes (default 2.0).")
    parser.add_argument("--single_layer", type=int, default=None,
                        help="If set, hook ONLY this layer through the full alpha sweep. "
                             "Overrides --modes. Writes rows with mode='layer_N'.")
    parser.add_argument("--all_tokens", action="store_true",
                        help="Modify ALL residual stream positions at each hooked layer "
                             "(LiReF protocol from Hong et al. 2025). Default modifies "
                             "only the last token. Affects the mode name suffix in CSV.")
    args = parser.parse_args()

    model_tag = args.model
    model_name = MODEL_NAMES[model_tag]
    alphas = ALPHAS if args.alphas is None else [float(x) for x in args.alphas.split(",")]
    scan_alpha = args.scan_alpha

    print(f"[load] {model_name}", flush=True)
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

    letter_token_ids = get_letter_token_ids(tokenizer)

    # Load per-source directions
    d_npz = np.load(CACHE / model_tag / "directions.npz", allow_pickle=True)
    per_source_dirs = d_npz["directions"].astype(np.float32)  # (n_src, n_dim)
    source_names = [str(s) for s in d_npz["source_names"]]
    bias_names = [str(b) for b in d_npz["bias_names"]]
    bias_to_indices = defaultdict(list)
    for i, b in enumerate(bias_names):
        bias_to_indices[b].append(i)

    # Random direction (single seed) for baseline
    rng = np.random.default_rng(SEED)
    random_full = rng.standard_normal(per_source_dirs.shape[1]).astype(np.float32)
    random_full /= np.linalg.norm(random_full)
    random_layers = split_direction_per_layer(random_full, n_layers, hidden_size)

    # Build LODO test list
    test_list = []
    for bias_type, indices in bias_to_indices.items():
        if len(indices) < 2:
            continue
        for held_out_idx in indices:
            test_list.append((bias_type, held_out_idx))
    if args.max_sources is not None:
        test_list = test_list[:args.max_sources]
    print(f"[plan] {len(test_list)} LODO test points  x  {n_layers + 1} layer modes  x  {len(alphas)} alphas  x  2 directions", flush=True)

    out_dir = CACHE / model_tag
    out_csv = out_dir / f"intervention_results{args.out_suffix}.csv"
    fields = [
        "model", "bias", "held_out_source", "mode", "alpha", "direction_type",
        "n_pairs", "n_flipped", "n_resisted",
        "overall_accuracy", "overall_target_rate",
        "flip_recovery", "flip_still_target",
        "rsst_preservation", "rsst_drift_to_target",
    ]
    if not out_csv.exists():
        out_csv.write_text(",".join(fields) + "\n")

    # Resume support: track already-saved (source, mode, alpha, dir_type)
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

        # LODO direction
        other = [i for i in bias_to_indices[bias_type] if i != held_out_idx]
        d_avg = per_source_dirs[other].mean(axis=0)
        d_avg = d_avg / (np.linalg.norm(d_avg) + 1e-8)
        d_layers_real = split_direction_per_layer(d_avg, n_layers, hidden_size)

        # Load BCT source + pairs_meta_all to get categories aligned with rows
        parts = held_out_src.split("__")
        bias_name = "__".join(parts[:-1])
        dataset_name = parts[-1]
        jsonl_path = DUMPS / bias_name / f"{dataset_name}_{bias_name}.jsonl"
        if not jsonl_path.exists():
            # try alternate naming for hindsight singleton
            jsonl_path = next(DUMPS.glob(f"{bias_name}/*{dataset_name}*"), None)
        rows, _n_skipped = load_bct_source(jsonl_path)

        pairs_meta_all = json.load(open(CACHE / model_tag / held_out_src / "pairs_meta_all.json"))
        if len(rows) != len(pairs_meta_all):
            print(f"  WARN: rows ({len(rows)}) != meta ({len(pairs_meta_all)}); skipping", flush=True)
            continue

        # Filter to flipped + resisted only
        keep = [i for i, m in enumerate(pairs_meta_all) if m["category"] in ("flipped", "resisted")]
        kept_rows = [rows[i] for i in keep]
        kept_meta = [pairs_meta_all[i] for i in keep]
        print(f"  pairs to evaluate: {len(kept_rows)} (flipped={sum(1 for m in kept_meta if m['category']=='flipped')}, resisted={sum(1 for m in kept_meta if m['category']=='resisted')})", flush=True)

        # Tokenize all biased messages once
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

        # === Build schedule ===
        schedule = []
        tok_suffix = "_all_tokens" if args.all_tokens else ""
        if args.single_layer is not None:
            # Single-layer mode: hook only this one layer, full alpha sweep
            schedule.append((f"layer_{args.single_layer}{tok_suffix}",
                            [args.single_layer], alphas))
        else:
            # All-layer mode: full alpha sweep
            schedule.append(("all_layers", list(range(n_layers)), alphas))
            # Per-layer scan: at scan_alpha only, on a subset or full layer set
            if args.modes == "all_and_subset":
                scan_layers = list(range(0, n_layers, 4))
            elif args.modes == "all_and_full_scan":
                scan_layers = list(range(n_layers))
            else:  # all_only
                scan_layers = []
            for ell in scan_layers:
                schedule.append((f"layer_{ell}", [ell], [scan_alpha]))

        baseline_recorded = False
        baseline_metrics = None

        with open(out_csv, "a") as fout:
            for mode, layers_to_hook, alpha_list in schedule:
                t_mode = time.time()
                for alpha in alpha_list:
                    if alpha == 0.0:
                        if not baseline_recorded:
                            baseline_metrics = evaluate_setup(
                                model, tokenizer, input_ids_list, correct_letters, bias_targets,
                                categories, d_layers_real, 0.0, [], letter_token_ids,
                                all_tokens=args.all_tokens,
                            )
                            baseline_recorded = True
                        for direction_type in ["real", "random"]:
                            key = (held_out_src, mode, f"{alpha}", direction_type)
                            if key in done:
                                continue
                            row = {
                                "model": model_tag,
                                "bias": bias_type,
                                "held_out_source": held_out_src,
                                "mode": mode,
                                "alpha": alpha,
                                "direction_type": direction_type,
                                **baseline_metrics,
                            }
                            fout.write(",".join(str(row[k]) for k in fields) + "\n")
                            done.add(key)
                        continue

                    for direction_type, d_layers in [("real", d_layers_real), ("random", random_layers)]:
                        key = (held_out_src, mode, f"{alpha}", direction_type)
                        if key in done:
                            continue
                        metrics = evaluate_setup(
                            model, tokenizer, input_ids_list, correct_letters, bias_targets,
                            categories, d_layers, alpha, layers_to_hook, letter_token_ids,
                            all_tokens=args.all_tokens,
                        )
                        row = {
                            "model": model_tag,
                            "bias": bias_type,
                            "held_out_source": held_out_src,
                            "mode": mode,
                            "alpha": alpha,
                            "direction_type": direction_type,
                            **metrics,
                        }
                        fout.write(",".join(str(row[k]) for k in fields) + "\n")
                        fout.flush()
                        done.add(key)
                print(f"  {mode}: {time.time() - t_mode:.1f}s", flush=True)
            print(f"  finished {held_out_src} (cumulative elapsed {time.time() - t0:.0f}s)", flush=True)

    print(f"\n[done] total elapsed: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
