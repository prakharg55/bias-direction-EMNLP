"""
Extract non-CoT direct-letter responses for every (bias x dataset) jsonl in BCT
dataset_dumps/test. Hidden states are saved ONLY for pairs that pass the strict-flipped
or strict-resisted filter; logits are saved for all pairs.

Per-pair categorization:
    target_eq_correct  bias_target == ground_truth (bias points at right answer; excluded)
    au_wrong           model picks the wrong letter under unbiased prompt (no flip possible)
    flipped            au == correct AND ab == bias_target  (strict flip)
    resisted           au == correct AND ab == correct      (strict resisted)
    other              au == correct AND ab is neither target nor correct (deviated to a 3rd option)

Cache layout per source:
    paper/cache/{model_tag}/{bias_name}__{dataset_name}/
        X_biased.npy        (n_filtered, n_layers * hidden_size)  fp16  filtered pairs only
        X_unbiased.npy      same
        logits_biased.npy   (n_total, len(LETTERS))               fp32  ALL pairs, logsumexp over variants
        logits_unbiased.npy same
        pairs_meta.json     filtered pairs metadata (matches X arrays index-for-index)
        pairs_meta_all.json all pairs metadata (matches logits arrays; carries category field)
        meta.json           source-level counts + model arch + letter token id map

Multi-turn biases:
    post_hoc              3-turn dialogue: a fake assistant turn pre-commits to a
                          wrong option, then the user asks for the answer. In the
                          non-CoT setting this is a self-anchoring bias (no rationale
                          to write). The reformatter in utils.py already strips the
                          trailing CoT instruction and leaves the assistant turn
                          intact, so it flows through the standard pipeline.

Skipped biases:
    positional_bias       LLM-as-judge format (not MCQ)
    are_you_sure          multi-turn dialogue with NOT_X targets (separate handling)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import logsumexp
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    LETTERS, get_letter_token_ids, load_bct_source,
    is_strict_flip, is_strict_nonflip,
)

REPO = Path(__file__).resolve().parents[1]
DUMPS = REPO / "dataset_dumps" / "test"
CACHE = Path(__file__).resolve().parent / "cache"

# Skipped: positional_bias is LLM-as-judge (not MCQ).
# Skipped: are_you_sure is a multi-turn dialogue with NOT_X negation targets that
# needs separate handling. post_hoc IS included: it is multi-turn but carries a
# concrete biased_option, so strict-flip categorization works unchanged, and the
# utils.py reformatter already handles its 3-turn structure.
SKIP_BIASES = {"positional_bias", "are_you_sure"}

MODEL_TAGS = {
    "meta-llama/Llama-3.1-8B-Instruct": "llama31_8b",
    "Qwen/Qwen2.5-7B-Instruct":         "qwen25_7b",
    "google/gemma-2-9b-it":             "gemma2_9b",
    "Qwen/Qwen2.5-7B":                  "qwen25_7b_base",
    "meta-llama/Llama-3.1-8B":          "llama31_8b_base",
    "google/gemma-2-9b":                "gemma2_9b_base",
    "mistralai/Mistral-7B-Instruct-v0.3": "mistral7b",
    "mistralai/Mistral-7B-v0.3":          "mistral7b_base",
    "allenai/OLMo-2-1124-7B-Instruct":    "olmo2_7b",
    "allenai/OLMo-2-1124-7B":             "olmo2_7b_base",
}

# Per-model load overrides (default = fp16, default attention, own tokenizer).
# Gemma-2 needs bfloat16 (fp16 overflows on Gemma-2) and eager attention
# (flash/sdpa skip Gemma-2's attention logit soft-capping, which corrupts both
# the hidden states and the logits we read). Each base model reuses its
# Instruct counterpart's tokenizer so base-vs-instruct prompts are byte-
# identical -- the only difference between the two runs is the model weights.
MODEL_LOAD = {
    "google/gemma-2-9b-it": {"dtype": torch.bfloat16, "attn": "eager"},
    "Qwen/Qwen2.5-7B":      {"dtype": torch.float16, "tokenizer": "Qwen/Qwen2.5-7B-Instruct"},
    "meta-llama/Llama-3.1-8B": {"tokenizer": "meta-llama/Llama-3.1-8B-Instruct"},
    "google/gemma-2-9b":   {"dtype": torch.bfloat16, "attn": "eager",
                            "tokenizer": "google/gemma-2-9b-it"},
    "allenai/OLMo-2-1124-7B-Instruct": {"dtype": torch.bfloat16},
    "allenai/OLMo-2-1124-7B": {"dtype": torch.bfloat16,
                               "tokenizer": "allenai/OLMo-2-1124-7B-Instruct"},
    "mistralai/Mistral-7B-v0.3": {"tokenizer": "mistralai/Mistral-7B-Instruct-v0.3"},
}


def get_model(name):
    print(f"Loading model: {name}", flush=True)
    cfg = MODEL_LOAD.get(name, {})
    tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer", name))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    load_kwargs = dict(torch_dtype=cfg.get("dtype", torch.float16), device_map="auto")
    if "attn" in cfg:
        load_kwargs["attn_implementation"] = cfg["attn"]
    model = AutoModelForCausalLM.from_pretrained(name, **load_kwargs)
    model.eval()
    return tok, model


@torch.no_grad()
def forward_pass(tokenizer, model, messages):
    """Return (per-layer last-token hidden state concat, full vocab last-token logits)."""
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt",
    )
    input_ids = inputs.to(model.device) if isinstance(inputs, torch.Tensor) else inputs["input_ids"].to(model.device)
    out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states[1:]  # skip embedding layer
    hidden = torch.cat([h[0, -1, :] for h in hs], dim=0).to(torch.float16).cpu().numpy()
    logits = out.logits[0, -1, :].float().cpu().numpy()
    return hidden, logits


def letter_logits(full_logits, letter_ids):
    """Combine token-variant logits per letter via log-sum-exp.

    Returns array of shape (len(LETTERS),) with the combined log-mass for each letter.
    """
    out = np.empty(len(LETTERS), dtype=np.float32)
    for i, L in enumerate(LETTERS):
        ids = letter_ids[L]
        out[i] = logsumexp([full_logits[j] for j in ids])
    return out


def list_sources():
    """Yield (bias_name, dataset_name, jsonl_path) for every (bias x dataset)."""
    for bias_dir in sorted(DUMPS.iterdir()):
        if not bias_dir.is_dir():
            continue
        bias_name = bias_dir.name
        if bias_name in SKIP_BIASES:
            continue
        for jsonl in sorted(bias_dir.glob("*.jsonl")):
            stem = jsonl.stem
            assert stem.endswith(f"_{bias_name}"), f"unexpected filename: {jsonl}"
            dataset_name = stem[: -len(bias_name) - 1]
            yield bias_name, dataset_name, jsonl


def extract_for_source(tokenizer, model, letter_ids, bias_name, dataset_name, jsonl_path, out_root):
    out_dir = out_root / f"{bias_name}__{dataset_name}"
    expected = ["X_biased.npy", "X_unbiased.npy", "logits_biased.npy",
                "logits_unbiased.npy", "meta.json",
                "pairs_meta.json", "pairs_meta_all.json"]
    if out_dir.exists() and all((out_dir / f).exists() for f in expected):
        print(f"  [skip] {bias_name}__{dataset_name} (already cached)", flush=True)
        return

    pairs, n_skipped = load_bct_source(jsonl_path)
    print(f"\n=== {bias_name}__{dataset_name}: {len(pairs)} pairs (skipped {n_skipped} malformed rows) ===", flush=True)
    if not pairs:
        return

    # Per-pair: always store logits + metadata. Hidden states only for filtered pairs.
    all_pairs_meta = []
    Lb_all, Lu_all = [], []
    Xb_filt, Xu_filt = [], []
    filtered_pairs_meta = []
    n_flipped = n_resisted = n_other = n_au_wrong = n_target_eq_correct = 0

    for i, p in enumerate(pairs):
        try:
            h_b, lg_b = forward_pass(tokenizer, model, p["biased_messages"])
            h_u, lg_u = forward_pass(tokenizer, model, p["unbiased_messages"])
        except Exception as e:
            print(f"  [error pair {i}]: {e}", flush=True)
            continue

        letter_lg_b = letter_logits(lg_b, letter_ids)
        letter_lg_u = letter_logits(lg_u, letter_ids)
        Lb_all.append(letter_lg_b); Lu_all.append(letter_lg_u)

        ab = LETTERS[int(np.argmax(letter_lg_b))]
        au = LETTERS[int(np.argmax(letter_lg_u))]
        t, c = p["bias_target"], p["correct_letter"]

        if t == c:
            category = "target_eq_correct"
            n_target_eq_correct += 1
        elif au != c:
            category = "au_wrong"
            n_au_wrong += 1
        elif is_strict_flip(t, c, ab, au):
            category = "flipped"
            n_flipped += 1
        elif is_strict_nonflip(t, c, ab, au):
            category = "resisted"
            n_resisted += 1
        else:
            category = "other"
            n_other += 1

        meta = {k: v for k, v in p.items()
                if k not in ("biased_messages", "unbiased_messages")}
        meta["biased_pred"] = ab
        meta["unbiased_pred"] = au
        meta["category"] = category
        all_pairs_meta.append(meta)

        if category in ("flipped", "resisted"):
            Xb_filt.append(h_b); Xu_filt.append(h_u)
            filtered_pairs_meta.append(meta)

        if (i + 1) % 50 == 0 or i == len(pairs) - 1:
            print(f"  {i+1}/{len(pairs)}  (flipped={n_flipped} resisted={n_resisted} "
                  f"au_wrong={n_au_wrong} other={n_other} t_eq_c={n_target_eq_correct})", flush=True)

    if not Lb_all:
        print(f"  [warn] no successful forward passes for {bias_name}__{dataset_name}", flush=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "logits_biased.npy", np.stack(Lb_all))
    np.save(out_dir / "logits_unbiased.npy", np.stack(Lu_all))
    if Xb_filt:
        np.save(out_dir / "X_biased.npy", np.stack(Xb_filt))
        np.save(out_dir / "X_unbiased.npy", np.stack(Xu_filt))
    else:
        # No filtered pairs in this source; still create empty arrays for downstream consistency.
        n_layers = model.config.num_hidden_layers
        hidden_size = model.config.hidden_size
        empty = np.zeros((0, n_layers * hidden_size), dtype=np.float16)
        np.save(out_dir / "X_biased.npy", empty)
        np.save(out_dir / "X_unbiased.npy", empty)

    with open(out_dir / "meta.json", "w") as f:
        json.dump({
            "bias_name": bias_name,
            "dataset_name": dataset_name,
            "n_total_pairs": len(all_pairs_meta),
            "n_filtered_pairs": len(filtered_pairs_meta),
            "n_flipped": n_flipped,
            "n_resisted": n_resisted,
            "n_au_wrong": n_au_wrong,
            "n_other": n_other,
            "n_target_eq_correct": n_target_eq_correct,
            "n_skipped_malformed": n_skipped,
            "n_layers": model.config.num_hidden_layers,
            "hidden_size": model.config.hidden_size,
            "letters": LETTERS,
            "letter_token_ids": {k: list(map(int, v)) for k, v in letter_ids.items()},
        }, f, indent=2)
    with open(out_dir / "pairs_meta_all.json", "w") as f:
        json.dump(all_pairs_meta, f, indent=2)
    with open(out_dir / "pairs_meta.json", "w") as f:
        json.dump(filtered_pairs_meta, f, indent=2)
    print(f"  Saved: {len(all_pairs_meta)} total pairs (logits), "
          f"{len(filtered_pairs_meta)} filtered (hidden states) to {out_dir}", flush=True)


def extract_for_source_xlabel(tokenizer, model, letter_ids, bias_name, dataset_name,
                              jsonl_path, out_root, instruct_tag):
    """xlabel mode: extract THIS model's hidden states for exactly the pairs the
    INSTRUCT model (instruct_tag) flipped/resisted on, carrying the instruct label.

    Used to probe base models: base models almost never flip, so they yield no
    flipped/resisted contrast of their own. Here we instead borrow the instruct
    model's labels and ask whether the base model's activations encode that same
    flipped-vs-resisted distinction.

    The output dir has the SAME layout as a normal source (X_biased.npy,
    pairs_meta.json with a 'category' field, etc.), so compute_directions and
    probe_lodo run on it unchanged. The 'category' field carries the INSTRUCT
    label; 'biased_pred'/'unbiased_pred' carry THIS model's own predictions.
    """
    out_dir = out_root / f"{bias_name}__{dataset_name}"
    expected = ["X_biased.npy", "X_unbiased.npy", "logits_biased.npy",
                "logits_unbiased.npy", "meta.json", "pairs_meta.json", "pairs_meta_all.json"]
    if out_dir.exists() and all((out_dir / f).exists() for f in expected):
        print(f"  [skip] {bias_name}__{dataset_name} (already cached)", flush=True)
        return

    pairs, n_skipped = load_bct_source(jsonl_path)

    # Instruct labels for these exact pairs. load_bct_source is deterministic and
    # model-independent, so base pair i corresponds to instruct pair i; we still
    # verify (length + per-index hash) and abort the source on any mismatch rather
    # than risk silently mislabeling.
    instruct_meta_path = CACHE / instruct_tag / f"{bias_name}__{dataset_name}" / "pairs_meta_all.json"
    if not instruct_meta_path.exists():
        print(f"  [skip] {bias_name}__{dataset_name}: no instruct meta at {instruct_meta_path}", flush=True)
        return
    instruct_pma = json.load(open(instruct_meta_path))
    if len(instruct_pma) != len(pairs):
        print(f"  [skip] {bias_name}__{dataset_name}: instruct meta len {len(instruct_pma)} "
              f"!= base pairs {len(pairs)}", flush=True)
        return
    hash_mismatch = sum(1 for i in range(len(pairs))
                        if pairs[i]["original_question_hash"] != instruct_pma[i]["original_question_hash"])
    if hash_mismatch:
        print(f"  [skip] {bias_name}__{dataset_name}: {hash_mismatch} per-index hash mismatches", flush=True)
        return

    selected = [(i, instruct_pma[i]["category"]) for i in range(len(pairs))
                if instruct_pma[i]["category"] in ("flipped", "resisted")]
    print(f"\n=== {bias_name}__{dataset_name}: {len(selected)} instruct-labeled pairs "
          f"(of {len(pairs)}; skipped {n_skipped} malformed) ===", flush=True)
    if not selected:
        print(f"  [warn] no instruct flipped/resisted pairs; skipping", flush=True)
        return

    Xb, Xu, Lb, Lu, meta_rows = [], [], [], [], []
    n_flipped = n_resisted = 0
    for n, (i, label) in enumerate(selected):
        p = pairs[i]
        try:
            h_b, lg_b = forward_pass(tokenizer, model, p["biased_messages"])
            h_u, lg_u = forward_pass(tokenizer, model, p["unbiased_messages"])
        except Exception as e:
            print(f"  [error pair {i}]: {e}", flush=True)
            continue
        letter_lg_b = letter_logits(lg_b, letter_ids)
        letter_lg_u = letter_logits(lg_u, letter_ids)
        Xb.append(h_b); Xu.append(h_u)
        Lb.append(letter_lg_b); Lu.append(letter_lg_u)
        ab = LETTERS[int(np.argmax(letter_lg_b))]
        au = LETTERS[int(np.argmax(letter_lg_u))]
        meta_rows.append({
            "bias_target": p["bias_target"],
            "correct_letter": p["correct_letter"],
            "bias_name": p["bias_name"],
            "original_dataset": p["original_dataset"],
            "original_question_hash": p["original_question_hash"],
            "biased_pred": ab,            # THIS model's own prediction
            "unbiased_pred": au,          # THIS model's own prediction
            "category": label,           # INSTRUCT label -> drives compute_directions/probe_lodo
            "instruct_category": label,
        })
        n_flipped += (label == "flipped")
        n_resisted += (label == "resisted")
        if (n + 1) % 100 == 0 or n == len(selected) - 1:
            print(f"  {n+1}/{len(selected)}  (flipped={n_flipped} resisted={n_resisted})", flush=True)

    if not Xb:
        print(f"  [warn] no successful forward passes for {bias_name}__{dataset_name}", flush=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X_biased.npy", np.stack(Xb))
    np.save(out_dir / "X_unbiased.npy", np.stack(Xu))
    np.save(out_dir / "logits_biased.npy", np.stack(Lb))
    np.save(out_dir / "logits_unbiased.npy", np.stack(Lu))
    with open(out_dir / "pairs_meta.json", "w") as f:
        json.dump(meta_rows, f, indent=2)
    with open(out_dir / "pairs_meta_all.json", "w") as f:
        json.dump(meta_rows, f, indent=2)
    # How often this model's own behavior matches the instruct label: a behavioral
    # sanity number (for a base model this should be LOW on flipped pairs).
    base_match = sum(1 for r in meta_rows
                     if (r["category"] == "flipped" and r["biased_pred"] == r["bias_target"])
                     or (r["category"] == "resisted" and r["biased_pred"] == r["correct_letter"]))
    with open(out_dir / "meta.json", "w") as f:
        json.dump({
            "bias_name": bias_name,
            "dataset_name": dataset_name,
            "mode": "xlabel",
            "instruct_label_source": instruct_tag,
            "n_total_pairs": len(meta_rows),
            "n_filtered_pairs": len(meta_rows),
            "n_flipped": n_flipped,
            "n_resisted": n_resisted,
            "n_this_model_matches_instruct_label": base_match,
            "n_layers": model.config.num_hidden_layers,
            "hidden_size": model.config.hidden_size,
            "letters": LETTERS,
            "letter_token_ids": {k: list(map(int, v)) for k, v in letter_ids.items()},
        }, f, indent=2)
    print(f"  Saved {len(meta_rows)} pairs (flipped={n_flipped} resisted={n_resisted}) to {out_dir}; "
          f"this model matches the instruct label on {base_match}/{len(meta_rows)}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODEL_TAGS.keys()))
    p.add_argument("--cache_root", default=None)
    p.add_argument("--hidden_pairs_from", default=None,
                   help="Instruct cache tag (e.g. llama31_8b). When set, run in xlabel "
                        "mode: extract this model's hidden states only for the pairs the "
                        "instruct model flipped/resisted on, carrying the instruct label. "
                        "Used to probe whether base models encode the bias distinction. "
                        "Pair --cache_root with a distinct dir (e.g. cache/llama31_8b_base_xlabel).")
    args = p.parse_args()

    tag = MODEL_TAGS[args.model]
    out_root = Path(args.cache_root) if args.cache_root else (CACHE / tag)
    out_root.mkdir(parents=True, exist_ok=True)

    sources = list(list_sources())
    print(f"Found {len(sources)} (bias x dataset) sources to extract:")
    for b, d, _ in sources:
        print(f"  - {b}__{d}")

    tokenizer, model = get_model(args.model)
    letter_ids = get_letter_token_ids(tokenizer)
    print(f"\nletter token ids:")
    for L in LETTERS[:8]:
        print(f"  {L}: {letter_ids[L]}")
    print(f"  ... (total {len(LETTERS)} letters)")
    print(f"n_layers={model.config.num_hidden_layers}, hidden_size={model.config.hidden_size}", flush=True)

    if args.hidden_pairs_from:
        print(f"\n*** xlabel mode: borrowing flipped/resisted labels from "
              f"instruct cache '{args.hidden_pairs_from}' ***", flush=True)
    for bias_name, dataset_name, jsonl_path in sources:
        if args.hidden_pairs_from:
            extract_for_source_xlabel(tokenizer, model, letter_ids, bias_name,
                                      dataset_name, jsonl_path, out_root,
                                      args.hidden_pairs_from)
        else:
            extract_for_source(tokenizer, model, letter_ids,
                               bias_name, dataset_name, jsonl_path, out_root)
    print("\nAll done.", flush=True)


if __name__ == "__main__":
    main()
