"""
Per-bias debiasing breakdown -- which biases are easier or harder to fix?

For each model at its tau=0.90 operating alpha (from debias_analysis), show
per-bias recovery and preservation. Surfaces existing debias_summary.json data.

Output: cache/per_bias_debias.json + console table.
"""
import json
from pathlib import Path

CACHE = Path(__file__).resolve().parent / "cache"

BIASES = ["suggested_answer", "post_hoc", "distractor_fact",
          "distractor_argument", "wrong_few_shot", "spurious_few_shot_squares"]
MODELS = [("llama31_8b",     "Llama"),
          ("qwen25_7b",      "Qwen"),
          ("gemma2_9b",      "Gemma"),
          ("mistral7b",      "Mistral"),
          ("olmo2_7b",       "OLMo"),
          ("qwen25_7b_base", "Qwen-base")]


def fmt(x):
    return f"{x:.3f}" if isinstance(x, float) else "  -- "


def main():
    db = json.load(open(CACHE / "debias_summary.json"))

    print(f"PER-BIAS RECOVERY at each model's tau>=0.90 operating alpha")
    print(f"(intervention: subtract alpha * LODO bias direction at every layer, real direction)")
    print()
    print(f"{'bias':<28s} " + "".join(f"{m[1]:>10s}" for m in MODELS))
    print("-" * (28 + 10 * len(MODELS)))
    out = {"recovery_by_bias_model": {}}
    for b in BIASES:
        line = f"{b:<28s} "
        out["recovery_by_bias_model"][b] = {}
        for tag, _ in MODELS:
            pb = db.get(tag, {}).get("per_bias_at_tau0.90", {}).get(b)
            r = pb["recovery"] if pb else None
            out["recovery_by_bias_model"][b][tag] = r
            line += f"{fmt(r):>10s}"
        print(line)

    # mean per bias across models (where available)
    print(f"\n{'mean across models':<28s} " + "".join(
        f"{'':>10s}" for _ in MODELS))
    print(f"{'-' * 28}")
    print(f"{'bias':<28s} {'mean':>10s}  {'n_models':>10s}")
    for b in BIASES:
        vals = [v for v in out["recovery_by_bias_model"][b].values()
                if isinstance(v, float)]
        m = sum(vals) / len(vals) if vals else None
        out["recovery_by_bias_model"][b]["__mean_across_models__"] = m
        print(f"{b:<28s} {fmt(m):>10s}  {len(vals):>10d}")

    # also surface preservation
    print(f"\n\nPRESERVATION at the same operating alpha (selectivity / 'do no harm')")
    print(f"{'bias':<28s} " + "".join(f"{m[1]:>10s}" for m in MODELS))
    print("-" * (28 + 10 * len(MODELS)))
    for b in BIASES:
        line = f"{b:<28s} "
        for tag, _ in MODELS:
            pb = db.get(tag, {}).get("per_bias_at_tau0.90", {}).get(b)
            p = pb["preservation"] if pb else None
            line += f"{fmt(p):>10s}"
        print(line)

    out_path = CACHE / "per_bias_debias.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
