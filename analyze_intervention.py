"""
Analyze intervention_results.csv.

Produces:
  - Per-source summary: best alpha, peak recovery rate, comparison with random
  - Per-bias mean recovery across LODO test points
  - Layer-wise scan: which layers give best recovery when intervened singly?
  - Selectivity: does intervention preserve resisted pairs' correct answers?
  - Saves a JSON summary at cache/{model_tag}/intervention_summary.json
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parent / "cache"


def load_csv(path):
    rows = []
    with open(path) as f:
        header = next(f).strip().split(",")
        for line in f:
            parts = line.strip().split(",")
            d = dict(zip(header, parts))
            for k in ("alpha",):
                d[k] = float(d[k])
            for k in ("n_pairs", "n_flipped", "n_resisted"):
                d[k] = int(d[k])
            for k in ("overall_accuracy", "overall_target_rate",
                      "flip_recovery", "flip_still_target",
                      "rsst_preservation", "rsst_drift_to_target"):
                d[k] = float(d[k]) if d[k] not in ("nan", "") else float("nan")
            rows.append(d)
    return rows


def summarize_for_model(model_tag):
    path = CACHE / model_tag / "intervention_results.csv"
    if not path.exists():
        print(f"[{model_tag}] no intervention_results.csv found")
        return None
    rows = load_csv(path)
    print(f"\n{'=' * 90}\n{model_tag}\n{'=' * 90}")
    print(f"Total rows: {len(rows)}")

    # Group by (source, mode, alpha, direction_type)
    by_source = defaultdict(list)
    for r in rows:
        by_source[r["held_out_source"]].append(r)
    print(f"Unique held-out sources: {len(by_source)}")

    # === Headline: per-source best alpha for all_layers mode (real direction) vs random ===
    print(f"\n--- All-layer intervention: best alpha per source (real vs random) ---")
    print(f"{'source':<55s} {'best_a_real':>12s} {'flip_recov_real':>16s} {'flip_recov_rand':>16s} {'lift':>8s}")
    headline = []
    for src, src_rows in by_source.items():
        all_layer_real = [r for r in src_rows if r["mode"] == "all_layers" and r["direction_type"] == "real" and r["alpha"] > 0]
        all_layer_rand = [r for r in src_rows if r["mode"] == "all_layers" and r["direction_type"] == "random" and r["alpha"] > 0]
        if not all_layer_real:
            continue
        best_real = max(all_layer_real, key=lambda r: r["flip_recovery"] if not np.isnan(r["flip_recovery"]) else -1)
        # Match alpha on random baseline
        best_rand = [r for r in all_layer_rand if r["alpha"] == best_real["alpha"]]
        rand_recov = best_rand[0]["flip_recovery"] if best_rand else float("nan")
        lift = best_real["flip_recovery"] - rand_recov if not np.isnan(rand_recov) else float("nan")
        print(f"  {src:<53s} {best_real['alpha']:>12.2f} {best_real['flip_recovery']:>16.3f} {rand_recov:>16.3f} {lift:>+8.3f}")
        headline.append({
            "source": src,
            "bias": best_real["bias"],
            "best_alpha": best_real["alpha"],
            "flip_recovery_real": best_real["flip_recovery"],
            "flip_recovery_random": rand_recov,
            "lift": lift,
            "rsst_preservation_real": best_real["rsst_preservation"],
        })

    # === Per-bias summary ===
    print(f"\n--- Per-bias mean recovery (all-layer, best alpha) ---")
    by_bias = defaultdict(list)
    for h in headline:
        by_bias[h["bias"]].append(h)
    for bias, hs in by_bias.items():
        mean_real = np.mean([h["flip_recovery_real"] for h in hs])
        mean_rand = np.mean([h["flip_recovery_random"] for h in hs])
        print(f"  {bias:<35s}  mean_real={mean_real:.3f}  mean_rand={mean_rand:.3f}  lift={mean_real - mean_rand:+.3f}")

    # === Selectivity: rsst preservation under best alpha ===
    print(f"\n--- Selectivity (resisted pairs still correct at best-alpha intervention) ---")
    rsst_vals = [h["rsst_preservation_real"] for h in headline if not np.isnan(h["rsst_preservation_real"])]
    if rsst_vals:
        print(f"  mean rsst_preservation under real direction: {np.mean(rsst_vals):.3f}  (1.0 = perfect preservation)")

    # === Layer-by-layer: best single-layer recovery per source ===
    print(f"\n--- Layer-by-layer scan: which single-layer intervention gives best recovery? ---")
    print(f"{'source':<55s} {'best_layer':>11s} {'recov':>7s}")
    for src, src_rows in by_source.items():
        single = [r for r in src_rows if r["mode"].startswith("layer_") and r["direction_type"] == "real" and r["alpha"] > 0]
        if not single:
            continue
        best = max(single, key=lambda r: r["flip_recovery"] if not np.isnan(r["flip_recovery"]) else -1)
        layer_idx = int(best["mode"].split("_")[1])
        print(f"  {src:<53s} {layer_idx:>11d} {best['flip_recovery']:>7.3f}")

    # === Save summary JSON ===
    overall = {
        "model": model_tag,
        "n_sources": len(headline),
        "mean_flip_recovery_real": float(np.mean([h["flip_recovery_real"] for h in headline])) if headline else None,
        "mean_flip_recovery_random": float(np.mean([h["flip_recovery_random"] for h in headline])) if headline else None,
        "mean_lift": float(np.mean([h["lift"] for h in headline if not np.isnan(h["lift"])])) if headline else None,
        "per_source": headline,
        "per_bias": {b: {
            "mean_real": float(np.mean([h["flip_recovery_real"] for h in hs])),
            "mean_random": float(np.mean([h["flip_recovery_random"] for h in hs])),
            "n_sources": len(hs),
        } for b, hs in by_bias.items()},
    }
    out_path = CACHE / model_tag / "intervention_summary.json"
    with open(out_path, "w") as f:
        json.dump(overall, f, indent=2)
    print(f"\nSaved summary: {out_path}")
    return overall


def main():
    for model_tag in ("mistral7b", "olmo2_7b"):
        summarize_for_model(model_tag)


if __name__ == "__main__":
    main()
