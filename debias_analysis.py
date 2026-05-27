"""
Debiasing operating-point analysis.

Reframes the causal intervention (subtract alpha * bias_direction at every layer,
real LODO direction) as a practical debiasing tool. The 15-alpha intervention
sweep is already computed; here we read it back and, per model, trace the
recovery-vs-preservation trade-off and pick a deployable operating point.

  recovery     = mean flip_recovery     = fraction of bias-induced errors corrected
                 (flipped pairs that now produce the correct answer)
  preservation = mean rsst_preservation = fraction of already-correct answers on
                 biased prompts kept correct -- the selectivity / "do no harm" side

Operating point = the single global alpha (alpha > 0, all-layer, real direction)
that maximizes recovery subject to preservation >= tau. Reported at tau = 0.95
and 0.90, each with the random-direction baseline at the same alpha for
direction-specificity.

Scope note: rsst_preservation measures preservation on biased prompts the model
already handled correctly. The dedicated unbiased-prompt selectivity test
(selectivity_results.csv) exists only for Llama/Qwen; this analysis uses the
all-6-model measure so the result is reported consistently across every family.

Input:  cache/{model}/intervention_results.csv  (all 6 models, full sweep)
Output: cache/debias_summary.json + console tables
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import analyze_intervention as ai  # reuse the tested CSV loader

CACHE = Path(__file__).resolve().parent / "cache"
MODELS = ["llama31_8b", "qwen25_7b", "gemma2_9b", "qwen25_7b_base", "mistral7b", "olmo2_7b"]
THRESHOLDS = [0.95, 0.90]


def _mean(rows, key):
    vals = [r[key] for r in rows if not np.isnan(r[key])]
    return float(np.mean(vals)) if vals else float("nan")


def curve_for_model(model):
    """Per-alpha mean recovery / preservation across held-out sources."""
    path = CACHE / model / "intervention_results.csv"
    if not path.exists():
        print(f"[{model}] no intervention_results.csv")
        return None
    rows = [r for r in ai.load_csv(path) if r["mode"] == "all_layers"]

    # baseline accuracy on biased prompts (no intervention, alpha = 0)
    base = [r for r in rows if r["alpha"] == 0.0]
    baseline_acc = _mean(base, "overall_accuracy")

    by = defaultdict(list)
    for r in rows:
        if r["alpha"] > 0:
            by[(r["alpha"], r["direction_type"])].append(r)

    curve = []
    for a in sorted({a for a, _ in by}):
        real = by.get((a, "real"), [])
        rand = by.get((a, "random"), [])
        curve.append({
            "alpha": a,
            "recovery_real": _mean(real, "flip_recovery"),
            "preservation_real": _mean(real, "rsst_preservation"),
            "drift_to_target_real": _mean(real, "rsst_drift_to_target"),
            "acc_real": _mean(real, "overall_accuracy"),
            "recovery_random": _mean(rand, "flip_recovery"),
            "preservation_random": _mean(rand, "rsst_preservation"),
            "n_sources": len(real),
        })
    return {"model": model, "baseline_acc": baseline_acc, "curve": curve}


def operating_point(curve, tau):
    """Highest-recovery alpha with preservation >= tau."""
    feasible = [c for c in curve
                if not np.isnan(c["preservation_real"])
                and not np.isnan(c["recovery_real"])
                and c["preservation_real"] >= tau]
    if not feasible:
        return None
    return max(feasible, key=lambda c: c["recovery_real"])


def per_bias_at(model, alpha):
    """Per-bias recovery/preservation at one alpha (real direction, all-layer)."""
    rows = [r for r in ai.load_csv(CACHE / model / "intervention_results.csv")
            if r["mode"] == "all_layers" and r["direction_type"] == "real"
            and r["alpha"] == alpha]
    by_bias = defaultdict(list)
    for r in rows:
        by_bias[r["bias"]].append(r)
    return {b: {"recovery": _mean(rs, "flip_recovery"),
                "preservation": _mean(rs, "rsst_preservation")}
            for b, rs in sorted(by_bias.items())}


def main():
    summary = {}
    for model in MODELS:
        cm = curve_for_model(model)
        if cm is None:
            continue
        curve = cm["curve"]

        print(f"\n{'=' * 88}\n{model}   (baseline accuracy on biased prompts: "
              f"{cm['baseline_acc']:.3f})\n{'=' * 88}")
        print(f"{'alpha':>7s} {'recovery':>9s} {'preserv':>8s} {'drift':>7s} "
              f"{'rand_rec':>9s} {'rand_prs':>9s}")
        for c in curve:
            print(f"{c['alpha']:>7.2f} {c['recovery_real']:>9.3f} {c['preservation_real']:>8.3f} "
                  f"{c['drift_to_target_real']:>7.3f} {c['recovery_random']:>9.3f} "
                  f"{c['preservation_random']:>9.3f}")

        # unconstrained best recovery (ignores selectivity) for contrast
        valid = [c for c in curve if not np.isnan(c["recovery_real"])]
        peak = max(valid, key=lambda c: c["recovery_real"]) if valid else None

        ops = {}
        for tau in THRESHOLDS:
            op = operating_point(curve, tau)
            ops[f"tau_{tau}"] = op
            if op:
                print(f"\n  operating point @ preservation>={tau}:  alpha={op['alpha']:.2f}  "
                      f"recovery={op['recovery_real']:.3f}  preservation={op['preservation_real']:.3f}  "
                      f"(random recovery at same alpha: {op['recovery_random']:.3f})")
        if peak:
            print(f"  [unconstrained peak recovery: alpha={peak['alpha']:.2f} "
                  f"recovery={peak['recovery_real']:.3f} at preservation={peak['preservation_real']:.3f}]")

        # per-bias breakdown at the tau=0.90 operating point
        op90 = ops.get("tau_0.9")
        per_bias = per_bias_at(model, op90["alpha"]) if op90 else {}

        summary[model] = {
            "baseline_acc_biased_prompts": cm["baseline_acc"],
            "curve": curve,
            "operating_points": {
                k: ({"alpha": v["alpha"], "recovery": v["recovery_real"],
                     "preservation": v["preservation_real"],
                     "random_recovery": v["recovery_random"],
                     "acc_after": v["acc_real"]} if v else None)
                for k, v in ops.items()
            },
            "unconstrained_peak": ({"alpha": peak["alpha"], "recovery": peak["recovery_real"],
                                    "preservation": peak["preservation_real"]} if peak else None),
            "per_bias_at_tau0.90": per_bias,
        }

    # headline table
    print(f"\n\n{'=' * 88}")
    print("DEBIASING OPERATING POINTS  (all-layer intervention, real LODO direction)")
    print(f"{'=' * 88}")
    print(f"{'model':<16s} {'baseline':>9s} | {'tau>=0.95: a/rec/prs':>26s} | {'tau>=0.90: a/rec/prs':>26s}")
    print("-" * 88)
    for model, s in summary.items():
        def cell(op):
            if not op:
                return f"{'(none)':>26s}"
            return f"a={op['alpha']:<5.2f} rec={op['recovery']:.2f} prs={op['preservation']:.2f}".rjust(26)
        o95 = s["operating_points"].get("tau_0.95")
        o90 = s["operating_points"].get("tau_0.9")
        print(f"{model:<16s} {s['baseline_acc_biased_prompts']:>9.3f} | {cell(o95)} | {cell(o90)}")

    out = CACHE / "debias_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
