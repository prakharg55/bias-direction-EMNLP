"""
Layer-wise emergence of the bias signal -- across all 6 model families.

For each model, plot the LODO AUROC averaged across (bias, held-out dataset) at
each transformer layer (using the cached per-source directions sliced into
per-layer chunks). Shows WHERE in depth the bias signal lives.

x-axis: relative layer depth (layer_idx / (n_layers - 1)) so models of different
        depth (Llama/Mistral/OLMo 32, Qwen 28, Gemma 42) line up.
y-axis: mean LODO AUROC across all (bias, dataset) test points at that layer.

Reads existing cache/{model}/layer_wise_summary.json (already computed for all
6 families). Writes a multi-line PDF + the underlying CSV.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CACHE = Path(__file__).resolve().parent / "cache"
FIG_OUT = Path(__file__).resolve().parent / "final_paper" / "figures"
MODELS = [
    ("llama31_8b",     "Llama-3.1-8B-Instruct"),
    ("qwen25_7b",      "Qwen2.5-7B-Instruct"),
    ("gemma2_9b",      "Gemma-2-9B-it"),
    ("mistral7b",      "Mistral-7B-Instruct"),
    ("olmo2_7b",       "OLMo-2-7B-Instruct"),
    ("qwen25_7b_base", "Qwen2.5-7B (base)"),
]


def main():
    # Sized for single-column ACL layout: ~3.3" wide column. We render at a
    # slightly larger physical size so the embedded text/markers remain
    # readable after LaTeX scales the image down.
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    csv_rows = ["model,layer,layer_frac,mean_lodo_auroc,mean_random_auroc,mean_lift"]
    summary_rows = []
    peak_fracs = []
    for tag, label in MODELS:
        p = CACHE / tag / "layer_wise_summary.json"
        if not p.exists():
            print(f"  [skip] no layer_wise_summary.json for {tag}")
            continue
        j = json.load(open(p))
        layers = sorted(int(k) for k in j["per_layer"].keys())
        n_layers = max(layers) + 1
        fracs = [l / (n_layers - 1) for l in layers]
        aurocs = [j["per_layer"][str(l)]["mean_auroc"] for l in layers]
        randoms = [j["per_layer"][str(l)]["mean_random_auroc"] for l in layers]
        for l, f, a, r in zip(layers, fracs, aurocs, randoms):
            csv_rows.append(f"{tag},{l},{f:.4f},{a:.4f},{r:.4f},{a-r:.4f}")
        # base model: dashed gray to de-emphasize (it's a control)
        is_base = "base" in tag
        style = dict(marker="o", markersize=2.2, linewidth=1.4 if is_base else 1.8,
                     linestyle="--" if is_base else "-",
                     color="gray" if is_base else None,
                     alpha=0.55 if is_base else 0.95)
        ax.plot(fracs, aurocs, label=label, **style)
        bl, ba = j.get("best_layer"), j.get("best_layer_mean_auroc")
        frac = bl / (n_layers - 1)
        summary_rows.append((label, n_layers, bl, frac, ba))
        if not is_base:
            peak_fracs.append(frac)

    # shaded band marking the consensus peak depth across instruct families
    if peak_fracs:
        lo, hi = min(peak_fracs), max(peak_fracs)
        ax.axvspan(lo, hi, color='#fbbf24', alpha=0.18, zorder=0,
                   label=f'peak band ({lo:.2f}--{hi:.2f})')
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.7, label="chance")
    ax.set_xlabel("Relative layer depth (0 = embed, 1 = final)", fontsize=10)
    ax.set_ylabel("Mean LODO AUROC", fontsize=10)
    # Title moved to LaTeX caption; saves vertical space at small figure sizes.
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=7.5, loc="lower right", ncol=2, framealpha=0.95,
              handlelength=1.4, handletextpad=0.5, columnspacing=0.9,
              borderpad=0.3, labelspacing=0.25)
    ax.set_ylim(0.45, None)
    plt.tight_layout()
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    for outdir in (CACHE, FIG_OUT):
        plt.savefig(outdir / "layer_wise_emergence.pdf")
        plt.savefig(outdir / "layer_wise_emergence.png", dpi=200)
    print(f"Saved figure to: {CACHE} and {FIG_OUT}")

    csv = CACHE / "layer_wise_curves.csv"
    open(csv, "w").write("\n".join(csv_rows) + "\n")
    print(f"Saved CSV:    {csv}")

    print(f"\n{'model':<24s} {'n_layers':>9s} {'best_layer':>11s} {'best_layer_frac':>16s} {'AUROC':>7s}")
    print("-" * 75)
    for label, nL, bl, frac, ba in summary_rows:
        print(f"{label:<24s} {nL:>9d} {bl:>11d} {frac:>16.2f} {ba:>7.3f}")

    json.dump(
        {"per_model": [{"label": l, "n_layers": nL, "best_layer": bl,
                        "best_layer_frac": frac, "best_layer_mean_auroc": ba}
                       for l, nL, bl, frac, ba in summary_rows]},
        open(CACHE / "layer_wise_emergence.json", "w"), indent=2,
    )


if __name__ == "__main__":
    main()
