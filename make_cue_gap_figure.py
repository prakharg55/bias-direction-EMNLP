"""Body figure: per-model contrast of base vs instruct LODO AUROC.

For each model family, plot three points on a single AUROC axis:
  - unbiased control AUROC on base activations (question-content baseline)
  - biased AUROC on base activations (does the bias cue add anything?)
  - biased AUROC on instruct activations (the bias direction we want to find)

The visual: base-biased sits right on top of unbiased-control, while
instruct-biased sits far to the right. The cue-attributable signal in
base is tiny; the large gap between base and instruct is what alignment
tuning installs.
"""
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

CACHE = Path(__file__).resolve().parent / "cache"
FIG = Path(__file__).resolve().parent / "final_paper" / "figures"

MODELS = [
    ("llama31_8b",  "Llama-3.1-8B"),
    ("gemma2_9b",   "Gemma-2-9B"),
    ("olmo2_7b",    "OLMo-2-7B"),
    ("mistral7b",   "Mistral-7B"),
    ("qwen25_7b",   "Qwen-2.5-7B"),
]


def main():
    ci = json.load(open(CACHE / "base_probe_ci.json"))

    rows = []
    for tag, label in MODELS:
        base = ci[f"{tag}_base_xlabel"]
        instr = json.load(open(CACHE / tag / "lodo_summary.json"))["overall"]
        rows.append({
            "label": label,
            "control": base["unbiased_mean"],
            "base":    base["biased_mean"],
            "instruct": instr["mean_lodo_auroc"],
            "p": base["wilcoxon_paired_p_one_sided"],
            "gap_ci": base["cue_attributable_gap_ci95"],
            "gap_mean": base["cue_attributable_gap_mean"],
        })

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    y = np.arange(len(rows))[::-1]

    # chance reference
    ax.axvline(0.5, color='gray', linewidth=0.8, linestyle=':', zorder=1,
               label='chance (0.5)')

    for yi, r in zip(y, rows):
        # connecting line from control to instruct
        ax.hlines(yi, r["control"], r["instruct"],
                  color='#cbd5e1', lw=1.8, zorder=2)
        # control marker (hollow gray)
        ax.plot(r["control"], yi, marker='o', markersize=9,
                mfc='white', mec='#475569', mew=1.4, zorder=4)
        # base biased marker (filled gray)
        ax.plot(r["base"], yi, marker='o', markersize=10,
                mfc='#94a3b8', mec='black', mew=0.6, zorder=4)
        # instruct biased marker (filled blue)
        ax.plot(r["instruct"], yi, marker='o', markersize=12,
                mfc='#1e40af', mec='black', mew=0.7, zorder=5)

    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=10)
    ax.set_xlabel("LODO AUROC (mean across biases x held-out datasets)", fontsize=10)
    ax.set_xlim(0.45, 0.88)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_title("Alignment tuning installs the bias representation:\nbase activations carry almost none of it",
                 fontsize=11.5, weight='bold', loc='left', pad=10)

    # legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='o', color='w', mfc='white', mec='#475569',
               mew=1.4, markersize=9,
               label='base, unbiased prompt (question-content control)'),
        Line2D([0], [0], marker='o', color='w', mfc='#94a3b8', mec='black',
               mew=0.6, markersize=10,
               label='base, biased prompt (cue signal in base activations)'),
        Line2D([0], [0], marker='o', color='w', mfc='#1e40af', mec='black',
               mew=0.7, markersize=12,
               label='instruct, biased prompt (the bias direction we extract)'),
    ]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.20),
              fontsize=8.5, framealpha=0.9, ncol=1)

    ax.grid(True, axis='x', alpha=0.3, linestyle=':', linewidth=0.5, zorder=0)

    plt.tight_layout()
    out_pdf = FIG / "cue_gap_dumbbell.pdf"
    out_png = FIG / "cue_gap_dumbbell.png"
    plt.savefig(out_pdf)
    plt.savefig(out_png, dpi=180)
    print(f"  saved: {out_pdf}")
    print(f"  saved: {out_png}")


if __name__ == "__main__":
    main()
