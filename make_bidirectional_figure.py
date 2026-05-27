"""Body figure: bidirectional intervention magnitude across 5 instruct families.

For each model, show the achievable range of P(bias_target):
  - Baseline (alpha=0):       how often the model caves with no intervention
  - Max induced (alpha<0):    how high we can push bias rate
  - Max debiased (alpha>0):   how low we can push bias rate

A horizontal bar from debiased-min to induced-max per model, with the
baseline marked. Numbers above the bar tell you, in one read, how much
the intervention can shift the bias rate in each direction.
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

CACHE = Path(__file__).resolve().parent / "cache"
FIG = Path(__file__).resolve().parent / "final_paper" / "figures"

# top-to-bottom display order
MODELS = [
    ("llama31_8b", "Llama-3.1-8B"),
    ("qwen25_7b",  "Qwen-2.5-7B"),
    ("gemma2_9b",  "Gemma-2-9B"),
    ("mistral7b",  "Mistral-7B"),
    ("olmo2_7b",   "OLMo-2-7B"),
]


def load_per_alpha(model_tag):
    """Returns {alpha: mean P(bias_target)} for real direction, averaged across LODO points."""
    by_point = defaultdict(dict)
    with open(CACHE / model_tag / "intervention_results.csv") as f:
        for row in csv.DictReader(f):
            if row['direction_type'] != 'real' or row['mode'] != 'all_layers':
                continue
            key = (row['bias'], row['held_out_source'])
            by_point[key][float(row['alpha'])] = float(row['overall_target_rate'])
    all_alphas = sorted({a for pt in by_point.values() for a in pt})
    return {a: float(np.mean([pt[a] for pt in by_point.values() if a in pt]))
            for a in all_alphas}


def summarize(means):
    """Return (baseline, induced_max, debiased_min) ignoring extreme-alpha breakdown.
    Use |alpha| <= 2 as the in-range region; extreme alpha collapses P(any letter)
    rather than acting causally on the bias direction.
    """
    baseline = means[0.0]
    in_range = {a: p for a, p in means.items() if -2 <= a <= 2}
    induced_alphas = [a for a in in_range if a < 0]
    debiased_alphas = [a for a in in_range if a > 0]
    induced_max = max(in_range[a] for a in induced_alphas) if induced_alphas else baseline
    debiased_min = min(in_range[a] for a in debiased_alphas) if debiased_alphas else baseline
    return baseline, induced_max, debiased_min


def main():
    fig, ax = plt.subplots(figsize=(7.0, 3.8))

    y = np.arange(len(MODELS))[::-1]

    for yi, (tag, label) in zip(y, MODELS):
        means = load_per_alpha(tag)
        base, ind, deb = summarize(means)

        # connecting bar from min to max
        ax.hlines(yi, deb, ind, color='#cbd5e1', lw=10, zorder=1)
        # debias marker (left, blue)
        ax.plot(deb, yi, marker='o', markersize=11, color='#1e40af',
                mec='black', mew=0.7, zorder=3)
        # baseline marker (middle, gray)
        ax.plot(base, yi, marker='D', markersize=8, color='#475569',
                mec='black', mew=0.7, zorder=3)
        # induce marker (right, red)
        ax.plot(ind, yi, marker='o', markersize=11, color='#991b1b',
                mec='black', mew=0.7, zorder=3)

        # numeric labels: percent of P(bias) at each end
        ax.text(deb - 0.012, yi, f"{deb*100:.0f}%", ha='right', va='center',
                fontsize=9, color='#1e40af', weight='bold')
        ax.text(ind + 0.012, yi, f"{ind*100:.0f}%", ha='left', va='center',
                fontsize=9, color='#991b1b', weight='bold')
        # baseline label, smaller, below the diamond
        ax.text(base, yi - 0.32, f"{base*100:.0f}%", ha='center', va='top',
                fontsize=8, color='#475569')

    ax.set_yticks(y)
    ax.set_yticklabels([m[1] for m in MODELS], fontsize=10)
    ax.set_xlabel(r"$P(\mathrm{bias\ target})$  achievable by steering along the bias direction",
                  fontsize=10)
    ax.set_xlim(0, 0.85)
    ax.set_ylim(-0.7, len(MODELS) - 0.3)

    # legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='o', color='w', mfc='#1e40af', mec='black',
               mew=0.7, markersize=10, label='debiased min ($\\alpha>0$)'),
        Line2D([0], [0], marker='D', color='w', mfc='#475569', mec='black',
               mew=0.7, markersize=8, label='no intervention ($\\alpha=0$)'),
        Line2D([0], [0], marker='o', color='w', mfc='#991b1b', mec='black',
               mew=0.7, markersize=10, label='induced max ($\\alpha<0$)'),
    ]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.18),
              fontsize=8.5, framealpha=0.9, ncol=3)

    ax.set_title("Intervention bidirectionally controls how often the model caves",
                 fontsize=11, weight='bold', loc='left', pad=10)
    ax.grid(True, axis='x', alpha=0.3, linestyle=':', linewidth=0.5, zorder=0)

    plt.tight_layout()
    out_pdf = FIG / "bidirectional_multi.pdf"
    out_png = FIG / "bidirectional_multi.png"
    plt.savefig(out_pdf)
    plt.savefig(out_png, dpi=180)
    print(f"  saved: {out_pdf}")
    print(f"  saved: {out_png}")


if __name__ == "__main__":
    main()
