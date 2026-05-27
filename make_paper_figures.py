"""Generate the two new paper figures:
   - cue_gap_forest.pdf: per-model cue-attributable gap (origin)
   - flip_rate_hero.pdf:  per-model x per-bias flip rate (hero, problem at a glance)
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

CACHE = Path(__file__).resolve().parent / "cache"
FIG = Path(__file__).resolve().parent / "final_paper" / "figures"

INSTRUCT = ["llama31_8b", "qwen25_7b", "gemma2_9b", "mistral7b", "olmo2_7b"]
INSTRUCT_LABEL = ["Llama-3.1-8B", "Qwen-2.5-7B", "Gemma-2-9B", "Mistral-7B", "OLMo-2-7B"]
BIASES = ["suggested_answer", "post_hoc", "distractor_fact",
          "distractor_argument", "wrong_few_shot", "spurious_few_shot_squares"]
BIAS_LABEL = ["Sugg.\nAns.", "Post\nHoc", "Dist.\nFact",
              "Dist.\nArg.", "Wrong\nFS", "Spur.\nSq."]

# -------- Fig: cue-gap forest (Pillar 1 origin) --------

def fig_cue_gap():
    ci = json.load(open(CACHE / "base_probe_ci.json"))
    order = ["llama31_8b_base_xlabel", "gemma2_9b_base_xlabel",
             "olmo2_7b_base_xlabel", "mistral7b_base_xlabel",
             "qwen25_7b_base_xlabel"]
    labels = ["Llama-3.1-8B", "Gemma-2-9B", "OLMo-2-7B", "Mistral-7B", "Qwen-2.5-7B"]
    gaps   = [ci[m]["cue_attributable_gap_mean"] for m in order]
    cis    = [ci[m]["cue_attributable_gap_ci95"] for m in order]
    pvals  = [ci[m]["wilcoxon_paired_p_one_sided"] for m in order]

    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    y = np.arange(len(order))[::-1]
    colors = ['#2b6cb0' if p < 0.05 else '#a0aec0' for p in pvals]
    for yi, (g, c, color) in enumerate(zip(gaps, cis, colors)):
        ax.errorbar([g], [y[yi]], xerr=[[g - c[0]], [c[1] - g]],
                    fmt='o', markersize=10, capsize=4, lw=1.5,
                    color=color, mfc=color, mec=color, mew=1.5)

    # reference: instruct lift over random
    ax.axvspan(0.20, 0.32, color='#48bb78', alpha=0.12,
               label='instruct lift over random')
    ax.axvline(0, color='black', lw=0.7, linestyle='--', label='chance')

    ax.set_yticks(y)
    ax.set_yticklabels([f"{lab}\n($p={p:.3f}$)" if p < 0.05
                        else f"{lab}\n($p={p:.2f}$, n.s.)"
                        for lab, p in zip(labels, pvals)],
                       fontsize=9)
    ax.set_xlabel("Cue-attributable LODO AUROC gap\n(biased $-$ unbiased control, per source)",
                  fontsize=10)
    ax.set_xlim(-0.10, 0.35)
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax.set_title("Base models lack the bias representation",
                 fontsize=11, weight='bold', loc='left')
    plt.tight_layout()
    out = FIG / "cue_gap_forest.pdf"
    plt.savefig(out)
    plt.savefig(FIG / "cue_gap_forest.png", dpi=200)
    print(f"  saved: {out}")


# -------- Fig: hero (per-model x per-bias flip rate) --------

def fig_hero():
    grid = np.full((len(INSTRUCT), len(BIASES)), np.nan)
    for i, m in enumerate(INSTRUCT):
        for j, b in enumerate(BIASES):
            n_flip = n_tot = 0
            for ds in ['hellaswag', 'logiqa', 'mmlu', 'truthfulqa']:
                p = CACHE / m / f"{b}__{ds}" / "meta.json"
                if not p.exists():
                    continue
                j_meta = json.load(open(p))
                n_flip += j_meta["n_flipped"]
                # denominator = pairs eligible to flip (model knew unbiased)
                n_tot += (j_meta["n_flipped"] + j_meta["n_resisted"]
                          + j_meta["n_other"])
            if n_tot > 0:
                grid[i, j] = 100.0 * n_flip / n_tot

    # Aspect 1.333:1 chosen to match fig1.png's height when both panels are
    # side-by-side in the combined intro figure (fig1 at 0.56\linewidth +
    # this at 0.42\linewidth, with fig1 having aspect 1.78).
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    im = ax.imshow(grid, aspect='auto', cmap='Reds', vmin=0, vmax=80)
    ax.set_xticks(range(len(BIASES)))
    ax.set_xticklabels(BIAS_LABEL, fontsize=9)
    ax.set_yticks(range(len(INSTRUCT)))
    ax.set_yticklabels(INSTRUCT_LABEL, fontsize=9)
    for i in range(len(INSTRUCT)):
        for j in range(len(BIASES)):
            v = grid[i, j]
            txt = f"{v:.0f}%" if not np.isnan(v) else "—"
            color = 'white' if (not np.isnan(v) and v > 45) else 'black'
            ax.text(j, i, txt, ha='center', va='center',
                    color=color, fontsize=9, weight='bold')
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("% of `model-knew' pairs the model flips on\nunder the bias",
                   fontsize=9)
    ax.set_title("Modern instruct LLMs cave to cue-induced biases at 10--70%",
                 fontsize=11, weight='bold', loc='left', pad=10)
    plt.tight_layout()
    out = FIG / "flip_rate_hero.pdf"
    plt.savefig(out)
    plt.savefig(FIG / "flip_rate_hero.png", dpi=200)
    print(f"  saved: {out}")


if __name__ == "__main__":
    fig_hero()
