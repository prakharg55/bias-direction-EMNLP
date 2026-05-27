"""Appendix figure: per-bias cosine matrices for all 5 instruct families.

A 1x5 grid showing the 7x7 per-bias cosine matrix per family, using the same
bias ordering as the body's 2-panel figure. Models that don't have a bias
(Gemma and OLMo lack the Spurious Few-Shot Hindsight singleton in the
cosine analysis) will show NaN in that row/col.
"""
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams['font.family'] = 'DejaVu Sans'

CACHE = Path(__file__).resolve().parent / "cache"
FIG = Path(__file__).resolve().parent / "final_paper" / "figures"

# Same ordering as the body 2-panel figure (make_cosine_figure.py)
BIAS_ORDER = [
    "distractor_argument",
    "spurious_few_shot_hindsight",
    "post_hoc",
    "distractor_fact",
    "spurious_few_shot_squares",
    "wrong_few_shot",
    "suggested_answer",
]
LABELS = ["dist-arg", "hindsight", "post-hoc", "dist-fact",
          "spurious sq.", "wrong-fs", "sugg-ans"]

MODELS = [
    ("llama31_8b",  "Llama-3.1-8B"),
    ("qwen25_7b",   "Qwen-2.5-7B"),
    ("gemma2_9b",   "Gemma-2-9B"),
    ("mistral7b",   "Mistral-7B"),
    ("olmo2_7b",    "OLMo-2-7B"),
]


def get_matrix(model):
    j = json.load(open(CACHE / model / "cross_bias_cosine.json"))
    cm = j["cos_matrix_point_estimate"]
    n = len(BIAS_ORDER)
    M = np.full((n, n), np.nan)
    for i, bi in enumerate(BIAS_ORDER):
        for k, bk in enumerate(BIAS_ORDER):
            if bi == bk:
                M[i, k] = 1.0
                continue
            for key in (f"{bi}__vs__{bk}", f"{bk}__vs__{bi}"):
                if key in cm:
                    M[i, k] = cm[key]
                    break
    return M


def panel(ax, M, title):
    im = ax.imshow(M, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    n = len(LABELS)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(LABELS, rotation=35, ha='right', fontsize=8)
    ax.set_yticklabels(LABELS, fontsize=8)
    for i in range(n):
        for j in range(n):
            v = M[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha='center', va='center',
                        color='gray', fontsize=8, alpha=0.5)
                continue
            color = 'white' if abs(v) > 0.55 else 'black'
            ax.text(j, i, f"{v:+.2f}", ha='center', va='center',
                    color=color, fontsize=7.5, weight='bold')
    ax.set_title(title, fontsize=11, weight='bold', pad=6)
    return im


def main():
    # 2x3 layout, hide the unused 6th cell
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.6))
    flat = axes.flatten()
    im = None
    for ax, (tag, label) in zip(flat, MODELS):
        M = get_matrix(tag)
        im = panel(ax, M, label)
    # hide the 6th (unused) subplot
    flat[5].set_visible(False)

    cax = fig.add_axes([0.71, 0.10, 0.012, 0.32])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Cosine similarity", fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.98, 1])
    out_pdf = FIG / "cosine_all_families.pdf"
    out_png = FIG / "cosine_all_families.png"
    plt.savefig(out_pdf)
    plt.savefig(out_png, dpi=170)
    print(f"  saved: {out_pdf}")
    print(f"  saved: {out_png}")


if __name__ == "__main__":
    main()
