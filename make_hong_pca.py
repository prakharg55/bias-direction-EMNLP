"""Hong-style Figure 2: 2D direction-aligned projection of last-token hidden
states per instruct model, restricted to a single bias type.

For each instruct family, take all pairs from the SINGLE_BIAS sources (4
datasets), slice X_biased to the model's best layer (from layer_wise_summary),
and project onto a task-aware 2D coordinate system:
  x-axis = projection onto the per-bias averaged diff-of-means direction
           d_bias (the discriminating axis itself)
  y-axis = top PC of the residual after removing the d_bias component
           (captures the dominant within-bias remaining variance)

Points colored by pair category (flipped/resisted), single marker shape
(uniform circle since only one bias is plotted). Black arrow along x-axis
labels d_bias; grey dashed line is a logistic-regression boundary fit on
the 2D projection.
"""
import os
# Limit BLAS threads to avoid contention when running on shared login nodes.
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')

import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('Agg')  # force non-interactive backend
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

CACHE = Path(__file__).resolve().parent / "cache"
FIG = Path(__file__).resolve().parent / "final_paper" / "figures"

INSTRUCT = ["llama31_8b", "qwen25_7b", "gemma2_9b", "mistral7b", "olmo2_7b"]
INSTRUCT_LABEL = ["Llama-3.1-8B", "Qwen-2.5-7B", "Gemma-2-9B",
                  "Mistral-7B", "OLMo-2-7B"]

SINGLE_BIAS = "suggested_answer"
SINGLE_BIAS_LABEL = "Suggested Answer"

BIASES = ["suggested_answer", "post_hoc", "distractor_fact",
          "distractor_argument", "wrong_few_shot",
          "spurious_few_shot_squares", "spurious_few_shot_hindsight"]
BIAS_LABEL = {
    "suggested_answer":               "Suggested Answer",
    "post_hoc":                       "Post Hoc",
    "distractor_fact":                "Distractor Fact",
    "distractor_argument":            "Distractor Argument",
    "wrong_few_shot":                 "Wrong Few-Shot",
    "spurious_few_shot_squares":      "Spurious Squares",
    "spurious_few_shot_hindsight":    "Spurious Hindsight",
}
BIAS_MARKERS = {
    "suggested_answer":            "o",
    "post_hoc":                    "s",
    "distractor_fact":             "^",
    "distractor_argument":         "v",
    "wrong_few_shot":              "D",
    "spurious_few_shot_squares":   "P",
    "spurious_few_shot_hindsight": "X",
}

FLIPPED_COLOR = "#d62728"
RESISTED_COLOR = "#1f77b4"
ARROW_COLOR = "#2d3748"
BOUNDARY_COLOR = "#4a5568"

RNG = np.random.default_rng(42)
MAX_POINTS_PER_PANEL = 2500


def best_layer_for_bias(model, bias_name):
    """Per-layer LODO probing AUROC peak restricted to one bias.

    Reads cache/{model}/layer_wise_lodo.csv, filters to rows whose bias matches
    bias_name, groups by layer, and returns the layer that maximizes mean AUROC
    across the held-out source datasets of that bias.
    """
    with open(CACHE / model / "layer_wise_lodo.csv") as f:
        rows = [r for r in csv.DictReader(f) if r['bias'] == bias_name]
    if not rows:
        raise ValueError(f'No layer-wise rows for {model}/{bias_name}')
    by_layer = defaultdict(list)
    for r in rows:
        by_layer[int(r['layer'])].append(float(r['auroc']))
    layer_means = {l: sum(v) / len(v) for l, v in by_layer.items()}
    return max(layer_means, key=layer_means.get)


def load_model_data(model):
    """Load bias-specific best-layer hidden states + labels + per-bias direction.
    Returns X (n_pairs, H), cats, biases (parallel arrays), d_layer (H,), best_layer (int).
    """
    best_layer = best_layer_for_bias(model, SINGLE_BIAS)

    meta_path = CACHE / model / "suggested_answer__mmlu" / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    H = meta['hidden_size']

    Xs, cats, biases = [], [], []
    for src_dir in sorted((CACHE / model).iterdir()):
        if not src_dir.is_dir() or not (src_dir / 'meta.json').exists():
            continue
        if '__' not in src_dir.name:
            continue
        bias_name = src_dir.name.split('__')[0]
        if bias_name != SINGLE_BIAS:
            continue
        X_biased_path = src_dir / 'X_biased.npy'
        if not X_biased_path.exists():
            continue
        X_biased = np.load(X_biased_path, mmap_mode='r')
        if X_biased.shape[0] == 0:
            continue
        X_layer = np.asarray(
            X_biased[:, best_layer*H:(best_layer+1)*H], dtype=np.float32
        )
        with open(src_dir / 'pairs_meta.json') as f:
            pm = json.load(f)
        assert len(pm) == X_layer.shape[0], f"{src_dir.name}: meta/X mismatch"
        cats.extend(p['category'] for p in pm)
        biases.extend(bias_name for _ in pm)
        Xs.append(X_layer)

    X = np.concatenate(Xs, axis=0)
    cats = np.array(cats)
    biases = np.array(biases)

    # Per-bias averaged direction: mean over the 4 source directions of
    # SINGLE_BIAS, normalized to unit length.
    dz = np.load(CACHE / model / "directions.npz")
    directions = dz['directions']                # (25, n_layers*H)
    source_names = list(dz['source_names'])
    mask = np.array(
        [name.split('__')[0] == SINGLE_BIAS for name in source_names]
    )
    d_mean = directions[mask].mean(axis=0)
    d_mean /= np.linalg.norm(d_mean)
    d_layer = d_mean[best_layer*H:(best_layer+1)*H].astype(np.float32)

    return X, cats, biases, d_layer, best_layer


def make_pca_panel(ax, X, cats, biases, d_layer, model_label, best_layer):
    # Direction-aligned projection.
    # x-axis: projection onto d_layer (the per-model averaged diff-of-means
    #         direction at the best layer).
    # y-axis: top PC of the residual after deflating along d_layer, computed
    #         on a subsample to keep memory and runtime bounded.
    n, H = X.shape
    print(f'    [panel] n={n} H={H}', flush=True)
    proj_d = X @ d_layer                            # (n,)
    print(f'    [panel] proj_d done', flush=True)

    sub_n = min(3000, n)
    sub_idx = RNG.choice(n, sub_n, replace=False)
    X_sub = X[sub_idx]
    X_sub_residual = X_sub - np.outer(proj_d[sub_idx], d_layer)
    pca_res = PCA(n_components=1)
    pca_res.fit(X_sub_residual)
    top_pc = pca_res.components_[0].astype(np.float32)
    del X_sub, X_sub_residual
    print(f'    [panel] PCA(1) on residual done', flush=True)

    mu_top = float(X.mean(axis=0) @ top_pc)
    proj_pc1 = X @ top_pc - mu_top                  # (n,) centered
    X_2d = np.column_stack([proj_d, proj_pc1])
    print(f'    [panel] X_2d done', flush=True)

    n = len(X)
    if n > MAX_POINTS_PER_PANEL:
        flip_idx = np.where(cats == 'flipped')[0]
        rsst_idx = np.where(cats == 'resisted')[0]
        ratio = len(flip_idx) / n
        n_flip = int(MAX_POINTS_PER_PANEL * ratio)
        n_rsst = MAX_POINTS_PER_PANEL - n_flip
        flip_sub = RNG.choice(flip_idx, min(n_flip, len(flip_idx)), replace=False)
        rsst_sub = RNG.choice(rsst_idx, min(n_rsst, len(rsst_idx)), replace=False)
        idx = np.concatenate([flip_sub, rsst_sub])
    else:
        idx = np.arange(n)

    for cat, color in [('flipped', FLIPPED_COLOR),
                       ('resisted', RESISTED_COLOR)]:
        mask = cats[idx] == cat
        if not mask.any():
            continue
        ax.scatter(
            X_2d[idx][mask, 0], X_2d[idx][mask, 1],
            c=color, marker='o',
            s=18, alpha=0.55, edgecolors='none',
        )

    # Diff-of-means decision boundary along d_bias: x0 = midpoint between
    # flipped-mean and resisted-mean projections. Equivalent to the
    # Bayes-optimal threshold for a 1D Gaussian discriminant with equal
    # priors and equal variances, which matches the paper's diff-of-means
    # framework. Robust against the default-LR-regularization artifact that
    # places the LR threshold outside the data range on Mistral (proj_d has
    # very small absolute scale + class imbalance + default L2 = degenerate
    # fit; verified that class-balanced LR converges to this same midpoint).
    y = (cats == 'flipped').astype(int)
    x0 = float(0.5 * (proj_d[y == 1].mean() + proj_d[y == 0].mean()))

    xlim = (X_2d[:, 0].min(), X_2d[:, 0].max())
    ylim = (X_2d[:, 1].min(), X_2d[:, 1].max())
    xpad = 0.05 * (xlim[1] - xlim[0])
    ypad = 0.05 * (ylim[1] - ylim[0])
    ax.set_xlim(xlim[0] - xpad, xlim[1] + xpad)
    ax.set_ylim(ylim[0] - ypad, ylim[1] + ypad)

    ax.axvline(x0, color=BOUNDARY_COLOR, ls='--', lw=1.6, alpha=0.85,
               zorder=4)

    # Arrow along x-axis (x-axis is by construction the d_bias direction).
    data_mid_x = (xlim[0] + xlim[1]) / 2
    data_mid_y = (ylim[0] + ylim[1]) / 2
    xrange = xlim[1] - xlim[0]
    arrow_len = 0.35 * xrange
    ax.annotate(
        '', xy=(data_mid_x + 0.5 * arrow_len, data_mid_y),
        xytext=(data_mid_x - 0.5 * arrow_len, data_mid_y),
        arrowprops=dict(arrowstyle='->', color=ARROW_COLOR, lw=2.4,
                        mutation_scale=18, alpha=0.9),
        zorder=5,
    )

    ax.set_xlabel(r'Projection onto $d_{\mathrm{bias}}$', fontsize=9)
    ax.set_ylabel('Top PC of residual', fontsize=9)
    ax.set_title(f'{model_label} (layer {best_layer})',
                 fontsize=10, weight='bold')
    ax.tick_params(labelsize=8)


def make_legend(ax):
    ax.axis('off')

    handles = [
        plt.Line2D([], [], marker='o', color='w', mfc=FLIPPED_COLOR,
                   markersize=11, label='Flipped'),
        plt.Line2D([], [], marker='o', color='w', mfc=RESISTED_COLOR,
                   markersize=11, label='Resisted'),
        plt.Line2D([], [], color=ARROW_COLOR, marker='>', lw=2.4,
                   markersize=10, label=r'$d_{\mathrm{bias}}$ (x-axis)'),
        plt.Line2D([], [], color=BOUNDARY_COLOR, ls='--', lw=1.8,
                   label='Class-midpoint boundary'),
    ]

    ax.legend(handles=handles, loc='center', frameon=False,
              fontsize=11, handlelength=1.8, labelspacing=0.9)
    ax.text(0.5, 0.92, f'Bias: {SINGLE_BIAS_LABEL}',
            ha='center', va='center', transform=ax.transAxes,
            fontsize=11, weight='bold', color='#2d3748')


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    axes_flat = axes.flatten()

    for i, model in enumerate(INSTRUCT):
        print(f'Processing {model}...', flush=True)
        X, cats, biases, d_layer, best_layer = load_model_data(model)
        n_flip = int((cats == 'flipped').sum())
        n_rsst = int((cats == 'resisted').sum())
        print(f'  n_pairs = {len(X)}  (flipped={n_flip}, resisted={n_rsst})',
              flush=True)
        make_pca_panel(axes_flat[i], X, cats, biases, d_layer,
                       INSTRUCT_LABEL[i], best_layer)

    make_legend(axes_flat[5])

    fig.tight_layout()
    out_pdf = FIG / 'hong_pca.pdf'
    out_png = FIG / 'hong_pca.png'
    fig.savefig(out_pdf, dpi=200, bbox_inches='tight')
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    print(f'Saved {out_pdf}', flush=True)
    print(f'Saved {out_png}', flush=True)


if __name__ == '__main__':
    main()
