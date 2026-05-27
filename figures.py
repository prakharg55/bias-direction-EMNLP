"""
Generate paper-quality figures from cached analysis outputs.

Figures generated (saved to paper/figures/):

  Part 1 (within-bias generalization):
    F1_source_cosine_{model}.pdf    21x21 source-level cosine heatmap, reordered by bias
    F1_lodo_auroc_{model}.pdf       Per-source LODO AUROC bar chart with random baseline

  Part 2 (cross-bias structure):
    F2_bias_cosine_{model}.pdf      6x6 per-bias cosine heatmap with bootstrap CIs
    F2_dendrogram_{model}.pdf       Hierarchical clustering dendrogram
    F2_cross_bias_probe_{model}.pdf 6x6 cross-bias probing AUROC heatmap

All figures saved as PDF (vector, EMNLP-friendly) and PNG (preview).
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.cluster.hierarchy import dendrogram

CACHE = Path(__file__).resolve().parent / "cache"
FIGS = Path(__file__).resolve().parent / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

MODEL_DISPLAY = {
    "llama31_8b": "Llama-3.1-8B",
    "qwen25_7b":  "Qwen-2.5-7B",
}

BIAS_DISPLAY = {
    "suggested_answer":             "suggested_answer",
    "distractor_argument":          "distractor_argument",
    "distractor_fact":              "distractor_fact",
    "wrong_few_shot":               "wrong_few_shot",
    "spurious_few_shot_squares":    "spurious_few_shot_squares",
    "spurious_few_shot_hindsight":  "spurious_few_shot_hindsight",
    "post_hoc":                     "post_hoc",
}

# Consistent color per bias (for the source heatmap row/col labels)
BIAS_COLORS = {
    "suggested_answer":             "#1f77b4",
    "distractor_argument":          "#ff7f0e",
    "distractor_fact":              "#2ca02c",
    "wrong_few_shot":               "#d62728",
    "spurious_few_shot_squares":    "#9467bd",
    "spurious_few_shot_hindsight":  "#8c564b",
    "post_hoc":                     "#e377c2",
}


def savefig(fig, name):
    pdf = FIGS / f"{name}.pdf"
    png = FIGS / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved: {pdf.name}, {png.name}")


# -------- F1: source-level cosine heatmap --------

def fig_source_cosine(model_tag):
    cm_npz = np.load(CACHE / model_tag / "cosine_matrix.npz", allow_pickle=True)
    cos = cm_npz["cosine"]
    sources = [str(s) for s in cm_npz["source_names"]]
    biases = [str(b) for b in cm_npz["bias_names"]]

    # Reorder by bias type, then by dataset name
    order = sorted(range(len(sources)), key=lambda i: (biases[i], sources[i]))
    cos_o = cos[np.ix_(order, order)]
    sources_o = [sources[i] for i in order]
    biases_o = [biases[i] for i in order]

    fig, ax = plt.subplots(figsize=(11, 9))
    vmax = 1.0
    vmin = -vmax
    im = ax.imshow(cos_o, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="equal")

    # Tick labels: just show dataset name per source (compact)
    short = [s.split("__")[-1] for s in sources_o]
    ax.set_xticks(range(len(short)))
    ax.set_yticks(range(len(short)))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(short, fontsize=7)

    # Color the tick labels by bias type
    for tick_label, b in zip(ax.get_xticklabels(), biases_o):
        tick_label.set_color(BIAS_COLORS.get(b, "black"))
    for tick_label, b in zip(ax.get_yticklabels(), biases_o):
        tick_label.set_color(BIAS_COLORS.get(b, "black"))

    # Bias-block boundaries
    boundaries = [0]
    cur_b = biases_o[0]
    for k, b in enumerate(biases_o):
        if b != cur_b:
            boundaries.append(k)
            cur_b = b
    boundaries.append(len(biases_o))
    for k in boundaries[1:-1]:
        ax.axhline(k - 0.5, color="black", linewidth=1.2)
        ax.axvline(k - 0.5, color="black", linewidth=1.2)

    # Bias group labels on the right
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        bias = biases_o[a]
        mid = (a + b - 1) / 2
        ax.text(len(short) + 0.6, mid, bias.replace("_", " "),
                va="center", ha="left", fontsize=8, color=BIAS_COLORS[bias])

    ax.set_title(f"Source-level direction cosine similarity ({MODEL_DISPLAY[model_tag]})",
                 fontsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.10)
    cbar.set_label("cosine similarity")
    fig.tight_layout()
    savefig(fig, f"F1_source_cosine_{model_tag}")


# -------- F1: LODO AUROC bar chart --------

def fig_lodo_auroc(model_tag):
    csv_path = CACHE / model_tag / "lodo_results.csv"
    rows = []
    with open(csv_path) as f:
        header = next(f).strip().split(",")
        for line in f:
            rows.append(dict(zip(header, line.strip().split(","))))

    # Group by bias for color and ordering
    by_bias = defaultdict(list)
    for r in rows:
        by_bias[r["bias"]].append(r)
    bias_order = sorted(by_bias.keys())

    # Larger figure with more vertical room for the bottom labels
    fig, ax = plt.subplots(figsize=(13, 6))
    x = 0
    xticks_pos = []
    xticks_lab = []
    bias_centers = []
    for bias in bias_order:
        rs = sorted(by_bias[bias], key=lambda r: r["held_out_dataset"])
        start = x
        for r in rs:
            auroc = float(r["lodo_auroc"])
            lo = float(r["lodo_ci_lo"])
            hi = float(r["lodo_ci_hi"])
            yerr = np.array([[auroc - lo], [hi - auroc]])
            ax.bar(x, auroc, color=BIAS_COLORS.get(bias, "gray"),
                   yerr=yerr, capsize=3, edgecolor="black", linewidth=0.6)
            xticks_pos.append(x)
            xticks_lab.append(r["held_out_dataset"])
            x += 1
        end = x - 1
        bias_centers.append(((start + end) / 2, bias, start, end))
        x += 1  # gap between bias groups

    # Random baseline band: span of per-source mean random-direction AUROCs
    # (each source's mean is averaged over 50 random directions, so this band reflects
    # how a typical random direction performs across our 20 test points).
    rand_means = [float(r["random_auroc_mean"]) for r in rows]
    ax.axhspan(min(rand_means), max(rand_means), color="lightgray", alpha=0.7,
               label=f"random-direction baseline (mean AUROC range {min(rand_means):.2f}--{max(rand_means):.2f})",
               zorder=0)
    ax.axhline(0.5, color="black", linestyle=":", linewidth=1.0,
               label="chance level (AUROC = 0.5)", zorder=1)

    ax.set_xticks(xticks_pos)
    ax.set_xticklabels(xticks_lab, rotation=45, ha="right", fontsize=10)
    ax.set_ylim(0.40, 1.0)
    ax.set_ylabel("LODO probing AUROC", fontsize=12)
    ax.set_title(f"Within-bias direction transfer (LODO) on {MODEL_DISPLAY[model_tag]}",
                 fontsize=12)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)

    # Add bracket + bias name BELOW the rotated x-tick labels.
    # We use figure-level placement so labels don't overlap the plot area.
    # First, give space at the bottom by adjusting subplot.
    fig.subplots_adjust(bottom=0.28)
    # Get the position of x-axis in figure coordinates after layout
    ax_y_bottom = ax.get_position().y0
    for center, bias, start, end in bias_centers:
        # Place bias bracket label well below x-tick labels using fixed figure y
        ax.annotate(
            bias.replace("_", " "),
            xy=(center, 0.0), xycoords=("data", "axes fraction"),
            xytext=(0, -65), textcoords="offset points",
            ha="center", va="top",
            color=BIAS_COLORS[bias], fontsize=11, fontweight="bold",
        )
        # Underline bracket showing the bias group span
        ax.annotate(
            "",
            xy=(start - 0.3, -0.21), xycoords=("data", "axes fraction"),
            xytext=(end + 0.3, -0.21),
            arrowprops=dict(arrowstyle="-", color=BIAS_COLORS[bias], linewidth=2),
        )

    savefig(fig, f"F1_lodo_auroc_{model_tag}")


# -------- F2: bias-level cosine heatmap with bootstrap CIs --------

def fig_bias_cosine(model_tag):
    cb = np.load(CACHE / model_tag / "cross_bias_cosine.npz", allow_pickle=True)
    cos = cb["cos_matrix"]
    ci_lo = cb["ci_lower"]
    ci_hi = cb["ci_upper"]
    biases = [str(b) for b in cb["bias_names"]]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cos, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")

    n = len(biases)
    for i in range(n):
        for j in range(n):
            if i == j:
                txt = "1.00"
            else:
                txt = f"{cos[i,j]:+.2f}\n[{ci_lo[i,j]:+.2f},{ci_hi[i,j]:+.2f}]"
            color = "white" if abs(cos[i, j]) > 0.5 else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=7)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([b.replace("_", "\n") for b in biases], fontsize=8, rotation=0)
    ax.set_yticklabels([b.replace("_", " ") for b in biases], fontsize=8)
    ax.set_title(f"Per-bias direction cosine similarity ({MODEL_DISPLAY[model_tag]})\n"
                 f"point estimate with bootstrap 95% CI [low, high]",
                 fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label="cosine similarity")
    fig.tight_layout()
    savefig(fig, f"F2_bias_cosine_{model_tag}")


# -------- F2: dendrogram --------

def fig_dendrogram(model_tag):
    cb = np.load(CACHE / model_tag / "cross_bias_cosine.npz", allow_pickle=True)
    Z = cb["linkage"]
    biases = [str(b) for b in cb["bias_names"]]

    fig, ax = plt.subplots(figsize=(8, 4))
    dendrogram(Z.astype(np.float64), labels=biases, ax=ax,
               leaf_rotation=20, color_threshold=0)
    for tick_label in ax.get_xticklabels():
        b = tick_label.get_text()
        tick_label.set_color(BIAS_COLORS.get(b, "black"))
    ax.set_ylabel("distance (1 - cosine)")
    ax.set_title(f"Hierarchical clustering of per-bias directions — {MODEL_DISPLAY[model_tag]}\n"
                 f"average linkage, cosine distance",
                 fontsize=10)
    fig.tight_layout()
    savefig(fig, f"F2_dendrogram_{model_tag}")


# -------- F2: PCA of per-source bias directions --------

def fig_pca_source_directions(model_tag):
    """2D PCA of the 21 per-source bias directions, colored by bias type.

    Visualizes the cross-bias structure from Section 6 in a single 2D scatter:
    sources from the same bias should cluster (within-bias generalization);
    biases that share representational subspace appear near each other.

    Llama: expected to be more scattered overall, with distractor_fact and
    suggested_answer near each other (cosine 0.56).
    Qwen: expected to show a 4-bias cluster (suggested_answer, distractor_fact,
    wrong_few_shot, spurious_few_shot_squares) plus distractor_argument and
    spurious_few_shot_hindsight as outliers.
    """
    d_npz = np.load(CACHE / model_tag / "directions.npz", allow_pickle=True)
    dirs = d_npz["directions"].astype(np.float32)        # (21, n_layers*H)
    source_names = [str(s) for s in d_npz["source_names"]]
    bias_names = [str(b) for b in d_npz["bias_names"]]

    # PCA via SVD on centered data
    centered = dirs - dirs.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    pcs = centered @ Vt[:2].T                            # (21, 2)
    var_explained = (S ** 2) / np.sum(S ** 2)

    # Marker per dataset
    MARKER = {
        "mmlu": "^",
        "hellaswag": "o",
        "logiqa": "s",
        "truthfulqa": "D",
        "hindsight_neglect": "*",
    }

    fig, ax = plt.subplots(figsize=(7.5, 6))
    for i, src in enumerate(source_names):
        bias = bias_names[i]
        dataset = src.split("__")[-1]
        color = BIAS_COLORS.get(bias, "gray")
        marker = MARKER.get(dataset, "o")
        size = 220 if dataset == "hindsight_neglect" else 140
        ax.scatter(pcs[i, 0], pcs[i, 1], c=[color], marker=marker, s=size,
                   edgecolors="black", linewidths=0.7, alpha=0.9, zorder=3)
        # Short dataset label next to marker
        short = {"hellaswag": "hs", "logiqa": "lq", "mmlu": "mm",
                 "truthfulqa": "tq", "hindsight_neglect": "hn"}.get(dataset, dataset[:4])
        ax.annotate(short, (pcs[i, 0], pcs[i, 1]),
                    fontsize=8, xytext=(6, 3), textcoords="offset points",
                    color="black", zorder=4)

    # Legend by bias type
    biases_seen = []
    for b in bias_names:
        if b not in biases_seen:
            biases_seen.append(b)
    bias_handles = [plt.Line2D([0], [0], marker="o", color="w",
                                markerfacecolor=BIAS_COLORS.get(b, "gray"),
                                markersize=10, label=b.replace("_", " "),
                                markeredgecolor="black", markeredgewidth=0.5)
                    for b in biases_seen]
    leg1 = ax.legend(handles=bias_handles, loc="upper left", fontsize=8.5,
                     title="bias type", title_fontsize=9, framealpha=0.95)
    ax.add_artist(leg1)
    # Second legend for marker shapes
    shape_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                   markeredgecolor="black", markersize=10, label="hellaswag"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
                   markeredgecolor="black", markersize=10, label="logiqa"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="gray",
                   markeredgecolor="black", markersize=10, label="mmlu"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="gray",
                   markeredgecolor="black", markersize=10, label="truthfulqa"),
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="gray",
                   markeredgecolor="black", markersize=12, label="hindsight"),
    ]
    ax.legend(handles=shape_handles, loc="lower right", fontsize=8.5,
              title="dataset", title_fontsize=9, framealpha=0.95)

    ax.axhline(0, color="black", linewidth=0.5, alpha=0.25)
    ax.axvline(0, color="black", linewidth=0.5, alpha=0.25)
    ax.set_xlabel(f"PC1 ({var_explained[0] * 100:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var_explained[1] * 100:.1f}% variance)", fontsize=11)
    ax.set_title(
        f"PCA of per-source bias directions — {MODEL_DISPLAY[model_tag]}",
        fontsize=12)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    savefig(fig, f"F2_pca_source_directions_{model_tag}")


# -------- F2: PCA of pair activations (LiReF-style) --------

# Best-AUROC layer per model (from Section 5.3 / Appendix C layer-wise probing).
BEST_LAYER_FOR_PCA = {"llama31_8b": 17, "qwen25_7b": 20}
MODEL_HIDDEN_SIZE = {"llama31_8b": 4096, "qwen25_7b": 3584}


def fig_pca_pair_activations(model_tag, n_per_source=60, seed=42):
    """LiReF-style 2D PCA of pair activations at the best layer, with
    per-source centering (residuals only).

    Why per-source centering: pair activations vary a lot by source (different
    prompts, different bias types). That across-source variance dominates the
    raw PCA and hides the within-source flipped-vs-resisted axis. We center
    each source by its own mean so the PCA reveals the *within-source*
    structure that the bias direction is supposed to capture.

    Points = pair hidden states at the best-AUROC layer (Llama 17, Qwen 20),
    pooled across all 21 sources, subsampled to n_per_source per source,
    and per-source mean-centered. Color: flipped (red) vs resisted (blue).
    Black arrow: diff-of-means direction from per-source-centered resisted
    centroid to per-source-centered flipped centroid.
    """
    layer = BEST_LAYER_FOR_PCA[model_tag]
    H = MODEL_HIDDEN_SIZE[model_tag]

    rng = np.random.default_rng(seed)
    cache_dir = CACHE / model_tag
    source_dirs = sorted(
        p.name for p in cache_dir.iterdir() if p.is_dir() and "__" in p.name
    )

    all_X, all_y = [], []
    for src in source_dirs:
        X_path = cache_dir / src / "X_biased.npy"
        meta_path = cache_dir / src / "pairs_meta.json"
        if not X_path.exists() or not meta_path.exists():
            continue
        X = np.load(X_path)
        with open(meta_path) as f:
            meta = json.load(f)
        if X.shape[0] != len(meta):
            continue
        # Slice to the best-AUROC layer only
        X_layer = X[:, layer * H : (layer + 1) * H].astype(np.float32)
        n_keep = min(n_per_source, len(meta))
        if len(meta) > n_keep:
            idx = rng.choice(len(meta), size=n_keep, replace=False)
            X_layer = X_layer[idx]
            meta_sub = [meta[i] for i in idx]
        else:
            meta_sub = meta
        # Per-source centering: subtract this source's mean (over kept pairs)
        # so the resulting points carry only within-source structure.
        X_layer = X_layer - X_layer.mean(axis=0, keepdims=True)
        for x, m in zip(X_layer, meta_sub):
            all_X.append(x)
            all_y.append(1 if m["category"] == "flipped" else 0)

    X_all = np.stack(all_X)              # (N, H), already per-source-centered
    y = np.array(all_y)
    n_flip = int((y == 1).sum())
    n_rsst = int((y == 0).sum())

    # Class centroids in the per-source-centered space
    mu_flip = X_all[y == 1].mean(axis=0)
    mu_rsst = X_all[y == 0].mean(axis=0)

    # PCA via SVD. Data is already mean-centered (per-source, and approximately
    # zero overall mean by construction since each source contributes zero-mean).
    _, S, Vt = np.linalg.svd(X_all, full_matrices=False)
    pcs = X_all @ Vt[:2].T               # (N, 2)
    var_explained = (S ** 2) / np.sum(S ** 2)

    # Project the two class centroids into the same PC plane
    flip_centroid_2d = mu_flip @ Vt[:2].T
    rsst_centroid_2d = mu_rsst @ Vt[:2].T

    fig, ax = plt.subplots(figsize=(7.5, 6))
    flipped_mask = y == 1
    ax.scatter(pcs[~flipped_mask, 0], pcs[~flipped_mask, 1],
               c="#1f77b4", s=10, alpha=0.30, edgecolors="none",
               label=f"resisted  (n={n_rsst})")
    ax.scatter(pcs[flipped_mask, 0], pcs[flipped_mask, 1],
               c="#d62728", s=10, alpha=0.30, edgecolors="none",
               label=f"flipped  (n={n_flip})")

    # Centroids
    ax.scatter(*rsst_centroid_2d, c="#1f77b4", s=260, marker="X",
               edgecolors="black", linewidths=2, zorder=10)
    ax.scatter(*flip_centroid_2d, c="#d62728", s=260, marker="X",
               edgecolors="black", linewidths=2, zorder=10)

    # Diff-of-means arrow from resisted to flipped centroid
    ax.annotate(
        "", xy=flip_centroid_2d, xytext=rsst_centroid_2d,
        arrowprops=dict(arrowstyle="->", color="black", lw=2.4, alpha=0.95),
        zorder=11,
    )
    midpt = ((rsst_centroid_2d[0] + flip_centroid_2d[0]) / 2,
             (rsst_centroid_2d[1] + flip_centroid_2d[1]) / 2)
    ax.annotate(
        "bias direction", xy=midpt, xytext=(10, 10),
        textcoords="offset points", fontsize=10.5, fontweight="bold",
        color="black",
    )

    ax.axhline(0, color="black", linewidth=0.5, alpha=0.2)
    ax.axvline(0, color="black", linewidth=0.5, alpha=0.2)
    ax.set_xlabel(f"PC1 ({var_explained[0] * 100:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var_explained[1] * 100:.1f}% variance)", fontsize=11)
    ax.set_title(
        f"PCA of pair activations (layer {layer}, per-source centered) — {MODEL_DISPLAY[model_tag]}\n"
        f"flipped vs resisted, pooled across 21 sources",
        fontsize=11)
    ax.legend(loc="best", fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    savefig(fig, f"F2_pca_pair_activations_{model_tag}")


# -------- F2: cross-bias probing AUROC heatmap --------

def fig_cross_bias_probe(model_tag):
    path = CACHE / model_tag / "cross_bias_probing.npz"
    if not path.exists():
        print(f"  [skip cross-bias probe figure: {path} missing]")
        return
    cp = np.load(path, allow_pickle=True)
    M = cp["matrix"]
    biases = [str(b) for b in cp["bias_names"]]
    n = len(biases)

    fig, ax = plt.subplots(figsize=(8, 7))
    # AUROC ranges 0-1; center at 0.5
    norm = mcolors.TwoSlopeNorm(vmin=0.4, vcenter=0.5, vmax=1.0)
    im = ax.imshow(M, cmap="RdBu_r", norm=norm, aspect="equal")

    for i in range(n):
        for j in range(n):
            v = M[i, j]
            if np.isnan(v):
                txt = "n/a"
                color = "gray"
            else:
                txt = f"{v:.3f}"
                color = "white" if abs(v - 0.5) > 0.25 else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=8)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([b.replace("_", "\n") for b in biases], fontsize=8)
    ax.set_yticklabels([b.replace("_", " ") for b in biases], fontsize=8)
    ax.set_xlabel("test (predict flips on this bias)", fontsize=9)
    ax.set_ylabel("train (use this bias's direction)", fontsize=9)
    ax.set_title(f"Cross-bias probing AUROC — {MODEL_DISPLAY[model_tag]}\n"
                 f"diagonal = within-bias LODO; off-diagonal = cross-bias transfer",
                 fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label="AUROC")
    fig.tight_layout()
    savefig(fig, f"F2_cross_bias_probe_{model_tag}")


# -------- F2: cross-bias intervention recovery heatmap --------

def fig_cross_bias_intervention(model_tag):
    """Heatmap of cross-bias causal intervention recovery.

    Rows = source bias (whose direction is used to intervene).
    Cols = target bias (whose flipped pairs are intervened on).
    Cell = mean flip-recovery rate across the target bias's held-out sources.
    Diagonal reproduces Section 7's within-bias recovery.
    """
    path = CACHE / model_tag / "cross_bias_intervention.csv"
    if not path.exists():
        print(f"  [skip cross-bias intervention figure: {path} missing]")
        return

    rows = []
    with open(path) as f:
        header = next(f).strip().split(",")
        for line in f:
            rows.append(dict(zip(header, line.strip().split(","))))
    real = [r for r in rows if r["direction_type"] == "real"]
    if not real:
        print(f"  [skip cross-bias intervention figure: no real rows for {model_tag}]")
        return

    biases = sorted(set(r["source_bias"] for r in real)
                    | set(r["target_bias"] for r in real))
    n = len(biases)
    bidx = {b: i for i, b in enumerate(biases)}

    # Build matrix: mean recovery per (source, target)
    acc = defaultdict(list)
    for r in real:
        acc[(r["source_bias"], r["target_bias"])].append(float(r["flip_recovery"]))
    M = np.full((n, n), np.nan, dtype=np.float32)
    for (sb, tb), vals in acc.items():
        M[bidx[sb], bidx[tb]] = float(np.mean(vals))

    fig, ax = plt.subplots(figsize=(8, 7))
    norm = mcolors.Normalize(vmin=0.0, vmax=max(0.4, np.nanmax(M)))
    im = ax.imshow(M, cmap="Blues", norm=norm, aspect="equal")

    for i in range(n):
        for j in range(n):
            v = M[i, j]
            if np.isnan(v):
                ax.text(j, i, "n/a", ha="center", va="center",
                        color="gray", fontsize=8)
            else:
                color = "white" if v > 0.5 * norm.vmax else "black"
                # Box the diagonal (within-bias) cells
                weight = "bold" if i == j else "normal"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=color, fontsize=9, fontweight=weight)
                if i == j:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                               fill=False, edgecolor="black",
                                               linewidth=2))

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([b.replace("_", "\n") for b in biases], fontsize=8)
    ax.set_yticklabels([b.replace("_", " ") for b in biases], fontsize=8)
    ax.set_xlabel("target: intervene on this bias's flipped pairs", fontsize=9)
    ax.set_ylabel("source: use this bias's direction", fontsize=9)
    ax.set_title(f"Cross-bias intervention recovery — {MODEL_DISPLAY[model_tag]}\n"
                 f"boxed diagonal = within-bias (Section 7); "
                 f"off-diagonal = causal cross-bias transfer",
                 fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label="flip-recovery rate")
    fig.tight_layout()
    savefig(fig, f"F2_cross_bias_intervention_{model_tag}")


# -------- F3: per-layer LODO AUROC curve --------

def fig_layer_wise(model_tag):
    """Per-layer LODO AUROC curve: real direction vs random baseline."""
    csv_path = CACHE / model_tag / "layer_wise_lodo.csv"
    if not csv_path.exists():
        print(f"  [skip layer-wise figure: {csv_path} missing]")
        return

    rows = []
    with open(csv_path) as f:
        header = next(f).strip().split(",")
        for line in f:
            parts = line.strip().split(",")
            d = dict(zip(header, parts))
            d["layer"] = int(d["layer"])
            d["auroc"] = float(d["auroc"])
            d["random_auroc"] = float(d["random_auroc"])
            rows.append(d)

    # Aggregate per layer
    by_layer = defaultdict(list)
    for r in rows:
        by_layer[r["layer"]].append(r)
    layers = sorted(by_layer.keys())

    mean_auroc = np.array([np.mean([r["auroc"] for r in by_layer[ell]]) for ell in layers])
    se_auroc = np.array([np.std([r["auroc"] for r in by_layer[ell]], ddof=1) / np.sqrt(len(by_layer[ell])) for ell in layers])
    mean_random = np.array([np.mean([r["random_auroc"] for r in by_layer[ell]]) for ell in layers])

    # Per-bias means
    by_layer_bias = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_layer_bias[r["layer"]][r["bias"]].append(r["auroc"])
    biases_present = sorted({r["bias"] for r in rows})
    n_test_points = len(by_layer[layers[0]])

    fig, ax = plt.subplots(figsize=(8, 5))

    # Per-bias thin lines (background)
    for bias in biases_present:
        per_layer_b = np.array([np.mean(by_layer_bias[ell][bias]) for ell in layers])
        ax.plot(layers, per_layer_b, color=BIAS_COLORS.get(bias, "gray"),
                alpha=0.4, linewidth=1.2, label=bias.replace("_", " "))

    # Main mean + SE band
    ax.fill_between(layers, mean_auroc - se_auroc, mean_auroc + se_auroc,
                    color="black", alpha=0.15, label="mean ± SE")
    ax.plot(layers, mean_auroc, color="black", linewidth=2.2, label=f"mean across {n_test_points} test points")

    # Random baseline
    ax.plot(layers, mean_random, color="gray", linestyle="--", linewidth=1.4,
            label="random baseline (mean)")
    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8, alpha=0.5)

    # Mark best layer
    best_ell = int(layers[np.argmax(mean_auroc)])
    ax.axvline(best_ell, color="red", linestyle=":", linewidth=1.0, alpha=0.6)
    ax.annotate(f"best layer = {best_ell}\nAUROC = {mean_auroc.max():.3f}",
                xy=(best_ell, mean_auroc.max()),
                xytext=(best_ell + 1, mean_auroc.max() - 0.04),
                fontsize=9,
                arrowprops=dict(arrowstyle="-", color="red", linewidth=0.8))

    ax.set_xlabel("transformer layer index", fontsize=11)
    ax.set_ylabel("LODO probing AUROC (layer-only)", fontsize=11)
    ax.set_title(f"Per-layer LODO probing AUROC on {MODEL_DISPLAY[model_tag]}",
                 fontsize=11)
    ax.set_ylim(0.40, 1.0)
    ax.set_xlim(min(layers) - 0.5, max(layers) + 0.5)
    ax.legend(loc="lower right", fontsize=8, ncol=2, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig(fig, f"F3_layer_wise_{model_tag}")


# -------- F4: intervention plots --------

def load_intervention_csv(model_tag):
    csv_path = CACHE / model_tag / "intervention_results.csv"
    if not csv_path.exists():
        return None
    rows = []
    with open(csv_path) as f:
        header = next(f).strip().split(",")
        for line in f:
            parts = line.strip().split(",")
            d = dict(zip(header, parts))
            d["alpha"] = float(d["alpha"])
            for k in ("overall_accuracy", "overall_target_rate",
                      "flip_recovery", "flip_still_target",
                      "rsst_preservation", "rsst_drift_to_target"):
                d[k] = float(d[k]) if d[k] not in ("nan", "") else float("nan")
            d["n_flipped"] = int(d["n_flipped"])
            d["n_resisted"] = int(d["n_resisted"])
            rows.append(d)
    return rows


def fig_intervention_alpha_sweep(model_tag):
    """Two-panel: flip-recovery and overall target rate vs alpha, real vs random.

    Covers the full bidirectional alpha range. Positive alpha subtracts the
    direction (debiases); negative alpha adds it (induces bias).
    """
    rows = load_intervention_csv(model_tag)
    if not rows:
        print(f"  [skip intervention alpha-sweep: no CSV for {model_tag}]")
        return
    rows = [r for r in rows if r["mode"] == "all_layers"]
    if not rows:
        return
    alphas = sorted({r["alpha"] for r in rows})

    def per_alpha(rows_subset, key):
        return [np.mean([r[key] for r in rows_subset if r["alpha"] == a]) for a in alphas]

    def per_alpha_se(rows_subset, key):
        out = []
        for a in alphas:
            v = [r[key] for r in rows_subset if r["alpha"] == a]
            out.append(np.std(v, ddof=1) / np.sqrt(max(1, len(v))) if len(v) > 1 else 0.0)
        return out

    real_rows = [r for r in rows if r["direction_type"] == "real"]
    rand_rows = [r for r in rows if r["direction_type"] == "random"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: flip recovery
    ax = axes[0]
    real_recov = per_alpha(real_rows, "flip_recovery")
    rand_recov = per_alpha(rand_rows, "flip_recovery")
    real_recov_se = per_alpha_se(real_rows, "flip_recovery")
    ax.errorbar(alphas, real_recov, yerr=real_recov_se, color="#1f77b4", linewidth=2.0,
                marker="o", markersize=6, capsize=4, label="real direction")
    ax.errorbar(alphas, rand_recov, yerr=None, color="gray", linestyle="--", linewidth=1.5,
                marker="s", markersize=5, label="random direction baseline")
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_xlabel(r"intervention strength $\alpha$", fontsize=11)
    ax.set_ylabel("mean recovery rate on flipped pairs", fontsize=11)
    ax.set_title("Recovery rate vs $\\alpha$", fontsize=11)
    ax.set_ylim(0, max(max(real_recov), max(rand_recov)) * 1.15 + 0.05)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right panel: overall bias-target rate (the headline bidirectional metric)
    ax = axes[1]
    real_target = per_alpha(real_rows, "overall_target_rate")
    rand_target = per_alpha(rand_rows, "overall_target_rate")
    real_target_se = per_alpha_se(real_rows, "overall_target_rate")
    ax.errorbar(alphas, real_target, yerr=real_target_se, color="#d62728", linewidth=2.0,
                marker="o", markersize=6, capsize=4, label="real direction")
    ax.errorbar(alphas, rand_target, yerr=None, color="gray", linestyle="--", linewidth=1.5,
                marker="s", markersize=5, label="random direction baseline")
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_xlabel(r"intervention strength $\alpha$", fontsize=11)
    ax.set_ylabel("P(bias-target) overall", fontsize=11)
    ax.set_title("Bias-target rate vs $\\alpha$", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Causal intervention on {MODEL_DISPLAY[model_tag]} "
                 r"($\alpha>0$: debias, $\alpha<0$: induce)", fontsize=12)
    fig.tight_layout()
    savefig(fig, f"F4_intervention_alpha_sweep_{model_tag}")


def fig_intervention_per_source(model_tag, fixed_alpha=0.5):
    """Per-source recovery bar chart at a fixed alpha (defaults to 0.5)."""
    rows = load_intervention_csv(model_tag)
    if not rows:
        return
    rows = [r for r in rows if r["mode"] == "all_layers" and r["alpha"] == fixed_alpha]
    if not rows:
        return
    by_source_real = {r["held_out_source"]: r for r in rows if r["direction_type"] == "real"}
    by_source_rand = {r["held_out_source"]: r for r in rows if r["direction_type"] == "random"}

    by_bias = defaultdict(list)
    for src, r in by_source_real.items():
        by_bias[r["bias"]].append((src, r))
    bias_order = sorted(by_bias.keys())

    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = 0
    xticks_pos = []
    xticks_lab = []
    bias_centers = []
    for bias in bias_order:
        srcs = sorted(by_bias[bias], key=lambda t: t[0])
        start = x
        for src, r in srcs:
            real_rec = r["flip_recovery"]
            rand_rec = by_source_rand.get(src, {}).get("flip_recovery", 0)
            ax.bar(x - 0.20, real_rec, width=0.4, color=BIAS_COLORS.get(bias, "gray"),
                   edgecolor="black", linewidth=0.5,
                   label="real direction" if x == 0 else None)
            ax.bar(x + 0.20, rand_rec, width=0.4, color="lightgray",
                   edgecolor="black", linewidth=0.5,
                   label="random direction" if x == 0 else None)
            xticks_pos.append(x)
            xticks_lab.append(src.split("__")[-1])
            x += 1
        end = x - 1
        bias_centers.append(((start + end) / 2, bias, start, end))
        x += 1

    ax.set_xticks(xticks_pos)
    ax.set_xticklabels(xticks_lab, rotation=45, ha="right", fontsize=10)
    ax.set_ylim(0, 0.6)
    ax.set_ylabel(f"recovery rate on flipped pairs (at $\\alpha={fixed_alpha}$)",
                  fontsize=11)
    ax.set_title(f"Per-source intervention recovery rate at $\\alpha={fixed_alpha}$ — "
                 f"{MODEL_DISPLAY[model_tag]}", fontsize=11)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Bias group labels below
    fig.subplots_adjust(bottom=0.28)
    for center, bias, start, end in bias_centers:
        ax.annotate(bias.replace("_", " "),
                    xy=(center, 0.0), xycoords=("data", "axes fraction"),
                    xytext=(0, -65), textcoords="offset points",
                    ha="center", va="top",
                    color=BIAS_COLORS[bias], fontsize=11, fontweight="bold")
        ax.annotate("",
                    xy=(start - 0.3, -0.21), xycoords=("data", "axes fraction"),
                    xytext=(end + 0.3, -0.21),
                    arrowprops=dict(arrowstyle="-", color=BIAS_COLORS[bias], linewidth=2))

    savefig(fig, f"F4_intervention_per_source_{model_tag}")


def fig_intervention_per_bias_bidirectional(
    sweet_alpha_per_model={"llama31_8b": 0.5, "qwen25_7b": 4.0},
    mode_per_model=None,
    output_suffix="",
):
    """Per-class bidirectional causal effects in a 2x2 panel grid.

    Rows separate the two metrics so they don't share a y-axis:
      - Top row: recovery rate on flipped pairs at alpha = +best (debias)
      - Bottom row: drift rate on resisted pairs at alpha = -best (induce)

    Columns are models (Llama, Qwen). Random-direction baseline is shown as a
    thin dashed horizontal line per panel with its value annotated.

    Recovery and drift are per-class metrics on different sub-populations
    (flipped vs resisted) with different "good" outcomes (correct vs
    bias-target). Splitting them into separate rows makes that explicit.

    Error bars are SEM across the 4 LODO sources per bias.
    """
    BLUE = "#1f4b8a"
    RED = "#a02525"
    EPS = 1e-6
    if mode_per_model is None:
        mode_per_model = {"llama31_8b": "all_layers", "qwen25_7b": "all_layers"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)

    for col_idx, model_tag in enumerate(("llama31_8b", "qwen25_7b")):
        ax = axes[col_idx]
        rows = load_intervention_csv(model_tag)
        if not rows:
            continue
        a_best = sweet_alpha_per_model[model_tag]
        mode = mode_per_model[model_tag]

        rows = [r for r in rows if r["mode"] == mode]

        # groups[bias] = dict of lists across LODO sources
        groups = defaultdict(lambda: {
            "real_recovery": [], "random_recovery": [],
            "real_drift": [], "random_drift": [],
        })
        for r in rows:
            bias = r["bias"]
            alpha = r["alpha"]
            dt = r["direction_type"]
            if abs(alpha - a_best) < EPS:
                v = r["flip_recovery"]
                if not (isinstance(v, float) and np.isnan(v)):
                    groups[bias][f"{dt}_recovery"].append(v)
            elif abs(alpha + a_best) < EPS:
                v = r["rsst_drift_to_target"]
                if not (isinstance(v, float) and np.isnan(v)):
                    groups[bias][f"{dt}_drift"].append(v)

        def stat(xs):
            if not xs:
                return 0.0, 0.0
            m = float(np.mean(xs))
            se = float(np.std(xs, ddof=1) / np.sqrt(len(xs))) if len(xs) > 1 else 0.0
            return m, se

        bias_order = sorted(groups.keys())
        n_bias = len(bias_order)
        xs = np.arange(n_bias)
        bar_w = 0.36
        bias_labels = [b.replace("_", "\n") for b in bias_order]

        # Two bars per bias: recovery (debias, +α) and drift (induce, -α)
        real_rec = [stat(groups[b]["real_recovery"]) for b in bias_order]
        real_drift = [stat(groups[b]["real_drift"]) for b in bias_order]
        rand_rec_all, rand_drift_all = [], []
        for b in bias_order:
            rand_rec_all.extend(groups[b]["random_recovery"])
            rand_drift_all.extend(groups[b]["random_drift"])
        rand_rec_mean = float(np.mean(rand_rec_all)) if rand_rec_all else 0.0
        rand_drift_mean = float(np.mean(rand_drift_all)) if rand_drift_all else 0.0

        ax.bar(xs - bar_w / 2, [m for m, _ in real_rec], width=bar_w,
               yerr=[se for _, se in real_rec], capsize=3,
               color=BLUE, edgecolor="black", linewidth=0.5,
               label=rf"recovery on flipped, $\alpha = +{a_best:g}$ (debias)")
        ax.bar(xs + bar_w / 2, [m for m, _ in real_drift], width=bar_w,
               yerr=[se for _, se in real_drift], capsize=3,
               color=RED, edgecolor="black", linewidth=0.5,
               label=rf"drift on resisted, $\alpha = -{a_best:g}$ (induce)")
        for x, (m, se) in zip(xs, real_rec):
            ax.text(x - bar_w / 2, m + se + 0.012, f"{m:.2f}",
                    ha="center", va="bottom", fontsize=9, color="black")
        for x, (m, se) in zip(xs, real_drift):
            ax.text(x + bar_w / 2, m + se + 0.012, f"{m:.2f}",
                    ha="center", va="bottom", fontsize=9, color="black")

        ax.axhline(rand_rec_mean, color=BLUE, linestyle="--",
                   linewidth=1.3, alpha=0.7,
                   label=f"random $d$ recovery (mean {rand_rec_mean:.2f})")
        ax.axhline(rand_drift_mean, color=RED, linestyle="--",
                   linewidth=1.3, alpha=0.7,
                   label=f"random $d$ drift (mean {rand_drift_mean:.2f})")

        ax.set_xticks(xs)
        ax.set_xticklabels(bias_labels, fontsize=9)
        ax.set_xlim(-0.6, n_bias - 0.4)
        ax.set_ylim(0, 0.5)
        if col_idx == 0:
            ax.set_ylabel("causal effect rate\n(on intended sub-population)",
                          fontsize=11)
        ax.set_title(MODEL_DISPLAY[model_tag], fontsize=12)
        ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Per-class causal effects of the bias direction: "
        r"blue $=$ recovery on flipped (after subtracting $d$); "
        r"red $=$ drift on resisted (after adding $d$)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    savefig(fig, f"F3_intervention_per_bias_bidirectional{output_suffix}")


def fig_intervention_bidirectional():
    """Headline bidirectional steering figure: bias-target rate vs alpha for
    both models on a single panel. Positive alpha debiases; negative induces.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, model_tag in zip(axes, ("llama31_8b", "qwen25_7b")):
        rows = load_intervention_csv(model_tag)
        if not rows:
            continue
        rows = [r for r in rows if r["mode"] == "all_layers"]
        alphas = sorted({r["alpha"] for r in rows})

        def per_alpha(rows_subset, key):
            return [np.mean([r[key] for r in rows_subset if r["alpha"] == a]) for a in alphas]

        def per_alpha_se(rows_subset, key):
            out = []
            for a in alphas:
                v = [r[key] for r in rows_subset if r["alpha"] == a]
                out.append(np.std(v, ddof=1) / np.sqrt(max(1, len(v))) if len(v) > 1 else 0.0)
            return out

        real_rows = [r for r in rows if r["direction_type"] == "real"]
        rand_rows = [r for r in rows if r["direction_type"] == "random"]

        real_target = per_alpha(real_rows, "overall_target_rate")
        rand_target = per_alpha(rand_rows, "overall_target_rate")
        real_target_se = per_alpha_se(real_rows, "overall_target_rate")

        ax.axvspan(min(alphas) - 0.5, 0, color="#d62728", alpha=0.06)
        ax.axvspan(0, max(alphas) + 0.5, color="#1f77b4", alpha=0.06)
        ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)

        ax.errorbar(alphas, real_target, yerr=real_target_se, color="#222222",
                    linewidth=2.0, marker="o", markersize=6, capsize=4,
                    label="real direction")
        ax.plot(alphas, rand_target, color="gray", linestyle="--", linewidth=1.5,
                marker="s", markersize=5, label="random direction")

        baseline = next((t for a, t in zip(alphas, real_target) if a == 0.0), None)
        if baseline is not None:
            ax.axhline(baseline, color="black", linestyle=":", linewidth=0.8, alpha=0.4)

        ax.set_xlim(min(alphas) - 0.5, max(alphas) + 0.5)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel(r"intervention strength $\alpha$", fontsize=11)
        if model_tag == "llama31_8b":
            ax.set_ylabel("P(bias-target) overall", fontsize=11)
        ax.set_title(MODEL_DISPLAY[model_tag], fontsize=12)
        ax.text(0.02, 0.97, r"$\alpha<0$: induce", transform=ax.transAxes,
                color="#a02525", fontsize=9.5, fontweight="bold",
                ha="left", va="top")
        ax.text(0.98, 0.97, r"$\alpha>0$: debias", transform=ax.transAxes,
                color="#1f4b8a", fontsize=9.5, fontweight="bold",
                ha="right", va="top")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Bidirectional causal steering along the bias direction",
                 fontsize=12.5, y=1.01)
    fig.tight_layout()
    savefig(fig, "F4_intervention_bidirectional")


def load_selectivity_csv(model_tag):
    csv_path = CACHE / model_tag / "selectivity_results.csv"
    if not csv_path.exists():
        return None
    rows = []
    with open(csv_path) as f:
        header = next(f).strip().split(",")
        for line in f:
            parts = line.strip().split(",")
            d = dict(zip(header, parts))
            d["alpha"] = float(d["alpha"])
            for k in ("unbiased_acc", "unbiased_target_rate",
                      "unbiased_acc_flipped", "unbiased_acc_resisted"):
                d[k] = float(d[k]) if d[k] not in ("nan", "") else float("nan")
            d["n_flipped"] = int(d["n_flipped"])
            d["n_resisted"] = int(d["n_resisted"])
            rows.append(d)
    return rows


def fig_intervention_selectivity(model_tag, sweet_alpha_per_model={"llama31_8b": 0.5, "qwen25_7b": 4.0},
                                  mode="all_layers", output_suffix=""):
    """Per-source selectivity check: P(correct on unbiased + intervention).

    For each source, two bars at the sweet-spot alpha:
      - real direction: P(correct on unbiased + alpha * d_bias subtracted)
      - random direction: P(correct on unbiased + alpha * d_random subtracted)

    Dotted line at 1.0 marks the no-intervention reference (by construction of
    filtered pairs, P(correct on unbiased, no intervention) = 1.0).

    If the real bar stays near 1.0, intervention is bias-specific (selective).
    If the real bar drops noticeably below the random bar, the bias direction
    has direction-specific collateral damage on clean inputs.
    """
    sel_rows = load_selectivity_csv(model_tag)
    if not sel_rows:
        print(f"  [skip selectivity: no CSV for {model_tag}]")
        return

    alpha = sweet_alpha_per_model[model_tag]
    EPS = 1e-6
    # Filter by mode (default 'all_layers'); old rows without 'mode' col were migrated
    sel_rows = [r for r in sel_rows if r.get("mode", "all_layers") == mode]
    sel_real = {r["held_out_source"]: r for r in sel_rows
                if r["direction_type"] == "real" and abs(r["alpha"] - alpha) < EPS}
    sel_rand = {r["held_out_source"]: r for r in sel_rows
                if r["direction_type"] == "random" and abs(r["alpha"] - alpha) < EPS}

    by_bias = defaultdict(list)
    for src in sel_real:
        bias = sel_real[src]["bias"]
        by_bias[bias].append(src)
    bias_order = sorted(by_bias.keys())

    fig, ax = plt.subplots(figsize=(13, 5.0))
    x = 0
    xticks_pos = []
    xticks_lab = []
    bias_centers = []
    bar_w = 0.38
    legend_done = False
    for bias in bias_order:
        srcs = sorted(by_bias[bias])
        start = x
        for src in srcs:
            r_real = sel_real.get(src)
            r_rand = sel_rand.get(src)
            if r_real is None or r_rand is None:
                continue
            bar_real = r_real["unbiased_acc"]
            bar_rand = r_rand["unbiased_acc"]
            color = BIAS_COLORS.get(bias, "gray")
            labels = [None, None]
            if not legend_done:
                labels = [r"real bias direction", r"random direction (control)"]
                legend_done = True
            ax.bar(x - bar_w / 2, bar_real, width=bar_w, color=color,
                   edgecolor="black", linewidth=0.5, label=labels[0])
            ax.bar(x + bar_w / 2, bar_rand, width=bar_w, color="lightgray",
                   edgecolor="black", linewidth=0.5, label=labels[1])
            xticks_pos.append(x)
            xticks_lab.append(src.split("__")[-1])
            x += 1
        end = x - 1
        bias_centers.append(((start + end) / 2, bias, start, end))
        x += 1

    ax.set_xticks(xticks_pos)
    ax.set_xticklabels(xticks_lab, rotation=45, ha="right", fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.text(len(xticks_pos), 1.005, "no-intervention reference (=1.0)",
            ha="right", va="bottom", fontsize=9, color="black", alpha=0.7,
            transform=ax.transData)
    ax.set_ylabel("P(correct letter) on unbiased prompt", fontsize=11)
    ax.set_title(f"Intervention selectivity on unbiased prompts — "
                 f"{MODEL_DISPLAY[model_tag]} ($\\alpha={alpha}$)", fontsize=11.5)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, axis="y")

    fig.subplots_adjust(bottom=0.28, top=0.92)
    for center, bias, start, end in bias_centers:
        ax.annotate(bias.replace("_", " "),
                    xy=(center, 0.0), xycoords=("data", "axes fraction"),
                    xytext=(0, -65), textcoords="offset points",
                    ha="center", va="top",
                    color=BIAS_COLORS[bias], fontsize=11, fontweight="bold")
        ax.annotate("",
                    xy=(start - 0.3, -0.21), xycoords=("data", "axes fraction"),
                    xytext=(end + 0.3, -0.21),
                    arrowprops=dict(arrowstyle="-", color=BIAS_COLORS[bias], linewidth=2))

    savefig(fig, f"F4_intervention_selectivity_{model_tag}{output_suffix}")


def fig_intervention_tradeoff(model_tag):
    """Recovery vs selectivity scatter: each point is (source, alpha)."""
    rows = load_intervention_csv(model_tag)
    if not rows:
        return
    rows = [r for r in rows if r["mode"] == "all_layers" and r["direction_type"] == "real" and r["alpha"] > 0]
    if not rows:
        return
    alphas = sorted({r["alpha"] for r in rows})
    cmap = plt.cm.viridis
    norm = mcolors.Normalize(vmin=min(alphas), vmax=max(alphas))

    fig, ax = plt.subplots(figsize=(7.5, 6))
    for r in rows:
        color = cmap(norm(r["alpha"]))
        ax.scatter(r["flip_recovery"], r["rsst_preservation"],
                   c=[color], s=60, alpha=0.85, edgecolors="black", linewidths=0.4)

    # Per-alpha mean point overlay
    for a in alphas:
        a_rows = [r for r in rows if r["alpha"] == a]
        mx = np.mean([r["flip_recovery"] for r in a_rows])
        my = np.mean([r["rsst_preservation"] for r in a_rows])
        ax.scatter([mx], [my], c=[cmap(norm(a))], s=300, marker="X",
                   edgecolors="black", linewidths=2, zorder=10)
        ax.annotate(f"$\\alpha={a}$", (mx, my), xytext=(7, 7),
                    textcoords="offset points", fontsize=10, fontweight="bold")

    ax.set_xlabel("recovery rate on flipped pairs", fontsize=11)
    ax.set_ylabel("preservation rate on resisted pairs", fontsize=11)
    ax.set_title(f"Recovery–selectivity trade-off on {MODEL_DISPLAY[model_tag]}\n"
                 f"each small circle = one source; large X = mean per $\\alpha$",
                 fontsize=11)
    ax.set_xlim(-0.02, 0.65)
    ax.set_ylim(0.0, 1.05)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                      label=r"intervention strength $\alpha$")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig(fig, f"F4_intervention_tradeoff_{model_tag}")


def main():
    print(f"Generating figures to {FIGS}/")
    for model_tag in ("llama31_8b", "qwen25_7b"):
        print(f"\n--- {model_tag} ---")
        fig_source_cosine(model_tag)
        fig_lodo_auroc(model_tag)
        fig_bias_cosine(model_tag)
        fig_dendrogram(model_tag)
        fig_cross_bias_probe(model_tag)
        fig_cross_bias_intervention(model_tag)
        fig_pca_source_directions(model_tag)
        fig_pca_pair_activations(model_tag)
        fig_layer_wise(model_tag)
        fig_intervention_alpha_sweep(model_tag)
        fig_intervention_per_source(model_tag)
        fig_intervention_tradeoff(model_tag)
        fig_intervention_selectivity(model_tag)

    print(f"\n--- both models ---")
    fig_intervention_per_bias_bidirectional()
    fig_intervention_bidirectional()


if __name__ == "__main__":
    main()
