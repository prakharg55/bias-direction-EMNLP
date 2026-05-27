# Code

Anonymous code bundle for the paper.

## Requirements

`torch >= 2.1`, `transformers >= 4.40`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `pandas`.

## Layout

Scripts expect `dataset_dumps/test/` (BCT prompt files from chua2024bias) one directory above this folder, and write to `cache/` and `figures/` inside it.

## Pipeline

1. `extract.py --model <hf_id>` — hidden states + meta per (bias, dataset)
2. `compute_directions.py`, `compute_bias_directions.py` — direction vectors
3. `probe_lodo.py`, `probe_lodo_per_layer.py` — LODO AUROC
4. `intervene.py`, `intervene_selectivity.py`, `intervene_cross_bias.py` — α-sweep + selectivity + cross-bias
5. `letter_balanced_control.py` — letter-balanced control
6. Base-probe: rerun `extract.py` with `--cache_root cache/<tag>_base_xlabel` on the matching base checkpoint

## Aggregation

`analyze_intervention.py`, `analyze_cross_bias_intervention.py`, `debias_analysis.py`, `per_bias_debias.py`, `cosine_within_bias.py`, `cross_bias_cosine.py`, `cross_bias_probing.py`, `cluster_significance.py`, `base_probe.py`, `base_probe_ci.py`, `instruction_rebuttal.py` — all read `cache/` and write JSON summaries.

## Figures → paper number

| Script | Fig |
|---|---|
| `make_paper_figures.py` | 1 |
| `make_hong_pca.py` | 2 |
| `layer_wise_figure.py` | 3 |
| `make_cue_gap_figure.py` | 4 |
| `make_bidirectional_figure.py` | 5 |
| `figures.py` | 6, 8, 9, 10, 11 |
| `make_cosine_all_families.py` | 7 |

## Tables → source

| Table | Source |
|---|---|
| 1 | `lodo_summary.json` |
| 2, 6 | per-source `meta.json` aggregated |
| 3 | `instruction_rebuttal.json` |
| 4 | `cluster_significance.json` |
| 5 | `debias_summary.json` |
| 7, 8 | per-source `meta.json` aggregated |
| 9 | `letter_balanced/summary.json` |
| 10 | `cosine_within_bias.json` |
| 11 | `cross_bias_cosine.json` |
