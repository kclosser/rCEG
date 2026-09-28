# Figure and table provenance

**Purpose.** Reviewer 4 could not determine which script produced the manuscript's
numbers (R4.7). This table gives, for every manuscript figure and table, the generating
script, the run directory, the inputs, and the output filename.

**Single source of truth.** Every number in the manuscript comes from
`runs/rceg_final_consolidated/`. Earlier directories — `pqr_full_domain_moe_*_LEAK094`,
`pqr_qmugs_strict_external_LEAK094`, `pqr_full_domain_moe_novel_split_regime` — remain on
disk for provenance only and **must not** be cited. Three figure scripts were found
reading from them and have been repointed; see "Source defects found and fixed" below.

**Reproducing everything.** `export PYTHONHASHSEED=0`, then follow `run_all.sh`.

---

## Frozen architecture

**rCEG = `piecewise_split`.** Selected on **validation** MAE averaged over the ten
repeated splits, before any test metric was consulted. Recorded in
`data/frozen_architecture.csv`.

| | validation | test |
|---|---|---|
| `piecewise_split` (**rCEG**) | 0.3441 ± 0.0159 | **0.3388 ± 0.0256**, R² 0.930 |
| `domain_expert` (ablation) | 0.3425 ± 0.0159 | 0.3354 ± 0.0262 |
| `full_rceg` (ablation) | 0.3465 ± 0.0222 | 0.3392 ± 0.0215 |

**The linear meta-learner is dropped.** With an honest cross-fitted validation score
(5-fold within the validation partition, since the stacker is fitted on it), `full_rceg`
does **not** beat `piecewise_split`: +0.0024 eV worse, better in only 5 of 10 repeats,
paired t p = 0.52. It is reported as an ablation row only.

`domain_expert` is 0.0017 eV better on validation, which is not significant.
`piecewise_split` is chosen on **interpretability** grounds — the learned breakpoint is
the paper's named contribution, so reporting `domain_expert` as rCEG would mean the model
in the abstract is not the model in the title. This is stated as an interpretability
choice, never as an accuracy claim.

Every downstream artifact reads the frozen config rather than re-deriving a "best"
model: `make_toc_graphic.py` and `extract_placeholder_values.py` both load
`frozen_architecture.csv`, so no published figure can silently reintroduce test-set
selection.

**Manuscript text to reword:** the Introduction sentence "the only weighted combination
anywhere in the model is a final linear meta-learner" describes a component that no
longer exists. Figure 2 needs no change — its Stage 2 box reads "Domain experts and
piecewise descriptor splits" and never depicted a stacking layer.


## Tables

Numbering below is FINAL and supersedes any earlier scheme. The manuscript was
renumbered mid-revision: what were previously called Tables 6 and 7 (the feature matrix
and hyperparameters) are now **S2** and **S3**; current Tables 6 and 7 are the ablation
and per-domain metrics.

| # | Content | Task | Script | Source file |
|---|---|---|---|---|
| 1 | QM9 vs PQR domain coverage | A2 | `audit_corpus_composition.py` | `corpus_composition.csv` |
| 2 | Confidence bin definitions | A6 | `pqr_full_domain_moe_qmugs_external.py` (`add_confidence`, `_cycle_conf`) | `confidence_thresholds_derived.csv` |
| 3 | Recompute sampling and convergence | A3 | `audit_recompute_outcomes.py` | `recompute_audit.csv` |
| 4 | Calibration performance by domain/source | A4 | `pqr_full_domain_moe_qmugs_external.py --emit-calibration-by-group` | `calibration_by_domain_and_source.csv` |
| 5 | Domain definitions and counts | A2 | `audit_corpus_composition.py` | `corpus_composition.csv` (`section=domain`) |
| 6 | Component ablation | C1, C2 | `pqr_full_domain_moe_qmugs_external.py --n-repeats 10` | `repeated_split_summary.csv` |
| 7 | Per-domain test metrics | C1 | same run | `mae_by_domain.csv` |
| 8 | Split-protocol comparison | C3 | same script, `--split-mode {random,scaffold,cluster}` | `split_protocol_comparison_ALL.csv` |
| 9 | Computational cost | D1 | `benchmark_computational_cost.py` | `timing_benchmark.csv` |
| S1 | Cleaning waterfall | A1 | `audit_cleaning_waterfall.py` | `cleaning_report_complete.csv`, `stage1_valence_failure_breakdown.csv` |
| S2 | 427-column feature matrix | static | — | verify against `load_pqr()` in the model script |
| S3 | Hyperparameters | static | — | verify against the `et()`, `rf()`, `hgb()`, `ridge_scaled()` definitions |

## Figures

| # | Content | Task | Script | Output (PNG + PDF, 600 dpi) |
|---|---|---|---|---|
| 1 | Reference descriptors vs gap | B1 | `make_figure1_reference_descriptor_scatter_final.py` | `figure1_reference_descriptor_scatter_final.*` + `figure1_data.csv` |
| 2 | Pipeline flowchart | B2 | `make_figure2_pipeline_flowchart.py` | `figure2_pipeline_flowchart.*` |
| 3 | Error vs boosting iteration | B3 | `make_training_iteration_error_figure_clean.py` (+ `emit_hgb_convergence_history.py`) | `figure_training_iteration_vs_error_clean_inset.*` + `figure3_data.csv` |
| 4 | Piecewise descriptor splits | — | `remake_piecewise_figure_clean.py` | `figure_piecewise_splits_named_best_available.*` |
| 5 | QMugs external validation | — | `plot_qmugs_external_validation.py --run runs/rceg_final_consolidated` | `figure_qmugs_predicted_vs_reference.png`, `figure_qmugs_mae_by_domain.png`, `figure_qmugs_published_context_comparison.png` |
| 6 | Per-domain error distributions | B4 | `make_figure6_domain_error_distributions.py` | `figure6_domain_error_distributions.*` + `figure6_data.csv` |
| TOC | Graphical abstract | B5 | `make_toc_graphic.py` | `toc_graphic.*` |

## Supporting scripts, not figure or table generators

| Script | Role |
|---|---|
| `resolve_lasso_feature_names.py` | Recovers the 375 descriptor names by value-matching (task E4) |
| `emit_hgb_convergence_history.py` | Produces Figure 3's convergence trace from the consolidated split |
| `run_geometry_sensitivity.py` / `.sh` | Geometry-sensitivity DFT (task A5, cluster only) |
| `improve_rceg_error*.py`, `validate_rceg_improvement.py` | Error-reduction study; **not** part of the reported architecture |

---

## Source defects found and fixed

Every figure bug found during this revision had the same root cause: scripts discovering
inputs by globbing across `runs/` or reading a stale cached CSV, rather than being pinned
to one run. All four are fixed; the pattern is worth checking for in any new script.

| Script | Defect | Fix |
|---|---|---|
| `make_figure1_reference_descriptor_scatter_final.py` | Joined reference sets on **raw SMILES strings**, so only 604 of 1,228 QM9 and 99 of 909 Psi4 molecules matched. Also read the pre-clean CSV, plotting 16 placeholder 0.0 eV gaps. | Canonicalise both sides before joining; take the gap from the strict corpus |
| `make_training_iteration_error_figure_clean.py` | Globbed for the newest convergence history (a **superseded May run**) and separately the newest metrics file, mixing two runs. This produced the 0.403 vs 0.402 mismatch. | Pinned to `emit_hgb_convergence_history.py` output from the consolidated split |
| `remake_piecewise_figure_clean.py` | Selected sources via `next()` over an **unsorted** `Path.glob()`, preferring a 27 MB stale cache; and hardcoded **three wrong descriptor names** plus two blanks | Sources pinned; names read from `piecewise_feature_name_map.csv`, with a guard that refuses placeholder labels |
| `plot_qmugs_external_validation.py` | Hardcoded to `runs/pqr_qmugs_strict_external_LEAK094` | `--run` flag defaulting to the consolidated run |

**Still carrying the pattern** (not used for current manuscript figures, but should be
archived rather than shipped): `make_error_vs_training_iteration.py`,
`make_training_iteration_error_figure.py`, `make_training_iteration_error_figure_inset.py`,
`remake_piecewise_figure_supplement_safe.py`,
`remake_piecewise_figure_clean_BEFORE_NAMED_AXES.py`, `generate_revised_paper_figures.py`,
`make_named_descriptor_figure*.py`, `make_training_label_descriptor_figure.py`,
`make_figure1_reference_descriptor_scatter_clean.py`.

**Do not use `enhanced_dataset_lasso.csv` for any new figure.** It is the pre-cleaning
descriptor table: it still contains the 4,719 rows cleaning removed, and it disagrees
with the training corpus on **1,409 rows**.

## Stale artifacts in `paper_figures_revised/`

These predate the consolidated run and must not be submitted:
`figure_qmugs_external_validation.png` and `..._data.csv` (27 June),
`figure_piecewise_splits_named_axes_data.csv` (27 June, 27 MB),
`figure_piecewise_splits_clean_no_titles.*`, `supplement_piecewise_internal_features.*`,
and every `figure1_reference_descriptor_scatter_clean.*`.
