## T5 — final figure manifest

All files below are the ones shipped in the deposit's `figures/`. Print size
assumes the stated dpi.

| Manuscript | File | Pixels | dpi | Print size | Vector? | Sidecar |
|---|---|---|---|---|---|---|
| Figure 1 | `figure1_reference_descriptor_scatter_final.png` | 7370×2557 | 600 | 12.3×4.3 in | yes | figure1_data.csv |
| Figure 2 | `figure2_pipeline_flowchart.png` | 7884×6083 | 600 | 13.1×10.1 in | yes | — |
| Figure 3 | `figure_training_iteration_vs_error_clean_inset.png` | 4355×2673 | 600 | 7.3×4.5 in | yes | figure_training_iteration_vs_error_clean_inset_data.csv |
| Figure 4 | `figure_piecewise_splits_named_best_available.png` | 4783×6588 | 600 | 8.0×11.0 in | yes | — |
| Figure 5a | `figure_qmugs_predicted_vs_reference.png` | 3837×3813 | 600 | 6.4×6.4 in | no | — |
| Figure 5b | `figure_qmugs_mae_by_domain.png` | 5239×2931 | 600 | 8.7×4.9 in | no | — |
| Figure 5c | `figure_qmugs_published_context_comparison.png` | 5937×3212 | 600 | 9.9×5.4 in | no | — |
| Figure 6 | `figure6_domain_error_distributions.png` | 5700×3055 | 600 | 9.5×5.1 in | yes | figure6_data.csv |
| TOC | `toc_graphic.png` | 1954×1053 | 600 | 3.3×1.8 in | yes | — |

Every figure is **600 dpi or vector**, comfortably above the 300 dpi bar.

### Figure 1 — confirmed clean

| Check | Result |
|---|---|
| Plotted from the strict 84,143-molecule corpus | **yes** |
| Panel counts read 1,228 and 909 | **yes** (`figure1_data_summary.csv`) |
| No point at 0 eV | **yes** — minimum gap in the sidecar is 6.14 eV, zero points below 0.01 |
| Exact mass not plotted | **yes** — panel dropped, reason recorded as collinearity with molecular weight (r = 0.9939) |
| Join method | RDKit canonical SMILES on both sides |

⚠️ **A stale sidecar was shipping alongside it and has been removed.**
`figure1_reference_descriptor_scatter_final_data.csv`, dated July, held 743 rows
**including the 16 placeholder zero-gap points the revision removed** and an
`exact_mass` column for the dropped panel. It was being picked up by a prefix
match and shipped next to the corrected figure — a reviewer opening it would
have found exactly the defect the response letter says was fixed. It is now
renamed `STALE_PRE_CLEANING_figure1_final_data_DO_NOT_USE.csv`, excluded from
the deposit, and guarded against by name. **`figure1_data.csv` (2,137 rows,
zero zero-gap points) is the current sidecar.**

The 2,137 rows here are 1,228 + 909 with the 14 dual-labelled molecules
appearing in both series. This is *not* the 2,137 error of DISCREPANCIES §A1 —
it is a plot of two series, not a deduplicated pool. Do not "correct" it.

### Figure 2 — confirmed, with one contradiction to resolve

| Check | Result |
|---|---|
| 600 dpi or vector | **yes** — 7884×6083 at 600 dpi, plus PDF |
| Calibration and rCEG as separate blocks | **yes** — "STAGE 1 — Calibration model", "STAGE 2 — rCEG prediction model" |
| QMugs excision shown | **yes** — 9,753 excised before any model development |
| Partition shown | **yes** — and it is **1,274 / 425 / 424**, not 1,273 / 425 / 425 |
| Final box carries a stale 0.347 | **no** — the figure hardcodes no MAE at all; its terminal box reads "Predicted gap" |

⚠️ **The partition in the figure is right and the one I had been quoting was
wrong.** The seed-42 split is 1,274 calibration / 425 validation / 424 test,
and it is identical in all ten repeats (membership varies, sizes do not).
`1,273 + 425 + 425` also sums to 2,123, which is why it went unnoticed. Every
document has been corrected to 1,274 / 425 / 424.

⚠️ **The inference block contradicts the B6 result.** Figure 2's
"INFERENCE, new molecule" path lists `PM7 semiempirical properties` as a
required input. B6 showed the five PQR scalars can be dropped for
**+0.0001 eV (p = 0.814)** — inference needs only SMILES. As drawn, the figure
concedes Reviewer 4's objection that the paper can now answer. Either mark
that box optional or remove it, and say so in the caption.

### Figure 3 — single-source confirmed, but the baseline is truncated

Legend, caption and `figure3_data.csv` all read `val_mae_min`, `test_mae` and
`best_iteration` from the same `global_hgb_convergence_summary.csv` row, so the
0.403-vs-0.402 drift that motivated the rewrite cannot recur. Values come from
the frozen run's primary split.

⚠️ **The curve has not converged.** `best_iteration` = 700 = `max_iter`, and
validation MAE is still falling at the cutoff (−0.005 eV per 100 iterations).
`global_hgb` is therefore an **under-trained** baseline, which slightly
flatters the ablation.

**It does not change the conclusion, and the arithmetic says so.** The gap to
rCEG is 0.066 eV; at the observed rate — which is decelerating — closing it
would need roughly 1,300 further iterations. Either raise `max_iter` and
re-run, or state in the caption that the baseline is shown at a fixed 700-
iteration budget. Do not describe the figure as showing convergence.

Note also that Figure 3 plots `global_hgb`, whose test MAE is 0.458 — this is
the ablation baseline, **not** the headline 0.338 eV. The caption should say so
plainly or a reader will read 0.458 as the paper's result.
