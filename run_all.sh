#!/usr/bin/env bash
# Reproduce every reported number and figure, top to bottom.
set -euo pipefail
export PYTHONHASHSEED=0
CODE=code
RUN=runs/rceg_final_consolidated
QC=${QC:-/Applications/anaconda3/envs/rceg_qc/bin/python}   # psi4 env
# Shared corpus/reference cache for steps 8, 9 and 12. Purely a speed-up: it is
# written on first use and read afterwards. Delete it to force a rebuild.
CACHE=${CACHE:-$RUN/inputs_cache.pkl}
mkdir -p "$RUN"

echo "1. descriptors from the raw PQR release"
python3 "$CODE/DescriptorPull3.py"

echo "2. cleaning"
python3 "$CODE/pqr_data_cleaner.py"
python3 "$CODE/pqr_diagnose_and_strict_clean.py"

echo "3. reference labels"
python3 "$CODE/make_qm9_gap_reference_from_xyz.py"
python3 "$CODE/prepare_qmugs_external_reference.py"

echo "4. audits (Tables 1-3, S1)"
python3 "$CODE/audit_cleaning_waterfall.py"
python3 "$CODE/audit_corpus_composition.py"
python3 "$CODE/audit_recompute_outcomes.py"

echo "5. descriptor-name resolution (needed before the piecewise figure)"
python3 "$CODE/resolve_lasso_feature_names.py" --n-sample 400 --outdir "$RUN"

echo "6. consolidated model run (Tables 4, 6, 7)"
python3 "$CODE/pqr_full_domain_moe_qmugs_external.py" \
    --pqr enhanced_dataset_lasso_STRICT.jsonl --qm9-ref qm9_gap_reference.csv \
    --extra-ref pqr_recomputed_reference.csv \
    --external-ref external_validation/qmugs/qmugs_external_reference.csv \
    --outdir "$RUN" --leakage-corr-threshold 0.94 \
    --seed 42 --n-repeats 10 --split-mode random \
    --emit-calibration-by-group --emit-permutation-control

echo "7. alternative split protocols (Table 8)"
for M in scaffold cluster; do
  python3 "$CODE/pqr_full_domain_moe_qmugs_external.py" \
      --pqr enhanced_dataset_lasso_STRICT.jsonl --qm9-ref qm9_gap_reference.csv \
      --extra-ref pqr_recomputed_reference.csv \
      --external-ref external_validation/qmugs/qmugs_external_reference.csv \
      --outdir "$RUN/split_$M" --leakage-corr-threshold 0.94 \
      --seed 42 --n-repeats 10 --split-mode "$M" --skip-legacy-single-run
done

echo "8. structure-only ablation (answers Reviewer 4)"
# --cache is optional and only a speed-up; without it the corpus and reference
# pool are rebuilt from source. The cache is shared with steps 9 and 12.
python3 "$CODE/run_B6_structure_only.py" --repeats 10 --seed 42 --cache "$CACHE"

echo "9. computational cost (Table 9) — run on an idle machine"
python3 "$CODE/benchmark_computational_cost.py" --n 100 --stratify domain \
    --cache "$CACHE"

echo "10. geometry sensitivity and same-hardware timing — REQUIRES psi4"
$QC "$CODE/run_tasks_G_and_T.py" --stage select
$QC "$CODE/run_tasks_G_and_T.py" --stage optimize
$QC "$CODE/run_tasks_G_and_T.py" --stage conformers
$QC "$CODE/run_tasks_G_and_T.py" --stage dft-timing
$QC "$CODE/run_tasks_G_and_T.py" --stage semiempirical
$QC "$CODE/run_tasks_G_and_T.py" --stage collect

echo "11a. inference latency (closes the Table 9 inference column)"
python3 "$CODE/run_C3_inference.py" --repeats 50 --threads 1 --cache "$CACHE"

echo "11. per-domain cost table — REQUIRES psi4"
# Writes every column of Table 9 except inference_ms, which step 11a measures.
# Representatives come from C3_representatives.csv (authoritative); see
# docs/DISCREPANCIES.md D3 for why they are not regenerated here.
$QC "$CODE/run_C3_timings.py"

echo "12. figures, last, so they read the finished run"
python3 "$CODE/emit_hgb_convergence_history.py" --cache "$CACHE"
python3 "$CODE/make_figure1_reference_descriptor_scatter_final.py"
# Revision-2 Figure 1: same data, axis labels with the method prefix
# dropped (PM6-vs-PM7 unresolved). Writes to runs/revision2/figures/.
python3 "$CODE/make_figure1.py" --dpi 600
python3 "$CODE/make_figure2_pipeline_flowchart.py"            --dpi 600
python3 "$CODE/make_training_iteration_error_figure_clean.py" --dpi 600
python3 "$CODE/remake_piecewise_figure_clean.py"
python3 "$CODE/plot_qmugs_external_validation.py"             --dpi 600
python3 "$CODE/make_figure6_domain_error_distributions.py"    --dpi 600
python3 "$CODE/make_toc_graphic.py"                           --dpi 600
echo "done — see docs/PROVENANCE.md for the figure/table to script map"
