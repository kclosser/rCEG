# rCEG JCTC Reproducibility Package

This package accompanies the manuscript:

**A split-architecture machine learning model for scalable HOMO-LUMO gap prediction: rCEG**

Prepared for review by Dr. Closser and for organization of the Data and Software Availability materials required for JCTC submission.

## Package overview

This is a lightweight but review-ready reproducibility package. It contains the main scripts, processed data tables, final figure files, environment file, Psi4 recomputation input geometries, and recomputation manifest needed to document and reproduce the main analyses reported in the manuscript.

The package is not intended to be a complete dump of every intermediate file generated during model development. Large model checkpoints, temporary cache files, and unnecessary intermediate runs were intentionally left out to keep the package manageable for review and repository upload.

## Current package contents

- `code/`: Python scripts used for descriptor processing, QM9 reference construction, Psi4 input generation, Psi4 recomputation, model training/evaluation, QMugs external analysis, and figure generation.
- `data/`: Processed machine-readable CSV/JSONL files used for reference calibration, recomputed reference labels, QMugs analysis, and figure data.
- `psi4_inputs/`: Psi4 recomputation input materials, including PQR XYZ geometries and the recomputation manifest.
- `figures/`: Final figure files and related figure outputs.
- `docs/`: File inventory.
- `environment.yml`: Conda environment export from the local analysis environment.

Approximate file counts in this package:
- Code files: 36
- Data files: 15
- Figure files: 7
- Psi4 XYZ input geometries: 947

## Psi4 recomputation details

The Psi4 recomputations were performed as **single-point B3LYP calculations** on pre-generated XYZ geometries.

The recomputation scripts used calls of the form:

```python
psi4.energy("b3lyp", molecule=mol, return_wfn=True)
```

No additional Psi4 geometry optimization was performed before extracting HOMO and LUMO orbital energies.

The package includes:
- `code/make_recompute_inputs.py`: script used to prepare recomputation input files.
- `code/run_psi4_recompute_batch.py`: batch script used to run Psi4 recomputations.
- `code/run_psi4_recompute_test.py`: smaller test version of the recomputation script.
- `psi4_inputs/xyz/`: XYZ geometries used as inputs for recomputation.
- `psi4_inputs/manifests/recompute_manifest.csv`: manifest connecting recomputation IDs, structures, and input files.
- `data/recompute_manifest.csv`: duplicate copy of the manifest included for convenience.
- `data/pqr_recomputed_reference.csv`: converged recomputation results.
- `data/pqr_recomputed_reference_bad.csv`: failed or non-converged recomputation records.

## Reference-label data

The manuscript uses a reference-calibration strategy involving:
1. PQR molecules overlapping with QM9.
2. PQR molecules independently recomputed using Psi4.
3. A combined reference-labeled pool used for calibration, validation, and held-out testing.

Relevant files include:
- `data/qm9_gap_reference.csv`
- `data/pqr_recomputed_reference.csv`
- `data/pqr_recomputed_reference_bad.csv`
- `data/recompute_manifest.csv`

## Main modeling and analysis scripts

Important scripts include:

- `code/pqr_calibrate_then_train.py`: calibration and model-training workflow.
- `code/pqr_final_hybrid_regime.py`: hybrid/regime model development script.
- `code/pqr_full_domain_moe_qmugs_external.py`: QMugs external evaluation workflow.
- `code/prepare_qmugs_external_reference.py`: preparation of QMugs overlap/reference files.
- `code/plot_qmugs_external_validation.py`: QMugs external-validation figure generation.
- `code/plot_qmugs_residual_offset.py`: signed residual/offset diagnostic plotting.
- `code/inspect_qmugs_crossfit_offsets.py`: inspection of cross-fitted offset values.
- `code/make_figure1_reference_descriptor_scatter_final.py`: Figure 1 generation.
- `code/make_training_iteration_error_figure_clean.py` and related scripts: training-iteration error figure generation.
- `code/make_training_label_descriptor_figure.py`: descriptor-versus-training-label figure generation.

Some older exploratory scripts are also included for transparency, but the final manuscript figures and metrics should be checked against the final scripts and processed data files before repository upload.

## Figure files and figure data

The `figures/` folder contains final or near-final versions of the paper figures, including:

- Figure 1 reference descriptor scatter plot.
- Training-iteration error plot with inset.
- Training-label versus interpretable descriptor plot.
- QMugs external validation plot.

The `data/` folder also includes figure-level CSV files used to generate several of these plots.

## External datasets

The original PQR, QM9, PubChem, and QMugs datasets are not redistributed in full in this lightweight package. They are publicly available from their original sources and are cited in the manuscript.

This package instead includes the processed machine-readable tables, recomputation outputs, figure data, and scripts needed to reproduce the reported analyses from the processed data used in the manuscript.

## Notes for JCTC Data and Software Availability statement

A possible manuscript statement is:

The data underlying this study are available in the published article, the Supporting Information, and at [repository link]. The scripts and code used to generate molecular descriptors, prepare Psi4 input files, run single-point Psi4 calculations, process HOMO-LUMO gap outputs, train and evaluate the rCEG models, perform the QMugs overlap analysis, and generate the figures are available at [repository link]. The repository contains processed machine-readable data tables, Psi4 recomputation input geometries, recomputation manifests, recomputed reference summaries, model-analysis scripts, figure-generation scripts, and figure data. The original PQR, QM9, PubChem, and QMugs datasets are publicly available from their original sources as cited in the manuscript.

## Suggested review checklist before upload

Before uploading to Zenodo, figshare, ACS Supporting Information, or another repository, please verify:

1. The manuscript numbers match the processed files in this package.
2. The final figure files match the figures included in the submitted manuscript.
3. The Data and Software Availability statement points to the final repository link.
4. Any large model checkpoint files intentionally omitted from this lightweight package are either not required for essential reproducibility or are uploaded separately.
5. The Psi4 input geometries and recomputation manifest are included, since the README states that they are included.
6. The manuscript clearly states that the Psi4 recomputations were single-point calculations and not geometry optimizations.

## Contact

Isaac Wang  
Kristina D. Closser
