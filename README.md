# rCEG Reproducibility Package

This package accompanies the manuscript:

**A split-architecture machine learning model for scalable HOMO-LUMO gap prediction: rCEG**

## Contents

- `code/`: scripts used to prepare descriptors, generate Psi4 inputs, run single-point Psi4 calculations, process HOMO-LUMO gap outputs, train and evaluate rCEG models, perform QMugs overlap analysis, and generate figures.
- `data/`: processed machine-readable tables used for calibration, held-out testing, external QMugs overlap evaluation, model metrics, and figure generation. Some data provided in zip files due to large size and enhanced_dataset_lasso_strict.jsonl is split into 11 pieces. 
- `figures/`: figure-generation outputs and plotted figure files.
- `models/`: saved model files, if present.
- `environment.yml` or `requirements.txt`: software environment information.

## Notes

The original PQR, QM9, PubChem, and QMugs datasets are publicly available from their original sources as cited in the manuscript. This package contains the processed data tables and scripts needed to reproduce the reported analyses and figures.

Psi4 recomputations were performed as single-point B3LYP calculations on pre-generated XYZ geometries. No additional Psi4 geometry optimization was performed before extracting HOMO and LUMO orbital energies.
