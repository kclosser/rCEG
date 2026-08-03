# rCEG JCTC Reproducibility Package

This package accompanies the manuscript:

**A split-architecture machine learning model for scalable HOMO-LUMO gap prediction: rCEG**

## Package overview

This is a lightweight but review-ready reproducibility package. It contains the main scripts, processed data tables, final figure files, environment file, and Psi4 input geometries needed to document and reproduce the main analyses reported in the manuscript.

## Current package contents

- `code/`: Python scripts used for descriptor processing, QM9 reference construction, Psi4 input generation, Psi4 recomputation, model training/evaluation, QMugs external analysis, and figure generation.
- `data/`: Processed machine-readable CSV/JSONL files used for reference calibration, recomputed reference labels, QMugs analysis, figure data and psi4 input files. Note some files are provided as zip files due to size. Additionally, the jsonl file was split into 11 parts to allow upload to github.
- `figures/`: Final figure files and related figure outputs.
- `environment.yml`: Conda environment export from the local analysis environment.

Approximate file counts in this package:
- Code files: 36
- Data files: 15
- Figure files: 7
- Psi4 XYZ input geometries: 947


## External datasets

The original PubChem, PQR, QM9, and QMugs datasets are not redistributed in full in this  package. They are publicly available from their original sources and are cited in the manuscript.

This package instead includes the processed machine-readable tables, figure data, and scripts needed to reproduce the reported analyses from the processed data used in the manuscript.


## Contact

Isaac Wang  
Kristina D. Closser
