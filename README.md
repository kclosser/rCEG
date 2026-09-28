# rCEG — reproducibility package

Accompanies **"A split-architecture machine learning model for scalable HOMO-LUMO gap
prediction: rCEG"**, Wang & Closser, *ACS Omega*.

1. `code/` holds **only** scripts that produced reported results. Superseded variants are
   in `code/archive/`, off the run path.
2. Every reported number comes from one run, `runs/rceg_final_consolidated/`.

## Layout

```
code/            scripts that produced reported results
code/archive/    superseded variants, not on the run path
data/            processed tables + LARGE_FILES_CHECKSUMS.csv
data/revision2/  revision-2 result tables, one file per spec task
data/revision2/paste/
                 paste-ready manuscript fragments, one per finding
data/revision2/figures/
                 revision-2 Figure 1 rebuild + its generating script
figures/         final figures at 600 dpi with _data.csv sidecars
psi4_inputs/     recomputation and geometry manifests
docs/            PROVENANCE.md, REVISION_LOG.md, the revision-2
                 documents below, and the two Claude-web specs
environment.yml  numpy >= 2.3 pinned, reason inline
run_all.sh       numbered 1-12, executable top to bottom
```

## The revision-2 documents

In `docs/`. `SUMMARY.md` and `SUPPORTING_INFORMATION.md` are **generated**
from the result tables and must not be hand-edited — rerun
`code/build_SI.py` and `code/build_SUMMARY.py` instead. The other two are
written by hand.

| Document | What it is |
|---|---|
| `SUPPORTING_INFORMATION.md` | The SI: hyperparameters, calibration diagnostics, the full leakage audit, the recomputation manifest with charge/multiplicity/basis, geometry sensitivity, split protocols, per-domain error deciles. Every table is annotated with the file and columns it was read from. |
| `SUMMARY.md` | Every number produced by the revision, mapped to the manuscript location it belongs in, plus an index of the paste fragments and result files. |
| `DISCREPANCIES.md` | Everything that disagrees with the submitted manuscript, is unresolved, or would mislead a reader. Read this first. |
| `REVISION2_EXECUTION_SPEC.md` | The full record of what was done, what was found, what was fixed, and how to regenerate it. |

**Figure 1.** Two builds exist. `figures/figure1_reference_descriptor_scatter_final.*`
is the September build (correct data: 2,137 points, n = 1,228 and 909, no
zero-gap records) but its axis labels name PM7.
`data/revision2/figures/figure1_reference_descriptor_scatter.*` is the
revision-2 build: identical data, method prefix dropped from the axis
labels because PM6-vs-PM7 is unresolved. **Use the revision-2 build.**
The July `*_clean.*` build is stale (743 points, 641/102, 16 zero-gap
records) and is not shipped.

Three items in `DISCREPANCIES.md` are **open and need an author decision**:
the PM6-versus-PM7 naming (resolvable only against the PQR publication §D1),
which inference timing Table 9 reports (§D2, and it moves the speedup
ratios), and a Table 9 caption note that the charged representative is an
open-shell dication (§D3).

`runs/rceg_final_consolidated/headline_metrics.csv` predates the final
ten-repeat run and reports 0.318 eV, which is not the headline number
(0.338 eV). `data/frozen_architecture.csv` is the authoritative record of what
was selected and why. See `DISCREPANCIES.md` §E3.

## Run

```bash
conda env create -f environment.yml && conda activate rceg
export PYTHONHASHSEED=0
bash run_all.sh
```

Four source datasets are **not** redistributed: PQR, QM9, QMugs, PubChem. Three large derived files are omitted for size;
`data/LARGE_FILES_CHECKSUMS.csv` gives SHA256 so a regenerated copy can be verified.

## Two non-optional environment requirements

**numpy ≥ 2.3.** Under numpy 1.26.4 on Python 3.14, `np.nanvar` silently returned 0.0 for
arrays above ~400k elements. sklearn's `VarianceThreshold` calls it internally.

**Scratch space for psi4.** 30 of the 38 original recomputation failures were
*"No space left on device"*, not chemistry. Note a known caveat in
`run_tasks_G_and_T.py`: psi4 reads `PSI_SCRATCH` at import, so setting it per molecule
does **not** relocate scratch — set it in the environment *before* launching Python.

## Reproducibility

- Branch-expert seeds use `zlib.crc32`, not the salted builtin `hash()`. Two runs at the
  same seed give byte-identical outputs (verified).
- Scaling and imputation are fitted inside each sklearn `Pipeline`, so no validation or
  test statistic influences training preprocessing.
- The QMugs external holdout is excised before leakage filtering, calibration,
  pseudo-labelling and expert training; the model script asserts this and fails loudly.

## Data and Software Availability

> The data underlying this study are available in the published article, the Supporting
> Information, and at [repository link]. Scripts to generate molecular descriptors,
> prepare and run Psi4 calculations, process HOMO-LUMO gap outputs, train and evaluate
> the rCEG models, perform the QMugs external analysis, quantify geometry sensitivity,
> and generate all figures are available at [repository link]. The repository contains
> processed machine-readable tables, recomputation manifests, model and figure scripts,
> figure data sidecars, and a figure-and-table provenance map. The original PQR, QM9,
> PubChem and QMugs datasets are publicly available from their original sources as cited.

Isaac Wang · Kristina D. Closser
