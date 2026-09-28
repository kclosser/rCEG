## B6 — structure-only ablation

Reviewer 4: *"The model is not structure-only. Four scalar inputs come directly from PQR ... rCEG cannot be applied to an arbitrary new molecule from SMILES alone."*

The objection is correct about the input specification. It has no measurable accuracy cost.

| configuration | features | MAE ± SD | Δ vs full | paired t | Cohen dz |
|---|---|---|---|---|---|
| `full` (reported rCEG) | 427 | 0.3384 ± 0.0258 | — | — | — |
| `drop_x0_x4` | 422 | 0.3385 ± 0.0254 | +0.00012 (+0.04%) | p = 0.814 | 0.08 |
| `drop_x2_x4` | 424 | 0.3390 ± 0.0257 | +0.00055 (+0.16%) | p = 0.230 | 0.41 |

**Interpretation.** Removing all five PQR-derived scalars (`x0`–`x4`) costs 0.3384 → 0.3385 eV, a change of **0.0001 eV** — roughly 200× smaller than the between-seed standard deviation of 0.026 eV. Removing only the three that genuinely require a semiempirical calculation (`x2` dipole, `x3` heat of formation, `x4` polarizability) costs 0.0006 eV.

Two reasons this is unsurprising. `x0` and `x1` are molecular mass and exact mass, both computable from SMILES and **already duplicated** in the RDKit block at `x380` and `x381`. And dipole, heat of formation and polarizability are largely redundant with the 422 structural descriptors, which already encode the size, polarity and heteroatom content those quantities track.

**Consequence for the manuscript.** rCEG can be run from SMILES alone at inference with no measurable loss of accuracy. The restriction of the external benchmark to the PQR∩QMugs intersection is therefore a property of how the benchmark was assembled, not a limitation of the method.

**Caveat that must be stated.** The *training labels* remain calibrated pseudo-labels derived from `pqr_gap`, itself a semiempirical quantity. The claim supported here is narrow and specific: **inference** on a new molecule requires only its SMILES. Training the model still required the PQR semiempirical data.

