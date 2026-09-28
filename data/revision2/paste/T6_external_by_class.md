## T6 — QMugs external validation by chemical class

All 9,753 molecules were excised from training by SMILES before any fitting. Offset alignment is five-fold cross-fitted, so no molecule contributes to the offset used to align it.

| Chemical class | n | Mean reference gap | Raw MAE | Offset-aligned MAE | R² (aligned) |
|---|---|---|---|---|---|
| qm9 like small organic | 134 | 10.70 eV | 4.13 eV | **0.436 eV** | 0.722 |
| near qm9 larger organic | 1,939 | 9.13 eV | 4.09 eV | **0.520 eV** | 0.659 |
| large neutral organic | 4,016 | 8.51 eV | 4.14 eV | **0.504 eV** | 0.557 |
| heteroatom rich non qm9 | 3,664 | 8.58 eV | 4.30 eV | **0.543 eV** | 0.479 |
| charged or radical | **0** | — | — | — | — |

**QMugs contains no charged-or-radical molecules**, so the model's weakest class is untested externally. That limits what the transfer result can claim and should be stated.

### The near-uniformity claim holds

Offset-aligned MAE spans **0.436 – 0.543 eV** across the four classes present — a spread of **0.107 eV**, against a mean of 0.501 eV. That is roughly a 21% relative spread, and the ordering does not track class difficulty internally (qm9-like is best both internally and here; the other three are within 0.04 eV of each other).

**The claim in the response letter is supported.** Once the protocol offset is removed, residual error is close to uniform across chemical classes — the model is not failing selectively on some chemistry, it is sitting on a constant protocol shift.

### The five cross-fit fold offsets

| Fold | n | Offset (eV) |
|---|---|---|
| 1 | 1,950 | 4.1755 |
| 2 | 1,951 | 4.1786 |
| 3 | 1,951 | 4.1820 |
| 4 | 1,951 | 4.1857 |
| 5 | 1,950 | 4.1863 |

Median **4.1820 eV**, range 4.1755–4.1863, spread **0.0108 eV**.

The offset is stable to about ±0.005 eV across folds — three orders of magnitude below the offset itself and an order below the residual MAE. **The alignment is estimating one genuine protocol constant, not overfitting per fold.** This is the strongest single piece of transfer evidence in the paper.

*Provenance note.* Fold assignments are not persisted in the run outputs. These five values were recovered exactly from `qmugs_external_predictions.csv` as the difference between the aligned and raw domain-expert predictions, which takes exactly five distinct values partitioning the 9,753 molecules into 1950/1951/1951/1951/1950. No value here is estimated.
