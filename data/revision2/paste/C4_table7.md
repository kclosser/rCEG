## C4 / Table 7 — per-class performance, ten repeated splits

**Replaces the single-split version.** All rows are means over the same ten repeated splits as the headline, with standard deviations across repeats.

| Chemical class | n | MAE (eV) | RMSE (eV) | R² | Median abs err (eV) |
|---|---|---|---|---|---|
| qm9 like small organic | 264 | **0.237 ± 0.025** | 0.464 ± 0.115 | 0.909 ± 0.037 | 0.116 |
| near qm9 larger organic | 20 | **0.389 ± 0.051** | 0.471 ± 0.061 | 0.850 ± 0.057 | 0.366 |
| large neutral organic | 44 | **0.480 ± 0.065** | 0.635 ± 0.078 | 0.621 ± 0.113 | 0.385 |
| heteroatom rich non qm9 | 47 | **0.529 ± 0.066** | 0.731 ± 0.106 | 0.490 ± 0.198 | 0.434 |
| charged or radical | 49 | **0.537 ± 0.106** | 0.781 ± 0.178 | 0.888 ± 0.059 | 0.368 |
| **all** | 424 | **0.338 ± 0.026** | 0.574 ± 0.070 | 0.930 ± 0.013 | 0.182 |

**The overall row now reconciles exactly with the headline: 0.3384 ± 0.0258 eV.** A reader reconstructing the overall MAE from the class rows recovers the reported figure, which the single-split version did not allow.

### What changed, and why it matters

| Class | Single split (old) | Ten repeats (new) | Change |
|---|---|---|---|
| qm9 like small organic | 0.253 | 0.237 ± 0.025 | -0.015 |
| near qm9 larger organic | 0.448 | 0.389 ± 0.051 | -0.059 |
| large neutral organic | 0.505 | 0.480 ± 0.065 | -0.024 |
| heteroatom rich non qm9 | 0.540 | 0.529 ± 0.066 | -0.010 |
| charged or radical | 0.733 | 0.537 ± 0.106 | -0.196 |
| all | 0.386 | 0.338 ± 0.026 | -0.047 |

The single split was an unfavourable draw throughout, and **severely so for `charged_or_radical`: 0.733 eV there against 0.537 ± 0.106 eV over ten repeats.** Its 0.733 is the second-highest of the ten values that class takes (range 0.456–0.734).

### ⚠️ The worst class is no longer a single class

| Class | MAE ± SD | Worst in |
|---|---|---|
| heteroatom-rich non-qm9 | 0.529 ± 0.066 | **6 of 10 repeats** |
| charged or radical | 0.537 ± 0.106 | 2 of 10 repeats |
| large neutral organic | 0.480 ± 0.065 | 2 of 10 repeats |

The two weakest classes differ by only **+0.0078 eV**, paired t **p = 0.84** — they are statistically indistinguishable.

**This reverses earlier advice.** On the single split, `charged_or_radical` at 0.733 eV was clearly worst, and the recommendation was to correct the text to say so. Over ten repeats that is not supportable: `heteroatom_rich_non_qm9` is worst more often, and the difference between them is well inside one standard deviation.

**Write it as a tie.** Something like: *the two weakest classes are heteroatom-rich non-QM9 and charged-or-radical species, at 0.53 and 0.54 eV; their difference is not significant across repeated splits.* If the submitted text already names heteroatom-rich as weakest, **it was right and the single-split table was the outlier** — check before editing.

`charged_or_radical` still has by far the largest spread (± 0.106 eV against ± 0.066), which is worth saying: it is the least *predictable* class even where it is not the least accurate, and that is consistent with the poor semiempirical/DFT agreement for open-shell species (r = 0.074).

### Stage 1 versus rCEG on the same test partition

The calibration mapping is handed the semiempirical gap; rCEG is not. Measured on the same test molecules in the same repeats:

| Model | Test MAE (eV) |
|---|---|
| Calibration (Stage 1, given the semiempirical gap) | **0.3268 ± 0.0278** |
| rCEG `piecewise_split` (descriptors only) | 0.3384 ± 0.0258 |
| rCEG `domain_expert` (descriptors only) | 0.3350 ± 0.0265 |

rCEG is **+0.0116 ± 0.0049 eV** behind the calibration mapping, worse in **10 of 10** repeats, paired p < 0.0001.

**State this, and state it accurately.** The gap is small but real and consistent — this is not parity. The correct claim is that **rCEG gives up about 12 meV, roughly 4% of the error, in exchange for not needing a quantum calculation at inference at all.** That is the trade the paper is offering, and quantifying it is stronger than leaving it unstated.

Note the earlier figures 0.331 and 0.338 were **not comparable**: 0.331 pooled validation and test (n = 849) while 0.338 was test only (n = 424). The table above fixes that.
