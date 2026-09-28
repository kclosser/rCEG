## C2 / Table 2 — confidence bands, including the applicability-domain row

The applicability-domain row was blank, leaving "low confidence" undefined. The cut-offs below are read from the assignment code (`add_confidence()`), and were reproduced exactly from the inputs.

### C2.1 Applicability domain

Distance metric: **1-nearest-neighbour Euclidean distance to the calibration set**, in RobustScaler space after median imputation. Thresholds are quantiles of the *held-out reference* molecules' distances, so the bands are defined by reference chemistry rather than chosen by hand.

| Band | Rule | Distance cut-off | n | % of corpus |
|---|---|---|---|---|
| `high_in_domain` | ≤ 90th percentile | ≤ 20.078 | 60,653 | 81.53% |
| `medium_near_domain` | > 90th, ≤ 97.5th | > 20.078 and ≤ 30.464 | 10,588 | 14.23% |
| `low_out_of_domain` | > 97.5th percentile | > 30.464 | 3,149 | 4.23% |

Exact thresholds: **high ≤ 20.0780**, **medium ≤ 30.4639**. Observed distances across the corpus span 0 to 1.2 × 10⁵, so the tail is very long — the `low_out_of_domain` band is genuinely far from anything the calibration set covers, not marginally outside it.

### C2.2 Cycle consistency (already in the table)

| Band | Rule | n | % of corpus |
|---|---|---|---|
| `high_cycle_consistency` | cycle_error <= 0.20 eV | 10,684 | 14.36% |
| `medium_cycle_consistency` | 0.20 < cycle_error <= 0.50 eV | 12,186 | 16.38% |
| `low_cycle_consistency` | cycle_error > 0.50 eV | 51,520 | 69.26% |

### C2.3 Combined confidence

Combination is **pessimistic**: a molecule is low if *either* axis is low, medium if either is medium, high only if both are high.

| Band | n | % of corpus |
|---|---|---|
| `high_confidence` | 9,451 | 12.70% |
| `medium_confidence` | 12,755 | 17.15% |
| `low_confidence` | 52,184 | 70.15% |

### ⚠️ Two provenance points that must appear in the caption

**1. The denominator is 74,390, not 84,143.** These bands are assigned over the **development corpus** — the strict corpus minus the 9,753 QMugs molecules excised before any model development. Quoting percentages against 84,143 would be wrong.

**2. These bands come from the legacy single-run split, not the repeated-split protocol.** `add_confidence()` is called in the original single-run code path, which partitions the reference pool with a plain 60/20/20 `train_test_split` into **1,273 / 425 / 425**. Every reported accuracy table (Tables 6, 7, 8) instead uses the grouped repeated-split protocol, whose partition is **1,274 / 425 / 424**.

Both partitions are real and both sum to 2,123. They are not interchangeable, and this is the origin of the two competing triples seen elsewhere in the revision. **The confidence bands are reproducible exactly** — recomputing from the legacy split returns 60,653 / 10,588 / 3,149 and the thresholds above to four decimal places — but the caption should state which split defines them, or a reader reconciling Table 2 against Table 7 will find partitions that do not match.

---

### Caption clause, ready to paste

> Confidence bands are assigned over the 74,390-molecule development corpus
> (the strict corpus after the 9,753-molecule QMugs external holdout is
> excised). Applicability-domain thresholds are the 90th and 97.5th
> percentiles of the 1-nearest-neighbour distance from held-out reference
> molecules to the calibration set, measured in RobustScaler space, giving
> cut-offs of 20.08 and 30.46. These bands derive from the single-run
> reference partition (1,273 calibration / 425 validation / 425 test); the
> accuracy tables use the repeated-split protocol, whose partition is
> 1,274 / 425 / 424.

Shorter, if space is tight:

> Bands are assigned over the 74,390-molecule development corpus, with
> applicability cut-offs at the 90th (20.08) and 97.5th (30.46) percentiles of
> held-out reference distances. This table derives from the single-run
> partition (1,273/425/425), not the repeated-split partition (1,274/425/424)
> used for the accuracy tables.
