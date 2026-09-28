# rCEG — final gaps report

Eight items, **all resolved**. Seven from existing outputs; C4 needed a
ten-repeat re-run, which was authorised and has completed.

Two items recomputed values from cached inputs through existing code paths
(C1's offsets, C2's thresholds); C2 reproduced its shipped file exactly. One
model run was performed: `run_C4_perclass_repeats.py`.

| Item | Outcome | Fragment |
|---|---|---|
| C1 | Resolved. Both sets, per class, with a significance test. **Corrects a labelling error.** | `paste/C1_table4.md` |
| C2 | Resolved and reproduced exactly. Denominator is 74,390, not 84,143. | `paste/C2_table2.md` |
| C3 | Extracted in full. | `paste/C3_hyperparameters.md` |
| C4 | **Re-run completed.** Ten-repeat rows with SDs; overall reconciles to 0.338. Worst class is a tie. | `paste/C4_table7.md` |
| C5 | Extracted; closes exactly. | `paste/C5_cleaning_waterfall.md` |
| C6 | Resolved — both values real, different bases. | `paste/C6_figure3.md` |
| C7 | Confirmed; one check fails (Figure 2). | `paste/C7_figure_manifest.md` |
| C8 | Resolved. **82,020 is wrong; use 72,267.** | `paste/C8_label_accounting.md` |

---

## C1 — calibration accuracy and offset, per class

**Resolved without refitting.** The per-class breakdown did exist for held-out
molecules (`B4_table5.csv`), and the offsets — being a property of the data
rather than of a fitted model — were recomputable for both partitions from the
split alone.

**Calibration accuracy on held-out reference molecules** (n = 849 per repeat,
mean over ten repeats). The ensemble never saw these:

| Class | n | MAE | RMSE | R² |
|---|---|---|---|---|
| qm9-like small organic | 524 | 0.226 | 0.433 | 0.922 |
| near-qm9 larger organic | 39 | 0.356 | 0.445 | 0.879 |
| large neutral organic | 91 | 0.473 | 0.624 | 0.647 |
| heteroatom-rich non-qm9 | 93 | 0.506 | 0.753 | 0.499 |
| charged or radical | 102 | 0.572 | 0.861 | 0.869 |
| **all** | 849 | **0.331** | 0.569 | 0.934 |

**Per-class offsets, both sets, over ten repeats:**

| Class | Calibration set | Held-out set |
|---|---|---|
| qm9-like small organic | 5.205 | 5.226 |
| near-qm9 larger organic | 5.404 | 5.349 |
| large neutral organic | 4.983 | 4.915 |
| heteroatom-rich non-qm9 | 4.760 | 4.737 |
| charged or radical | 6.396 | 6.613 |
| **spread** | **1.637 eV** | **1.876 eV** |

The two agree class by class to within 0.22 eV, so the structure is a property
of the chemistry rather than of one partition.

**The offset differs significantly across classes.** One-way ANOVA
F = 37.38, p = 8.3 × 10⁻²⁹; Kruskal–Wallis H = 91.35, p = 6.8 × 10⁻¹⁹;
η² = **0.150**. Chemical class accounts for ~15% of the variance in the
correction.

This is the direct answer to Reviewer 3 and **the justification for
conditioning the calibration on descriptors**. A global constant would misplace
`charged_or_radical` by ~+1.2 eV and `heteroatom_rich_non_qm9` by ~−0.6 eV
relative to the corpus mean — several times the model's 0.338 eV error. The
manuscript should make this argument explicitly rather than treating it as an
implementation detail.

### A labelling error this corrected

Earlier revision documents described the 1.876 eV spread as coming from the
**calibration set**. It does not: `calibration_by_domain_and_source.csv` is
computed on held-out molecules only (`ref_val` + `ref_test`), as its emitting
code states. The correct pairing is **1.637 (calibration) / 1.876 (held-out)**.

A third figure, 2.310 eV, appears in `B4_domain_independence_test.csv` — n =
425, no MAE column, no generating script in the repository. **Superseded; do
not quote it.** SI, SUMMARY and DISCREPANCIES have been corrected.

---

## C2 — applicability-domain cut-offs

**Resolved and reproduced exactly.** Distance metric is 1-NN Euclidean distance
to the calibration set in RobustScaler space after median imputation.
Thresholds are quantiles of held-out reference distances.

| Band | Percentile | Distance | n | % |
|---|---|---|---|---|
| `high_in_domain` | ≤ 90th | ≤ **20.0780** | 60,653 | 81.53% |
| `medium_near_domain` | > 90th, ≤ 97.5th | ≤ **30.4639** | 10,588 | 14.23% |
| `low_out_of_domain` | > 97.5th | > 30.4639 | 3,149 | 4.23% |

Recomputing from the inputs returns these counts exactly and the thresholds to
four decimal places.

### Two provenance points for the caption

**The denominator is 74,390, not 84,143.** The spec asked for fractions of the
strict corpus, but the bands are assigned over the **development corpus** —
after the 9,753 QMugs molecules are excised. Quoting against 84,143 would be
wrong.

**These bands come from the legacy single-run split.** `add_confidence()` is
called in the original single-run path, which partitions the reference pool
with a plain 60/20/20 `train_test_split` into **1,273 / 425 / 425**. Every
reported accuracy table uses the grouped repeated-split protocol, whose
partition is **1,274 / 425 / 424**.

This also explains the two competing triples seen elsewhere in the revision:
both are real, from different code paths. Neither was invented.

---

## C3 — hyperparameters

Extracted in full from the model source: prediction stage, all eight
calibration candidates, the four ablation baselines, the seven architecture
thresholds, and the five determinism guards. A one-sentence Methods summary is
included at the top of the fragment.

Nothing was missing and nothing had to be inferred.

---

## C4 — per-class performance, re-run over ten repeats

**Resolved by a new model run.** RMSE was already in the frozen output; the
missing piece was that the table was a **single split**. Re-running the frozen
protocol with per-class collection gives averaged rows with standard
deviations:

| Class | n | MAE (eV) | RMSE (eV) | R² |
|---|---|---|---|---|
| qm9-like small organic | 264 | **0.237 ± 0.025** | 0.464 ± 0.115 | 0.909 ± 0.037 |
| near-qm9 larger organic | 20 | **0.389 ± 0.051** | 0.471 ± 0.061 | 0.850 ± 0.057 |
| large neutral organic | 44 | **0.480 ± 0.065** | 0.635 ± 0.078 | 0.621 ± 0.113 |
| heteroatom-rich non-qm9 | 47 | **0.529 ± 0.066** | 0.731 ± 0.106 | 0.490 ± 0.198 |
| charged or radical | 49 | **0.537 ± 0.106** | 0.781 ± 0.178 | 0.888 ± 0.059 |
| **all** | 424 | **0.338 ± 0.026** | 0.574 ± 0.070 | 0.930 ± 0.013 |

**The overall row reconciles exactly with the headline (0.3384 ± 0.0258).**
The single-split version gave 0.386, which is what prompted the re-run.

The re-run reproduces the published single split exactly as repeat 0
(0.2526 / 0.4481 / 0.5046 / 0.5395 / 0.7329, overall 0.3859), confirming it
uses the frozen architecture rather than a reimplementation.

### The published split was an unfavourable draw, badly so for one class

| Class | Single split | Ten repeats | Change |
|---|---|---|---|
| qm9-like small organic | 0.253 | 0.237 | −0.016 |
| near-qm9 larger organic | 0.448 | 0.389 | −0.059 |
| large neutral organic | 0.505 | 0.480 | −0.025 |
| heteroatom-rich non-qm9 | 0.540 | 0.529 | −0.011 |
| **charged or radical** | **0.733** | **0.537** | **−0.196** |

`charged_or_radical` at 0.733 sits near the top of its own 0.456–0.734 range
across repeats. Every class improves on averaging, so the published table
understated the model throughout.

### ⚠️ This reverses an earlier recommendation

The two weakest classes are now **statistically indistinguishable**:
0.529 ± 0.066 against 0.537 ± 0.106, differing by 0.008 eV with paired
p = 0.84. Heteroatom-rich is worst in 6 of 10 repeats, charged-or-radical in
2, large neutral organic in 2.

Earlier advice in this file and in `DISCREPANCIES.md` §A2 was to correct the
text to name `charged_or_radical` as worst. **That was based on the single
split and is not supportable.** Write it as a tie, naming both — and if the
submitted text already names heteroatom-rich as weakest, it was right and the
table was the outlier.

`charged_or_radical` retains much the largest spread (± 0.106 vs ± 0.066), so
it is the least *predictable* class even where not the least accurate —
consistent with the poor open-shell agreement (r = 0.074).

### Stage 1 versus rCEG, measured on the same partition

| Model | Test MAE (eV) |
|---|---|
| Calibration (Stage 1, **given** the semiempirical gap) | **0.327 ± 0.028** |
| rCEG `piecewise_split` (descriptors only) | 0.338 ± 0.026 |
| rCEG `domain_expert` (descriptors only) | 0.335 ± 0.027 |

rCEG is **+0.0116 ± 0.0049 eV** behind, worse in **10 of 10** repeats, paired
p < 0.0001. **This is not parity**, and should not be written as such. The
accurate claim: rCEG gives up about **12 meV, roughly 4% of the error**, in
exchange for needing no quantum calculation at inference.

The previously circulating pair 0.331 vs 0.338 was not comparable — 0.331
pooled validation and test (n = 849), 0.338 was test only (n = 424).

### The two source rows

- QM9-labelled: **0.227 eV** (n = 232). The draft carries 0.229; the frozen run
  gives 0.2275. **Adjust to 0.227.**
- Psi4-recomputed: **0.580 eV** (n = 191). **Confirmed exactly.**

These remain single-split; the re-run collected per-class, not per-source.
A third source row with n = 1 should be footnoted or dropped.

---

## C5 — cleaning waterfall

Extracted and closes exactly: 106,066 − 21,923 = 84,143, no unexplained
residual. All fourteen rules are listed, including the five that removed
nothing — a rule checked and found not to fire is part of the audit.

Largest contributors: valence-sanitisation failures 10,089 (46%), placeholder
or sub-threshold gaps 4,719, exact duplicates 3,328, multi-fragment entries
2,005, conflicting duplicates 1,012.

---

## C6 — which baseline Figure 3 plots

**Both circulating values are real and measure different things.**

| Value | What it is | Basis |
|---|---|---|
| 0.405 ± 0.024 eV | `global_hgb` in the ablation table | Ten repeated splits |
| **0.458 eV** | `global_hgb` at the best validation iteration of the warm-start curve | **One split** — this is what Figure 3 annotates |
| 0.456 eV | same model with early stopping instead of warm-start checkpointing | One split |

Caption-ready: test MAE **0.4579**, validation minimum **0.4369** at iteration
**700**, cap 700, early stopping **not** triggered, slope over the last 100
iterations **−0.00527 eV**.

**Quote the gap on a consistent basis.** Ten-repeat: 0.405 − 0.338 =
**0.066 eV**. On the single split the figure plots: 0.458 − 0.386 =
**0.072 eV**. The frequently quoted 0.066 is the ten-repeat comparison — do not
pair it with the figure's 0.458, and do not compare 0.458 against the
ten-repeat headline.

**The baseline is truncated, not converged**: best iteration equals the cap and
validation MAE is still falling. It is shown under-trained, which slightly
flatters the ablation. This does not change the conclusion — closing the gap at
the observed, decelerating rate would need roughly 1,300 further iterations —
but the figure should be described as a fixed 700-iteration budget, not as
convergence.

---

## C7 — figure files

All nine manuscript figures are 600 dpi or vector, each with a data sidecar
where one is meaningful.

**Figure 1 — all four checks pass.** Reference-labelled molecules, legend
counts 1,228 and 909, exact mass not plotted (panel dropped for collinearity,
r = 0.9939), no point at 0 eV (minimum gap 6.14).

**Figure 2 — three of four pass, one fails.**

- Vector/600 dpi: pass.
- Calibration and rCEG as separate blocks: pass.
- No stale 0.347 in the final box: **pass** — the figure hardcodes no MAE at
  all; its terminal box reads "Predicted gap".
- Does not mark semiempirical inputs as required at inference: **FAIL.** The
  inference path lists `PM7 semiempirical properties` as required, which
  concedes the objection the structure-only result (+0.0001 eV, p = 0.814)
  answers.

**This is the only figure change still outstanding.**

A separate defect found earlier and already fixed: a stale July Figure 1
sidecar holding the 16 zero-gap points the revision removed was shipping in the
deposit. Renamed, excluded, and guarded by name.

---

## C8 — pseudo-label count

**Resolved. 82,020 is wrong.** It equals 84,143 − 2,123 and omits the 9,753
QMugs molecules excised before any fitting.

| Step | Count |
|---|---|
| PQR release | 106,066 |
| − cleaning | 21,923 |
| = strict corpus | 84,143 |
| − QMugs excised | 9,753 |
| **= development corpus** | **74,390** |

By `label_source` in `all_pqr_predictions_with_label_source.csv`: 1,273 real
anchors + **73,117** pseudo-labels = 74,390.

Two defensible figures depending on what the sentence claims:

| Reading | Anchors | Pseudo | Total |
|---|---|---|---|
| Labels assigned across the development corpus | 1,273 | 73,117 | 74,390 |
| **Molecules actually trained on, per repeat** | 1,274 | **72,267** | 73,541 |

**The manuscript should carry 72,267.** The sentence says *pseudo-labelled
training targets*, and that is the count the model trains on. It also matches
Figure 2, which already shows 73,541 = 1,274 + 72,267 — so adopting it makes
text and figure agree.

State the 9,753 excision explicitly wherever corpus sizes appear; its absence
is what produced the error.

---

## What could not be produced

| Item | Reason |
|---|---|
| Per-repeat SDs for the C1 calibration MAE/RMSE columns | The run aggregated before writing. The C1 *offsets* were recomputed and do carry SDs. |
| Per-**source** SDs in Table 7 (QM9 vs Psi4 rows) | The re-run collected per-class, not per-source. Those two rows remain single-split. |

---

## Corrections this round made to earlier revision documents

1. **The 1.876 eV spread is held-out, not calibration-set.** Correct pairing:
   1.637 / 1.876. The 2.310 eV figure is superseded. (SI, SUMMARY,
   DISCREPANCIES corrected.)
2. **1,273 / 425 / 425 was not an invention.** It is the legacy single-run
   split, which defines Table 2's confidence bands. The reported accuracy
   tables use 1,274 / 425 / 424. Both are real; DISC-B9 rewritten to say so.

---

## Outstanding author actions after this round

| # | Action |
|---|---|
| C4 | ~~Decide on a re-run~~ — **done**; paste the ten-repeat table, and write the worst class as a tie |
| C4 | Adjust the QM9 source row from 0.229 to **0.227** |
| C7 | Mark the semiempirical inputs optional in Figure 2's inference path |
| C6 | Figure 3 caption: fixed 700-iteration budget; state it plots 0.458, not 0.338 |
| C8 | Replace 82,020 with **72,267** |
| C2 | Table 2 caption: denominator 74,390, and bands derive from the legacy split |
| C1 | Add the significance result as the stated justification for descriptor-conditioned calibration |

Plus the ten manuscript edits already listed in `SPEC_FOR_CLAUDE_WEB.md`.
