# rCEG — reviewer coverage check

**Paste this into Claude web together with `SPEC_FOR_CLAUDE_WEB.md`, the
manuscript, and — importantly — the actual reviewer reports.**

`SPEC_FOR_CLAUDE_WEB.md` tells you *what to change*. This file tells you *what
each reviewer asked for and whether it has been answered*, so nothing reaches
resubmission uncovered.

---

## 0. A limitation you must work around

**I do not have the reviewer reports.** The map in §2 was reconstructed from
concerns recorded across the analysis files during the revision. It is
accurate as far as it goes, and every "answered by" cell below points at a
real, verified result.

But it **cannot certify completeness**, for two reasons:

1. **Reviewer 1 appears nowhere in the recorded material.** Either they raised
   nothing that reached the analysis, or their points were folded in without
   attribution. §2.1 is therefore empty and must be filled from the report.
2. A concern that was noted and quietly handled — or noted and forgotten —
   would leave no trace in these files.

**So your first task is §1, not §2.** Read the reports, enumerate every
discrete request, and match them against §2. Anything in the reports that is
not in §2 is a **gap** and must be reported, not assumed covered.

---

## 1. What to do first

1. Extract every discrete request from each reviewer report. Number them
   R1.1, R1.2, R2.1 … A single paragraph often contains two or three.
2. Match each against §2 below.
3. Classify every one:

| Verdict | Meaning |
|---|---|
| **ANSWERED** | A verified result addresses it; §2 names the evidence |
| **ANSWERED, NEEDS WRITING** | The result exists but no manuscript text carries it yet |
| **PARTIAL** | Addressed in part; say precisely what is missing |
| **GAP** | In the reports, absent from §2. **Flag loudly; do not improvise an answer** |
| **DECLINED** | A deliberate decision not to do what was asked; must be argued in the letter |

4. Produce a coverage table before drafting anything.
5. **Report every GAP before writing the letter.** A gap found now is
   recoverable; one found by an editor is not.

---

## 2. Concerns recorded during the revision

### 2.1 Reviewer 1

**Nothing recorded.** One item is known only second-hand: a request that the
**cleaned dataset be defined in the Methods**, which is answered by the
itemised waterfall in `paste/C5_cleaning_waterfall.md` (all fourteen rules,
106,066 → 84,143, closing exactly with no residual).

Treat this section as **unverified**. Populate it from the report and expect
gaps.

### 2.2 Reviewer 2

| # | Concern | Status | Answered by |
|---|---|---|---|
| 2a | Reference-pool arithmetic does not add up | **ANSWERED** — the pool is **2,123**, not 2,137. 1,228 + 909 − 14, matching 1,274 + 425 + 424 | `paste/A8_pool_arithmetic.md`; edit **E1** |
| 2b | The calibration model's own accuracy on reference molecules it never saw, **resolved by class** | **ANSWERED** — per-class MAE/RMSE/R² on held-out molecules; overall **0.331 eV** | `paste/C1_table4.md` §C1.1 |

Reviewer 2 was right on 2a. Concede it plainly.

### 2.3 Reviewer 3

| # | Concern | Status | Answered by |
|---|---|---|---|
| 3a | Is the semiempirical→DFT correction domain-independent? | **ANSWERED DECISIVELY** — it is **not**. ANOVA F = 37.38, p = 8.3 × 10⁻²⁹; η² = 0.150 | `paste/C1_table4.md` §C1.3; edit **E13** |
| 3b | PQR-minus-reference differences reported separately by domain | **ANSWERED** — per class, on both the calibration and held-out sets | `paste/C1_table4.md` §C1.2, `paste/B4_domain_independence.md` |
| 3c | Error **distributions**, not only summary statistics | **ANSWERED** — deciles p10–p95 and max, per class | `paste/B2_error_deciles.md` |
| 3d | Sample size, MAE, **RMSE** and R² per class | **ANSWERED** — all four, now over ten repeats with SDs | `paste/C4_table7.md` |

**3a is the strongest answer in the revision.** It converts descriptor-
conditioned calibration from a design preference into a requirement: a single
global offset would misplace charged species by ~+1.2 eV, several times the
model's own error. Lead with it.

### 2.4 Reviewer 4

| # | Concern | Status | Answered by |
|---|---|---|---|
| 4a | "The model is not structure-only — four scalar inputs come from PQR, so rCEG cannot be applied to an arbitrary molecule from SMILES alone" | **ANSWERED** — dropping all five costs **+0.0001 eV, p = 0.814** | `paste/B6_structure_only.md`; edit **E10** |
| 4b | Cannot determine which script produced which number | **ANSWERED** — figure/table→script provenance map, plus a 225-file deposit with `run_all.sh` | `docs/PROVENANCE.md`, the reproducibility package |
| 4c | A DFT reference time is needed alongside the inference time | **ANSWERED** — re-timed on the same machine, **116.8 s** (n=20) vs 119 s from cluster logs; agree within 2% | `paste/T1_dft_timing.md` |
| 4d | A molecule outside PQR needs a semiempirical step first, so the in-PQR speedup is not the honest one | **ANSWERED** — both columns reported; and 4a shows the step is skippable entirely | `paste/C3_table9.md`, `paste/T3_speedups.md` |
| 4e | "Low confidence" is undefined — the applicability row is blank | **ANSWERED** — cut-offs 20.08 / 30.46 at the 90th and 97.5th percentiles, with band populations | `paste/C2_table2.md`; edit **E12** |
| 4f | Absence of repeated splits | **ANSWERED** — everything is ten repeated splits; per-class rows now too | `paste/C4_table7.md`, Tables 6–8 |

### 2.5 Raised by more than one reviewer

| Concern | Status | Answered by |
|---|---|---|
| The data is **PM6**, not PM7 | **OPEN — needs the authors.** The PQR FAQ says PM6 as implemented in MOPAC; the distributed field is named `pm7`. Documentation outranks the field label | `paste/A3_method_name.md`; §4 of the edit spec |
| Near-duplicates could inflate performance | **ANSWERED** — scaffold splitting costs only **+7.3%** while cutting median nearest-neighbour similarity 0.455 → 0.347 | `paste/B5`/Table 8 data |

---

## 3. Answers that exist but no reviewer asked for

Volunteer these. Several pre-empt obvious follow-ups, and two are corrections
a referee would otherwise catch.

| Result | Why it matters |
|---|---|
| Geometry sensitivity **0.440 eV**, exceeding the model's own error | Forces accuracy to one decimal (E5). Damaging if a referee finds it first |
| Conformational floor **0.076 eV**; cross-protocol floor **0.295 eV** | Shows the model sits near its reference noise floor |
| **176 of 375** descriptors are fingerprint bits | Narrows the interpretability claim (E7) before someone else does |
| **None of the 5 breakpoints** is a fingerprint bit | The interpretability claim that survives, and it is the stronger one |
| Pseudo-label count 82,020 → **72,267** | An arithmetic error in the Methods (E9) |
| Calibration candidates ten/six → **eight/four** | A miscount in the Methods (E11) |
| Stage 1 reaches **0.327 eV** vs rCEG **0.338 eV** on the same partition | Quantifies what is given up to avoid a quantum calculation (E15) |
| QMugs has **no charged-or-radical molecules** | The weakest class is untested externally — state it before a referee asks |

---

## 4. Things to be careful about in the letter

Four places where a confident sentence would be wrong.

**The worst class is a tie.** Over ten repeats, heteroatom-rich non-QM9
(0.529 ± 0.066) and charged-or-radical (0.537 ± 0.106) differ by 0.008 eV,
paired p = 0.84. Do **not** claim charged-or-radical is worst — that came from
a single split where it scored 0.733, near the top of its own 0.456–0.734
range. If the submitted text already named heteroatom-rich, it was right.

**Stage 1 versus rCEG is not parity.** rCEG is +0.0116 ± 0.0049 eV behind, in
10 of 10 repeats, p < 0.0001. Say rCEG gives up ~12 meV (~4%) to need no
quantum calculation. Do not write "matches" or "equals".

**Two partitions are both real.** 1,274/425/424 for the accuracy tables;
1,273/425/425 for Table 2's confidence bands. Neither is an error.

**Figure 3 plots 0.458 eV**, the single-split ablation baseline — not the
ten-repeat 0.405, and not the headline 0.338. Its curve has not converged
(best iteration = the 700 cap). Do not call it a convergence plot.

---

## 5. Still open — must be resolved or declared

| # | Item | Who |
|---|---|---|
| 1 | **PM6 vs PM7** — verify the PQR FAQ, then adopt PM6 and note the `pm7` field name | Authors |
| 2 | Per-**source** SDs in Table 7 (the 0.227 QM9 and 0.580 Psi4 rows are still single-split) | Optional re-run |
| 3 | Per-repeat SDs on the calibration MAE/RMSE columns in §C1.1 | Would need a re-run |

Items 2 and 3 are **disclosure, not blockers**: state that those specific rows
are single-split rather than implying a spread that was not measured.

---

## 6. What to hand back

1. **The coverage table from §1** — every numbered reviewer request with a
   verdict. This is the primary deliverable; produce it before the letter.
2. **Every GAP, listed separately and prominently.**
3. The response-to-reviewers letter, organised by reviewer, each point citing
   the specific number and where it now appears in the manuscript.
4. Anything in §2 you could not locate in the revised manuscript — meaning the
   result exists but never made it into the text.

**A gap reported now is recoverable. Do not close one by inventing an answer,
and do not mark something ANSWERED because a result merely exists — it has to
appear in the manuscript.**
