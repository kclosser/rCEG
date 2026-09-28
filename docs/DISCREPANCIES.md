# Discrepancies, blockers and things the authors must decide

Everything found during revision 2 that disagrees with the submitted
manuscript, is unresolved, or would mislead a reader if left as it stands.

> **Note on section letters.** The A/B/C/D/E sections below are a severity
> grouping local to this document. They are **not** the revision spec's task
> IDs, which happen to use the same letters for different things — spec task B6
> is the structure-only ablation, whereas §B6 here is a Figure 2 defect. When
> citing an item from this file elsewhere, prefix it `DISC-`.

Ordered by how much damage it would do if it reached the referees unaddressed.
Each entry states the finding, the evidence, and the recommended action. Where
a finding sounds alarming, the number that bounds it is given in the same
entry — nothing here is a threat to the validity of the paper, and several
items are the paper's strongest material once stated correctly.

---

## A. Numbers in the manuscript that are wrong

### A1. The reference pool is 2,123, not 2,137

**Evidence.** 1,228 QM9-overlap molecules + 909 Psi4-recomputed − 14 carrying
both labels = 2,123. The split sizes confirm it independently:
1,274 + 425 + 424 = 2,123, identical in all ten repeats.

**Status.** Certain. Reviewer 2 raised this and is correct.

**Action.** Correct every occurrence of 2,137. Check the Methods text, any
table caption quoting the pool size, and the SI.

<sub>`A8_dual_labelled.csv`, `paste/A8_pool_arithmetic.md`</sub>

---

### A2. The worst class is a tie, and the published table overstated it

**Superseded by the ten-repeat re-run.** The published per-class table was a
single split. Averaged over the same ten repeats as the headline:

| Class | Single split | Ten repeats | Worst in |
|---|---|---|---|
| qm9-like small organic | 0.253 | **0.237 ± 0.025** | — |
| near-qm9 larger organic | 0.448 | **0.389 ± 0.051** | — |
| large neutral organic | 0.505 | **0.480 ± 0.065** | 2 of 10 |
| heteroatom-rich non-qm9 | 0.540 | **0.529 ± 0.066** | **6 of 10** |
| charged or radical | **0.733** | **0.537 ± 0.106** | 2 of 10 |
| all | 0.386 | **0.338 ± 0.026** | — |

The two weakest classes differ by **0.008 eV, paired p = 0.84** — statistically
indistinguishable.

**This reverses the earlier recommendation in this file.** On the single split,
`charged_or_radical` at 0.733 eV was clearly worst and the advice was to
correct the text to say so. That is not supportable over ten repeats: its
0.733 is near the top of its own 0.456–0.734 range, and
`heteroatom_rich_non_qm9` is worst more often.

**Action.** Write the two weakest classes as a tie, naming both. **If the
submitted text already names heteroatom-rich as weakest, it was right and the
single-split table was the outlier** — verify before editing.

Worth keeping: `charged_or_radical` has much the largest spread (± 0.106 vs
± 0.066), so it is the least *predictable* class even where not the least
accurate. That is consistent with the poor semiempirical/DFT agreement for
open-shell species (§C2 below, r = 0.074).

<sub>`C4_perclass_summary.csv`, `paste/C4_table7.md`</sub>

---

### A3. The Introduction still describes the meta-learner, which was dropped

**Evidence.** The meta-learner (`full_rceg`) was **not** retained. On
validation it is +0.0053 eV *worse* than the selected `piecewise_split`
configuration, better in only 3 of 10 repeats, paired t-test p = 0.12. It
survives in the ablation table only.

**Action.** Rewrite the Introduction's architecture description to the frozen
two-stage form: calibration → pseudo-labels → domain experts with piecewise
descriptor splits. Check the abstract and any architecture figure caption for
the same language. Figure 2's Stage 2 box already reads "Domain experts and
piecewise splits" and needs no change.

<sub>`frozen_architecture.csv`, cols `meta_learner_verdict`, `why_not_domain_expert`</sub>

---

### A4. The open-shell "0.2 eV" matches no computed quantity

**Evidence.** The draft cites roughly 0.2 eV in the context of open-shell
agreement. Nothing measured here is 0.2 eV. The actual open-shell numbers:

| Subset | n | Pearson r | mean &#124;Δ&#124; (eV) |
|---|---|---|---|
| multiplicity > 1 (open-shell) | 99 | 0.074 | 8.30 |
| neutral closed-shell | 660 | 0.694 | 5.05 |

**Action.** Either locate what the 0.2 eV referred to and label it, or remove
it. Do not leave an unattributed number in a revision whose referees are
checking arithmetic. The honest replacement is §C2 below.

<sub>`A6_openshell_agreement.csv`</sub>

---

### A5. The Methods miscount the calibration candidates

**Evidence.** The Methods state **ten calibration candidates, six of them
descriptor-based**. The frozen model defines **eight, four of them
descriptor-based**:

| Kind | n | Candidates |
|---|---|---|
| Gap-only | 4 | `gap_linear_robust`, `gap_ridge_robust`, `gap_ridge_standard`, `gap_ridge_power` |
| Descriptor-based | 4 | `desc_ridge_robust`, `desc_hgb`, `desc_et`, `desc_rf` |

Read directly from the `cands` dictionary in `fit_calibration_ensemble`.

**Action.** Correct to **eight candidates, four descriptor-based**, and note
that the top three by validation MAE are retained and combined with weights
proportional to 1/MAE. The full listing is in `paste/C3_hyperparameters.md`.

Nothing else changes: the selection rule, the ensemble size and every reported
number are unaffected. This is a counting error in the prose, not in the
model.

<sub>`pqr_full_domain_moe_qmugs_external.py::fit_calibration_ensemble`</sub>

---

## B. Things that change how results must be *stated*

### B1. Geometry sensitivity exceeds the model's own error

**The single most consequential finding of this revision.**

**Evidence.** Re-optimizing at B3LYP and recomputing the gap, versus the
UFF-optimized geometry the pipeline actually consumes:

| Quantity | Value |
|---|---|
| mean &#124;Δ gap&#124; | **0.440 eV** |
| median &#124;Δ gap&#124; | 0.279 eV |
| SD / max | 0.607 / 3.591 eV |
| n converged | 37 of 50 |

The mean shift (0.440 eV) is **larger than the reported test MAE
(0.338 eV)**.

**What this does and does not mean.** It does *not* invalidate the model: the
reported error is a correct measurement of the pipeline as specified, and the
pipeline specifies UFF geometries at both training and inference, so the
comparison is internally consistent. What it means is that the *third
significant figure is not meaningful*, because a defensible change to the
input geometry moves the answer by more than the error being quoted.

**Action.**
1. Quote accuracy to **one decimal place — 0.3 eV**, not 0.338 eV, in the
   abstract and conclusions. Keep the precise value in tables where the
   protocol is fully specified.
2. State explicitly that the accuracy is conditional on the UFF-optimized
   input geometry.
3. Report this measurement with its sample size and the caveats in B2.

<sub>`C1_geometry_shift.csv`, `geometry_sensitivity.csv`</sub>

---

### B2. The geometry sample is 37, not the 45 the protocol asked for — and it is biased in both directions

**Evidence.** 50 molecules attempted, 37 optimizations converged. The
shortfall is not random with respect to the quantity being measured:

- **Upward bias.** Optimizer failures concentrate in large, flexible molecules
  with shallow potential-energy surfaces — exactly the molecules whose gaps
  move *least* between geometries. Dropping them *raises* the reported mean.
- **Downward bias.** The optimizations that did converge used `gau_loose`
  criteria, which stop nearer the starting geometry and therefore *understate*
  the true difference.

These do not cancel in any quantified way, and it is not possible to say from
this sample which dominates.

**Action.** Report 0.4 eV as an order-of-magnitude sensitivity with n = 37 and
both caveats stated, not as a clean measurement. Do not report the per-domain
breakdown as if it were reliable — it rests on 4–10 molecules per domain, and
its largest value sits on `qm9_like_small_organic` purely because of a single
3.59 eV outlier, which would otherwise invite exactly the wrong conclusion.

<sub>SI §S7.2</sub>

---

### B3. The speedup spans three orders of magnitude; a single figure overstates precision

**Evidence.** Measured per-domain, all timings single-threaded on one machine:

| Domain | B3LYP | rCEG end-to-end | speedup |
|---|---|---|---|
| `qm9_like_small_organic` | 10.9 s | 11.2 ms | 972× |
| `near_qm9_larger_organic` | 40.1 s | 12.2 ms | 3,280× |
| `heteroatom_rich_non_qm9` | 42.4 s | 12.4 ms | 3,409× |
| `large_neutral_organic` | 129.8 s | 14.0 ms | 9,294× |
| `charged_or_radical` | 256.2 s | 15.1 ms | 16,979× |

The range is **972× to 16,979×**, because B3LYP cost grows steeply with system
size while rCEG inference is essentially flat.

**Action.** Quote the range, or quote a named domain. A single averaged
speedup is not a property of the method and will be challenged.

<sub>`C3_timings.csv`</sub>

---

### B4. 176 of the 375 selected descriptors are Morgan fingerprint bits

**Evidence.** All 375 LASSO-block names were recovered by value-matching
against a regenerated descriptor pool (362 uniquely, 13 to a small set of
value-identical candidates). Of those, **176 (47%) are Morgan fingerprint
bits** — substructure indicators, not named physicochemical quantities.

**Action.** Narrow any claim that the selected feature set is chemically
interpretable. The claim that *survives intact* is the stronger and more
specific one: **none of the five learned piecewise breakpoints falls on a
Morgan bit.** All five are named descriptors — TPSA, Hall–Kier alpha, a BCUT2D
eigenvalue, Ipc, and a VSA_EState bin. The paper's named contribution is
therefore interpretable even though the bulk feature block is not, and that
distinction should be made explicitly rather than glossed.

<sub>`A4_lasso_feature_names.csv`, `A4_piecewise_feature_map.csv`, SI §S5</sub>

---

### B5. Two different calibration "spreads" exist and must not be conflated

**Evidence, corrected.** Earlier revision documents paired these wrongly.
`calibration_by_domain_and_source.csv` is computed on **held-out** molecules
only (`ref_val` + `ref_test`), as its emitting code states — so its 1.876 eV
spread is a held-out number, not a calibration-set one.

Recomputed over the same ten repeats, the correct pairing is:

| Set | Per-class offset spread |
|---|---|
| Calibration | **1.637 eV** |
| Held-out (validation + test) | **1.876 eV** |

A third figure, **2.310 eV**, comes from `B4_domain_independence_test.csv`:
n = 425 (a single partition), no MAE column, and **no generating script in the
repository**. It is superseded.

**Action.** Use `paste/C1_table4.md`, which gives both sets per class with the
significance test. Name the set whenever a spread is quoted; never average
them; drop 2.310 eV.

<sub>`calibration_offsets.csv`, `B4_domain_independence_test.csv`</sub>

---

## C. Findings that strengthen the paper if stated correctly

### C1. The model is effectively structure-only at inference

**Evidence.** Dropping all five PQR semiempirical scalars (`x0`–`x4`) changes
test MAE by **+0.0001 eV (+0.0%)**, far inside the ±0.026 eV split-to-split
SD (p = 0.81). Dropping only the three that genuinely require a semiempirical
calculation costs +0.0006 eV.

This directly answers Reviewer 4's objection that rCEG cannot be applied to an
arbitrary molecule from SMILES alone. **At inference, it can.**

**Required caveat.** Training labels are calibrated pseudo-labels derived from
`pqr_gap`, itself a semiempirical quantity. The result licenses a statement
about inference, not about training. State both halves.

<sub>`B6_summary.csv`, SI §S10</sub>

---

### C2. `charged_or_radical` is hard because the input data is bad there, not because the model fails

**Evidence.** PQR and DFT barely agree on open-shell molecules: Pearson
r = **0.074** over 99 molecules, against r = 0.694 for 660 neutral
closed-shell molecules.

This is the mechanism behind the worst-domain result in A2. It is a property
of the semiempirical input, and no learner consuming that input could do much
better on those molecules.

**Action.** Use this to replace the unattributed 0.2 eV in A4, and to reframe
the worst-domain sentence. It converts an apparent weakness into a
characterized limitation of the source data.

<sub>`A6_openshell_agreement.csv`</sub>

---

### C3. Two reference protocols disagree by about as much as the model errs

**Evidence.** On the 14 molecules labelled under both QM9 and Psi4 protocols,
mean absolute disagreement is **0.295 eV** (median 0.276, max 0.680). The
manuscript's 0.295 eV is confirmed.

Separately, the conformational floor — the spread of PQR gaps among conformers
of the same molecule, across 9,697 molecules with ≥3 conformers — has a median
within-molecule SD of **0.076 eV**.

**Action.** Both belong in the Discussion as bounds on achievable accuracy.
Together with B1 they make a coherent argument that the model is operating
near the noise floor of its own reference data, which is a far better framing
than defending a third decimal place.

<sub>`A8_dual_labelled.csv`, `C2_conformer_floor.csv`</sub>

---

### C4. The timing comparison was cross-hardware; re-measured, it is unchanged

**Evidence.** The manuscript's 119 s B3LYP reference came from SLURM cluster
logs while rCEG timings were measured on a laptop. Re-timed on the same
machine as every other number: **116.8 s median (n = 20)** versus 119 s
(n = 332) from the cluster.

**The two agree to within 2%, so no conclusion changes.**

**Action.** Keep the result, but state the hardware in the Table 9 caption and
present it as a like-for-like measurement on one machine. The reviewer's
methodological objection was valid even though the number survives it.

<sub>`T1_timing_dft_local.csv`, `paste/T1_dft_timing.md`</sub>

---

## D. Unresolved — needs an author decision

### D1. PM6 versus PM7 is still open

**Evidence.** The PQR download exposes the properties under a field literally
named `pm7`, and no other method string appears anywhere in the first 5,000
records. But **two referees independently described the data as PM6**, and a
field name is weaker evidence than the source publication.

**This cannot be resolved from the data files.** It requires checking the PQR
publication and documentation directly.

**New evidence, supplied by the authors.** The PQR project FAQ at
`pqr.pitt.edu` states the properties are **PM6 as implemented in MOPAC**. That
is project documentation and it outranks a field label in the download, so the
balance of evidence now favours **PM6**, consistent with what both referees
said.

**Action.** Verify the FAQ statement directly and cite it, then use **PM6**
consistently throughout, adding one sentence noting that the distributed data
exposes these properties under a field named `pm7` — otherwise a reader
checking the raw download will think the manuscript is wrong. This is the only
remaining item where the analysis cannot settle the question on its own: it
rests on the source project's documentation, not on anything measurable here.

<sub>`paste/A3_method_name.md`</sub>

---

### ~~D2~~ RESOLVED. Table 9's inference figure is single-molecule latency, and it now reproduces

**Two separate problems, found during final assembly.**

**(a) The inference time is not per-domain.** `C3_timings.csv` reports
`inference_ms = 9.496` for all five domains — identical to three decimal
places. It is a single measurement broadcast to every row, not five per-domain
measurements. Inference cost is genuinely near-flat in molecule size, so this
is defensible, but the table presents it as if measured per domain.

**(b) The script that produced it is not in the repository.**
`run_C3_timings.py` documents that inference was measured separately by
`run_C3_inference.py`, because the psi4 environment has no scikit-learn. **No
such file exists** in the repository or the working directory. The 9.496 ms
figure therefore cannot currently be reproduced from the deposit.

**Additionally, three different inference numbers exist in the run outputs**,
measuring three different things, and they are easy to confuse:

| Value | What it measures | Source |
|---|---|---|
| 0.40 ms | batched, amortized per molecule over 20 | `timing_benchmark.csv` |
| 9.50 ms | the figure used in Table 9 | `C3_timings.csv` |
| 38.3 ms | single-molecule cold latency | `timing_benchmark.csv` |

`paste/T1_dft_timing.md` **mislabels these**: it presents the end-to-end total
of 42.2 ms as "descriptors 3.1 ms + inference 0.40 ms", which does not sum
(3.5 ≠ 42.2). The 42.2 ms total is in fact descriptors (3.1 ms) plus
*single-molecule latency* (38.3 ms). The parenthetical uses the batched
figure; the total uses the unbatched one.

**RESOLVED by T3.** `run_C3_inference.py` has been written, committed and added
to `run_all.sh`. Re-measuring on the same hardware gives a **median
single-molecule latency of 9.611 ms**, against the tabulated 9.496 ms — a 1.2%
difference. **Table 9's inference column is the single-molecule latency**, and
it is now regenerable.

Two further points settle the rest of the item:

- **The single repeated value is justified.** Measured latency spans only
  9.57–9.79 ms from 8 to 27 heavy atoms, because cost is set by the fitted
  forest rather than the molecule. State this in the caption rather than
  leaving five identical numbers unexplained.
- **The published speedup range is confirmed.** Re-measurement gives
  965×–16,655× under the single-molecule definition, against the published
  972×–16,979×. Batched inference (0.073–0.133 ms/molecule) would instead give
  6,008×–45,020×.

**Recommendation: keep the single-molecule definition.** It is what the
published range already uses and it is the conservative choice. The
parenthetical in `paste/T1_dft_timing.md` has been corrected to
"single-molecule inference 38.3 ms".

**One number remains unexplained and should not be quoted.**
`timing_benchmark.csv` records `inference_single_molecule_latency` = 38.3 ms,
four times what T3 measures under the same definition on the same hardware. It
routed through a different code path and is not reproduced by the committed
script.

<sub>`C3_timings.csv`, `timing_benchmark.csv`, `paste/T1_dft_timing.md`</sub>

---

### ~~D3~~ RESOLVED as documented. Table 9's representatives are pinned, and the caption now names the dication

**Evidence.** The five representative molecules were chosen ad hoc and the
selection left in `/tmp/c3_reps.csv`, with no generating script. The file has
now been preserved as `C3_representatives.csv`, and
`run_C3_timings.py::select_representatives()` reconstructs the documented
criterion — one molecule per domain at the domain's median heavy-atom count.

**The reconstruction does not reproduce the original picks.** It agrees on
heavy-atom count in 4 of 5 domains but selects different molecules, and for
`heteroatom_rich_non_qm9` it computes a median of 19 where the original used
18 — implying the original median was taken over a different population:

| Domain | published | reconstructed |
|---|---|---|
| `charged_or_radical` | pqr_0166 (22 heavy, **charge +2, mult 3**) | pqr_0007 (22 heavy, charge +1, mult 1) |
| `heteroatom_rich_non_qm9` | pqr_0399 (18) | pqr_0251 (19) |
| `large_neutral_organic` | pqr_0502 (27) | pqr_0502 (27) — same |
| `near_qm9_larger_organic` | pqr_0845 (14) | pqr_0788 (14) |
| `qm9_like_small_organic` | pqr_0871 (8) | pqr_0852 (8) |

**Why this matters beyond bookkeeping.** The published
`charged_or_radical` representative is an **open-shell dication**. Its 256 s
B3LYP cost is what produces the headline **16,979×** speedup — the top of the
range quoted in §B3. That cost is driven as much by the unrestricted
open-shell reference and the +2 charge as by its 22 heavy atoms; a neutral
closed-shell molecule of the same size is far cheaper, and picking one would
lower the maximum speedup substantially.

**RESOLVED by T4.** `C3_representatives.csv` is authoritative and
`run_C3_timings.py` prefers it, so Table 9 as published is reproducible. A
ready-to-paste caption naming the dication is in `paste/T4_table9_caption.md`,
together with the full identity of all five representatives (ID, SMILES,
heavy-atom count, charge, multiplicity, basis, B3LYP time). Each sits exactly
at its class's median heavy-atom count.

The reconstruction remains a reconstruction: if a fresh selection is ever
made, **every row must be re-timed.**

<sub>`C3_representatives.csv`, `run_C3_timings.py::select_representatives`</sub>

---

### ~~B6~~ FIXED. Figure 2's inference path now marks the semiempirical step optional

**Evidence.** Figure 2's "INFERENCE, new molecule" block lists
`PM7 semiempirical properties` as a required input. The structure-only
ablation (spec task B6) shows the
five PQR scalars can be dropped for **+0.0001 eV (p = 0.814)**: inference needs
only a SMILES string.

As drawn, the figure concedes exactly the objection — Reviewer 4's "rCEG cannot
be applied to an arbitrary new molecule from SMILES alone" — that the paper is
now in a position to answer.

**FIXED.** `make_figure2_pipeline_flowchart.py` now draws that box with a
dashed outline labelled **"PM7 semiempirical properties (optional)"**, and adds
a dashed bypass arc from SMILES directly to the descriptor step, annotated
*"structure-only route: no quantum calculation"*. The figure has been
regenerated at 600 dpi with its PDF.

The rest of the figure already made the point — Stage 2 reads "inputs are
molecular descriptors only — no gap of any kind" — so only the inference row
was inconsistent.

**Remaining for the authors:** swap the regenerated file into the manuscript
and add one clause to the caption stating that the semiempirical inputs are
not required at inference.

<sub>`B6_summary.csv`, `make_figure2_pipeline_flowchart.py`, `paste/T5_figure_manifest.md`</sub>

---

### B7. Figure 3's baseline has not converged

**Evidence.** `best_iteration` = 700 = `max_iter`, and the validation curve is
still descending at the cutoff, at −0.005 eV per 100 iterations. `global_hgb`
is therefore shown under-trained, which slightly flatters the ablation.

**It does not change the conclusion.** The gap from `global_hgb` (0.405 eV) to
rCEG (0.338 eV) is 0.066 eV. At the observed rate — which is decelerating —
closing it would take roughly 1,300 further iterations. The baseline is
truncated, not merely unlucky.

**Action.** Either raise `max_iter` and re-run, or state in the caption that
the baseline is shown at a fixed 700-iteration budget. **Do not call the figure
a convergence plot.** Also note that it plots `global_hgb` at 0.458 eV test
MAE, not the headline 0.338 eV — say so, or a reader will take 0.458 for the
paper's result.

<sub>`global_hgb_convergence_summary.csv`, `global_hgb_convergence_history.csv`</sub>

---

### B8. A stale Figure 1 sidecar was shipping in the deposit

**Evidence.** `figure1_reference_descriptor_scatter_final_data.csv`, dated
July, held 743 rows **including the 16 placeholder zero-gap points the revision
removed**, plus an `exact_mass` column for the panel since dropped. A prefix
match in `build_deposit.py` was shipping it beside the corrected figure, so a
reviewer opening it would have found precisely the defect the response letter
says was fixed.

**Fixed.** Renamed to `STALE_PRE_CLEANING_figure1_final_data_DO_NOT_USE.csv`,
excluded from the deposit, and guarded by name in `STALE_DO_NOT_SHIP`.
`figure1_data.csv` (2,137 rows, zero zero-gap points) is the current sidecar.

⚠️ Those 2,137 rows are 1,228 + 909 with the 14 dual-labelled molecules in both
series. This is **not** the 2,137 error of §A1 — it is a two-series plot, not a
deduplicated pool. Do not "correct" it.

---

### B9. Two reference-pool partitions exist, and they belong to different tables

**Both triples are real. They come from different code paths.**

| Partition | Produced by | Used for |
|---|---|---|
| **1,274 / 425 / 424** | `grouped_three_way_split`, the repeated-split protocol | **Every reported accuracy table** (6, 7, 8) and the headline |
| 1,273 / 425 / 425 | plain 60/20/20 `train_test_split` in the legacy single-run path | `add_confidence()` — i.e. **Table 2's confidence bands** |

Both sum to 2,123. Earlier revision documents quoted the legacy triple while
describing the repeated-split results, which is the actual error; the numbers
themselves were not invented. Figure 2 shows 1,274 / 425 / 424 and is right for
the tables it accompanies.

**Action.** Use **1,274 / 425 / 424** wherever the reported tables are being
described — this is now corrected throughout the revision documents. Where
Table 2 is described, state that its bands derive from the legacy single-run
split, or a reader reconciling the two tables will find partitions that do not
match. See `paste/C2_table2.md`.

---

## E. Infrastructure and reproducibility notes

### E1. `PSI_SCRATCH` is read at import time — this caused most recompute failures

**Evidence.** Psi4 reads `PSI_SCRATCH` when the module is imported. Setting it
per molecule inside an already-running Python process has no effect: the real
scratch went to `/tmp/psi.<pid>.*` while the cleanup routine dutifully removed
the empty directories it had been told about. 14 GB was reclaimed by hand.

This is the direct cause of the `disk_or_scratch_exhausted` failures that
account for 30 of the 38 recompute failures.

**Action.** Export `PSI_SCRATCH` **before** the interpreter starts. Documented
in SI §S16; keep it there — anyone reproducing this will otherwise hit the
same wall.

---

### E2. NumPy 1.26.4 silently corrupts results on this platform

**Evidence.** Under NumPy 1.26.4 with Python 3.14, `np.nanvar` returned 0.0
for arrays above roughly 400,000 elements — silently, with no error. This
corrupts variance-based feature filtering. Results were regenerated under
NumPy 2.5.3 and verified to reproduce bit-identically.

**Already handled.** `rCEG_ACSOmega_reproducibility_package/environment.yml`
pins `numpy>=2.3,<3` with the reason recorded in a comment block, and the
psi4 environment is kept separate because psi4 pins an older NumPy. Also
stated in SI §S16. **Do not relax this pin.**

---

### E3. Stale files that will produce wrong numbers if read

| File | Problem |
|---|---|
| `runs/rceg_final_consolidated/headline_metrics.csv` | Predates the final ten-repeat run. Reports **0.318 eV**, which is *not* the headline number (0.338 eV). Retained for audit history only. |
| `runs/rceg_final_consolidated/PLACEHOLDER_VALUES.md` | Same vintage; several values superseded. |
| `runs/revision2/B5_scaffold_split.csv` | **Contained the `random` row, not the scaffold row** — mislabelled during porting. Deleted during final assembly and replaced by `B5_split_protocols.csv`, which carries all three protocols from `split_protocol_comparison_ALL.csv`. |

**Action.** Read headline numbers only from `frozen_architecture.csv` and
`B3_ablation.csv`. The `*_LEAK094` directories are provenance only and must
never be cited.

---

### E4. The working filenames use an obsolete table numbering

`B2_table3.csv` holds per-domain performance, which is manuscript **Table 7**.
`B4_table5.csv` holds calibration by domain, which is manuscript **Table 4**.

The filenames were left unchanged so they still match the task IDs in the
revision spec. The mapping in `SUMMARY.md` §11 is authoritative; the filenames
are not.

---

## Summary of required author actions

| # | Action | Blocking? |
|---|---|---|
| A1 | Correct 2,137 → 2,123 everywhere | Yes |
| A2 | Worst class is a **tie** (0.529 vs 0.537, p = 0.84); name both. Check whether the text was already right | Yes |
| A3 | Rewrite the Introduction to drop the meta-learner | Yes |
| A4 | Remove or attribute the open-shell "0.2 eV" | Yes |
| A5 | Calibration candidates: ten/six → **eight/four** | Yes |
| B1 | Quote accuracy as 0.3 eV; state the geometry condition | Yes |
| B3 | Quote the speedup as a range | Yes |
| B4 | Narrow the interpretability claim to the breakpoints | Yes |
| D1 | Adopt **PM6** per the PQR FAQ; note the `pm7` field name | Yes — verify the FAQ, then mechanical |
| ~~D2~~ | ~~Decide the Table 9 inference figure~~ — **resolved (T3)**; keep single-molecule | No |
| ~~D3~~ | ~~Table 9 caption~~ — **resolved (T4)**; paste `T4_table9_caption.md` | No |
| ~~B6~~ | ~~Mark PM7 optional in Figure 2~~ — **figure fixed and regenerated**; swap the file in and add a caption clause | File swap only |
| B7 | Figure 3 caption: fixed 700-iteration budget, and it plots 0.458 not 0.338 | Yes |
| ~~B8~~ | ~~Stale Figure 1 sidecar in the deposit~~ — **fixed** | No |
| B9 | Use 1,274/425/424 for the reported tables; note Table 2 uses the legacy 1,273/425/425 split | Yes — one sentence in the Table 2 caption |
| B2, B5, C1–C4 | Stated correctly in the text and SI | Editorial |
| E1–E4 | Already handled; keep the SI notes | No |
