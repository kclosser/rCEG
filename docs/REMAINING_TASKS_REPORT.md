# rCEG revision 2 — remaining-tasks report

Six tasks: three lookups, two confirmations, one re-measurement. **All six
complete.** DISC-D2 and DISC-D3 are closed. One item that was expected to be a
correction turned out not to be, and four new defects surfaced — three in the
figures, one of my own making.

| Task | Outcome | Fragment |
|---|---|---|
| T1 | Verified. The targeted column is confirmed against the manifest, not derived. **The 1.1% is correct — do not change it.** | `paste/T1_table3.md` |
| T2 | **No split feature is a Morgan bit.** All five resolve uniquely. | `paste/T2_split_features.md` |
| T3 | **9.611 ms reproduces the tabulated 9.496 ms.** DISC-D2 closed. | `paste/T3_speedups.md` |
| T4 | Representatives pinned and fully described; caption written. DISC-D3 closed. | `paste/T4_table9_caption.md` |
| T5 | Figures confirmed, **three defects found**. | `paste/T5_figure_manifest.md` |
| T6 | Near-uniformity claim **supported**; five fold offsets recovered exactly. | `paste/T6_external_by_class.md` |

---

## T1 — Table 3 targeted column

**Verified, and stronger than expected.** The targeted counts are not merely
derived from converged-plus-failures: `recompute_manifest.csv` has exactly 947
rows with per-class counts identical to the table. Sums close at 947 / 909 /
38, and the failure split matches the reported 21 / 16 / 1 / 0 / 0 exactly.

| Class | Population | Targeted | Converged | Failed | Sampled |
|---|---|---|---|---|---|
| qm9-like small organic | 4,949 | 100 | 100 | 0 | 2.0% |
| near-qm9 larger organic | 25,498 | 100 | 100 | 0 | 0.4% |
| large neutral organic | 22,193 | 248 | 227 | 21 | 1.1% |
| heteroatom-rich non-qm9 | 24,530 | 249 | 233 | 16 | 1.0% |
| charged or radical | 6,973 | 250 | 249 | 1 | 3.6% |
| **total** | **84,143** | **947** | **909** | **38** | **1.1%** |

### The expected correction is not a correction

The spec anticipated changing 1.1% to 1.0% for large neutral organic, on the
grounds that 227/22,193 = 1.02%. That uses the **converged** count. The
manuscript's 1.1% uses the **targeted** count: 248/22,193 = 1.12%. Both are
right; they answer different questions.

**Recommendation: keep 1.1% and label the column "fraction of class
targeted".** That is the sampling-design quantity, independent of how many
later hit a disk limit. The defect is an ambiguous label, not a wrong digit.

---

## T2 — split descriptors

**None of the five learned breakpoints is a Morgan fingerprint bit**, and all
five resolved uniquely — none falls among the 13 ambiguous names.

| Class | Feature | Descriptor | Cut |
|---|---|---|---|
| qm9-like small organic | `x15` | HallKierAlpha | −0.400 |
| near-qm9 larger organic | `x13` | TPSA | 12.530 |
| large neutral organic | `x29` | VSA_EState6 | 0.000 |
| heteroatom-rich non-qm9 | `x260` | Ipc | 1,186.695 |
| charged or radical | `x47` | BCUT2D_MRLOW | −0.451 |

So the narrow interpretability claim survives intact even though 176 of 375
selected descriptors are fingerprint bits. This is the difference between a
chemical result and a plot, and it lands on the chemical side.

---

## T3 — inference latency (closes DISC-D2)

`run_C3_inference.py` written, committed, added to `run_all.sh` as step 11a,
and it runs from the deposit with no hardcoded paths.

Hardware: Apple M4 Pro, 12 logical cores, 25.8 GB, single-threaded.

**Median single-molecule latency 9.611 ms against the tabulated 9.496 ms — a
1.2% difference.** Table 9's inference column is the single-molecule latency,
and it is now regenerable.

| Class | Single-molecule | Batched, per molecule |
|---|---|---|
| qm9-like small organic | 9.57 ms | 0.073 ms |
| near-qm9 larger organic | 9.61 ms | 0.114 ms |
| large neutral organic | 9.72 ms | 0.133 ms |
| heteroatom-rich non-qm9 | 9.58 ms | 0.131 ms |
| charged or radical | 9.79 ms | 0.102 ms |

**The single repeated value in Table 9 is justified.** Latency spans only
9.57–9.79 ms from 8 to 27 heavy atoms, because cost is set by the fitted forest
rather than the molecule. Say so in the caption rather than leaving five
identical numbers unexplained.

**Speedups under both definitions:**

- single-molecule **965× – 16,655×** (reproduces the published 972×–16,979×)
- batched **6,008× – 45,020×**

**Recommendation: keep single-molecule.** It is what the published range
already uses and it is the conservative choice.

**Still unexplained:** `timing_benchmark.csv`'s 38.3 ms
`inference_single_molecule_latency` is 4× what T3 measures under the same
definition on the same hardware. Different code path, not reproduced by the
committed script. Do not quote it.

### A note on why the model had to be retrained

There was no serialized production model to load. The frozen run was not
invoked with `--save-models`, and that flag would not have helped: it persists
`sorted(domain_models.keys())` — the expert *names*, not the fitted
estimators. The script therefore rebuilds the Stage-2 experts through the
model module's own constructors on the same split and pseudo-labels, so the
timed object has production structure.

---

## T4 — Table 9 representatives (closes DISC-D3)

All five described in full, each sitting **exactly at its class's median
heavy-atom count**.

| Class | ID | Heavy | Charge | Mult. | Basis | B3LYP |
|---|---|---|---|---|---|---|
| qm9-like small organic | `pqr_0871` | 8 | 0 | 1 | 6-31G(d,p) | 10.9 s |
| near-qm9 larger organic | `pqr_0845` | 14 | 0 | 1 | 6-31G(d,p) | 40.0 s |
| large neutral organic | `pqr_0502` | 27 | 0 | 1 | 6-31G(d,p) | 129.8 s |
| heteroatom-rich non-qm9 | `pqr_0399` | 18 | 0 | 1 | def2-SVP | 42.4 s |
| charged or radical | `pqr_0166` | 22 | **+2** | **3** | def2-SVP | 256.2 s |

`C3_representatives.csv` is accepted as authoritative rather than forcing
`select_representatives()` to reproduce it, because the original tie-breaking
is unknown and inventing one would be guessing. A caption naming the dication
is in `paste/T4_table9_caption.md`.

---

## T5 — figures

All nine manuscript figures are **600 dpi or vector**. Figure 1 confirmed clean
on every check: strict corpus, panel counts 1,228 and 909, no point at 0 eV,
exact-mass panel dropped for collinearity (r = 0.9939).

Three defects found.

### A stale Figure 1 sidecar was shipping in the deposit — fixed

`figure1_reference_descriptor_scatter_final_data.csv`, dated July, held 743
rows **including the 16 placeholder zero-gap points the revision removed**,
plus the dropped `exact_mass` column. A prefix match was shipping it beside the
corrected figure, so a reviewer opening it would have found precisely the
defect the response letter says was fixed. Renamed, excluded, and guarded by
name. `figure1_data.csv` (2,137 rows, zero zero-gap points) is current.

### Figure 2 contradicts the structure-only result — needs an edit

Its "INFERENCE, new molecule" path lists `PM7 semiempirical properties` as
required. The structure-only ablation shows the PQR scalars can be dropped for
+0.0001 eV (p = 0.814). **As drawn, the figure concedes the objection the
paper can now answer.** Mark the box optional or remove it.

Otherwise Figure 2 checks out: separate Stage 1 / Stage 2 blocks, QMugs
excision shown, and — contrary to expectation — **no stale 0.347 anywhere**.
It hardcodes no MAE at all; its terminal box reads "Predicted gap".

### Figure 3's baseline has not converged — needs a caption change

`best_iteration` = 700 = `max_iter`, with validation MAE still falling at
−0.005 eV per 100 iterations. `global_hgb` is shown under-trained.

**This does not change the conclusion.** The gap to rCEG is 0.066 eV; at the
observed, decelerating rate that would need ~1,300 more iterations. But the
figure should not be called a convergence plot. Its legend, caption and sidecar
do all read from one variable, so the 0.403-vs-0.402 drift cannot recur.

Also: Figure 3 plots `global_hgb` at **0.458 eV**, not the headline 0.338 eV.
Say so, or a reader will take 0.458 for the paper's result.

---

## T6 — external validation by class

**The near-uniformity claim is supported.**

| Class | n | Raw MAE | Offset-aligned MAE | R² |
|---|---|---|---|---|
| qm9-like small organic | 134 | 4.13 | **0.436** | 0.722 |
| near-qm9 larger organic | 1,939 | 4.09 | **0.520** | 0.659 |
| large neutral organic | 4,016 | 4.14 | **0.504** | 0.557 |
| heteroatom-rich non-qm9 | 3,664 | 4.30 | **0.543** | 0.479 |
| charged or radical | **0** | — | — | — |

Aligned MAE spans 0.436–0.543 eV, a spread of 0.107 eV on a mean of 0.501 —
about 21% relative, with three of four classes within 0.04 eV of each other.

**QMugs contains no charged-or-radical molecules**, so the model's weakest
class is untested externally. State this; it bounds the transfer claim.

### The five cross-fit fold offsets

4.1755, 4.1786, 4.1820, 4.1857, 4.1863 eV — median **4.1820**, spread
**0.0108 eV** over folds of 1950/1951/1951/1951/1950.

Stable to about ±0.005 eV: three orders of magnitude below the offset itself
and an order below the residual MAE. **The alignment estimates one genuine
protocol constant rather than overfitting per fold** — the strongest single
piece of transfer evidence in the paper.

*Provenance.* Fold assignments are not persisted. These five values were
recovered exactly from `qmugs_external_predictions.csv` as the aligned-minus-raw
prediction difference, which takes exactly five distinct values partitioning
the 9,753 molecules. Nothing here is estimated.

---

## A correction of my own

Earlier revision documents — including `DISCREPANCIES.md`, `SUMMARY.md` and
the A8 fragment — stated the reference-pool split as **1,273 / 425 / 425**. The
actual split is **1,274 / 425 / 424**, identical in all ten repeats. Both
triples sum to 2,123, which is why it survived review. Figure 2 had it right
all along; I was checking the figure against a wrong number.

All revision documents are corrected. The manuscript should be checked for the
same slip. Filed as DISC-B9.

---

## What could not be reconciled

| Item | Status |
|---|---|
| `timing_benchmark.csv` 38.3 ms inference | 4× the T3 measurement under the same definition. Different code path, not reproduced. **Do not quote.** |
| `select_representatives()` vs the original picks | Reconstruction agrees on heavy-atom count in 4 of 5 classes but selects different molecules; original tie-break unknown. Preserved file kept as authoritative rather than guessing. |
| Fold assignments for the QMugs cross-fit | Not persisted. Offsets recovered exactly by arithmetic, but the assignments themselves are gone. |

---

## Remaining author actions

**DISC-D1 is now near-mechanical.** The PQR FAQ at `pqr.pitt.edu` states PM6 as
implemented in MOPAC — project documentation, which outranks the `pm7` field
label and agrees with both referees. Verify the FAQ, cite it, adopt **PM6**
throughout, and add one sentence noting the field name so a reader checking the
raw download is not confused.

Everything else is manuscript text:

| # | Action |
|---|---|
| A1 | 2,137 → 2,123 |
| A2 | Worst-domain sentence → `charged_or_radical` |
| A3 | Drop the meta-learner from the Introduction |
| A4 | Remove or attribute the open-shell "0.2 eV" |
| B1 | Quote accuracy as 0.3 eV; state the geometry condition |
| B3 | Quote the speedup as a range |
| B4 | Narrow the interpretability claim to the breakpoints |
| B6 | Mark PM7 optional in Figure 2's inference path |
| B7 | Figure 3 caption: fixed 700-iteration budget; it plots 0.458 |
| B9 | Check the manuscript for 1,273 / 425 / 425 |
