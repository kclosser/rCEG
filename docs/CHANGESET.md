# rCEG — changeset

Everything changed in the final assembly phase of revision 2, with the reason
for each change and how to verify it.

**Scope.** This covers work from the point where all modelling was already
frozen: the three assembly documents, the deposit repair, tasks T1–T6, and
tasks C1–C8. The earlier revision's measurement history and the defects found
during it are recorded separately in `REVISION2_EXECUTION_SPEC.md` §4.

**Not a git repo.** The record is file mtimes and this document. A pre-change
copy of the deposit is in the session scratch directory as
`deposit_backup_prerev2` (143 files).

**One model run was performed**, and only one: `run_C3_inference.py` retrains
the Stage-2 experts to time inference. Nothing else refits anything. Two other
items recomputed values from cached inputs through existing code paths (C1
offsets, C2 thresholds); C2 reproduced its shipped file exactly.

---

## 1. New files

### 1.1 Scripts

| File | Lines | Purpose |
|---|---|---|
| `build_SI.py` | 815 | Generates the Supporting Information from the result CSVs |
| `build_SUMMARY.py` | 601 | Generates the number-to-manuscript map |
| `rceg_inputs.py` | 114 | Shared corpus/reference loader — **fixes three unrunnable scripts**, see §3.1 |
| `run_C3_inference.py` | 248 | Measures inference latency; closes DISC-D2 |

`build_SI.py` and `build_SUMMARY.py` read every value from a named column of a
named file and annotate each table with its source, so the documents cannot
drift from the runs they describe. Edit the builder, never the output.

### 1.2 Documents

| File | Lines | Purpose |
|---|---|---|
| `runs/revision2/SUPPORTING_INFORMATION.md` | 464 | The SI, 16 sections — **generated** |
| `runs/revision2/SUMMARY.md` | 292 | Every number mapped to its manuscript location — **generated** |
| `runs/revision2/DISCREPANCIES.md` | 593 | What contradicts the draft, and what is open — hand-maintained |
| `runs/revision2/REMAINING_TASKS_REPORT.md` | 257 | Report for tasks T1–T6 |
| `runs/revision2/FINAL_GAPS_REPORT.md` | 307 | Report for tasks C1–C8 |
| `docs/REVISION2_EXECUTION_SPEC.md` | 308 | Record of the whole revision |
| `SPEC_FOR_CLAUDE_WEB.md` | 442 | Self-contained manuscript-revision spec |
| `docs/CHANGESET.md` | — | This file |

### 1.3 Result tables

`runs/revision2/`: `B5_split_protocols.csv`, `C3_representatives.csv`,
`C3_inference_timings.csv`, `C3_speedups_both_definitions.csv`,
`C3_inference_hardware.json`, `C1_calibration_offsets_by_class.csv`.

### 1.4 Paste fragments

14 new fragments in `runs/revision2/paste/`: `T1_table3`,
`T2_split_features`, `T3_speedups`, `T4_table9_caption`,
`T5_figure_manifest`, `T6_external_by_class`, `C1_table4`, `C2_table2`,
`C3_hyperparameters`, `C4_table7`, `C5_cleaning_waterfall`, `C6_figure3`,
`C7_figure_manifest`, `C8_label_accounting`.

---

## 2. Modified files

| File | Change |
|---|---|
| `build_deposit.py` | Ships revision-2 data, docs and scripts; drops stale files; adds a build-time guard (§3.2, §3.3) |
| `emit_hgb_convergence_history.py` | Hardcoded scratch path removed; uses `rceg_inputs` (§3.1) |
| `benchmark_computational_cost.py` | Same |
| `run_B6_structure_only.py` | Same |
| `run_C3_timings.py` | Representative selection made explicit and self-contained (§3.4) |
| `docs/DEPOSIT_run_all.sh` | 10 steps → 12, plus step 11a; adds a shared cache variable |
| `docs/DEPOSIT_README.md` | Documents the revision-2 data, the five documents, and the deliberate exclusions |
| `RUN_ORDER.md` | 11 steps → 13; removes a duplicated stage; adds the inference and document-build steps |
| `README.md` | Adds a "Start here for the revision" section and both filename traps |
| `REVISION_BRIEF_FOR_CLAUDE_WEB.md` | Marked **superseded** at the top (§4.4) |
| `runs/revision2/paste/T1_dft_timing.md` | Arithmetic error corrected (§4.3) |
| `runs/revision2/paste/A8_pool_arithmetic.md` | Partition corrected (§4.2) |
| `runs/revision2/REVISION2_LEDGER.json` | Marked complete; ID-collision warning added |

---

## 3. Defects found and fixed

### 3.1 Three scripts could not run outside the originating machine

**Found:** `emit_hgb_convergence_history.py`,
`benchmark_computational_cost.py` and `run_B6_structure_only.py` each loaded a
pickle from a hardcoded session scratch path:

```python
SCRATCH = Path("/private/tmp/claude-501/.../scratchpad")
cache = pickle.load(open(SCRATCH / "cache.pkl", "rb"))
```

The first two are in `run_all.sh`, so **steps 8 and 10 — Figure 3's
convergence trace and the Table 9 cost benchmark — would have failed with
`FileNotFoundError` for anyone reproducing from the deposit.**

**Fixed:** added `rceg_inputs.py`, which rebuilds the corpus and merged
reference pool from source using the model's own `load_pqr` and
`load_reference`. `--cache` restores the speed-up: read if present, written if
absent. Behaviour is unchanged when a cache exists.

**Verify:** `grep -rl claude-501 code/*.py` inside the deposit returns only
`rceg_inputs.py`, where the string appears in a docstring explaining the bug.
All four scripts pass a `--help` import test from inside the deposit.

### 3.2 The deposit shipped a stale headline

`headline_metrics.csv` predates the final ten-repeat run and reports
**0.318 eV**, contradicting the manuscript's own abstract (0.338 eV). It was
in the copy list.

**Fixed:** removed; `frozen_architecture.csv` ships instead. Added
`STALE_DO_NOT_SHIP`, which fails the build if a listed file reappears.

### 3.3 The deposit shipped a stale Figure 1 sidecar

`figure1_reference_descriptor_scatter_final_data.csv`, dated July, held 743
rows **including the 16 placeholder zero-gap points the revision removed**,
plus an `exact_mass` column for a panel since dropped. A `KEEP_FIG` prefix
match was shipping it beside the corrected figure — a reviewer opening it would
have found precisely the defect the response letter says was fixed.

**Fixed:** renamed to `STALE_PRE_CLEANING_figure1_final_data_DO_NOT_USE.csv`,
excluded by name, and the figure copy loop now checks `STALE_DO_NOT_SHIP`.
`figure1_data.csv` (2,137 rows, zero zero-gap points) is the current sidecar.

### 3.4 Table 9's representatives had no generating script

The five representative molecules were chosen ad hoc and left in
`/tmp/c3_reps.csv`. Once that file cleared, Table 9 could not be regenerated.

**Fixed:** preserved as `C3_representatives.csv` (authoritative; the script
prefers it) and added `select_representatives()`, which reconstructs the
documented criterion.

**Stated honestly:** the reconstruction **does not reproduce the original
picks**. It agrees on heavy-atom count in four of five classes but selects
different molecules, and computes a heteroatom median of 19 against the
original 18. Rather than silently adopt a new set, the original is kept and the
reconstruction is documented as one.

### 3.5 `B5_scaffold_split.csv` contained the wrong row

It held the `random` row, not the scaffold row — a porting error that would
have published random-split numbers as scaffold numbers.

**Fixed:** deleted, replaced by `B5_split_protocols.csv` carrying all three
protocols from `split_protocol_comparison_ALL.csv`.

### 3.6 `run_all.sh` omitted steps

B6, C3 and four `run_tasks_G_and_T` stages were missing entirely.

**Fixed:** now 12 steps, every referenced script verified to exist, shell
syntax checked.

---

## 4. Corrections to numbers

Each of these was wrong in a document I produced, or wrong in the manuscript
and now documented.

### 4.1 The calibration spread was mislabelled

**Was:** "1.876 eV over the calibration set, 2.310 eV over held-out."

**Is:** `calibration_by_domain_and_source.csv` is computed on **held-out
molecules only** (`ref_val` + `ref_test`), as its emitting code states. So
1.876 is a held-out number. Recomputed over the same ten repeats, the correct
pairing is **1.637 eV (calibration) / 1.876 eV (held-out)**.

The 2.310 eV figure comes from `B4_domain_independence_test.csv`: n = 425, no
MAE column, **no generating script in the repository**. Superseded.

**Propagated to:** `build_SI.py`, `build_SUMMARY.py`, `DISCREPANCIES.md`.
Authoritative replacement is `paste/C1_table4.md`.

### 4.2 Two reference-pool partitions exist

**Was:** I recorded 1,273 / 425 / 425, then "corrected" it to 1,274 / 425 / 424
as though the first were invented.

**Is:** both are real, from different code paths.

| Partition | Produced by | Governs |
|---|---|---|
| **1,274 / 425 / 424** | `grouped_three_way_split`, repeated-split protocol | Every reported accuracy table, and the headline |
| 1,273 / 425 / 425 | plain 60/20/20 `train_test_split`, legacy single-run path | `add_confidence()` — Table 2's confidence bands |

Both sum to 2,123. The actual error was quoting the legacy triple while
describing repeated-split results. Figure 2 shows 1,274 / 425 / 424 and is
correct for the tables it accompanies.

**Propagated to:** `build_SI.py`, `build_SUMMARY.py`, `DISCREPANCIES.md` §B9
(rewritten), `paste/A8_pool_arithmetic.md`.

### 4.3 The T1 timing fragment did not sum

**Was:** "descriptors 3.1 ms + inference 0.40 ms | 42.2 ms" — which is 3.5, not
42.2. The parenthetical used the *batched* figure while the total used
*single-molecule latency*.

**Is:** "descriptors 3.1 ms + single-molecule inference 38.3 ms". The fragment
now carries a table distinguishing all three inference figures.

### 4.4 The revision brief listed resolved work as blocked

`REVISION_BRIEF_FOR_CLAUDE_WEB.md` still listed `[[GEOM_SHIFT]]`,
`[[GEOM_SD]]`, `[[GEOM_MAX]]`, `[[GEOM_DOMAIN]]` and `[[CONF_SPREAD]]` under
"Still blocked, do not invent values." All five had since been measured. It
was warning readers off an older spec while being stale itself.

**Fixed:** values filled in, and the file marked superseded by
`SPEC_FOR_CLAUDE_WEB.md`.

### 4.5 The geometry median

An intermediate value of 0.317 eV was quoted at 38 molecules. The final figure
over the 37 converged molecules is **0.279 eV**. Mean 0.440 is unchanged.

---

## 5. Findings recorded, not fixed

These are manuscript-side and need the document or an author decision. Full
argument in `DISCREPANCIES.md`.

| # | Finding |
|---|---|
| A1 | Reference pool is 2,123, not 2,137 |
| A2 | Text and table disagree on the worst class; the table is right |
| A3 | The Introduction still describes the dropped meta-learner |
| A4 | The open-shell "0.2 eV" matches no computed quantity |
| B1 | Geometry shift (0.440) exceeds the model's own error (0.338) → quote 0.3 eV |
| B3 | Speedup spans 972×–16,979×; quote the range |
| B4 | 176 of 375 descriptors are Morgan bits → narrow the interpretability claim |
| B6 | Figure 2's inference path contradicts the structure-only result |
| B7 | Figure 3's baseline has not converged; it plots 0.458, not 0.338 |
| C4 | Table 7 is a single split; per-class SDs would need a new model run |
| C8 | 82,020 pseudo-labels is wrong; use 72,267 |
| D1 | PM6 vs PM7 — the PQR FAQ says PM6; verify and adopt |

---

## 6. Deposit

**143 files → 219 files.**

| Added | Removed |
|---|---|
| 5 revision-2 documents in `docs/` | `headline_metrics.csv` (stale, §3.2) |
| 56 tables and fragments in `data/revision2/` | Stale Figure 1 sidecar (§3.3) |
| 6 scripts: the three new ones plus B6, C3 timings, G_and_T | |
| `SPEC_FOR_CLAUDE_WEB.md` | |

Build passes its exclusion rules: no `.pt`, no `__pycache__`, no `.DS_Store`,
no Word lock file, no neural-network-track scripts, no stale files.

---

## 7. Naming hazards introduced or documented

Three collisions that will cause mistakes if not known. All are documented in
the affected files.

1. **Revision-2 filenames use an obsolete table numbering.** `B2_table3.csv`
   is manuscript **Table 7**; `B4_table5.csv` is **Table 4**. Kept so they
   still match the spec's task IDs. `SUMMARY.md` §11 is authoritative.
2. **Ledger task IDs D1–D4 are unrelated to DISCREPANCIES §D1–D3.** Ledger D1
   is external validation; DISC-D1 is PM6-vs-PM7. Prefix `DISC-`. Recorded in
   the ledger's `id_collision_warning`.
3. **DISCREPANCIES section letters are not spec task IDs.** Spec task B6 is the
   structure-only ablation; DISC-B6 is a Figure 2 defect. A note at the top of
   the file says so.

---

## 8. How to regenerate

```bash
export PYTHONHASHSEED=0
python3 run_C3_inference.py --repeats 50 --threads 1   # only if re-timing
python3 build_SI.py
python3 build_SUMMARY.py
python3 build_deposit.py --clean
```

`DISCREPANCIES.md`, `REMAINING_TASKS_REPORT.md`, `FINAL_GAPS_REPORT.md`,
`REVISION2_EXECUTION_SPEC.md`, `SPEC_FOR_CLAUDE_WEB.md` and this file are
hand-maintained; the rest are generated.

---

## 9. Verification run after the changes

| Check | Result |
|---|---|
| Every file path referenced in the documents exists | Pass, except two referenced *because* they are missing |
| Key numbers consistent across all documents | Pass |
| All 23 numbers in `SPEC_FOR_CLAUDE_WEB.md` match source CSVs | Pass |
| Stale 0.318 appears only inside stale-file warnings | Pass |
| No `1,273` outside the correction notices | Pass |
| Deposit exclusion rules | Pass |
| All four repaired scripts import from inside the deposit | Pass |
| `run_all.sh` syntax and script existence | Pass |
| C2 thresholds reproduce the shipped file | Pass, to four decimal places |
