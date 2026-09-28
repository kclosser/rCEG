# rCEG revision 2 — execution specification

**Manuscript:** *A split-architecture machine learning model for scalable
HOMO–LUMO gap prediction: rCEG* — Wang & Closser, *ACS Omega* (revision 2).

**Status: analysis complete.** Every measurement in the revision spec has run.
What remains is ten edits to the manuscript text, three of which need an
author decision. Nothing is blocked on further computation.

This document is the record of what was done, what was found, what was fixed,
and how to regenerate all of it. It is written to be self-contained: someone
picking this up cold should not need the conversation that produced it.

---

## 0. Ground rules that governed the work

These were set by the author and held throughout. They are recorded because
they explain several decisions that would otherwise look odd.

| Rule | How it was honoured |
|---|---|
| **Never overwrite an existing run.** | All new output went to `runs/revision2/`. `runs/pqr_full_domain_moe_piecewise_split_LEAK094/` and `runs/pqr_qmugs_strict_external_LEAK094/` were treated as read-only. |
| **Never invent a number.** | Where an input was missing or disagreed with the spec, the task stopped and the finding was written down rather than estimated. Three such cases became DISC-D1, DISC-D2 and DISC-D3. |
| **Never select on the test set.** | The architecture was frozen on validation MAE. `frozen_architecture.csv` records `test_never_used_for_selection = True`. |
| **Determinism.** | Every command ran with `PYTHONHASHSEED=0`. Byte-identical reruns were verified. |
| **Report, don't reconcile.** | Where analysis disagreed with the manuscript, the manuscript was flagged for change — the analysis was never adjusted to agree with it. |
| **Do not run the leaking script.** | `pqr_full_domain_moe_BEFORE_NO_LEAK_PATCH.py` was never executed. |
| **Do not touch the abandoned NN track.** | The `HLGModelTrain*` / `schnet3d` / `.pt` checkpoint family was neither run nor modified, and is excluded from the deposit by prefix rule. |

One rule was added mid-revision at the author's instruction, after an earlier
over-escalation: **when flagging something alarming, give the number that
bounds it in the same message.** `DISCREPANCIES.md` is written that way
throughout.

---

## 1. What exists now

### 1.1 Documents — read in this order

| File | Lines | What it is |
|---|---|---|
| `runs/revision2/DISCREPANCIES.md` | 457 | Everything that contradicts the submitted draft, is unresolved, or would mislead a reader. **Read first.** |
| `runs/revision2/SUMMARY.md` | 278 | Every number, mapped to the manuscript location it belongs in, plus the fragment and file indexes. |
| `runs/revision2/SUPPORTING_INFORMATION.md` | 464 | The SI itself, 16 sections. |

`SUMMARY.md` and `SUPPORTING_INFORMATION.md` are **generated** by
`build_SUMMARY.py` and `build_SI.py` from the result CSVs — every table is
annotated with the file and columns it was read from, so no value can drift
from the run that produced it. Edit the builder, never the output.
`DISCREPANCIES.md` is maintained by hand because it is argument, not data.

### 1.2 Result tables

29 CSVs in `runs/revision2/`, one per spec task, plus 11 paste-ready
Markdown fragments in `runs/revision2/paste/`. Indexed in `SUMMARY.md` §10–11.

### 1.3 Scripts

| Script | Lines | Role |
|---|---|---|
| `build_SI.py` | 811 | Generates the SI from result CSVs |
| `build_SUMMARY.py` | 598 | Generates the number-to-manuscript map |
| `build_deposit.py` | 393 | Rebuilds the reproducibility package |
| `run_tasks_G_and_T.py` | 558 | Geometry sensitivity, conformers, DFT and semiempirical timing |
| `run_C3_timings.py` | 235 | Per-domain cost, Table 9 |
| `run_B6_structure_only.py` | 160 | Structure-only ablation |
| `rceg_inputs.py` | 114 | Shared corpus/reference loader (new; see §4.2) |

### 1.4 Deposit

`rCEG_ACSOmega_reproducibility_package/` — 198 files, 22 MB.

```
code/            30 scripts that produced reported results (29 .py + 1 .sh)
code/archive/    43 superseded variants, off the run path
data/            49 tables + LARGE_FILES_CHECKSUMS.csv
data/revision2/  39 revision tables and paste fragments
docs/            PROVENANCE, REVISION_LOG + the three documents above
figures/         24 files, 600 dpi, with _data.csv sidecars
psi4_inputs/     4 manifests
run_all.sh       12 numbered steps, executable top to bottom
```

---

## 2. What was measured

All on the frozen run `runs/rceg_final_consolidated/`, ten repeated splits,
seeds 42 + 1000·i.

### 2.1 Headline

| Quantity | Value |
|---|---|
| Test MAE | **0.338 ± 0.026 eV** |
| Test R² | 0.930 |
| Validation MAE (the selection criterion) | 0.344 ± 0.016 eV |
| Selected configuration | `piecewise_split` |

The meta-learner was **dropped**: `full_rceg` is +0.0053 eV *worse* than
`piecewise_split` on validation, better in only 3 of 10 repeats, paired
p = 0.12. `domain_expert` is 0.0013 eV better but not significantly;
`piecewise_split` was kept on interpretability grounds, the learned breakpoint
being the paper's named contribution.

### 2.2 Task-by-task

| Task | Result |
|---|---|
| **A1** recompute reconciliation | 947 targeted / 909 converged / 38 failed; 30 of the failures were disk exhaustion, not chemistry |
| **A2** cleaning waterfall | Closes to exactly 21,923 removed, zero unexplained; 10,089 are valence rejections |
| **A3** PM6 vs PM7 | Field is literally named `pm7`; no other method string in 5,000 records. **Evidence is weak — unresolved, see DISC-D1** |
| **A4** descriptor provenance | 375/375 LASSO names recovered (362 unique, 13 ambiguous). Producer is `DescriptorPull3.py`, not `MordredPull2.py`. **176 of 375 are Morgan bits** |
| **A5** leakage audit | Max abs correlation anywhere = 0.534; 5 features exceed 0.5; none approaches the 0.94 excision threshold |
| **A6** open-shell agreement | r = 0.074 over 99 open-shell molecules vs r = 0.694 for 660 neutral closed-shell |
| **A7** zero-gap contamination | **0** of 84,143 in the strict corpus. The earlier Figure 1 band came from plotting the pre-cleaning table |
| **A8** reference pool | 1,228 + 909 − 14 = **2,123**, not the manuscript's 2,137. Cross-protocol floor 0.295 eV confirmed |
| **A9** SCF settings | `df` / 1e-6 / 1e-6 / maxiter 150 / RKS for closed shell, UKS for open |
| **B2** per-domain | Best `qm9_like_small_organic` 0.253 eV; worst `charged_or_radical` 0.733 eV |
| **B3** ablation | 11 configurations; both `domain_permuted` controls degrade to ~0.44 eV |
| **B4** calibration by domain | Offset spread 1.876 eV (calibration set) / 2.310 eV (held-out reference) |
| **B5** split protocols | random 0.335, scaffold 0.360 (+7.3%), cluster 0.337 (+0.5%) |
| **B6** structure-only | Dropping all five PQR scalars costs **+0.0001 eV (p = 0.814)** |
| **C1** geometry sensitivity | mean **0.440 eV**, median 0.279, n = 37 of 50 converged |
| **C2** conformational floor | median within-molecule SD **0.076 eV**, n = 9,697 |
| **C3** cost, Table 9 | speedups **972× – 16,979×** across domains |
| **T1** DFT timing | 116.8 s local (n=20) vs 119 s cluster (n=332) — agree within 2% |
| **D1** external validation | QMugs frozen 4.19 eV (R² −14.5); 5-fold cross-fit offset-aligned **0.521 eV** (R² 0.600) |

---

## 3. Findings requiring manuscript action

Full argument and evidence in `DISCREPANCIES.md`. Summarised here by whether
they need a decision or just an edit.

### 3.1 Wrong numbers — mechanical edits

| # | Finding |
|---|---|
| A1 | Reference pool is **2,123**, not 2,137 |
| A2 | Text and per-domain table disagree on the worst domain; the table is right (`charged_or_radical`) |
| A3 | The Introduction still describes the dropped meta-learner |
| A4 | The open-shell "0.2 eV" matches no computed quantity |

### 3.2 Restatements — how results must be phrased

| # | Finding |
|---|---|
| B1 | Geometry shift (0.440 eV) **exceeds the model's own error** (0.338 eV) → quote accuracy to one decimal, 0.3 eV, and state the UFF-geometry condition |
| B2 | Geometry sample is n=37 not 45, biased in **both** directions (optimizer failures raise the mean; `gau_loose` lowers it) |
| B3 | Speedup spans 972×–16,979× → quote the range, not a single figure |
| B4 | 176 of 375 descriptors are Morgan bits → narrow the interpretability claim |
| B5 | Two calibration spreads (1.876 / 2.310 eV) on different molecule sets → name the set, never average |

### 3.3 Findings that strengthen the paper

| # | Finding |
|---|---|
| C1 | Inference is effectively **structure-only** — answers Reviewer 4 directly |
| C2 | `charged_or_radical` is weak because the *input data* is bad there (r = 0.074), not because the model fails |
| C3 | Two reference protocols disagree by 0.295 eV — the model operates near its own reference noise floor |
| C4 | The cross-hardware timing objection was valid, but re-measurement changed nothing (116.8 vs 119 s) |

### 3.4 Open — need an author decision

| # | Decision | Why it can't be settled here |
|---|---|---|
| **DISC-D1** | PM6 vs PM7 | Two referees said PM6; the data field says `pm7`. A field name is weaker evidence than the source publication. Requires checking the PQR paper. **Do not find-and-replace on the field name alone.** |
| **DISC-D2** | Which inference figure Table 9 reports | Three exist — 0.40 ms (batched), 9.50 ms (Table 9), 38.3 ms (single-molecule). Batching moves it ~95×, and the speedup ratios follow. `run_C3_inference.py` was never preserved. |
| **DISC-D3** | Table 9 caption | The `charged_or_radical` representative is an **open-shell dication**; its 256 s B3LYP cost drives the headline 16,979×. Caption should say so, or the figure reads as typical. |

---

## 4. Defects found and fixed

These were not in the revision spec. They were found while executing it and
would have caused wrong results or an unusable deposit.

### 4.1 Analysis correctness

| Defect | Consequence | Fix |
|---|---|---|
| **NumPy 1.26.4 / Python 3.14** returned `np.nanvar` = 0.0 for arrays above ~400k elements, silently | `VarianceThreshold` calls it internally, so every descriptor pipeline was corrupted without raising | Upgraded to 2.5.3; results regenerated and verified bit-identical. `environment.yml` pins `numpy>=2.3,<3` |
| **Non-determinism** traced to `RandomForest` with `n_jobs=-1` accumulating tree predictions in non-fixed order | A 1e-17 difference in calibration weights was amplified by near-tied ExtraTrees splits into **0.096 eV** on individual predictions | Three guards: quantize candidate MAEs to 12 dp, round pseudo-labels to 9 dp, force `n_jobs=1` at predict. Seeds use `zlib.crc32`, not salted `hash()` |
| **`bisect.py`** in the repo shadowed the stdlib module | Broke `import pandas` | Renamed to `pipeline_checkpoints.py` |
| **`B5_scaffold_split.csv`** contained the `random` row, not the scaffold row | Would have published random-split numbers as scaffold numbers | Replaced by `B5_split_protocols.csv` carrying all three protocols from `split_protocol_comparison_ALL.csv` |

### 4.2 Deposit could not be run by a third party

Found during final assembly. Three scripts loaded a pickle from a hardcoded
session scratch path:

```python
SCRATCH = Path("/private/tmp/claude-501/.../scratchpad")
cache = pickle.load(open(SCRATCH / "cache.pkl", "rb"))
```

`emit_hgb_convergence_history.py` and `benchmark_computational_cost.py` are
both in `run_all.sh`, so **steps 8 and 10 — Figure 3's convergence trace and
the Table 9 cost benchmark — would have died with `FileNotFoundError`** for
anyone outside the originating machine. `run_B6_structure_only.py` had the
same defect.

Fixed by adding `rceg_inputs.py`, a shared loader that rebuilds the corpus and
merged reference pool from source using the model's own `load_pqr` and
`load_reference`. `--cache` keeps the speed-up optional: read if present,
written if absent. Behaviour is unchanged when a cache exists.

### 4.3 Other deposit defects

| Defect | Fix |
|---|---|
| `build_deposit.py --clean` destroyed `README.md`, `run_all.sh`, `environment.yml` (they had been written by hand *after* the build) | All three are now **generated inside the build** from `docs/DEPOSIT_*` templates, so `--clean` cannot orphan them |
| The deposit shipped `headline_metrics.csv`, which predates the final run and reports **0.318 eV** — contradicting the manuscript's own abstract | Removed from the copy list; `frozen_architecture.csv` ships instead. A build-time guard (`STALE_DO_NOT_SHIP`) now fails if it reappears |
| `run_all.sh` omitted B6, C3 and four `run_tasks_G_and_T` stages | Now 12 steps; every referenced script verified present |
| Table 9's representative molecules lived only in `/tmp/c3_reps.csv` with no generating script | Preserved as `C3_representatives.csv`; `select_representatives()` reconstructs the criterion — **but does not reproduce the original picks**, which is why the preserved file is authoritative (DISC-D3) |
| `PSI_SCRATCH` is read by psi4 **at import**, so per-molecule setting never relocated scratch | Documented in SI §S16 and the deposit README: export before launching Python. This caused 30 of the 38 recompute failures |

---

## 5. Verification performed

| Check | Result |
|---|---|
| Byte-identical reruns after the determinism fixes | Pass |
| E4 and A2 reproduce under NumPy 2.5.3 | Pass |
| Cleaning waterfall closes to 21,923, zero unexplained | Pass |
| QMugs excision asserted in code before any fitting | Pass — script fails loudly otherwise |
| Every file path referenced in the three documents exists | Pass, except two referenced *because* they are missing (DISC-D2) |
| Key numbers consistent across all three documents | Pass — 0.338, 2,123, 0.440, 0.279, 972, 176, 0.074, 9,697 |
| Stale 0.318 appears only inside stale-file warnings | Pass |
| Deposit exclusion rules (no `.pt`, `__pycache__`, `.DS_Store`, NN track, stale files) | Pass |
| All four revision scripts import cleanly *from inside the deposit* | Pass |
| `run_all.sh` shell syntax and script existence | Pass |

Two known non-reproductions, both deliberate and documented rather than
papered over:

1. `select_representatives()` does not reproduce the original Table 9
   molecules (DISC-D3).
2. `run_C3_inference.py` does not exist, so the 9.496 ms inference figure
   cannot currently be regenerated (DISC-D2).

---

## 6. How to regenerate everything

Full detail in `RUN_ORDER.md` (13 steps). Condensed:

```bash
export PYTHONHASHSEED=0

# 1-7  descriptors, cleaning, reference labels, audits, model, splits, ablations
# 8-11 cost, geometry (psi4), per-domain table, figures

# 12. the revision documents
python3 build_SI.py        # -> runs/revision2/SUPPORTING_INFORMATION.md
python3 build_SUMMARY.py   # -> runs/revision2/SUMMARY.md

# 13. the deposit
python3 build_deposit.py --clean
```

Two environments are required: the default `python3` (numpy ≥ 2.3) and
`rceg_qc` for psi4/xtb. Export `PSI_SCRATCH` to a large volume **before**
launching any psi4 work.

Runtime is dominated by geometry sensitivity: ~30 h across three parallel
workers, against ~2.5 h for the consolidated model run. Start geometry first
if running everything.

---

## 7. Naming hazards

Three collisions that will cause mistakes if not known:

1. **Revision-2 filenames use an obsolete table numbering.**
   `B2_table3.csv` is manuscript **Table 7**. `B4_table5.csv` is manuscript
   **Table 4**. The names were kept so they still match the spec's task IDs.
   The mapping in `SUMMARY.md` §11 is authoritative.

2. **Ledger task IDs D1–D4 are unrelated to DISCREPANCIES sections D1–D3.**
   Ledger D1 is external validation; DISCREPANCIES D1 is PM6-vs-PM7. Prefix
   the latter `DISC-` when referring to it. Recorded in the ledger's
   `id_collision_warning` field.

3. **Three inference timings exist** — 0.40 ms (batched), 9.50 ms (Table 9),
   38.3 ms (single-molecule latency). They differ by ~95×. Always state which.

---

## 8. Current status

| Category | State |
|---|---|
| Measurements | **Complete.** 22 ledger tasks, all done |
| Documents | **Complete.** SI, SUMMARY, DISCREPANCIES written and cross-verified |
| Deposit | **Complete.** 198 files, rebuilt, all checks pass |
| Manuscript edits | **Not started** — 7 mechanical, 3 needing a decision |

The seven mechanical edits (§3.1, §3.2) are changes to the paper text and
cannot be made from the analysis side. The three decisions (§3.4) need the
PQR publication, a choice about batching, and a caption judgement
respectively.
