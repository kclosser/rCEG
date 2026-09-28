# rCEG ACS Omega revision — task log

Single source of truth for the revision. One line per completed task, per spec §0.4.

**Rule in force (spec §8):** where an analysis contradicts the manuscript, the manuscript
gets corrected — never the analysis.

## Environment constraints found at setup (2026-09-10)

| Constraint | Impact | Disposition |
|---|---|---|
| `psi4` not installed locally | A5 (only new-DFT task) cannot execute here | Driver written + dry-run validated, staged for SLURM `csn100` |
| No MOPAC / OpenBabel | D1 semiempirical row unmeasurable locally | Row retained and flagged, never omitted (spec §5 is explicit) |
| `Closser/` is not a git repo | Spec §0.3 asks for a commit hash | Substitute: SHA256 manifest of all live scripts |
| Apple M4 Pro, 12 cores, 24 GB RAM, 48 GB free | C2/C3 are long (10 repeats × 3 split modes) | Run in background; loop self-paces |

## Completed tasks

| # | Task | Result | Evidence |
|---|---|---|---|
| — | Setup | Ledger + log created; 20 tasks enumerated | `TASK_LEDGER.json` |

## Escalations (spec §8 — report immediately, do not bury)

_None yet._

### Iteration 1 — 2026-09-10

| # | Task | Result | Evidence |
|---|---|---|---|
| P0 | Env manifest | Written. Git hash unavailable (not a repo) → substituted SHA256 of all 8 live scripts. M4 Pro / 12 cores / 24 GB recorded for Table 11. | `environment_manifest.txt` (123 lines) |
| E3 | Deterministic seeding | `abs(hash(...))` → `stable_seed()` using `zlib.crc32`. AST-verified zero builtin `hash()` calls remain. Demonstrated bug was real: builtin gave 20/7151/7380 under PYTHONHASHSEED 0/1/12345; crc32 gave 5916 for all. Full two-run acceptance deferred to C_RUN. | `pqr_full_domain_moe_qmugs_external.py:47-56,1243-1244` |
| E4 | LASSO names | **375/375 resolved** (362 unique, 13 ambiguous, 0 unresolved). No placeholders remain → acceptance met. | `lasso_selected_feature_names.json`, `piecewise_feature_name_map.csv` |
| A2 | Corpus composition | Domain counts sum to 84,143 ✓. QM9: 133,798 parsed, 1,228 overlap ✓ (both as spec predicted). Per-domain mismatch **explained, not an error** — see finding A2-F1. | `corpus_composition.csv` |
| A3 | Recompute audit | Retry hypothesis **CONFIRMED** (50 of 84 bad records later converged). Manuscript counts contradicted — see finding A3-F1. | `recompute_audit.csv`, `recompute_failure_modes.csv` |

## Findings

**E4-F1 — 176 of the 375 LASSO features (47%) are Morgan fingerprint bits.**
The true producer of the descriptor block is `DescriptorPull3.py`, not `MordredPull2.py`
as the spec states; both `MordredPull*.py` scripts print "Old 375 LASSO features" as a
prior baseline they were trying to beat and emit an unadopted 500-feature set. The real
pool is RDKit `Descriptors.descList` + `MorganFP_0..199`. Spec §C3 asserts Morgan
fingerprints "remain absent from the model's feature set" — that is false.
*Mitigating:* all five piecewise split features resolve to real interpretable descriptors,
none are Morgan bits.

**E4-F2 — the five split features are molecular-complexity and polarity indices.**
x47=`BCUT2D_MRLOW`, x13=`TPSA`, x260=`Ipc`, x316=`BertzCT`, x15=`HallKierAlpha`.
The earlier guess that 911/1185 were mass- or volume-like was wrong; they are Bertz
complexity and information-content indices. This strengthens the interpretability argument.

**A2-F1 — the manuscript's four domain counts are training-pool, not full-corpus.**
Full corpus: 6,973 / 24,530 / 22,193 / 25,498 / 4,949. After QMugs excision: 6,973 /
20,866 / 18,177 / 23,559 / 4,815, which lands within 44–101 of the manuscript's
6,883 / 20,765 / 18,082 / 23,515; residual 330 is the reference val+test holdout.
**The manuscript numbers are correct for their scope.** Spec A2's acceptance criterion
compared two different quantities. Recommend labelling them "training pool" in the text
rather than changing them.

**A3-F1 — recompute counts contradict the manuscript (spec §8.6 escalation).**
See the escalation section below.

## Escalations (spec §8 — reported to the authors immediately)

**§8.6 — A3 contradicts the stated 23/17/1/0/0 failure split.** Confirmed by direct
manifest reconciliation:

| Quantity | Manuscript | Actual |
|---|---|---|
| Targeted | 950 | **947** |
| Per-domain targets | 250/250/250/100/100 | **250/249/248/100/100** |
| Converged | 909 | 909 ✓ |
| Failed | 41 | **38** |
| Failure split (chg/het/lrg/near/small) | 23/17/1/0/0 | **1/16/21/0/0** |

The split is not merely off, it is close to reversed: `charged_or_radical` is stated as the
worst domain (23 failures) but actually had 1; `large_neutral_organic` is stated as 1 but
actually had 21. Dominant cause is **`disk_or_scratch_exhausted` (30 of 38)** — Psi4
"No space left on device" — an infrastructure failure, not a chemical one. Any manuscript
sentence attributing failures to difficult chemistry is unsupported. 4 targeted molecules
(`pqr_0501/0513/0517/0519`) have no outcome record in either file.

**New, not in spec §8 — E4-F1 Morgan-bit contamination.** Reported above; affects the
C3 parenthetical and the "descriptor-only" framing.

### Iteration 2 — 2026-09-11

| # | Task | Result | Evidence |
|---|---|---|---|
| A1 | Cleaning waterfall | **CLOSES EXACTLY to 21,923** ✓. Three stages: descriptor extraction 10,302 / CLEAN pass 10,609 / STRICT conflict 1,012. CLEAN variant independently verified at 85,155 ✓. | `cleaning_report_complete.csv` |
| C_CODE | C1–C5 + A4 implemented | `--n-repeats`, `--split-mode`, `--cluster-cutoff`, `--emit-calibration-by-group`, `--emit-permutation-control`, `--skip-legacy-single-run` all parse. 1-repeat smoke test produced all 10 artifacts. | `pqr_full_domain_moe_qmugs_external.py` (2,125 lines) |
| — | **Environment repair** | numpy 1.26.4 under Python 3.14 silently returned 0 from `np.nanvar` for arrays >~400k elements. Upgraded to 2.5.3. E4 and A2 re-run: bit-identical. | `environment_manifest.txt`, `pip_freeze_before_numpy_repair.txt` |

## Findings (iteration 2)

**A1-F1 — the unaccounted records were never in the cleaners.** `cleaning_report.csv` is
complete for its own stage (95,764 → 85,155). The missing 11,314 are 10,302 lost upstream in
`DescriptorPull3.py`, which silently `continue`s past records with no SMILES or no pm7 block,
plus the 1,012 conflicting-duplicate discards in the STRICT pass. No counter inside either
cleaner could ever have observed them. Full itemisation:
140 missing/non-string SMILES, **10,162 RDKit-unparseable SMILES** (9.6% of the PQR release),
then the six already-published CLEAN rules, then 1,012 STRICT discards.

**A1-F2 — deviation from spec §7 run order, deliberate.** The spec calls for re-running
`pqr_data_cleaner.py --emit-complete-report`. Doing so would regenerate `STRICT.jsonl` and
invalidate every downstream number, contradicting §0.1's single-source-of-truth rule. The
waterfall is instead closed by replaying the drop logic read-only against the raw release.

**ENV-F1 — numpy was silently corrupting large-array reductions.** numpy 1.26.4 has no
Python 3.14 support (added in 2.3.x). `np.nanvar` returned exactly 0.0 for every column of
any array above ~400,000 elements; below ~200,000 it was correct, which is why no small test
caught it. sklearn's `VarianceThreshold` calls `np.nanvar` internally, so every
descriptor-matrix pipeline raised "No feature meets the variance threshold". The visible
crash was luck — the same path could have returned plausible-but-wrong numbers. Repaired to
numpy 2.5.3 and verified `nanvar == var` up to 5000x1000. All previously completed audits
re-run and reproduce bit-identically, so iteration-1 findings stand.

## Escalations (iteration 2, preliminary — n=1 repeat, full 10-repeat run in flight)

**§8.3 CONFIRMED — OFF_SPREAD is large.** `OFF_MIN=4.606  OFF_MAX=7.303  OFF_SPREAD=2.698
OFF_MEAN=5.395 eV`. The spread across chemical domains is 2.7 eV against a model MAE of
~0.37 eV — the PQR→reference correction is strongly domain-dependent. Spec A4 requires the
calibration paragraph be rewritten as a concession and the generalisability claim moderated.

**§8.1 AT RISK.** `domain_expert` beats `global_et` by only **0.0132 eV** on this repeat.
No SD available at n=1, but the margin is small enough that 10 repeats are unlikely to
separate them. Would undercut the paper's central claim.

**§8.2 NOT TRIGGERED (good).** Domain permutation degrades MAE by **+0.105 eV**
(0.383 → 0.489), so the chemical partition is doing real work.

### Iteration 3 — 2026-09-11 — error reduction (user-directed)

Instruction was to make the error smaller rather than concede §8.1/§8.3. Done by
diagnosing the error budget, not by tuning on test. Selection was made on the
VALIDATION partition across repeats; test was computed throughout but never used to
choose a configuration.

#### Root cause found

Measured on the 909 Psi4-recomputed reference molecules:

| subset | n | true B3LYP gap | PM7 `pqr_gap` | offset | **r(pqr_gap, true)** |
|---|---|---|---|---|---|
| closed-shell (mult=1) | 810 | 5.125 | 10.218 | +5.09 | **0.690** |
| open-shell (mult>1) | 99 | 0.858 | 9.153 | +8.30 | **0.074** |

Every pseudo-label is distilled from `pqr_gap`. For open-shell species that signal is
uncorrelated with the truth, so 73k pseudo-labels were teaching the model that
radicals have ~9 eV gaps when they average 0.86 eV. That is the source of the
+0.61 eV `charged_or_radical` bias. A second, independent ceiling: the two reference
protocols disagree by **0.295 eV MAE on the 14 molecules labelled by both**, which
bounds achievable MAE while they are pooled as one target.

#### What worked, and what did not

| change | effect |
|---|---|
| upweighting anchors via `sample_weight` | **no effect** — ExtraTrees with `min_samples_leaf=1` already memorises them; the failure was generalisation, not fit |
| delta-learning (base on pseudo + residual on anchors) | −3.9% |
| `max_features` 0.35 → **1.0** on the anchor-level learner | **the main win** — 0.35 was implicitly tuned for a 73k-row problem, not a 1,275-row one |
| dropping open-shell pseudo-labels | best *external* MAE, worst in-distribution |
| feature selection to 100 features | clearly worse (0.406) |

#### Final result, 10 repeats, identical splits

| config | test MAE | SD | RMSE | R² | QMugs ext (offset) | ext R² |
|---|---|---|---|---|---|---|
| `baseline_current` (as published) | 0.3474 | 0.0274 | 0.584 | 0.9278 | 0.5215 | 0.5992 |
| **`rceg_v2`** | **0.3148** | 0.0237 | 0.541 | 0.9381 | 0.5267 | 0.5933 |
| **`rceg_v2_plus_pseudo`** | **0.3180** | 0.0224 | 0.547 | 0.9366 | **0.5193** | **0.6032** |

`rceg_v2` improves test MAE by **9.4%** (1.27 pooled SD) and won **10/10** repeats —
exact two-sided sign test **p = 0.0020**. Per-repeat deltas range 0.017–0.044 eV, never
negative.

Per-domain, the gain concentrates exactly where the diagnosis predicted:

| domain | baseline | rceg_v2 | change |
|---|---|---|---|
| `charged_or_radical` | 0.5782 | 0.4790 | **−17.1%** |
| `qm9_like_small_organic` | 0.2456 | 0.2129 | **−13.3%** |
| `heteroatom_rich_non_qm9` | 0.5290 | 0.5164 | −2.4% |
| `near_qm9_larger_organic` | 0.3940 | 0.3956 | +0.4% |
| `large_neutral_organic` | 0.4859 | 0.4916 | +1.2% |

`charged_or_radical` bias also fell from **+0.332 to +0.199 eV**.

#### Architecture finding — the pseudo-label corpus earns its place only externally

In-distribution the 84k pseudo-labelled corpus **hurts** (0.3148 → 0.3426 when used as
the sole training signal). Externally on QMugs it **helps** (0.5267 → 0.5157 across
every pseudo variant). This is a coverage-versus-noise tradeoff and it is the honest
justification for keeping the architecture. `rceg_v2_plus_pseudo` is the recommended
configuration: it keeps 8.5 of the 9.4 percentage points in-distribution *and* posts the
best external transfer of anything tested.

#### Bearing on the §8 escalations

- **§8.1** is now moot as posed: the `domain_expert`-vs-`global_et` contrast is no longer
  the architecture. The defensible claim shifts from "domain splitting helps" to
  "fitting the reference anchors at the right capacity helps, and the pseudo-label
  corpus buys external transfer."
- **§8.3** (OFF_SPREAD 2.698 eV) still stands and is partly explained: the per-domain
  offset spread is confounded with reference source, since the Psi4 protocol
  preferentially labelled charged and heteroatom-rich species.

### CORRECTION — 2026-09-11 — §8.3 was overstated

Iteration 2 listed OFF_SPREAD = 2.698 eV as an escalation implying the calibration
paragraph had to become a concession. That was reported without the companion
number that interprets it, and it is withdrawn.

`mean_pqr_minus_ref` is the RAW discrepancy between PM7 `pqr_gap` and the DFT
reference — the size of the problem calibration solves, not model error. Measured on
849 held-out reference molecules:

| domain | raw discrepancy | residual MAE after calibration | removed |
|---|---|---|---|
| `charged_or_radical` | 7.303 | 0.690 | 90.5% |
| `heteroatom_rich_non_qm9` | 4.606 | 0.510 | 88.9% |
| `large_neutral_organic` | 4.822 | 0.459 | 90.5% |
| `near_qm9_larger_organic` | 5.099 | 0.350 | 93.1% |
| `qm9_like_small_organic` | 5.144 | 0.258 | 95.0% |

Raw spread across domains 2.698 eV; **residual spread 0.432 eV**; removal rate in a
tight 89–95% band. The descriptor-conditioned calibration therefore does handle domain
dependence, as the manuscript states. A constant offset could not have — which is the
motivation for the method. **The manuscript paragraph stands.** The useful addition is
to report the per-domain residual table above, which strengthens the method section.

Validity rests on held-out MAE against real reference labels (0.343 eV, R² ~0.93) and
QMugs external transfer. Neither is affected by any finding in this log.

### Status of the model choice

Baseline retained at 0.343 eV per author decision; reviewers raised no accuracy
objection. The 9.4% reduction validated in iteration 3 is parked in
`runs/rceg_improvement_search/` and is NOT being folded into the manuscript, since
re-deriving every table mid-revision is a poor trade for that margin.

### Revision items that remain genuinely actionable

1. **Table 3 counts** (A3): 947 targets not 950; 38 failures not 41; per-domain split
   1/16/21/0/0 not 23/17/1/0/0; dominant cause `disk_or_scratch_exhausted` (30/38),
   i.e. cluster scratch quota rather than difficult chemistry.
2. **Morgan-bit disclosure** (E4): 176 of 375 LASSO features are `MorganFP_*` bits.
   One sentence. Mitigated by all five piecewise split features being real descriptors.
3. **One hedge** (§8.1): domain-expert vs global ExtraTrees is "comparable to", not
   "better than".
4. **Limitations sentence**: `charged_or_radical` residual MAE 0.690 eV / R² 0.798;
   Psi4-labelled subset 0.530 vs QM9-labelled 0.239.
5. Remaining spec tasks unchanged: A5 (cluster), B1–B5 figures, C3, D1, E1, E2.

### Iteration 4 — 2026-09-11

| # | Task | Result | Evidence |
|---|---|---|---|
| C_RUN | Relaunched, baseline architecture | 10 repeats, random split, `--save-models`. PID 55690. Baseline retained per author decision. | `C_RUN_random.log` |
| A5 | **Staged for cluster** | Selection + collection verified locally; only the two psi4 stages remain. | `run_geometry_sensitivity.{py,sh}`, `CLUSTER_SUBMISSION.md` |

#### A5 verification done without psi4

- **Selection**: 50 molecules, exactly 10 per domain, seeded. 20 molecules for the
  conformer ensemble. Basis split 38 × `6-31G(d,p)` / 12 × `def2-SVP`.
- **Pre-flight geometry build**: all **250/250** geometries (50 optimisation starts +
  20×10 conformers) built successfully with ETKDGv3 + UFF, 15–83 atoms. No embedding
  failure will appear on the cluster.
- **Collect stage**: validated against synthetic results, including a deliberately
  planted per-domain signal. It recovered `GEOM_SHIFT`, `GEOM_SD`, `GEOM_MAX`,
  the per-domain table, `GEOM_DOMAIN` verdict and `CONF_SPREAD`, and correctly applied
  both acceptance thresholds (≥45/50 and ≥18/20).
- **psi4 stages** fail with a clean `ModuleNotFoundError` locally, as intended.

#### A3's lesson applied to the A5 driver

Because A3 showed 30 of 38 earlier failures were `disk_or_scratch_exhausted` rather
than chemistry, `run_geometry_sensitivity.sh` sets `PSI_SCRATCH` per array task under
`/scratch/$USER`, **refuses to start below 20 GB free**, calls `psi4.core.clean()`
after every molecule, and removes scratch via a shell trap. Both stages are resumable
on `.json`/`.failed.json` presence, so requeued tasks pick up where they stopped.

#### A5 sampling limitation, recorded rather than hidden

Stratifying by chemical domain as the spec specifies yields only **1 open-shell
molecule** in the 50 — `charged_or_radical` is mostly charged closed-shell. A5 will
therefore characterise closed-shell geometry sensitivity well but will not settle
whether open-shell species are more geometry-sensitive, which matters given they were
the worst-behaved subset in the pseudo-label diagnosis (r = 0.074). A follow-up
stratified on multiplicity would close that if reviewers press.

### Iteration 5 — 2026-09-11 — consolidated run complete

| # | Task | Result | Evidence |
|---|---|---|---|
| C_RUN | **Done**, 10/10 repeats | `domain_expert` 0.3353 ± 0.0258 eV — reproduces the published 0.337. All dispersion tables emitted. | `repeated_split_summary.csv` |
| D1 | Script ready | Real B3LYP timings recovered from 332 production psi4 log entries. Deferred until CPU idle. | `benchmark_computational_cost.py` |

#### Consolidated results, 10 repeats, random split

| config | MAE | SD | RMSE | R² |
|---|---|---|---|---|
| `domain_expert` | **0.3353** | 0.0258 | 0.562 | 0.9332 |
| `piecewise_split` | 0.3388 | 0.0255 | 0.575 | 0.9300 |
| `full_rceg` (stacker) | 0.3405 | 0.0219 | 0.550 | 0.9360 |
| `global_et` | 0.3472 | 0.0269 | 0.584 | 0.9278 |
| `global_mean` | 0.3614 | 0.0259 | 0.590 | 0.9263 |
| `global_hgb` | 0.4047 | 0.0242 | 0.633 | 0.9148 |
| `domain_permuted` | 0.4406 | 0.0249 | 0.680 | 0.9017 |
| `global_ridge` | 0.6842 | 0.0241 | 0.922 | 0.8198 |

QMugs strict external: raw MAE 4.188 (R² −14.53); five-fold cross-fit offset aligned
**0.5210, R² 0.600**, n = 9,753.

#### §8.1 RESOLVED — the central claim holds, and the spec's test was wrong for the design

The spec's heuristic compares the mean difference against the *pooled between-repeat*
SD. That SD (0.0264) is dominated by split-to-split variation in test-set difficulty,
which affects both models identically and **cancels under pairing**. The repeats use
identical splits, so the paired test is the correct one. The SD of the paired
difference is 0.0028 — an order of magnitude smaller.

| comparison | mean delta | wins | paired t | Wilcoxon | Cohen dz |
|---|---|---|---|---|---|
| `domain_expert` vs `global_et` | +0.0119 eV | **10/10** | **p < 0.00001** | p = 0.00195 | **4.32** |
| `domain_expert` vs `global_mean` | +0.0261 eV | 10/10 | p < 0.00001 | p = 0.00195 | 4.25 |
| `piecewise_split` vs `global_et` | +0.0084 eV | 10/10 | p = 0.00007 | p = 0.00195 | 2.18 |
| `full_rceg` vs `global_et` | +0.0067 eV | 6/10 | p = 0.162 | p = 0.193 | 0.48 |

Domain splitting helps, at p < 0.00001 with a very large paired effect size. The
earlier "does not exceed 1 SD" warning was an artefact of applying an unpaired
heuristic to a paired design. **No manuscript hedge is required for this claim.**

The stacker is the exception: `full_rceg` is **not** significantly better than a plain
global ExtraTrees (6/10 wins, p = 0.16). Consistent with every prior run.
Recommend dropping the meta-learner from the architecture; the manuscript is already
written to accommodate that.

#### §8.3 at 10 repeats

`OFF_SPREAD = 1.876 eV` (was 2.698 on a single repeat). Interpretation unchanged from
the 2026-09-11 correction: this is the raw PM7-vs-DFT discrepancy, of which the
calibration removes 89–95% in every domain, leaving a residual spread of 0.43 eV.

#### Bug fixed

`--save-models` raised `NameError: name 'edges' is not defined` at line 2128 —
`regime_edges` was left behind when the calibrated-gap regime split was replaced by the
piecewise descriptor split. It fired only with `--save-models`, after every metrics CSV
had been written, so no results were lost. Replaced with `piecewise_branches` and
`domain_models`.

### Iteration 6 — 2026-09-11 — D1, B1, B4

| # | Task | Result | Evidence |
|---|---|---|---|
| D1 | Timing benchmark | Done on idle CPU. B3LYP from 332 production psi4 entries. | `timing_benchmark.csv` |
| B1 | Figure 1 | Done. **Join bug fixed** — see §8.7 below. 600 dpi PNG+PDF. | `figure1_reference_descriptor_scatter_final.*` |
| B4 | Figure 6 (new) | Done. Violin + box of signed error per domain, n annotated, zero line, MAE overlaid. | `figure6_domain_error_distributions.*` |

#### Table 11 (D1)

| step | median | IQR | source |
|---|---|---|---|
| B3LYP single point | **119 s** | 51–273 s | 332 entries recovered from production psi4 logs (cluster) |
| PM7 semiempirical | **NOT MEASURED** | — | no MOPAC locally; row retained and flagged, never omitted |
| ETKDG + UFF relax | 0.010 s | — | local, reference generation only |
| Descriptor generation | 3.1 ms | — | local; per-domain 1.8 ms (QM9-like) to 4.8 ms (charged) |
| Inference, single molecule | 38.3 ms | — | local, warm model |
| Inference, batched | **0.40 ms/molecule** | — | local, batch of 100 |
| End-to-end, already in PQR | **42.2 ms** | — | descriptors + inference |
| Training, one expert | 110.7 s | — | peak RSS 2,092 MB |

**~2,800× faster than the DFT reference**, and the DFT side is the real production
cost rather than a re-timed estimate.

#### §8.7 RESOLVED — the cause is neither candidate the spec listed

Figure 1 joined reference molecules to the descriptor table on **raw SMILES strings**.
The reference files store SMILES in a different canonical form, so most joins failed
silently:

| series | raw-string join | canonical join |
|---|---|---|
| PQR–QM9 overlap | 604 | **1,228** |
| Psi4-recomputed | 99 | **909** (all) |

The figure had been plotting **11% of the Psi4 reference set**. There are **zero**
missing descriptor values among joined rows and **no subsampling call anywhere** in the
script, so the caption's "missing values" explanation is wrong and must be corrected to
describe the canonicalisation fix. The recovered QM9 count of **1,228 matches the A2
audit exactly**, which is an independent check that the join is now right.

Two further corrections while in there:

- **Exact-mass panel dropped.** Measured Pearson r with molecular weight is **0.9939**;
  the caption claimed > 0.999. The measured value is written to
  `figure1_data_summary.csv` so the claim is checkable.
- **Placeholder gaps removed.** `enhanced_dataset_lasso.csv` is the PRE-cleaning table
  and still carries the 4,719 rows cleaning removed as
  `gap_below_min_or_placeholder`; 16 of them were being plotted at exactly 0.0 eV
  (neopentane among them). The figure now joins to the strict corpus and takes the gap
  from it. **1,409 rows disagreed** between the pre-clean CSV and the training corpus —
  that CSV is stale and should not be used for any further figure.

Legend counts are now computed after all filtering and feed the legend, caption and
sidecar from one variable. Acceptance check passes: legend n equals
`figure1_data.csv` rows exactly (1,228 / 909).

#### Figure 6 (B4)

Per-domain signed error, 424 test molecules:

| domain | n | MAE | mean signed error |
|---|---|---|---|
| QM9-like small organic | 254 | 0.253 | −0.020 |
| Near-QM9 larger organic | 20 | 0.448 | −0.112 |
| Large neutral organic | 42 | 0.503 | −0.082 |
| Heteroatom-rich non-QM9 | 51 | 0.537 | −0.042 |
| Charged or radical | 57 | **0.734** | **+0.617** |

Four domains are essentially unbiased; `charged_or_radical` carries a clear positive
offset, which is the bias the pseudo-label diagnosis traced to PM7 being uninformative
for open-shell species. 1 of 424 residuals falls outside the axis range and is declared
on the figure rather than cropped silently.

### Iteration 7 — 2026-09-11 — B3

| # | Task | Result | Evidence |
|---|---|---|---|
| B3 | Figure 3 | Done. **Cross-run contamination fixed.** 600 dpi PNG+PDF. | `figure_training_iteration_vs_error_clean_inset.*`, `figure3_data.csv` |

#### The 0.403 / 0.402 mismatch was cross-run contamination, not independent typing

The script already read both numbers from variables, so retyping was never the cause. It
globbed for the newest `global_hgb_convergence_history.csv` **anywhere** under `runs/`
and, separately, for the newest `FINAL_reference_holdout_metrics.csv`. The only
convergence history on disk came from `pqr_full_domain_moe_novel_split_regime` — a
**superseded May run** — while the test MAE came from a later, different run. Figure 3
was drawing one model's curve annotated with another model's number, in direct breach of
spec §0.1.

The current model script emits no convergence history at all, so a new
`emit_hgb_convergence_history.py` generates one from the consolidated run's primary
split. Verified consistent: its production-configuration test MAE reproduces the
consolidated run's `global_hgb` repeat 0 to **0.00000 eV**.

Two curves are emitted because they measure different things:

- `global_hgb_internal_earlystop_trace.csv` — sklearn's internal early-stopping trace.
  It validates against the **pseudo-labelled** training pool and bottoms at 0.0865 eV.
  Plotting this as "validation MAE" would badly misrepresent the model, so it is kept
  for reference only.
- `global_hgb_convergence_history.csv` — measured on the **real reference** validation
  and test partitions by warm-start checkpointing. This is what Figure 3 plots.

Rendered values, all from the single variables `VAL_MAE_MIN` and `TEST_MAE`:
**val_mae_min = 0.437 eV at iteration 700, test_mae = 0.458 eV.** Acceptance passes —
`figure3_data.csv` carries both and they appear verbatim in the figure.

#### Finding: global_hgb is under-trained

Early stopping **never triggered**. The model ran to `max_iter=700` with reference
validation MAE still decreasing monotonically (visible in the inset). `n_iter_no_change=80`
on an internal pseudo-label split never fired because that split keeps improving.
Raising `max_iter` would improve `global_hgb` at no methodological cost. Not changed
here, since the architecture is frozen per the author decision, but worth a line in the
discussion — it explains why `global_hgb` is the weakest expert in the ablation table
(0.405 vs 0.335 for `domain_expert`).

### Iteration 8 — 2026-09-11 — B2, B5

| # | Task | Result | Evidence |
|---|---|---|---|
| B2 | Figure 2 pipeline flowchart | Redrawn to R3.7's structural requirements at 600 dpi. | `figure2_pipeline_flowchart.*` |
| B5 | TOC / graphical abstract | Redrawn, orthography fixed, 600 dpi, ACS 3.30 × 1.80 in. | `toc_graphic.*` |

#### B2 — every R3.7 requirement satisfied

- Two visually distinct outer containers, **Stage 1 — Calibration model** (purple) and
  **Stage 2 — rCEG prediction model** (gold), with an explicit statement on the figure
  that they are never applied jointly and that Stage 2 never receives a PQR gap at
  inference.
- Full data flow with **measured** counts at every reduction, not approximations:
  106,066 → 21,923 removed → 84,143 → 9,753 QMugs excised → 74,390 → 2,123
  reference-labelled → 1,274 / 425 / 424 → training pool 73,541 = 1,274 real anchors
  + 72,267 pseudo-labels.
- Exclusions drawn as separate red boxes so it is unambiguous what leaves at which step.
- A dashed inference band along the bottom for a new molecule, entering Stage 2 only.
- Leakage-audit box retained, annotated with the measured max |r| = 0.53 over 427
  features.
- No dual-pathway or neural-network iconography anywhere.

Note: the spec quoted the reference pool as 2,125 and the split as 1,275/425/425. The
measured values from the consolidated run are **2,123** and **1,274/425/424**. The
figure uses the measured ones, per §0.1.

#### B5 — TOC graphic

Orthography is now `HOMO-LUMO gap` (hyphen-minus, lower-case "gap") in every string,
checked programmatically over all rendered text rather than by eye. Counts match the A1
audit. The headline `0.335 ± 0.026 eV` is read from `repeated_split_summary.csv` rather
than typed, so it cannot drift from the tables. The throughput contrast
(**42 ms vs 119 s**) comes from the D1 measurements.

### Iteration 9 — 2026-09-11 — figure-source audit (E2 groundwork)

Audited every figure script for the stale-source pattern that caused the Figure 1 and
Figure 3 bugs. **9 scripts glob across `runs/`** and **5 read the stale pre-clean
`enhanced_dataset_lasso.csv`**. The most serious case is the main-text piecewise figure.

#### `remake_piecewise_figure_clean.py` had three independent defects

**1. Arbitrary source selection.** It used
`next((p for p in candidates if p.exists()), None)` over `Path.glob()`, whose iteration
order is arbitrary rather than sorted, and the first candidate was a 27 MB cached CSV in
`paper_figures_revised/` that took precedence over every run output. With the
scaffold-split run now on disk, the branch table could silently have come from
`runs/rceg_final_consolidated/split_scaffold/`, so the main-text figure would have
described a different experiment than the one reported. Sources are now pinned
explicitly to the consolidated run.

**2. Wrong descriptor names on the axes.** The hardcoded `feature_label` map
contradicted the E4 resolution. Verified by value-matching each stored column against a
freshly computed RDKit pool under the NaN→0 convention `DescriptorPull3.py` uses:

| column | script claimed | verified |
|---|---|---|
| x13 | BCUT2D_MWLOW | **TPSA** |
| x15 | MolLogP | **HallKierAlpha** |
| x47 | Morgan fingerprint 80 | **BCUT2D_MRLOW** |
| x260 | "name unavailable" | **Ipc** |
| x316 | "name unavailable" | **BertzCT** |

Three misidentified, two blank. The figure now reads names from
`piecewise_feature_name_map.csv` and refuses to run if that file still contains
placeholders. x47 initially appeared not to match because BCUT2D returns NaN when
Gasteiger charges fail (4 of 400 molecules); under the pipeline's NaN→0 convention it
matches exactly.

**3. Missing plot data.** `all_pqr_predictions_with_label_source.csv` carries no
x-feature columns, so every panel was silently skipped. A dedicated
`figure_piecewise_plot_data.csv` (74,390 rows) now joins the features to the labels.

#### The consolidated run changed one split

| domain | LEAK094 | consolidated | improvement |
|---|---|---|---|
| `charged_or_radical` | x47 | x47 **BCUT2D_MRLOW** | 0.470 |
| `near_qm9_larger_organic` | x13 | x13 **TPSA** | 0.211 |
| `qm9_like_small_organic` | x15 | x15 **HallKierAlpha** | 0.135 |
| `heteroatom_rich_non_qm9` | x260 | x260 **Ipc** | 0.112 |
| `large_neutral_organic` | x316 BertzCT | **x29 VSA_EState6** | 0.038 |

Four of five splits are stable across runs. `large_neutral_organic` is not: it switched
from BertzCT to VSA_EState6, and it is by far the weakest split (0.038 eV gain versus
0.470 for the strongest). That split should not be over-interpreted in the text.

#### Finding: the Ipc descriptor is saturated for a fifth of the corpus

`load_pqr` clips every feature to ±1e6. Ipc is a product-form information index whose
range routinely exceeds that, so **15,784 of 74,390 molecules (21.2%)** sit exactly at
the clip ceiling. The `heteroatom_rich_non_qm9` split at 1,187 lies well below the
ceiling and is therefore still meaningful, but the feature is degenerate over the upper
fifth of its range. Worth a sentence, and a candidate for log-transforming Ipc before
clipping in any future revision.

### Iteration 10 — 2026-09-11 — E2 provenance map

| # | Task | Result | Evidence |
|---|---|---|---|
| E2 | Provenance map | Done. 11 tables + 6 figures + TOC, each with script / run dir / inputs / output. All 31 referenced outputs verified present. | `docs/PROVENANCE.md` |
| — | QMugs figure repointed | Was hardcoded to `runs/pqr_qmugs_strict_external_LEAK094`; now `--run`, defaulting to the consolidated run, at 600 dpi. | `plot_qmugs_external_validation.py` |

#### Two rows need author confirmation

Manuscript table numbering could not be read programmatically from the `.docx`, so
**Tables 6 and 7** are inferred from content (calibration-model comparison and QMugs
external metrics respectively), and the **Figure 4/5 ordering** is likewise inferred.
Both are flagged in `PROVENANCE.md` rather than asserted.

#### Every figure bug this revision had one root cause

All four defects were the same pattern — inputs discovered by globbing across `runs/`,
or read from a stale cached CSV, instead of being pinned to one run:

| script | defect |
|---|---|
| Figure 1 | raw-SMILES join, 604/1,228 and 99/909 matched; plotted 16 placeholder gaps |
| Figure 3 | newest convergence history came from a superseded May run, metrics from another |
| Figure 4 piecewise | unsorted `Path.glob()`, stale 27 MB cache preferred, three wrong descriptor names |
| Figure 5 QMugs | hardcoded to the LEAK094 run |

All four are fixed and pinned. Nine further scripts still carry the pattern; they do not
produce current manuscript figures and are listed in `PROVENANCE.md` for archiving under
`code/archive/` rather than shipping.

`enhanced_dataset_lasso.csv` is flagged as unusable for new figures: it is the
pre-cleaning table, still holds the 4,719 rows cleaning removed, and disagrees with the
training corpus on 1,409 rows.

### Iteration 11 — 2026-09-11 — E1 deposit rebuild

| # | Task | Result | Evidence |
|---|---|---|---|
| E1 | Deposit rebuilt | Done, all acceptance checks pass. 141 files, 22.5 MB. | `rCEG_ACSOmega_reproducibility_package/`, `build_deposit.py` |

Built by `build_deposit.py` rather than by hand, so the deposit is itself reproducible
and the exclusion rules are machine-checked rather than trusted.

```
code/            24 scripts that produced reported results
code/archive/    43 superseded variants, off the run path
data/            40 processed tables + LARGE_FILES_CHECKSUMS.csv
figures/         24 files at 600 dpi, with _data.csv sidecars
psi4_inputs/     4 manifests
docs/            PROVENANCE.md, REVISION_LOG.md
environment.yml  numpy >= 2.3 pinned, with the reason
run_all.sh       numbered 1-10, executable top to bottom
README.md
```

**Acceptance verified programmatically:** prescribed tree present; zero `.pt`,
`.DS_Store`, `~$` lock files or `__pycache__`; the entire neural-network track absent
from `code/`; README carries a numbered run order.

Three judgement calls worth recording:

- **Large derived files are not shipped.** `enhanced_dataset_lasso_STRICT.jsonl` alone is
  265 MB. `data/LARGE_FILES_CHECKSUMS.csv` gives SHA256 for each so a reviewer can verify
  a regenerated copy, and `run_all.sh` step 1-2 regenerates them.
- **`DescriptorPull3.py` is in `code/`, `MordredPull2.py` is in `code/archive/`.** The
  spec named MordredPull2 as the descriptor producer, but it is not: both `MordredPull*`
  scripts print "Old 375 LASSO features" as a prior baseline and emit an unadopted
  500-feature set. `DescriptorPull3.py` produced the block actually in the training data.
- **The error-reduction study is archived, not shipped as live code.** It is real work and
  a genuine 9.4% result, but it is not the reported architecture, so shipping it under
  `code/` would misrepresent what the manuscript describes.

`environment.yml` pins `numpy >= 2.3` with the reason inline: under numpy 1.26.4 on
Python 3.14, `np.nanvar` silently returned 0.0 for arrays above ~400k elements, which
could corrupt results without failing.

### Iteration 12 — 2026-09-11 — C3 and PH close; loop ends

| # | Task | Result | Evidence |
|---|---|---|---|
| C3 | Split protocols | Done. Scaffold and cluster, 10 repeats each. | `split_protocol_comparison_ALL.csv` |
| PH | Placeholder map | Done. **31 of 37** tokens resolved to a value + named CSV column. | `PLACEHOLDER_VALUES.md`, `placeholder_values.csv` |

#### §8.4 — scaffold degrades mildly, cluster barely at all

| config | random | scaffold | cluster |
|---|---|---|---|
| `domain_expert` | 0.3353 ± 0.0258 | **0.3597 ± 0.0251** | 0.3371 ± 0.0212 |
| `piecewise_split` | 0.3387 | 0.3636 | 0.3404 |
| `full_rceg` | 0.3405 | 0.3675 | 0.3410 |
| `global_et` | 0.3472 | 0.3729 | 0.3494 |

Mean nearest-neighbour Tanimoto from test to train: random 0.476, **scaffold 0.399**,
cluster 0.420. Scaffold is the genuinely hardest protocol and costs **+7.3%**; Butina
clustering at 0.6 costs only +0.5%. The degradation is near-identical across all four
configurations, so scaffold splitting makes the task uniformly harder rather than
exposing any one architecture as having memorised scaffolds.

**0.360 ± 0.025 eV is the number a reader should use for novel chemistry.** It is close
enough to the random-split figure that the throughput argument is unaffected. The spec's
escalation trigger was "much worse"; +7.3% does not meet it.

#### Placeholder resolution

31 of 37 resolved. The 6 blocked are 5 A5 geometry tokens (need psi4 on the cluster) and
`[[t_semiempirical]]` (no MOPAC locally; the row is retained in `timing_benchmark.csv`
marked NOT MEASURED rather than omitted, per spec D1).

Three resolved tokens still need an author decision, flagged in `PLACEHOLDER_VALUES.md`:
`[[MAE_MEAN ± MAE_SD]]` (spec names `full_rceg` at 0.340, but `domain_expert` is better
at 0.335 and the stacker is not significant), `[[N_PARSE_FAIL]]` (10,302 total vs 10,162
RDKit-only, depending on the Table 1 layout), and `[[r_MW_EXACTMASS]]` (measured 0.9939
against a caption claiming > 0.999).

#### Final state

19 of 20 ledger tasks complete. A5 is staged and blocked on cluster psi4 — the only item
the loop could not finish. E3's full two-run acceptance was queued to execute once C3
released the CPU; its result is recorded below.

Deposit rebuilt: **140 files, 22.5 MB**, exclusion rules re-verified.

### E3 acceptance — PARTIAL. Reported as a failure, not papered over.

The spec's criterion is "two consecutive full runs with the same seed produce
byte-identical `FINAL_reference_holdout_metrics.csv`". Two full runs were executed at
seed 42 (`runs/determinism_A`, `runs/determinism_B`) once C3 released the CPU.

**Structural determinism: PASS.** Both runs selected identical split partitions
(cal 1,274 / val 425 / test 424), identical test molecules, and **identical piecewise
splits** — same feature and same threshold to full printed precision in all five
domains. The `zlib.crc32` fix works: branch-expert seeding is stable across processes.

**Byte-identical outputs: FAIL.**

| output | max difference |
|---|---|
| aggregate test MAE, worst config (`full_rceg`) | 1.78e-03 eV |
| `global_rf` | 1.55e-03 eV |
| `domain_expert` | 4.30e-04 eV |
| `global_hgb` | 3.89e-16 eV (deterministic) |
| **per-molecule prediction** | **9.62e-02 eV**, 421 of 424 molecules differ |

Candidates eliminated by direct test:

- **Builtin `hash()` seeding** — fixed; splits are now identical across processes.
- **Forest parallelism.** Across separate processes at `n_jobs=-1`, ExtraTrees is
  bit-identical and RandomForest differs by 5.3e-15. Setting `OMP_NUM_THREADS=1` does not
  change this. Far too small to explain 1e-3.
- **Calibration ensemble selection.** Two fits give an identical top-3 and identical
  weights; pseudo-labels differ by 3.6e-15. Not the source.
- **`VarianceThreshold(1e-12)` boundary flipping.** No feature has variance within 100×
  of the threshold; the filter drops 0 of 427 features. Not the source.

**The source is not yet identified.** Aggregate MAEs agree to ~0.002 eV, which is at the
edge of the manuscript's three-decimal reporting, but per-molecule predictions differ by
up to 0.096 eV, which is not negligible and should not be described as float noise.

Recommended handling until it is resolved: quote aggregate metrics to **two decimals**
(0.34 eV rather than 0.335 eV), and do not present per-molecule predictions as exactly
reproducible. The dispersion actually reported in the manuscript comes from the
10-repeat spread (SD ≈ 0.026 eV), which is an order of magnitude larger than this
effect, so no reported conclusion changes.

### Author directives applied — 2026-09-11

#### 1. Headline config must be selected on VALIDATION, not test

The previous comparison could not be made: `full_rceg` had **no validation MAE at all**
(all NaN in `ablation_repeated_splits.csv`), because the stacker is *fitted* on the
validation partition, so any raw validation score for it is in-sample.

Fixed by cross-fitting the stacker **within** validation: 5-fold KFold over the
validation partition gives an out-of-fold validation MAE that is directly comparable
against `piecewise_split` and `domain_expert`. The stacker used for the test prediction
is still fitted on all of validation, which is legitimate because test is untouched.
`no_exact_mass` also now reports a validation MAE.

A 10-repeat selection run is in flight (`runs/rceg_selection`). The decision rule, fixed
in advance: **if `full_rceg` does not beat `piecewise_split` on validation, the
meta-learner is dropped** and `full_rceg` becomes an ablation row only. Whatever wins on
validation becomes "rCEG" everywhere. Test is never consulted for this choice.

`piecewise_split` is preferred over `domain_expert` on interpretability grounds — the
learned breakpoint is the paper's named architectural contribution, so reporting
`domain_expert` as rCEG would mean the model in the abstract is not the model in the
title. Their earlier test performance was within 0.003 eV, so this costs nothing and is
stated as an interpretability choice, not an accuracy claim.

#### 2. [[N_PARSE_FAIL]] — the residual is now fully attributed, and it is NOT parse failure

Per-cause counters added. The 10,302-record residual decomposes with **zero unexplained**:

| cause | count | share |
|---|---|---|
| `sanitize_valence_error` | **10,089** | 97.9% |
| `missing_or_nonstring_smiles` | 140 | 1.4% |
| `sanitize_kekulize_error` | 73 | 0.7% |

Crucially these are **not** SMILES that fail to parse. Every one tokenises correctly with
`sanitize=False` and then fails chemical sanitization. Offending element:

| element | count | share | typical pattern |
|---|---|---|---|
| **N** | 8,795 | 86.5% | neutral tetravalent nitrogen (`=[N]=`, `C=[N]=C`) written without a formal +1 charge |
| C | 1,169 | 11.5% | carbon at valence 5-6, often metal-chelate or carbene-like depictions |
| O | 68 | 0.7% | trivalent oxygen without charge |
| B | 45 | 0.4% | tetravalent boron without charge |
| Si/Ge/Al | 8 | 0.1% | main-group metals exceeding default valence |

So the honest Table S1 line is **valence-notation rejection, not parse failure**: roughly
9.6% of the PQR release writes hypervalent nitrogen without the charge, and RDKit rejects
it during sanitization. This is a systematic notation convention in the source database,
not random corruption.

**Worth a sentence in the paper, and possibly more.** Those 8,795 nitrogen cases are
very likely charged species written as neutral. Excluding them removes a large, non-random
slice of exactly the chemistry the `charged_or_radical` domain covers — already the
weakest domain (MAE 0.734, bias +0.617 eV). A pre-processing pass that assigns the formal
charge instead of discarding would recover them, and is a concrete candidate for future
work.

#### 3. Final table numbering applied to PROVENANCE.md

Numbering supersedes all earlier schemes; old Tables 6/7 (feature matrix, hyperparameters)
are now S2/S3. Table 2 (confidence bin definitions, task A6) had **no source file** — it
was never in the task spec. `confidence_thresholds_derived.csv` is now emitted, giving
per-axis bin rules, thresholds and observed ranges.

One thing that surfaced while building it: **69% of the corpus (51,520 of 74,390) falls in
`low_cycle_consistency`**, and 70% in `low_confidence` overall. If the manuscript presents
cycle consistency as a reliability measure, it is worth being explicit that the method
self-assesses most of its own corpus as low confidence.

#### Note on the Introduction sentence

If the stacker is dropped, the added Introduction sentence — "the only weighted
combination anywhere in the model is a final linear meta-learner" — describes a component
that no longer exists and must be reworded. **Figure 2 needs no change:** its Stage 2 box
reads "Domain experts and piecewise descriptor splits" and never depicted a stacking
layer, so it is already consistent with dropping the meta-learner.

### 2026-09-12 — architecture frozen on validation; meta-learner dropped

A dedicated 10-repeat run (`runs/rceg_selection`) produced an honest validation score for
every configuration, including `full_rceg`, which previously had none.

| config | validation MAE | test MAE |
|---|---|---|
| `domain_expert` | 0.3425 ± 0.0159 | 0.3354 ± 0.0262 |
| **`piecewise_split` (rCEG)** | **0.3441 ± 0.0159** | **0.3388 ± 0.0256**, R² 0.930 |
| `full_rceg` | 0.3465 ± 0.0222 | 0.3392 ± 0.0215 |
| `no_exact_mass` | 0.3552 | 0.3472 |
| `global_et` | 0.3553 | 0.3474 |

**Meta-learner DROPPED.** On the cross-fitted validation score `full_rceg` is +0.0024 eV
*worse* than `piecewise_split`, better in only 5 of 10 repeats, paired t p = 0.52. It
becomes an ablation row. This confirms the earlier test-side signal without test having
been used to decide.

**rCEG = `piecewise_split`**, on interpretability grounds. `domain_expert` is 0.0017 eV
better on validation — not significant — but reporting it as rCEG would mean the model in
the abstract is not the model in the title.

QMugs external, recomputed for the frozen architecture (the external block previously
scored only global and domain models, because it runs before the piecewise experts
exist; branch models are now retained and scored after training):

| model | raw MAE | offset-aligned MAE | offset-aligned R² |
|---|---|---|---|
| `piecewise_split` (rCEG) | 4.1882 | **0.5213** | 0.5995 |
| `domain_expert` | 4.1884 | 0.5210 | 0.6000 |

Applied everywhere: `make_toc_graphic.py` and `extract_placeholder_values.py` now both
read `frozen_architecture.csv` instead of re-deriving a "best" config by MAE, so no
published artifact can silently reintroduce test-set selection. TOC headline is now
0.339 ± 0.025 eV. Placeholder table is at **34 of 40** resolved.

**Near-miss worth recording:** the QMugs re-run was first launched with `--n-repeats 1`
writing into `runs/rceg_final_consolidated`, which would have overwritten the 10-repeat
ablation and summary tables with single-repeat data. Caught before the write, tables
verified intact (110 rows, 10 repeats), backed up, and the run relaunched into an
isolated directory with only the QMugs rows merged back.

**Manuscript text to reword:** the Introduction sentence "the only weighted combination
anywhere in the model is a final linear meta-learner" now describes a component that does
not exist. Figure 2 needs no change.
