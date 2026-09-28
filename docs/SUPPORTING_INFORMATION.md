# Supporting Information

**A split-architecture machine learning model for scalable HOMO–LUMO gap prediction: rCEG**

Wang & Closser

Every table below is generated directly from the result files of the frozen run by `build_SI.py`. Each is annotated with the file and columns it was read from, so any value can be traced to the run that produced it.

---

## S1. Model hyperparameters

All learners are wrapped in a pipeline of median imputation → near-zero-variance filtering (threshold 1e-12) → the estimator. Tree learners are not scaled, because split thresholds are scale invariant; linear learners are, because ridge penalties are not.

### S1.1 Estimators

| Role | Estimator | Hyperparameters |
|---|---|---|
| Domain expert, piecewise branch expert | `ExtraTreesRegressor` | `n_estimators=500`, `max_features=0.35`, `min_samples_leaf=1` |
| Calibration candidate `desc_et` | `ExtraTreesRegressor` | `n_estimators=700`, `max_features=0.35`, `min_samples_leaf=1` |
| Calibration candidate `desc_rf`, ablation `global_rf` | `RandomForestRegressor` | `n_estimators=400` (500 in calibration), `max_features=0.35`, `min_samples_leaf=1` |
| Calibration candidate `desc_hgb`, ablation `global_hgb` | `HistGradientBoostingRegressor` | `max_iter=700`, `learning_rate=0.035`, `max_leaf_nodes=31`, `l2_regularization=0.03`, `loss='absolute_error'`, `early_stopping=True`, `validation_fraction=0.15`, `n_iter_no_change=80` |
| Linear/ridge calibration, `global_ridge` | `RidgeCV` | `alphas=np.logspace(-6, 5, 60)`, generalized cross-validation |

<sub>Source: `pqr_full_domain_moe_qmugs_external.py`, functions `et`, `rf`, `hgb`, `ridge_scaled`, `pipe`</sub>

### S1.2 Architecture thresholds

| Parameter | Value | Meaning |
|---|---|---|
| `min_domain` | 500 | Minimum training molecules before a domain gets its own expert; smaller domains fall back to the global mean model. |
| `min_piecewise` | 1000 | Minimum training molecules before a domain is considered for a piecewise split. |
| `min_leaf` | `max(250, min(750, n_domain // 12))` | Minimum molecules on each side of a candidate breakpoint. |
| `max_features` (split search) | 50 | Candidate descriptors ranked by absolute correlation with the training label. |
| `top_k` (calibration ensemble) | 3 | Candidates retained, weighted by 1/(validation MAE). |

<sub>Source: `pqr_full_domain_moe_qmugs_external.py`, `_train_domain_and_piecewise`, `_best_piecewise_split`, `fit_calibration_ensemble`</sub>

### S1.3 Determinism

Three guards make repeated runs byte-identical. They were added after a reproducibility audit found that a 1e-17 difference in calibration ensemble weights — caused by `RandomForestRegressor` with `n_jobs=-1` accumulating tree predictions in a non-fixed order — was amplified by near-tied ExtraTrees splits into differences of up to 0.096 eV on individual predictions.

1. Candidate validation MAEs are quantized to 12 decimal places before ranking and weighting, removing the sensitivity at source (1e-12 eV is roughly ten orders of magnitude below anything reported).
2. Pseudo-labels are rounded to 9 decimal places.
3. `DeterministicPipeline` forces `n_jobs=1` at predict time.
4. Seeds derive from `zlib.crc32` of a string key rather than Python's salted `hash()`; all runs set `PYTHONHASHSEED=0`.

Byte-identical reruns were verified after these changes.

---

## S2. Calibration diagnostics by chemical domain

### S2.1 Calibrated accuracy and raw PQR offset, per domain

| Domain | n | MAE (eV) | RMSE (eV) | R² | mean PQR − reference (eV) | SD (eV) |
|---|---|---|---|---|---|---|
| `qm9_like_small_organic` | 524 | 0.226 | 0.433 | 0.922 | 5.226 | 1.828 |
| `near_qm9_larger_organic` | 39 | 0.356 | 0.445 | 0.879 | 5.349 | 1.799 |
| `large_neutral_organic` | 91 | 0.473 | 0.624 | 0.647 | 4.915 | 1.435 |
| `heteroatom_rich_non_qm9` | 93 | 0.506 | 0.753 | 0.499 | 4.737 | 1.605 |
| `charged_or_radical` | 102 | 0.572 | 0.861 | 0.869 | 6.613 | 2.281 |
| **all** | 849 | **0.331** | 0.569 | 0.934 | 5.312 | 1.897 |

Counts are means over the 10 repeated splits, so they are not integers.

<sub>Source: `runs/revision2/B4_table5.csv`</sub>

### S2.2 Is one global offset sufficient?

The raw PQR-minus-reference offset is not constant across chemical domains. Across the HELD-OUT reference molecules (validation + test; `calibration_by_domain_and_source.csv` is computed on held-out only) the per-domain mean offset ranges from **4.737 eV** to **6.613 eV** — a spread of **1.876 eV** about a mean of **5.368 eV**.

<sub>Source: `runs/rceg_final_consolidated/calibration_offsets.csv`, cols OFF_MIN / OFF_MAX / OFF_MEAN / OFF_SPREAD</sub>

On the held-out reference molecules the same quantity behaves the same way:

| Domain | n | mean offset (eV) | SD (eV) | median (eV) |
|---|---|---|---|---|
| `qm9_like_small_organic` | 283 | 5.192 | 1.804 | 5.877 |
| `near_qm9_larger_organic` | 16 | 5.822 | 2.248 | 6.009 |
| `large_neutral_organic` | 40 | 5.005 | 1.316 | 4.639 |
| `heteroatom_rich_non_qm9` | 51 | 4.517 | 1.561 | 3.990 |
| `charged_or_radical` | 35 | 6.827 | 2.529 | 6.897 |

Spread of the per-domain means: **2.310 eV** (`charged_or_radical` highest, `heteroatom_rich_non_qm9` lowest).

This is the justification for calibrating rather than subtracting a single constant: a global offset would carry an error of order 1 eV into the domains at the ends of this range, several times the final model error.

<sub>Source: `runs/revision2/B4_domain_independence_test.csv`</sub>

*Note on the spreads.* Both numbers above are over **held-out** reference molecules: 1.876 eV pooling validation and test over ten repeats, and 2.310 eV from a single-partition file (n = 425) that has no generating script and is superseded. The matching calibration-set spread, recomputed over the same ten repeats, is **1.637 eV** — see `paste/C1_table4.md`, which is the authoritative per-class breakdown. Name the set whenever either is quoted, and never average them.

---

## S3. Feature-leakage audit

Every one of the 427 input features was correlated against four leakage-relevant targets (`gap`, `homo`, `lumo`, `lumo_minus_homo`) across the full labelled pool. Any feature exceeding |r| = 0.94 with the target is excised before training; the audit confirms that no feature comes close to that bar.

- Largest absolute correlation anywhere in the matrix: **|r| = 0.534**
- Features exceeding |r| = 0.5: **5**
- Features exceeding the excision threshold |r| = 0.94: **0**

### S3.1 Twenty strongest feature–target correlations

| Feature | Descriptor | Target | r |
|---|---|---|---|
| `x12` | FractionCSP3 | `gap` | +0.534 |
| `x396` | — | `gap` | +0.534 |
| `x396` | — | `lumo_minus_homo` | +0.534 |
| `x12` | FractionCSP3 | `lumo_minus_homo` | +0.534 |
| `x163` | SMR_VSA7 | `gap` | -0.530 |
| `x163` | SMR_VSA7 | `lumo_minus_homo` | -0.530 |
| `x15` | HallKierAlpha | `gap` | +0.519 |
| `x15` | HallKierAlpha | `lumo_minus_homo` | +0.519 |
| `x136` | SlogP_VSA6 | `gap` | -0.518 |
| `x136` | SlogP_VSA6 | `lumo_minus_homo` | -0.518 |
| `x12` | FractionCSP3 | `lumo` | +0.513 |
| `x396` | — | `lumo` | +0.513 |
| `x397` | — | `gap` | -0.484 |
| `x397` | — | `lumo_minus_homo` | -0.484 |
| `x398` | — | `gap` | -0.483 |
| `x398` | — | `lumo_minus_homo` | -0.483 |
| `x382` | — | `gap` | -0.479 |
| `x91` | HeavyAtomMolWt | `gap` | -0.479 |
| `x382` | — | `lumo_minus_homo` | -0.479 |
| `x91` | HeavyAtomMolWt | `lumo_minus_homo` | -0.479 |

The full 1708-row table is `runs/revision2/A5_leakage_audit.csv`.

<sub>Source: `runs/revision2/A5_leakage_audit.csv`</sub>

---

## S4. Descriptor provenance and naming

The 427-column feature matrix is assembled as follows.

| Block | Columns | Content |
|---|---|---|
| PQR semiempirical scalars | `x0`–`x4` | molecular mass, exact mass, dipole moment, heat of formation, polarizability |
| LASSO-selected block | `x5`–`x379` | 375 descriptors selected from a pool of RDKit `Descriptors.descList` plus 200 Morgan fingerprint bits (radius 2, 1024-bit) |
| RDKit block | `x380`–`x410` | 31 further RDKit descriptors |
| Bond-step topology | `x411`–`x426` | 16 bond-step counts |

The original column names were lost when the LASSO block was written as bare `x` indices. They were recovered by regenerating the full descriptor pool and value-matching each column against it: **375 of 375 resolved**, 362 uniquely and 13 to a small set of value-identical candidates.

**176 of the 375 LASSO-selected features (47%) are Morgan fingerprint bits**, and 199 are interpretable physicochemical descriptors. This bears directly on interpretability claims: most of the selected block is substructure indicators rather than named chemical quantities.

The five learned piecewise breakpoints, however, fall on named descriptors: **0 of 5 split features is a Morgan bit** (§S5).

<sub>Source: `runs/revision2/A4_lasso_feature_names.csv`, col `resolved_name`</sub>

---

## S5. Learned piecewise breakpoints

| Domain | n (train) | Split feature | Descriptor | Breakpoint | left n | right n | MAE improvement (eV) |
|---|---|---|---|---|---|---|---|
| `qm9_like_small_organic` | 4,264 | `x15` | Hall-Kier alpha shape correction | -0.400 | 2,277 | 1,987 | 0.135 |
| `near_qm9_larger_organic` | 23,523 | `x13` | Topological polar surface area (A^2) | 12.530 | 5,333 | 18,190 | 0.211 |
| `large_neutral_organic` | 18,098 | `x29` | EState-weighted van der Waals surface area, bin 6 | 0.000 | 7,244 | 10,854 | 0.038 |
| `heteroatom_rich_non_qm9` | 20,773 | `x260` | Information content of the characteristic polynomial | 1186.695 | 8,309 | 12,464 | 0.112 |
| `charged_or_radical` | 6,882 | `x47` | BCUT2D eigenvalue, molar-refractivity weighted (lowest) | -0.451 | 3,957 | 2,925 | 0.470 |

Improvement is the reduction in single-line MAE when one linear fit across the domain is replaced by two fits either side of the breakpoint — the criterion the breakpoint was selected on.

<sub>Source: `runs/rceg_final_consolidated/piecewise_split_architecture_branches.csv`</sub>
 · <sub>Source: `runs/revision2/A4_piecewise_feature_map.csv`, descriptor names</sub>

---

## S6. Reference recomputation manifest

**947 molecules** were targeted for Psi4 B3LYP recomputation, selected as the lowest-confidence predictions of the preceding cycle. **909 converged** and **38 failed**.

### S6.1 Outcome by domain

| Domain | targeted | converged | failed | dominant failure mode |
|---|---|---|---|---|
| `qm9_like_small_organic` | 100 | 100 | 0 | none |
| `near_qm9_larger_organic` | 100 | 100 | 0 | none |
| `large_neutral_organic` | 248 | 227 | 21 | disk_or_scratch_exhausted=17 |
| `heteroatom_rich_non_qm9` | 249 | 233 | 16 | disk_or_scratch_exhausted=13; other=3 |
| `charged_or_radical` | 250 | 249 | 1 | manually_marked_failed=1 |
| **total** | **947** | **909** | **38** | |

Most failures are `disk_or_scratch_exhausted`, an infrastructure limit rather than a chemical one, so the failed set is not systematically different in chemistry from the converged set. The failure-record file contains more rows than there were final failures because molecules that failed once and succeeded on retry each left a record; the reconciliation column `n_records_that_later_converged` accounts for the difference.

<sub>Source: `runs/revision2/A1_recompute_reconciliation.csv`</sub>

### S6.2 Charge and multiplicity distribution

| Charge | n | | Multiplicity | n |
|---|---|---|---|---|
| +0 | 700 | | 1 | 848 |
| +1 | 191 | | 2 | 50 |
| +2 | 53 | | 3 | 46 |
| +3 | 3 | | 4 | 2 |
|  |  | | 5 | 1 |

Of the 947 targeted molecules, **247 carry a non-zero formal charge** and **99 are open-shell** (multiplicity > 1). Open-shell species use an unrestricted Kohn–Sham (UKS) reference; closed-shell species use RKS.

<sub>Source: `runs/pqr_full_domain_moe_qm9_cycle/recompute_inputs/recompute_manifest.csv`, cols `charge`, `multiplicity`</sub>

### S6.3 Basis set and SCF settings

| Setting | Value |
|---|---|
| Functional | B3LYP |
| Basis (H, C, N, O, F only) | 6-31G(d,p) |
| Basis (any other element) | def2-SVP |
| Reference | RKS (multiplicity 1), UKS (multiplicity > 1) |
| `scf_type` | `df` (density fitting), `df_scf_guess=True` |
| `e_convergence` | 1e-6 |
| `d_convergence` | 1e-6 |
| `maxiter` | 150 |
| Geometry | as supplied in the manifest XYZ; `no_reorient`, `no_com` |

The basis set is chosen per molecule by element composition, so a single manuscript sentence naming one basis is incomplete; both must be stated.

<sub>Source: `run_C3_timings.py`, functions `basis_for`, `main`</sub> · <sub>Source: `runs/revision2/paste/A9_scf_settings.md`</sub>

---

## S7. Geometry sensitivity

The production pipeline consumes a single UFF-optimized conformer. This measures what is lost relative to a B3LYP-optimized geometry: a random sample of 50 molecules, stratified across the five domains, was re-optimized at B3LYP and the gap recomputed at both geometries.

- Molecules attempted: **50**
- Geometry optimizations that converged: **37**
- Mean |Δ gap|: **0.440 eV**
- Median |Δ gap|: **0.279 eV**
- SD: **0.607 eV**; maximum: **3.591 eV**

The mean and the median differ by more than a factor of 1.5 because the distribution is long-tailed: most molecules shift little and a few shift a great deal. Both should be quoted.

### S7.1 By domain

| Domain | n converged | mean &#124;Δ gap&#124; (eV) |
|---|---|---|
| `qm9_like_small_organic` | 10 | 0.744 |
| `near_qm9_larger_organic` | 8 | 0.355 |
| `large_neutral_organic` | 6 | 0.211 |
| `heteroatom_rich_non_qm9` | 9 | 0.305 |
| `charged_or_radical` | 4 | 0.497 |

Per-domain means rest on very few molecules each and should be read as indicative only. The largest per-domain mean sits on `qm9_like_small_organic`, which is the *most* accurate domain for the model itself; it is driven by a single 3.59 eV outlier and is not evidence that small organics are geometry-sensitive in general.

### S7.2 Convergence shortfall and its direction

37 of 50 optimizations converged, below the 45 the protocol asked for. The failures are not random with respect to the quantity being measured, and they push the estimate in **both** directions:

- **Upward bias.** Optimizer failures concentrate in large, flexible molecules with shallow potential-energy surfaces. Those are exactly the molecules whose gaps move *least* between geometries, so dropping them raises the reported mean.
- **Downward bias.** Converged optimizations used `gau_loose` criteria. A looser convergence threshold stops nearer the starting geometry, which understates the true difference.

These do not cancel in any quantified way. The figure should be reported with the sample size and both caveats attached, not as a clean measurement.

<sub>Source: `runs/rceg_final_consolidated/geometry_sensitivity.csv`</sub> · <sub>Source: `runs/revision2/C1_geometry_shift.csv`, per-molecule, including failures</sub>

### S7.3 Conformational floor

A second, independent limit: where the corpus contains multiple conformers of the same molecule, the spread of PQR gaps among them is irreducible noise that no model trained on a single conformer can predict.

- Molecules with ≥3 conformers: **9,697**
- Median within-molecule SD: **0.076 eV** (IQR 0.028–0.160)
- Mean: 0.112 eV; 90th percentile: 0.265 eV; maximum: 1.071 eV

<sub>Source: `runs/revision2/C2_conformer_floor.csv`</sub>

---

## S8. Split-protocol comparison

Random splitting can flatter a model by placing close analogues on both sides of the split. Two stricter protocols test that: Bemis–Murcko scaffold splitting, and Butina clustering on Morgan fingerprints. All three use the same ten repeats and the same architecture.

| Protocol | test MAE ± SD (eV) | RMSE | R² | n test | median max Tanimoto, test→train | Δ vs random |
|---|---|---|---|---|---|---|
| random | 0.335 ± 0.026 | 0.562 | 0.933 | 424 | 0.455 | — |
| scaffold | 0.360 ± 0.025 | 0.570 | 0.925 | 408 | 0.347 | +0.024 (+7.3%) |
| cluster | 0.337 ± 0.021 | 0.565 | 0.933 | 423 | 0.421 | +0.002 (+0.5%) |

Scaffold splitting costs **7.3%** accuracy, and it does so while cutting the median nearest-neighbour similarity from 0.455 to 0.347. The degradation is real but modest, which is the substantive answer to the reviewer's concern: performance does not depend on near-duplicates across the split.

<sub>Source: `runs/revision2/B5_split_protocols.csv`</sub>

---

## S9. Error distribution by domain

Per-domain MAE hides a strongly skewed error distribution. The percentiles below are of absolute error on the held-out test molecules.

| Domain | n | MAE | p10 | p20 | p50 | p80 | p90 | p95 | max |
|---|---|---|---|---|---|---|---|---|---|
| `qm9_like_small_organic` | 254 | 0.253 | 0.014 | 0.028 | 0.117 | 0.335 | 0.499 | 0.799 | 7.227 |
| `near_qm9_larger_organic` | 20 | 0.448 | 0.064 | 0.164 | 0.375 | 0.661 | 0.828 | 0.936 | 1.196 |
| `large_neutral_organic` | 42 | 0.505 | 0.108 | 0.157 | 0.424 | 0.734 | 0.942 | 1.290 | 1.929 |
| `heteroatom_rich_non_qm9` | 51 | 0.539 | 0.082 | 0.159 | 0.498 | 0.854 | 0.969 | 1.291 | 2.569 |
| `charged_or_radical` | 57 | 0.733 | 0.088 | 0.156 | 0.458 | 1.113 | 1.395 | 1.947 | 4.778 |

The median error is far below the mean in every domain. For `qm9_like_small_organic` the median absolute error is 0.117 eV against an MAE of 0.253 eV — a factor of two — because a small number of molecules carry very large errors (maximum 7.23 eV). Reporting the median alongside the mean is the honest summary.

<sub>Source: `runs/revision2/B2_error_deciles.csv`</sub>

---

## S10. Structure-only ablation

Five inputs (`x0`–`x4`) come from the PQR semiempirical calculation rather than from the SMILES string. This tests how much they contribute, on the same ten repeated splits.

| Configuration | features | test MAE ± SD (eV) | Δ vs full |
|---|---|---|---|
| `full` — all inputs (the reported rCEG) | 427 | 0.3384 ± 0.026 | — |
| `drop_x0_x4` — all five PQR scalars removed | 422 | 0.3385 ± 0.025 | +0.0001 (+0.0%) |
| `drop_x2_x4` — only the three genuinely semiempirical scalars removed | 424 | 0.3390 ± 0.026 | +0.0006 (+0.2%) |

`x0` and `x1` are molecular mass and exact mass, both computable from SMILES alone and both already duplicated in the RDKit block at `x380`/`x381`; removing them removes nothing. The three inputs that genuinely require a semiempirical calculation are `x2` (dipole moment), `x3` (heat of formation) and `x4` (polarizability), which `drop_x2_x4` isolates.

Dropping all five changes test MAE by +0.0001 eV (+0.0%) — far inside the ±0.026 eV split-to-split SD. **Inference on a new molecule needs only its SMILES string.**

One caveat must accompany that claim either way: the *training* labels are calibrated pseudo-labels derived from `pqr_gap`, itself a semiempirical quantity. The structure-only result licenses a narrow statement about inference, not about training.

<sub>Source: `runs/revision2/B6_summary.csv`</sub> · <sub>Source: `runs/revision2/B6_structure_only.csv`, per-repeat</sub>

---

## S11. Open-shell agreement between PQR and reference

| Subset | n | Pearson r | Spearman ρ | mean &#124;Δ&#124; (eV) | median &#124;Δ&#124; (eV) |
|---|---|---|---|---|---|
| all recomputed | 909 | 0.589 | 0.586 | 5.442 | 4.962 |
| domain == charged_or_radical | 249 | 0.505 | 0.266 | 6.487 | 6.495 |
| charge != 0 | 246 | 0.513 | 0.277 | 6.452 | 6.473 |
| multiplicity > 1 (open-shell) | 99 | 0.074 | 0.124 | 8.295 | 7.878 |
| neutral closed-shell | 660 | 0.694 | 0.629 | 5.047 | 4.478 |

**The semiempirical and DFT descriptions of open-shell molecules barely agree**: r = 0.074 over 99 molecules, against r = 0.694 for 660 neutral closed-shell molecules. This is the mechanism behind `charged_or_radical` being the model's weakest domain, and it is a property of the input data, not of the learner.

<sub>Source: `runs/revision2/A6_openshell_agreement.csv`</sub>

---

## S12. Corpus and reference-pool arithmetic

### S12.1 Cleaning waterfall

| Stage | Rule | Records removed |
|---|---|---|
| 1_descriptor_extraction | `missing_or_nonstring_smiles` | 140 |
| 1_descriptor_extraction | `missing_pm7_block` | 0 |
| 1_descriptor_extraction | `numeric_coercion_error` | 0 |
| 1_descriptor_extraction | `not_a_dict` | 0 |
| 1_descriptor_extraction | `missing_descriptor_block` | 0 |
| 1_descriptor_extraction | `sanitize_valence_error` | 10,089 |
| 1_descriptor_extraction | `sanitize_kekulize_error` | 73 |
| 2_clean_pass | `gap_below_min_or_placeholder` | 4,719 |
| 2_clean_pass | `duplicate_rows_removed` | 3,328 |
| 2_clean_pass | `multi_fragment` | 2,005 |
| 2_clean_pass | `target_percentile_trim` | 446 |
| 2_clean_pass | `isolated_hydrogen` | 101 |
| 2_clean_pass | `gap_above_hard_max` | 10 |
| 3_strict_pass | `conflicting_duplicate_discard` | 1,012 |
| | **total removed** | **21,923** |

Rules with a count of zero are retained in the table deliberately: they were checked and found not to fire, which is itself part of the audit.

<sub>Source: `runs/revision2/A2_cleaning_counts.csv`</sub> · <sub>Source: `runs/revision2/A2_stage1_valence_breakdown.csv`, valence detail</sub>

### S12.2 Reference pool

| Quantity | Value |
|---|---|
| QM9 overlap with the strict corpus | 1,228 |
| Psi4 B3LYP recomputed in the strict corpus | 909 |
| molecules carrying **both** labels | 14 |
| **reference pool (union)** | **2,123** |

1,228 + 909 − 14 = 2,123, which matches the split sizes 1,274 + 425 + 424 = 2,123. The figure 2,137 that appears in the submitted manuscript is incorrect.

### S12.3 Cross-protocol label floor

On the 14 molecules labelled under both protocols, the mean absolute difference between the QM9 and Psi4 gaps is **0.295 eV** (median 0.276, maximum 0.680).

This is a floor on achievable accuracy: two defensible reference protocols disagree with each other by about 0.3 eV on the same molecules, which is the same order as the model's own error. No model trained against a mixture of the two can be expected to do much better, and reporting accuracy below this floor would be meaningless.

<sub>Source: `runs/revision2/A8_dual_labelled.csv`</sub>

---

## S13. External validation on QMugs

All 9,753 QMugs molecules used here were excised from training by SMILES before any fitting, and the excision is asserted in code.

Two evaluations are reported, and the difference between them is the point:

| Evaluation | model | MAE (eV) | RMSE | R² |
|---|---|---|---|---|
| strict frozen transfer | `global_hgb` | 4.183 | 4.237 | -14.48 |
| strict frozen transfer | `global_et` | 4.202 | 4.257 | -14.63 |
| strict frozen transfer | `global_rf` | 4.196 | 4.252 | -14.59 |
| strict frozen transfer | `global_mean` | 4.194 | 4.248 | -14.56 |
| strict frozen transfer | `domain_expert` | 4.189 | 4.244 | -14.53 |
| strict frozen transfer | `piecewise_split` | 4.188 | 4.243 | -14.53 |
| 5-fold cross-fit, offset-aligned | `domain_expert` | **0.521** | 0.681 | 0.600 |
| 5-fold cross-fit, offset-aligned | `piecewise_split` | **0.521** | 0.681 | 0.600 |

Applied frozen, the model is **not transferable**: the MAE of about 4.2 eV and the large negative R² reflect an almost constant offset between the QMugs reference protocol and this corpus's labels. Once that offset is estimated on held-out QMugs folds — five-fold cross-fitting, so no molecule contributes to its own alignment — the MAE falls to **0.521 eV** with R² **0.600**.

The honest reading is that rCEG transfers in *ranking and shape* but not in absolute placement, and that a new reference protocol requires a handful of labelled anchors to re-align. That is a limitation of the calibration design and should be stated as one.

<sub>Source: `runs/revision2/D1_external_metrics.csv`</sub> · <sub>Source: `runs/revision2/D1_external_by_domain.csv`, per-domain</sub>

---

## S14. Full ablation table

All configurations on the same ten repeated random splits.

| Configuration | MAE ± SD (eV) | min | max | RMSE | R² |
|---|---|---|---|---|---|
| `domain_expert` | 0.335 ± 0.026 | 0.298 | 0.383 | 0.562 | 0.933 |
| `full_rceg` | 0.338 ± 0.023 | 0.305 | 0.375 | 0.548 | 0.936 |
| `piecewise_split` | 0.338 ± 0.026 | 0.299 | 0.386 | 0.574 | 0.930 |
| `no_exact_mass` | 0.347 ± 0.027 | 0.308 | 0.396 | 0.584 | 0.928 |
| `global_et` | 0.348 ± 0.027 | 0.310 | 0.398 | 0.585 | 0.928 |
| `global_rf` | 0.349 ± 0.025 | 0.317 | 0.401 | 0.578 | 0.929 |
| `global_mean` | 0.361 ± 0.026 | 0.330 | 0.414 | 0.590 | 0.926 |
| `global_hgb` | 0.405 ± 0.024 | 0.379 | 0.456 | 0.633 | 0.915 |
| `domain_permuted_piecewise` | 0.437 ± 0.023 | 0.403 | 0.481 | 0.675 | 0.903 |
| `domain_permuted` | 0.440 ± 0.026 | 0.403 | 0.491 | 0.679 | 0.902 |
| `global_ridge` | 0.684 ± 0.024 | 0.650 | 0.725 | 0.922 | 0.820 |

The two `domain_permuted` rows are the control: domain labels are shuffled while preserving domain sizes, so the architecture is identical and only the chemical meaning of the partition is destroyed. Their degradation is the evidence that the split architecture is doing chemical work rather than acting as an arbitrary ensemble.

<sub>Source: `runs/revision2/B3_ablation.csv`</sub> · <sub>Source: `runs/revision2/B3_ablation_per_seed.csv`, per-repeat</sub>

---

## S15. Computational cost

One representative molecule per domain, chosen at that domain's median heavy-atom count. **All timings single-threaded on one machine (Apple M4 Pro, 12 logical cores)** so that the ratio is not contaminated by comparing a cluster job against a laptop job.

| domain | mol_id | n_heavy_atoms | charge | multiplicity | basis | descriptors_ms | semiempirical_gfn2xtb_ms | b3lyp_s | b3lyp_converged | inference_ms | end_to_end_in_pqr_ms | end_to_end_novel_ms | speedup_in_pqr | speedup_novel |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| charged_or_radical | pqr_0166 | 22 | 2 | 3 | def2-SVP | 5.589 | 26.202 | 256.21 | True | 9.496 | 15.09 | 41.29 | 16979 | 6205 |
| heteroatom_rich_non_qm9 | pqr_0399 | 18 | 0 | 1 | def2-SVP | 2.94 | 44.691 | 42.41 | True | 9.496 | 12.44 | 57.13 | 3409 | 742 |
| large_neutral_organic | pqr_0502 | 27 | 0 | 1 | 6-31G(d,p) | 4.475 | 65.304 | 129.84 | True | 9.496 | 13.97 | 79.27 | 9294 | 1638 |
| near_qm9_larger_organic | pqr_0845 | 14 | 0 | 1 | 6-31G(d,p) | 2.717 | 48.322 | 40.05 | True | 9.496 | 12.21 | 60.54 | 3280 | 662 |
| qm9_like_small_organic | pqr_0871 | 8 | 0 | 1 | 6-31G(d,p) | 1.744 | 43.817 | 10.92 | True | 9.496 | 11.24 | 55.06 | 972 | 198 |

The speedup is not a single number. Across the five representatives it spans roughly three orders of magnitude, because the reference B3LYP cost grows steeply with system size while rCEG inference is essentially flat. Quote the range, or quote a specific domain, but do not quote a single average as though it were a property of the method.

<sub>Source: `runs/revision2/C3_timings.csv`</sub> · <sub>Source: `runs/revision2/T1_timing_dft_local.csv`, independent n=20 DFT timing</sub>

---

## S16. Reproducing these results

The deposit (`rCEG_ACSOmega_reproducibility_package/`) contains `run_all.sh`, `environment.yml` and a `README.md` describing the run order. Every command sets `PYTHONHASHSEED=0`. The environment actually used is captured verbatim in `runs/rceg_final_consolidated/environment_manifest.txt`.

Two environment notes matter for reproduction:

1. **NumPy version.** Under NumPy 1.26.4 with Python 3.14, `np.nanvar` silently returned 0.0 for arrays above roughly 400,000 elements on this platform, corrupting variance-based filtering without raising. The results reported here were produced under NumPy 2.5.3 and re-verified to reproduce bit-identically. Do not run this code on NumPy 1.26.4; `environment.yml` pins `numpy>=2.3,<3`.
2. **Psi4 scratch.** `PSI_SCRATCH` is read by Psi4 at import time. Setting it per molecule inside a running Python process has no effect; it must be exported before the interpreter starts. Otherwise scratch accumulates in `/tmp/psi.<pid>.*` and can exhaust the disk mid-run — which is the origin of most of the recomputation failures in §S6.1.

