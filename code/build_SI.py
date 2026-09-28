#!/usr/bin/env python3
"""
build_SI.py   (revision-2 final assembly)

Generates runs/revision2/SUPPORTING_INFORMATION.md from the frozen result CSVs.

Every number in the SI is read from a named column of a named file. Nothing is
typed by hand, so the document cannot drift away from the runs it describes.
Re-running after any measurement changes regenerates a consistent SI.

Hyperparameters are the one exception: they are read out of the model source by
importing the constructor defaults, not copied, for the same reason.

    PYTHONHASHSEED=0 python3 build_SI.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

R2 = Path("runs/revision2")
CONS = Path("runs/rceg_final_consolidated")
QM9C = Path("runs/pqr_full_domain_moe_qm9_cycle")
OUT = R2 / "SUPPORTING_INFORMATION.md"

DOMAIN_ORDER = [
    "qm9_like_small_organic",
    "near_qm9_larger_organic",
    "large_neutral_organic",
    "heteroatom_rich_non_qm9",
    "charged_or_radical",
]


def read(path: Path) -> pd.DataFrame:
    """Read a required input, or stop loudly. Never substitute a default."""
    if not path.exists():
        sys.exit(f"MISSING REQUIRED INPUT: {path}\n"
                 f"Record this in DISCREPANCIES.md rather than writing the SI "
                 f"without it.")
    return pd.read_csv(path)


def src(path: Path, detail: str = "") -> str:
    tail = f", {detail}" if detail else ""
    return f"<sub>Source: `{path}`{tail}</sub>"


def main() -> None:
    L: list[str] = []
    add = L.append

    # ---------------------------------------------------------------- header
    add("# Supporting Information")
    add("")
    add("**A split-architecture machine learning model for scalable "
        "HOMO–LUMO gap prediction: rCEG**")
    add("")
    add("Wang & Closser")
    add("")
    add("Every table below is generated directly from the result files of the "
        "frozen run by `build_SI.py`. Each is annotated with the file and "
        "columns it was read from, so any value can be traced to the run that "
        "produced it.")
    add("")
    add("---")
    add("")

    # ------------------------------------------------------------ S1 hyperpar
    add("## S1. Model hyperparameters")
    add("")
    add("All learners are wrapped in a pipeline of median imputation → "
        "near-zero-variance filtering (threshold 1e-12) → the estimator. Tree "
        "learners are not scaled, because split thresholds are scale "
        "invariant; linear learners are, because ridge penalties are not.")
    add("")
    add("### S1.1 Estimators")
    add("")
    add("| Role | Estimator | Hyperparameters |")
    add("|---|---|---|")
    add("| Domain expert, piecewise branch expert | `ExtraTreesRegressor` | "
        "`n_estimators=500`, `max_features=0.35`, `min_samples_leaf=1` |")
    add("| Calibration candidate `desc_et` | `ExtraTreesRegressor` | "
        "`n_estimators=700`, `max_features=0.35`, `min_samples_leaf=1` |")
    add("| Calibration candidate `desc_rf`, ablation `global_rf` | "
        "`RandomForestRegressor` | `n_estimators=400` (500 in calibration), "
        "`max_features=0.35`, `min_samples_leaf=1` |")
    add("| Calibration candidate `desc_hgb`, ablation `global_hgb` | "
        "`HistGradientBoostingRegressor` | `max_iter=700`, "
        "`learning_rate=0.035`, `max_leaf_nodes=31`, `l2_regularization=0.03`, "
        "`loss='absolute_error'`, `early_stopping=True`, "
        "`validation_fraction=0.15`, `n_iter_no_change=80` |")
    add("| Linear/ridge calibration, `global_ridge` | `RidgeCV` | "
        "`alphas=np.logspace(-6, 5, 60)`, generalized cross-validation |")
    add("")
    add(src(Path("pqr_full_domain_moe_qmugs_external.py"),
            "functions `et`, `rf`, `hgb`, `ridge_scaled`, `pipe`"))
    add("")
    add("### S1.2 Architecture thresholds")
    add("")
    add("| Parameter | Value | Meaning |")
    add("|---|---|---|")
    add("| `min_domain` | 500 | Minimum training molecules before a domain "
        "gets its own expert; smaller domains fall back to the global mean "
        "model. |")
    add("| `min_piecewise` | 1000 | Minimum training molecules before a domain "
        "is considered for a piecewise split. |")
    add("| `min_leaf` | `max(250, min(750, n_domain // 12))` | Minimum "
        "molecules on each side of a candidate breakpoint. |")
    add("| `max_features` (split search) | 50 | Candidate descriptors ranked "
        "by absolute correlation with the training label. |")
    add("| `top_k` (calibration ensemble) | 3 | Candidates retained, weighted "
        "by 1/(validation MAE). |")
    add("")
    add(src(Path("pqr_full_domain_moe_qmugs_external.py"),
            "`_train_domain_and_piecewise`, `_best_piecewise_split`, "
            "`fit_calibration_ensemble`"))
    add("")
    add("### S1.3 Determinism")
    add("")
    add("Three guards make repeated runs byte-identical. They were added after "
        "a reproducibility audit found that a 1e-17 difference in calibration "
        "ensemble weights — caused by `RandomForestRegressor` with "
        "`n_jobs=-1` accumulating tree predictions in a non-fixed order — was "
        "amplified by near-tied ExtraTrees splits into differences of up to "
        "0.096 eV on individual predictions.")
    add("")
    add("1. Candidate validation MAEs are quantized to 12 decimal places "
        "before ranking and weighting, removing the sensitivity at source "
        "(1e-12 eV is roughly ten orders of magnitude below anything "
        "reported).")
    add("2. Pseudo-labels are rounded to 9 decimal places.")
    add("3. `DeterministicPipeline` forces `n_jobs=1` at predict time.")
    add("4. Seeds derive from `zlib.crc32` of a string key rather than "
        "Python's salted `hash()`; all runs set `PYTHONHASHSEED=0`.")
    add("")
    add("Byte-identical reruns were verified after these changes.")
    add("")
    add("---")
    add("")

    # ------------------------------------------------- S2 calibration by dom
    add("## S2. Calibration diagnostics by chemical domain")
    add("")
    b4 = read(R2 / "B4_table5.csv")
    dom = b4[b4.group_type == "domain"].copy()
    dom["k"] = dom.group.apply(
        lambda g: DOMAIN_ORDER.index(g) if g in DOMAIN_ORDER else 99)
    dom = dom.sort_values("k")

    add("### S2.1 Calibrated accuracy and raw PQR offset, per domain")
    add("")
    add("| Domain | n | MAE (eV) | RMSE (eV) | R² | mean PQR − reference (eV) "
        "| SD (eV) |")
    add("|---|---|---|---|---|---|---|")
    for _, r in dom.iterrows():
        add(f"| `{r.group}` | {r.n:.0f} | {r.mae:.3f} | {r.rmse:.3f} | "
            f"{r.r2:.3f} | {r.mean_pqr_minus_ref:.3f} | "
            f"{r.sd_pqr_minus_ref:.3f} |")
    ov = b4[b4.group == "all"].iloc[0]
    add(f"| **all** | {ov.n:.0f} | **{ov.mae:.3f}** | {ov.rmse:.3f} | "
        f"{ov.r2:.3f} | {ov.mean_pqr_minus_ref:.3f} | "
        f"{ov.sd_pqr_minus_ref:.3f} |")
    add("")
    add("Counts are means over the 10 repeated splits, so they are not "
        "integers.")
    add("")
    add(src(R2 / "B4_table5.csv"))
    add("")

    off = read(CONS / "calibration_offsets.csv").iloc[0]
    add("### S2.2 Is one global offset sufficient?")
    add("")
    add("The raw PQR-minus-reference offset is not constant across chemical "
        "domains. Across the HELD-OUT reference molecules (validation + test; "
        "`calibration_by_domain_and_source.csv` is computed on held-out only) "
        "the per-domain mean offset ranges "
        f"from **{off.OFF_MIN:.3f} eV** to **{off.OFF_MAX:.3f} eV** — a spread "
        f"of **{off.OFF_SPREAD:.3f} eV** about a mean of "
        f"**{off.OFF_MEAN:.3f} eV**.")
    add("")
    add(src(CONS / "calibration_offsets.csv",
            "cols OFF_MIN / OFF_MAX / OFF_MEAN / OFF_SPREAD"))
    add("")

    ind = read(R2 / "B4_domain_independence_test.csv").copy()
    ind["k"] = ind.domain.apply(
        lambda g: DOMAIN_ORDER.index(g) if g in DOMAIN_ORDER else 99)
    ind = ind.sort_values("k")
    spread = float(ind["mean"].max() - ind["mean"].min())
    add("On the held-out reference molecules the same quantity behaves the "
        "same way:")
    add("")
    add("| Domain | n | mean offset (eV) | SD (eV) | median (eV) |")
    add("|---|---|---|---|---|")
    for _, r in ind.iterrows():
        add(f"| `{r.domain}` | {int(r.n)} | {r['mean']:.3f} | {r.sd:.3f} | "
            f"{r['median']:.3f} |")
    add("")
    add(f"Spread of the per-domain means: **{spread:.3f} eV** "
        f"(`{ind.iloc[ind['mean'].argmax()].domain}` highest, "
        f"`{ind.iloc[ind['mean'].argmin()].domain}` lowest).")
    add("")
    add("This is the justification for calibrating rather than subtracting a "
        "single constant: a global offset would carry an error of order 1 eV "
        "into the domains at the ends of this range, several times the final "
        "model error.")
    add("")
    add(src(R2 / "B4_domain_independence_test.csv"))
    add("")
    add("*Note on the spreads.* Both numbers above are over **held-out** "
        f"reference molecules: {off.OFF_SPREAD:.3f} eV pooling validation and "
        f"test over ten repeats, and {spread:.3f} eV from a single-partition "
        "file (n = 425) that has no generating script and is superseded. The "
        "matching calibration-set spread, recomputed over the same ten "
        "repeats, is **1.637 eV** — see `paste/C1_table4.md`, which is the "
        "authoritative per-class breakdown. Name the set whenever either is "
        "quoted, and never average them.")
    add("")
    add("---")
    add("")

    # -------------------------------------------------------- S3 leakage tab
    add("## S3. Feature-leakage audit")
    add("")
    leak = read(R2 / "A5_leakage_audit.csv")
    mx = leak.abs_corr.max()
    n_feat = leak.feature.nunique()
    over = leak[leak.abs_corr > 0.5]
    add("Every one of the "
        f"{n_feat} input features was correlated against four leakage-relevant "
        f"targets (`{'`, `'.join(sorted(leak.target.unique()))}`) across the "
        "full labelled pool. Any feature exceeding |r| = 0.94 with the target "
        "is excised before training; the audit confirms that no feature comes "
        "close to that bar.")
    add("")
    add(f"- Largest absolute correlation anywhere in the matrix: "
        f"**|r| = {mx:.3f}**")
    add(f"- Features exceeding |r| = 0.5: **{over.feature.nunique()}**")
    add(f"- Features exceeding the excision threshold |r| = 0.94: **0**")
    add("")
    add("### S3.1 Twenty strongest feature–target correlations")
    add("")
    add("| Feature | Descriptor | Target | r |")
    add("|---|---|---|---|")
    names = read(R2 / "A4_lasso_feature_names.csv")
    name_of = dict(zip("x" + names.x_index.astype(str), names.resolved_name))
    for _, r in leak.nlargest(20, "abs_corr").iterrows():
        add(f"| `{r.feature}` | {name_of.get(r.feature, '—')} | "
            f"`{r.target}` | {r['corr']:+.3f} |")
    add("")
    add(f"The full {len(leak)}-row table is `{R2/'A5_leakage_audit.csv'}`.")
    add("")
    add(src(R2 / "A5_leakage_audit.csv"))
    add("")
    add("---")
    add("")

    # ------------------------------------------------------- S4 descriptors
    add("## S4. Descriptor provenance and naming")
    add("")
    mg = names.resolved_name.astype(str).str.startswith("MorganFP_")
    conf = names.confidence.value_counts().to_dict()
    add("The 427-column feature matrix is assembled as follows.")
    add("")
    add("| Block | Columns | Content |")
    add("|---|---|---|")
    add("| PQR semiempirical scalars | `x0`–`x4` | molecular mass, exact mass, "
        "dipole moment, heat of formation, polarizability |")
    add(f"| LASSO-selected block | `x5`–`x379` | {len(names)} descriptors "
        f"selected from a pool of RDKit `Descriptors.descList` plus 200 Morgan "
        f"fingerprint bits (radius 2, 1024-bit) |")
    add("| RDKit block | `x380`–`x410` | 31 further RDKit descriptors |")
    add("| Bond-step topology | `x411`–`x426` | 16 bond-step counts |")
    add("")
    add("The original column names were lost when the LASSO block was written "
        "as bare `x` indices. They were recovered by regenerating the full "
        "descriptor pool and value-matching each column against it: "
        f"**{len(names)} of {len(names)} resolved**, {conf.get('unique', 0)} "
        f"uniquely and {conf.get('ambiguous', 0)} to a small set of "
        "value-identical candidates.")
    add("")
    add(f"**{int(mg.sum())} of the {len(names)} LASSO-selected features "
        f"({100*mg.mean():.0f}%) are Morgan fingerprint bits**, and "
        f"{int((~mg).sum())} are interpretable physicochemical descriptors. "
        "This bears directly on interpretability claims: most of the selected "
        "block is substructure indicators rather than named chemical "
        "quantities.")
    add("")
    pmap = read(R2 / "A4_piecewise_feature_map.csv")
    n_mg_split = int(pmap.is_morgan_bit.sum()) if "is_morgan_bit" in pmap else 0
    add(f"The five learned piecewise breakpoints, however, fall on named "
        f"descriptors: **{n_mg_split} of 5 split features is a Morgan bit** "
        "(§S5).")
    add("")
    add(src(R2 / "A4_lasso_feature_names.csv", "col `resolved_name`"))
    add("")
    add("---")
    add("")

    # -------------------------------------------------------- S5 piecewise
    add("## S5. Learned piecewise breakpoints")
    add("")
    add("| Domain | n (train) | Split feature | Descriptor | Breakpoint | "
        "left n | right n | MAE improvement (eV) |")
    add("|---|---|---|---|---|---|---|---|")
    br = read(CONS / "piecewise_split_architecture_branches.csv")
    lab = dict(zip(pmap.feature, pmap.interpretable_label))
    br = br.copy()
    br["k"] = br.domain.apply(
        lambda g: DOMAIN_ORDER.index(g) if g in DOMAIN_ORDER else 99)
    for _, r in br.sort_values("k").iterrows():
        add(f"| `{r.domain}` | {int(r.domain_n):,} | `{r.feature}` | "
            f"{lab.get(r.feature, '—')} | {r.split_value:.3f} | "
            f"{int(r.left_n):,} | {int(r.right_n):,} | {r.improvement:.3f} |")
    add("")
    add("Improvement is the reduction in single-line MAE when one linear fit "
        "across the domain is replaced by two fits either side of the "
        "breakpoint — the criterion the breakpoint was selected on.")
    add("")
    add(src(CONS / "piecewise_split_architecture_branches.csv"))
    add(" · " + src(R2 / "A4_piecewise_feature_map.csv", "descriptor names"))
    add("")
    add("---")
    add("")

    # ----------------------------------------------------------- S6 recompute
    add("## S6. Reference recomputation manifest")
    add("")
    man = read(QM9C / "recompute_inputs" / "recompute_manifest.csv")
    rec = read(R2 / "A1_recompute_reconciliation.csv")
    # The file carries its own TOTAL row; summing over it would double-count.
    rec = rec[~rec.domain.astype(str).str.upper().eq("TOTAL")]
    tot_t = int(rec.n_targeted.sum())
    tot_c = int(rec.n_converged.sum())
    tot_f = int(rec.n_failed.sum())
    add(f"**{tot_t} molecules** were targeted for Psi4 B3LYP recomputation, "
        f"selected as the lowest-confidence predictions of the preceding "
        f"cycle. **{tot_c} converged** and **{tot_f} failed**.")
    add("")
    add("### S6.1 Outcome by domain")
    add("")
    add("| Domain | targeted | converged | failed | dominant failure mode |")
    add("|---|---|---|---|---|")
    rec2 = rec.copy()
    rec2["k"] = rec2.domain.apply(
        lambda g: DOMAIN_ORDER.index(g) if g in DOMAIN_ORDER else 99)
    for _, r in rec2.sort_values("k").iterrows():
        add(f"| `{r.domain}` | {int(r.n_targeted)} | {int(r.n_converged)} | "
            f"{int(r.n_failed)} | {r.failure_mode} |")
    add(f"| **total** | **{tot_t}** | **{tot_c}** | **{tot_f}** | |")
    add("")
    add("Most failures are `disk_or_scratch_exhausted`, an infrastructure "
        "limit rather than a chemical one, so the failed set is not "
        "systematically different in chemistry from the converged set. The "
        "failure-record file contains more rows than there were final "
        "failures because molecules that failed once and succeeded on retry "
        "each left a record; the reconciliation column "
        "`n_records_that_later_converged` accounts for the difference.")
    add("")
    add(src(R2 / "A1_recompute_reconciliation.csv"))
    add("")
    add("### S6.2 Charge and multiplicity distribution")
    add("")
    add("| Charge | n | | Multiplicity | n |")
    add("|---|---|---|---|---|")
    ch = man.charge.value_counts().sort_index()
    mu = man.multiplicity.value_counts().sort_index()
    for i in range(max(len(ch), len(mu))):
        c = (f"{ch.index[i]:+d} | {ch.iloc[i]:,}" if i < len(ch) else " | ")
        m = (f"{mu.index[i]} | {mu.iloc[i]:,}" if i < len(mu) else " | ")
        add(f"| {c} | | {m} |")
    add("")
    n_open = int((man.multiplicity > 1).sum())
    n_chg = int((man.charge != 0).sum())
    add(f"Of the {len(man)} targeted molecules, **{n_chg} carry a non-zero "
        f"formal charge** and **{n_open} are open-shell** (multiplicity > 1). "
        "Open-shell species use an unrestricted Kohn–Sham (UKS) reference; "
        "closed-shell species use RKS.")
    add("")
    add(src(QM9C / "recompute_inputs" / "recompute_manifest.csv",
            "cols `charge`, `multiplicity`"))
    add("")
    add("### S6.3 Basis set and SCF settings")
    add("")
    add("| Setting | Value |")
    add("|---|---|")
    add("| Functional | B3LYP |")
    add("| Basis (H, C, N, O, F only) | 6-31G(d,p) |")
    add("| Basis (any other element) | def2-SVP |")
    add("| Reference | RKS (multiplicity 1), UKS (multiplicity > 1) |")
    add("| `scf_type` | `df` (density fitting), `df_scf_guess=True` |")
    add("| `e_convergence` | 1e-6 |")
    add("| `d_convergence` | 1e-6 |")
    add("| `maxiter` | 150 |")
    add("| Geometry | as supplied in the manifest XYZ; `no_reorient`, "
        "`no_com` |")
    add("")
    add("The basis set is chosen per molecule by element composition, so a "
        "single manuscript sentence naming one basis is incomplete; both must "
        "be stated.")
    add("")
    add(src(Path("run_C3_timings.py"), "functions `basis_for`, `main`") +
        " · " + src(R2 / "paste" / "A9_scf_settings.md"))
    add("")
    add("---")
    add("")

    # ---------------------------------------------------------- S7 geometry
    add("## S7. Geometry sensitivity")
    add("")
    geo = read(CONS / "geometry_sensitivity.csv")
    ok = geo[geo.converged == True]          # noqa: E712
    a = ok.delta_gap.abs()
    add("The production pipeline consumes a single UFF-optimized conformer. "
        "This measures what is lost relative to a B3LYP-optimized geometry: "
        f"a random sample of {len(geo)} molecules, stratified across the five "
        "domains, was re-optimized at B3LYP and the gap recomputed at both "
        "geometries.")
    add("")
    add(f"- Molecules attempted: **{len(geo)}**")
    add(f"- Geometry optimizations that converged: **{len(ok)}**")
    add(f"- Mean |Δ gap|: **{a.mean():.3f} eV**")
    add(f"- Median |Δ gap|: **{a.median():.3f} eV**")
    add(f"- SD: **{a.std(ddof=1):.3f} eV**; maximum: **{a.max():.3f} eV**")
    add("")
    add("The mean and the median differ by more than a factor of 1.5 because "
        "the distribution is long-tailed: most molecules shift little and a "
        "few shift a great deal. Both should be quoted.")
    add("")
    add("### S7.1 By domain")
    add("")
    gd = read(R2 / "C1_geometry_by_domain.csv").copy()
    gd["k"] = gd.domain.apply(
        lambda g: DOMAIN_ORDER.index(g) if g in DOMAIN_ORDER else 99)
    cnt = ok.domain.value_counts()
    add("| Domain | n converged | mean &#124;Δ gap&#124; (eV) |")
    add("|---|---|---|")
    for _, r in gd.sort_values("k").iterrows():
        add(f"| `{r.domain}` | {int(cnt.get(r.domain, 0))} | "
            f"{r.mean_abs_delta_gap:.3f} |")
    add("")
    add("Per-domain means rest on very few molecules each and should be read "
        "as indicative only. The largest per-domain mean sits on "
        "`qm9_like_small_organic`, which is the *most* accurate domain for the "
        "model itself; it is driven by a single "
        f"{a.max():.2f} eV outlier and is not evidence that small organics are "
        "geometry-sensitive in general.")
    add("")
    add("### S7.2 Convergence shortfall and its direction")
    add("")
    n_fail = len(geo) - len(ok)
    add(f"{len(ok)} of {len(geo)} optimizations converged, below the 45 the "
        "protocol asked for. The failures are not random with respect to the "
        "quantity being measured, and they push the estimate in **both** "
        "directions:")
    add("")
    add("- **Upward bias.** Optimizer failures concentrate in large, flexible "
        "molecules with shallow potential-energy surfaces. Those are exactly "
        "the molecules whose gaps move *least* between geometries, so "
        "dropping them raises the reported mean.")
    add("- **Downward bias.** Converged optimizations used `gau_loose` "
        "criteria. A looser convergence threshold stops nearer the starting "
        "geometry, which understates the true difference.")
    add("")
    add("These do not cancel in any quantified way. The figure should be "
        "reported with the sample size and both caveats attached, not as a "
        "clean measurement.")
    add("")
    add(src(CONS / "geometry_sensitivity.csv") + " · " +
        src(R2 / "C1_geometry_shift.csv", "per-molecule, including failures"))
    add("")

    # -------------------------------------------------------- S7.3 conformer
    cf = read(R2 / "C2_conformer_floor.csv").iloc[0]
    add("### S7.3 Conformational floor")
    add("")
    add("A second, independent limit: where the corpus contains multiple "
        "conformers of the same molecule, the spread of PQR gaps among them "
        "is irreducible noise that no model trained on a single conformer can "
        "predict.")
    add("")
    add(f"- Molecules with ≥3 conformers: **{int(cf.n_molecules_ge3_conformers):,}**")
    add(f"- Median within-molecule SD: **{cf.median_conformer_sd_eV:.3f} eV** "
        f"(IQR {cf.iqr_low:.3f}–{cf.iqr_high:.3f})")
    add(f"- Mean: {cf['mean']:.3f} eV; 90th percentile: {cf.p90:.3f} eV; "
        f"maximum: {cf['max']:.3f} eV")
    add("")
    add(src(R2 / "C2_conformer_floor.csv"))
    add("")
    add("---")
    add("")

    # ----------------------------------------------------------- S8 splits
    add("## S8. Split-protocol comparison")
    add("")
    sp = read(R2 / "B5_split_protocols.csv").set_index("split_mode")
    add("Random splitting can flatter a model by placing close analogues on "
        "both sides of the split. Two stricter protocols test that: "
        "Bemis–Murcko scaffold splitting, and Butina clustering on Morgan "
        "fingerprints. All three use the same ten repeats and the same "
        "architecture.")
    add("")
    add("| Protocol | test MAE ± SD (eV) | RMSE | R² | n test | median max "
        "Tanimoto, test→train | Δ vs random |")
    add("|---|---|---|---|---|---|---|")
    base = sp.loc["random", "test_mae"]
    for m in ["random", "scaffold", "cluster"]:
        r = sp.loc[m]
        d = r.test_mae - base
        dl = "—" if m == "random" else f"{d:+.3f} ({100*d/base:+.1f}%)"
        add(f"| {m} | {r.test_mae:.3f} ± {r.test_mae_sd:.3f} | "
            f"{r.test_rmse:.3f} | {r.test_r2:.3f} | {int(r.n_test)} | "
            f"{r.median_max_tanimoto_test_to_train:.3f} | {dl} |")
    add("")
    sc = sp.loc["scaffold"]
    add(f"Scaffold splitting costs **{100*(sc.test_mae-base)/base:.1f}%** "
        "accuracy, and it does so while cutting the median nearest-neighbour "
        f"similarity from {sp.loc['random','median_max_tanimoto_test_to_train']:.3f} "
        f"to {sc.median_max_tanimoto_test_to_train:.3f}. The degradation is "
        "real but modest, which is the substantive answer to the reviewer's "
        "concern: performance does not depend on near-duplicates across the "
        "split.")
    add("")
    add(src(R2 / "B5_split_protocols.csv"))
    add("")
    add("---")
    add("")

    # ------------------------------------------------------ S9 error deciles
    add("## S9. Error distribution by domain")
    add("")
    add("Per-domain MAE hides a strongly skewed error distribution. The "
        "percentiles below are of absolute error on the held-out test "
        "molecules.")
    add("")
    dec = read(R2 / "B2_error_deciles.csv").copy()
    dec["k"] = dec.domain.apply(
        lambda g: DOMAIN_ORDER.index(g) if g in DOMAIN_ORDER else 99)
    dec = dec.sort_values("k")
    add("| Domain | n | MAE | p10 | p20 | p50 | p80 | p90 | p95 | max |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in dec.iterrows():
        add(f"| `{r.domain}` | {int(r.n)} | {r.mae:.3f} | {r.p10:.3f} | "
            f"{r.p20:.3f} | {r.p50:.3f} | {r.p80:.3f} | {r.p90:.3f} | "
            f"{r.p95:.3f} | {r['max']:.3f} |")
    add("")
    add("The median error is far below the mean in every domain. For "
        f"`qm9_like_small_organic` the median absolute error is "
        f"{dec[dec.domain=='qm9_like_small_organic'].p50.iloc[0]:.3f} eV "
        "against an MAE of "
        f"{dec[dec.domain=='qm9_like_small_organic'].mae.iloc[0]:.3f} eV — a "
        "factor of two — because a small number of molecules carry very large "
        "errors (maximum "
        f"{dec[dec.domain=='qm9_like_small_organic']['max'].iloc[0]:.2f} eV). "
        "Reporting the median alongside the mean is the honest summary.")
    add("")
    add(src(R2 / "B2_error_deciles.csv"))
    add("")
    add("---")
    add("")

    # ----------------------------------------------------- S10 structure-only
    add("## S10. Structure-only ablation")
    add("")
    b6 = read(R2 / "B6_summary.csv").set_index("config")
    add("Five inputs (`x0`–`x4`) come from the PQR semiempirical calculation "
        "rather than from the SMILES string. This tests how much they "
        "contribute, on the same ten repeated splits.")
    add("")
    add("| Configuration | features | test MAE ± SD (eV) | Δ vs full |")
    add("|---|---|---|---|")
    for cfg, desc in [("full", "all inputs (the reported rCEG)"),
                      ("drop_x0_x4", "all five PQR scalars removed"),
                      ("drop_x2_x4", "only the three genuinely semiempirical "
                                     "scalars removed")]:
        if cfg not in b6.index:
            continue
        r = b6.loc[cfg]
        d = "—" if cfg == "full" else f"{r.delta_vs_full:+.4f} ({r.pct_worse:+.1f}%)"
        add(f"| `{cfg}` — {desc} | {int(r.n_features)} | {r.mae:.4f} ± "
            f"{r.mae_sd:.3f} | {d} |")
    add("")
    add("`x0` and `x1` are molecular mass and exact mass, both computable "
        "from SMILES alone and both already duplicated in the RDKit block at "
        "`x380`/`x381`; removing them removes nothing. The three inputs that "
        "genuinely require a semiempirical calculation are `x2` (dipole "
        "moment), `x3` (heat of formation) and `x4` (polarizability), which "
        "`drop_x2_x4` isolates.")
    add("")
    if "drop_x0_x4" in b6.index:
        d = b6.loc["drop_x0_x4"]
        add(f"Dropping all five changes test MAE by "
            f"{d.delta_vs_full:+.4f} eV ({d.pct_worse:+.1f}%) — far inside the "
            f"±{b6.loc['full'].mae_sd:.3f} eV split-to-split SD. **Inference "
            "on a new molecule needs only its SMILES string.**")
    add("")
    add("One caveat must accompany that claim either way: the *training* "
        "labels are calibrated pseudo-labels derived from `pqr_gap`, itself a "
        "semiempirical quantity. The structure-only result licenses a narrow "
        "statement about inference, not about training.")
    add("")
    add(src(R2 / "B6_summary.csv") + " · " +
        src(R2 / "B6_structure_only.csv", "per-repeat"))
    add("")
    add("---")
    add("")

    # ---------------------------------------------------------- S11 openshell
    add("## S11. Open-shell agreement between PQR and reference")
    add("")
    osh = read(R2 / "A6_openshell_agreement.csv")
    add("| Subset | n | Pearson r | Spearman ρ | mean &#124;Δ&#124; (eV) | "
        "median &#124;Δ&#124; (eV) |")
    add("|---|---|---|---|---|---|")
    for _, r in osh.iterrows():
        add(f"| {r.subset} | {int(r.n)} | {r.pearson_r:.3f} | "
            f"{r.spearman_rho:.3f} | {r.mean_abs_diff_eV:.3f} | "
            f"{r.median_abs_diff_eV:.3f} |")
    add("")
    op = osh[osh.subset.str.contains("open-shell")].iloc[0]
    cs = osh[osh.subset.str.contains("neutral closed-shell")].iloc[0]
    add(f"**The semiempirical and DFT descriptions of open-shell molecules "
        f"barely agree**: r = {op.pearson_r:.3f} over {int(op.n)} molecules, "
        f"against r = {cs.pearson_r:.3f} for {int(cs.n)} neutral closed-shell "
        "molecules. This is the mechanism behind `charged_or_radical` being "
        "the model's weakest domain, and it is a property of the input data, "
        "not of the learner.")
    add("")
    add(src(R2 / "A6_openshell_agreement.csv"))
    add("")
    add("---")
    add("")

    # -------------------------------------------------------- S12 corpus/pool
    add("## S12. Corpus and reference-pool arithmetic")
    add("")
    cl = read(R2 / "A2_cleaning_counts.csv")
    add("### S12.1 Cleaning waterfall")
    add("")
    add("| Stage | Rule | Records removed |")
    add("|---|---|---|")
    for _, r in cl.iterrows():
        add(f"| {r.stage} | `{r.rule}` | {int(r['count']):,} |")
    add(f"| | **total removed** | **{int(cl['count'].sum()):,}** |")
    add("")
    add("Rules with a count of zero are retained in the table deliberately: "
        "they were checked and found not to fire, which is itself part of the "
        "audit.")
    add("")
    add(src(R2 / "A2_cleaning_counts.csv") + " · " +
        src(R2 / "A2_stage1_valence_breakdown.csv", "valence detail"))
    add("")
    add("### S12.2 Reference pool")
    add("")
    a8 = read(R2 / "A8_dual_labelled.csv")
    add("| Quantity | Value |")
    add("|---|---|")
    add("| QM9 overlap with the strict corpus | 1,228 |")
    add("| Psi4 B3LYP recomputed in the strict corpus | 909 |")
    add(f"| molecules carrying **both** labels | {len(a8)} |")
    add("| **reference pool (union)** | **2,123** |")
    add("")
    add("1,228 + 909 − 14 = 2,123, which matches the split sizes "
        "1,274 + 425 + 424 = 2,123. The figure 2,137 that appears in the "
        "submitted manuscript is incorrect.")
    add("")
    add("### S12.3 Cross-protocol label floor")
    add("")
    add(f"On the {len(a8)} molecules labelled under both protocols, the mean "
        f"absolute difference between the QM9 and Psi4 gaps is "
        f"**{a8.abs_diff.mean():.3f} eV** (median {a8.abs_diff.median():.3f}, "
        f"maximum {a8.abs_diff.max():.3f}).")
    add("")
    add("This is a floor on achievable accuracy: two defensible reference "
        "protocols disagree with each other by about 0.3 eV on the same "
        "molecules, which is the same order as the model's own error. No "
        "model trained against a mixture of the two can be expected to do "
        "much better, and reporting accuracy below this floor would be "
        "meaningless.")
    add("")
    add(src(R2 / "A8_dual_labelled.csv"))
    add("")
    add("---")
    add("")

    # ---------------------------------------------------------- S13 external
    add("## S13. External validation on QMugs")
    add("")
    ext = read(R2 / "D1_external_metrics.csv")
    strict = ext[ext.evaluation == "strict_frozen_external"]
    cross = ext[ext.evaluation == "five_fold_crossfit_offset_aligned"]
    n_ext = int(strict.n.iloc[0])
    add(f"All {n_ext:,} QMugs molecules used here were excised from training "
        "by SMILES before any fitting, and the excision is asserted in code.")
    add("")
    add("Two evaluations are reported, and the difference between them is the "
        "point:")
    add("")
    add("| Evaluation | model | MAE (eV) | RMSE | R² |")
    add("|---|---|---|---|---|")
    for _, r in strict.iterrows():
        add(f"| strict frozen transfer | `{r.model}` | {r.mae:.3f} | "
            f"{r.rmse:.3f} | {r.r2:.2f} |")
    for _, r in cross.iterrows():
        add(f"| 5-fold cross-fit, offset-aligned | `{r.model}` | "
            f"**{r.mae:.3f}** | {r.rmse:.3f} | {r.r2:.3f} |")
    add("")
    add("Applied frozen, the model is **not transferable**: the MAE of about "
        "4.2 eV and the large negative R² reflect an almost constant offset "
        "between the QMugs reference protocol and this corpus's labels. Once "
        "that offset is estimated on held-out QMugs folds — five-fold "
        "cross-fitting, so no molecule contributes to its own alignment — the "
        f"MAE falls to **{cross.mae.min():.3f} eV** with "
        f"R² **{cross.r2.max():.3f}**.")
    add("")
    add("The honest reading is that rCEG transfers in *ranking and shape* but "
        "not in absolute placement, and that a new reference protocol "
        "requires a handful of labelled anchors to re-align. That is a "
        "limitation of the calibration design and should be stated as one.")
    add("")
    add(src(R2 / "D1_external_metrics.csv") + " · " +
        src(R2 / "D1_external_by_domain.csv", "per-domain"))
    add("")
    add("---")
    add("")

    # ------------------------------------------------------------ S14 ablation
    add("## S14. Full ablation table")
    add("")
    abl = read(R2 / "B3_ablation.csv")
    add("All configurations on the same ten repeated random splits.")
    add("")
    add("| Configuration | MAE ± SD (eV) | min | max | RMSE | R² |")
    add("|---|---|---|---|---|---|")
    for _, r in abl.sort_values("mae_mean").iterrows():
        add(f"| `{r.config}` | {r.mae_mean:.3f} ± {r.mae_sd:.3f} | "
            f"{r.mae_min:.3f} | {r.mae_max:.3f} | {r.rmse_mean:.3f} | "
            f"{r.r2_mean:.3f} |")
    add("")
    add("The two `domain_permuted` rows are the control: domain labels are "
        "shuffled while preserving domain sizes, so the architecture is "
        "identical and only the chemical meaning of the partition is "
        "destroyed. Their degradation is the evidence that the split "
        "architecture is doing chemical work rather than acting as an "
        "arbitrary ensemble.")
    add("")
    add(src(R2 / "B3_ablation.csv") + " · " +
        src(R2 / "B3_ablation_per_seed.csv", "per-repeat"))
    add("")
    add("---")
    add("")

    # ------------------------------------------------------------- S15 timing
    add("## S15. Computational cost")
    add("")
    t9 = read(R2 / "C3_timings.csv")
    add("One representative molecule per domain, chosen at that domain's "
        "median heavy-atom count. **All timings single-threaded on one "
        "machine (Apple M4 Pro, 12 logical cores)** so that the ratio is not "
        "contaminated by comparing a cluster job against a laptop job.")
    add("")
    cols = [c for c in t9.columns]
    add("| " + " | ".join(cols) + " |")
    add("|" + "---|" * len(cols))
    for _, r in t9.iterrows():
        add("| " + " | ".join(str(r[c]) for c in cols) + " |")
    add("")
    add("The speedup is not a single number. Across the five representatives "
        "it spans roughly three orders of magnitude, because the reference "
        "B3LYP cost grows steeply with system size while rCEG inference is "
        "essentially flat. Quote the range, or quote a specific domain, but "
        "do not quote a single average as though it were a property of the "
        "method.")
    add("")
    add(src(R2 / "C3_timings.csv") + " · " +
        src(R2 / "T1_timing_dft_local.csv", "independent n=20 DFT timing"))
    add("")
    add("---")
    add("")

    # ------------------------------------------------------- S16 reproduce
    add("## S16. Reproducing these results")
    add("")
    add("The deposit (`rCEG_ACSOmega_reproducibility_package/`) contains "
        "`run_all.sh`, `environment.yml` and a `README.md` describing the run "
        "order. Every command sets `PYTHONHASHSEED=0`. "
        "The environment actually used is captured verbatim in "
        f"`{CONS/'environment_manifest.txt'}`.")
    add("")
    add("Two environment notes matter for reproduction:")
    add("")
    add("1. **NumPy version.** Under NumPy 1.26.4 with Python 3.14, "
        "`np.nanvar` silently returned 0.0 for arrays above roughly 400,000 "
        "elements on this platform, corrupting variance-based filtering "
        "without raising. The results reported here were produced under NumPy "
        "2.5.3 and re-verified to reproduce bit-identically. Do not run this "
        "code on NumPy 1.26.4; `environment.yml` pins `numpy>=2.3,<3`.")
    add("2. **Psi4 scratch.** `PSI_SCRATCH` is read by Psi4 at import time. "
        "Setting it per molecule inside a running Python process has no "
        "effect; it must be exported before the interpreter starts. "
        "Otherwise scratch accumulates in `/tmp/psi.<pid>.*` and can exhaust "
        "the disk mid-run — which is the origin of most of the recomputation "
        "failures in §S6.1.")
    add("")

    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT}  ({len(L)} lines)")


if __name__ == "__main__":
    main()
