#!/usr/bin/env python3
"""
improve_rceg_error3.py

Round 3. Two jobs.

1. Push in-distribution MAE below the round-2 winner (0.3174 val / 0.3193 test,
   ExtraTrees max_features=1.0 on anchors only) by refining the anchor-level
   learner, ensembling, and re-admitting the pseudo-labels ONLY where they
   carry signal.

2. Answer the question that decides whether the paper's architecture survives:
   does the 84k pseudo-label corpus buy anything on the EXTERNAL QMugs
   benchmark, even though it buys nothing in-distribution?

Why pseudo-labels get reweighted rather than deleted
----------------------------------------------------
Measured on the 909 Psi4-recomputed molecules:

    closed-shell (mult=1)  pqr_gap vs true B3LYP gap:  r = 0.690, offset 5.09 eV
    open-shell   (mult>1)  pqr_gap vs true B3LYP gap:  r = 0.074, offset 8.30 eV

PM7 reports ~9.2 eV for open-shell species whose true B3LYP frontier gap
averages 0.86 eV. The pseudo-label is not merely offset there, it is
uninformative. Training on it injects exactly the +0.61 eV bias seen in the
charged_or_radical domain. So pseudo-labels are weighted by a reliability
proxy computable from descriptors alone (radical-electron count), which is
what the paper's existing confidence machinery was already measuring but
never fed back into training.
"""

from __future__ import annotations

import argparse
import importlib.util
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

SCRATCH = Path("/private/tmp/claude-501/-Users-isaacwang-Downloads/"
               "4ed5c178-2e52-4633-a5ab-976fc101c95e/scratchpad")


def load_model_module():
    spec = importlib.util.spec_from_file_location(
        "rceg", "pqr_full_domain_moe_qmugs_external.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def imp(model):
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("model", model)])


def et(seed, n=800, mf=1.0, leaf=1, bootstrap=False, max_samples=None):
    return imp(ExtraTreesRegressor(
        n_estimators=n, max_features=mf, min_samples_leaf=leaf,
        bootstrap=bootstrap, max_samples=max_samples,
        random_state=seed, n_jobs=-1))


def mets(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    return (float(mean_absolute_error(y, p)),
            float(np.sqrt(np.mean((p - y) ** 2))),
            float(r2_score(y, p)), float(np.mean(p - y)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="runs/rceg_improvement_search")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    m = load_model_module()

    cache = pickle.load(open(SCRATCH / "cache.pkl", "rb"))
    pqr, audit, ref = cache["pqr"], cache["audit"], cache["ref"]

    extref = m.load_reference(
        "external_validation/qmugs/qmugs_external_reference.csv", "qmugs")
    external = pqr.merge(extref, on="smiles", how="inner")
    extset = set(external["smiles"])
    pqr = pqr[~pqr.smiles.isin(extset)].reset_index(drop=True)
    audit = audit[~audit.smiles.isin(extset)].reset_index(drop=True)
    ref = ref[~ref.smiles.isin(extset)].reset_index(drop=True)

    feat = m.leakage_filter(pqr, audit, 0.94, outdir)
    ref_pqr = pqr.merge(ref, on="smiles", how="inner")
    groups = m.assign_split_groups(list(ref_pqr["smiles"]), "random", seed=args.seed)

    Xext = external[feat].to_numpy()
    yext = external["ref_gap"].to_numpy()
    print(f"external QMugs holdout: {len(external):,} molecules")

    # radical-electron count is descriptor-derived, so usable as a
    # training-time reliability proxy without touching inference features
    rad_col = "x54"   # NumRadicalElectrons, resolved in task E4
    rows = []

    for rep in range(args.repeats):
        rseed = args.seed + 1000 * rep
        cal_i, val_i, test_i = m.grouped_three_way_split(groups, rseed)
        ref_cal, ref_val, ref_test = (ref_pqr.iloc[cal_i].copy(),
                                      ref_pqr.iloc[val_i].copy(),
                                      ref_pqr.iloc[test_i].copy())
        ya, yv, yt = (ref_cal["ref_gap"].to_numpy(), ref_val["ref_gap"].to_numpy(),
                      ref_test["ref_gap"].to_numpy())
        Xa, Xv, Xt = (ref_cal[feat].to_numpy(), ref_val[feat].to_numpy(),
                      ref_test[feat].to_numpy())
        print(f"\n=== repeat {rep} seed={rseed} ===")

        def rec(name, pv, pt, pe=None):
            mv, mt = mets(yv, pv), mets(yt, pt)
            r = {"config": name, "repeat": rep, "val_mae": mv[0],
                 "test_mae": mt[0], "test_rmse": mt[1], "test_r2": mt[2],
                 "test_bias": mt[3]}
            if pe is not None:
                me = mets(yext, pe)
                off = np.median(yext - pe)
                ma = mets(yext, pe + off)
                r.update({"ext_mae_raw": me[0], "ext_r2_raw": me[2],
                          "ext_mae_offset": ma[0], "ext_r2_offset": ma[2]})
            rows.append(r)
            return r

        # ---------- refine the anchor-level learner ----------
        best_local = None
        for n_est in [800, 2000]:
            for leaf in [1, 2]:
                mdl = et(rseed + 410, n=n_est, mf=1.0, leaf=leaf)
                mdl.fit(Xa, ya)
                r = rec(f"direct_et{n_est}_leaf{leaf}_mf1.0",
                        mdl.predict(Xv), mdl.predict(Xt), mdl.predict(Xext))
                if best_local is None or r["val_mae"] < best_local[0]:
                    best_local = (r["val_mae"], mdl)

        for ms in [0.7, 0.9]:
            mdl = et(rseed + 415, n=1200, mf=1.0, leaf=1,
                     bootstrap=True, max_samples=ms)
            mdl.fit(Xa, ya)
            rec(f"direct_et_bootstrap{ms}", mdl.predict(Xv), mdl.predict(Xt),
                mdl.predict(Xext))

        rfm = imp(RandomForestRegressor(n_estimators=800, max_features=1.0,
                                        min_samples_leaf=1,
                                        random_state=rseed + 420, n_jobs=-1))
        rfm.fit(Xa, ya)
        rec("direct_rf_mf1.0", rfm.predict(Xv), rfm.predict(Xt), rfm.predict(Xext))

        # ---------- feature selection ----------
        for k in [100, 200]:
            sel = Pipeline([("imp", SimpleImputer(strategy="median")),
                            ("sel", SelectKBest(mutual_info_regression, k=k)),
                            ("model", ExtraTreesRegressor(
                                n_estimators=800, max_features=1.0,
                                min_samples_leaf=1, random_state=rseed + 430,
                                n_jobs=-1))])
            sel.fit(Xa, ya)
            rec(f"direct_selk{k}", sel.predict(Xv), sel.predict(Xt), sel.predict(Xext))

        # ---------- ensemble of ET + RF ----------
        eb = best_local[1]
        rec("ensemble_et_rf",
            0.5 * eb.predict(Xv) + 0.5 * rfm.predict(Xv),
            0.5 * eb.predict(Xt) + 0.5 * rfm.predict(Xt),
            0.5 * eb.predict(Xext) + 0.5 * rfm.predict(Xext))

        # ---------- pseudo-labels, full and reliability-weighted ----------
        t0 = time.time()
        cal_predict, *_ = m.fit_calibration_ensemble(ref_cal, ref_val, feat, rseed)
        pseudo = cal_predict(pqr)
        holdout = set(ref_val["smiles"]) | set(ref_test["smiles"])
        keep = ~pqr["smiles"].isin(holdout).to_numpy()
        pool = pqr.loc[keep]
        Xp = pool[feat].to_numpy()
        yp = pseudo[keep].copy()
        anchors = dict(zip(ref_cal["smiles"], ref_cal["ref_gap"]))
        is_anch = pool["smiles"].isin(anchors).to_numpy()
        yp[is_anch] = pool.loc[is_anch, "smiles"].map(anchors).to_numpy()
        openshell = pd.to_numeric(pool[rad_col], errors="coerce").fillna(0).to_numpy() > 0
        print(f"  pool={len(pool):,} anchors={is_anch.sum():,} "
              f"open-shell={openshell.sum():,} ({time.time()-t0:.0f}s)")

        for tag, w in [
            ("pseudo_full_mf1.0", np.ones(len(yp))),
            ("pseudo_openshell_dropped", np.where(openshell & ~is_anch, 0.0, 1.0)),
            ("pseudo_reliability_w", np.where(is_anch, 50.0,
                                              np.where(openshell, 0.0, 1.0))),
        ]:
            mdl = et(rseed + 440, n=400, mf=1.0, leaf=1)
            nz = w > 0
            mdl.fit(Xp[nz], yp[nz], model__sample_weight=w[nz])
            rec(tag, mdl.predict(Xv), mdl.predict(Xt), mdl.predict(Xext))

        for r in rows[-12:]:
            if r["repeat"] == rep:
                print(f"    {r['config']:30s} val={r['val_mae']:.4f} "
                      f"test={r['test_mae']:.4f} "
                      f"extoff={r.get('ext_mae_offset', float('nan')):.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "improvement_round3_raw.csv", index=False)
    summ = (df.groupby("config")
              .agg(val_mae=("val_mae", "mean"), val_sd=("val_mae", "std"),
                   test_mae=("test_mae", "mean"), test_sd=("test_mae", "std"),
                   test_r2=("test_r2", "mean"), bias=("test_bias", "mean"),
                   ext_mae_raw=("ext_mae_raw", "mean"),
                   ext_mae_offset=("ext_mae_offset", "mean"),
                   ext_r2_offset=("ext_r2_offset", "mean"),
                   n=("val_mae", "size"))
              .reset_index().sort_values("val_mae"))
    summ.to_csv(outdir / "improvement_round3_summary.csv", index=False)

    print("\n" + "=" * 104)
    print("ROUND 3 -- ranked by validation MAE; external QMugs shown for the "
          "architecture question")
    print("=" * 104)
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
