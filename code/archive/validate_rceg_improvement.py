#!/usr/bin/env python3
"""
validate_rceg_improvement.py

Final confirmation of the error reduction, at 10 repeats, with the per-domain
and per-source breakdown that shows where the gain comes from.

Compared on identical splits:
  baseline_current   the architecture as published: ExtraTrees(mf=0.35) over
                     the 73k pseudo-labelled corpus with real anchors substituted
  rceg_v2            ExtraTrees(mf=1.0, 2000 trees) fitted on the reference
                     anchors, selected on validation in rounds 2-3
  rceg_v2_plus_pseudo rceg_v2 averaged with a reliability-weighted pseudo-label
                     model, to see whether external transfer can be kept

Selection already happened on validation in earlier rounds. This run exists to
report the chosen configuration honestly at n=10 with dispersion.
"""

from __future__ import annotations

import argparse
import importlib.util
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
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


def et(seed, n=2000, mf=1.0, leaf=1):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("model", ExtraTreesRegressor(
                         n_estimators=n, max_features=mf, min_samples_leaf=leaf,
                         random_state=seed, n_jobs=-1))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="runs/rceg_final_consolidated")
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
    Xext, yext = external[feat].to_numpy(), external["ref_gap"].to_numpy()

    rows, per_mol = [], []

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

        cal_predict, *_ = m.fit_calibration_ensemble(ref_cal, ref_val, feat, rseed)
        pseudo = cal_predict(pqr)
        holdout = set(ref_val["smiles"]) | set(ref_test["smiles"])
        keep = ~pqr["smiles"].isin(holdout).to_numpy()
        pool = pqr.loc[keep]
        Xp, yp = pool[feat].to_numpy(), pseudo[keep].copy()
        anchors = dict(zip(ref_cal["smiles"], ref_cal["ref_gap"]))
        is_anch = pool["smiles"].isin(anchors).to_numpy()
        yp[is_anch] = pool.loc[is_anch, "smiles"].map(anchors).to_numpy()
        openshell = pd.to_numeric(pool["x54"], errors="coerce").fillna(0).to_numpy() > 0

        preds = {}

        base = et(rseed + 110, n=700, mf=0.35)
        base.fit(Xp, yp)
        preds["baseline_current"] = (base.predict(Xt), base.predict(Xext))

        v2 = et(rseed + 410, n=2000, mf=1.0)
        v2.fit(Xa, ya)
        preds["rceg_v2"] = (v2.predict(Xt), v2.predict(Xext))

        w = np.where(is_anch, 50.0, np.where(openshell, 0.0, 1.0))
        nz = w > 0
        pw = et(rseed + 440, n=400, mf=1.0)
        pw.fit(Xp[nz], yp[nz], model__sample_weight=w[nz])
        preds["rceg_v2_plus_pseudo"] = (
            0.5 * v2.predict(Xt) + 0.5 * pw.predict(Xt),
            0.5 * v2.predict(Xext) + 0.5 * pw.predict(Xext))

        for name, (pt, pe) in preds.items():
            off = np.median(yext - pe)
            rows.append({
                "config": name, "repeat": rep,
                "test_mae": mean_absolute_error(yt, pt),
                "test_rmse": float(np.sqrt(np.mean((pt - yt) ** 2))),
                "test_r2": r2_score(yt, pt),
                "test_bias": float(np.mean(pt - yt)),
                "ext_mae_offset": mean_absolute_error(yext, pe + off),
                "ext_r2_offset": r2_score(yext, pe + off),
            })
            d = ref_test[["smiles", "ref_gap", "domain"]].copy()
            if "ref_source" in ref_test.columns:
                d["ref_source"] = ref_test["ref_source"].values
            d["config"] = name; d["repeat"] = rep; d["pred"] = pt
            d["abs_err"] = np.abs(pt - yt); d["signed_err"] = pt - yt
            per_mol.append(d)
        print(f"repeat {rep}: " + "  ".join(
            f"{k}={mean_absolute_error(yt, v[0]):.4f}" for k, v in preds.items()))

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "improvement_validation_raw.csv", index=False)
    pm = pd.concat(per_mol, ignore_index=True)
    pm.to_csv(outdir / "improvement_validation_per_molecule.csv", index=False)

    summ = (df.groupby("config")
              .agg(test_mae=("test_mae", "mean"), test_mae_sd=("test_mae", "std"),
                   test_rmse=("test_rmse", "mean"), test_r2=("test_r2", "mean"),
                   bias=("test_bias", "mean"),
                   ext_mae_offset=("ext_mae_offset", "mean"),
                   ext_r2_offset=("ext_r2_offset", "mean"),
                   n=("test_mae", "size"))
              .reset_index().sort_values("test_mae"))
    summ.to_csv(outdir / "improvement_validation_summary.csv", index=False)

    print("\n" + "=" * 100)
    print(f"FINAL COMPARISON, {args.repeats} repeats, identical splits")
    print("=" * 100)
    print(summ.to_string(index=False))

    bl = summ[summ.config == "baseline_current"].iloc[0]
    for _, r in summ.iterrows():
        if r.config == "baseline_current":
            continue
        d = 100 * (bl.test_mae - r.test_mae) / bl.test_mae
        pooled = float(np.sqrt((bl.test_mae_sd ** 2 + r.test_mae_sd ** 2) / 2))
        sd_units = (bl.test_mae - r.test_mae) / pooled if pooled else float("nan")
        print(f"\n{r.config}: test MAE {bl.test_mae:.4f} -> {r.test_mae:.4f} "
              f"({d:+.1f}%, {sd_units:.2f} pooled SD)")

    print("\n" + "-" * 100)
    print("PER-DOMAIN AND PER-SOURCE (pooled over repeats)")
    print("-" * 100)
    for col in ["domain", "ref_source"]:
        if col not in pm.columns:
            continue
        t = (pm.groupby(["config", col])["abs_err"].mean().unstack(0))
        cols = [c for c in ["baseline_current", "rceg_v2", "rceg_v2_plus_pseudo"]
                if c in t.columns]
        t = t[cols]
        t["improvement_%"] = 100 * (t["baseline_current"] - t["rceg_v2"]) / t["baseline_current"]
        print(f"\n{col}:")
        print(t.round(4).to_string())

    b = (pm[pm.config.isin(["baseline_current", "rceg_v2"])]
         .groupby(["config", "domain"])["signed_err"].mean().unstack(0))
    print("\nmean signed error by domain (bias):")
    print(b.round(4).to_string())


if __name__ == "__main__":
    main()
