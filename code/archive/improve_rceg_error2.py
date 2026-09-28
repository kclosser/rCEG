#!/usr/bin/env python3
"""
improve_rceg_error2.py

Round 2 of the error-reduction search.

Round 1 established:
  * delta-learning (base on pseudo-labels + residual fitted on anchors)
    beats the baseline in 3/3 repeats, -3.9% MAE
  * anchor_only (1,275 real anchors, zero pseudo-labels) TIES it
  * upweighting anchors does nothing, because ExtraTrees already memorises
    them; the failure is generalisation, not fit

So the useful signal lives almost entirely in a 1,275-row problem. That is
cheap to fit, which means the anchor-level learner can be tuned properly --
it was previously inheriting hyperparameters chosen for a 73k-row problem.

Efficiency
----------
The expensive base model is fitted ONCE per repeat and its predictions are
cached; every anchor-level candidate is then a sub-second fit against those
cached predictions.

Discipline
----------
Selection is on VALIDATION mean MAE across repeats. Test is computed and
printed but never used to choose. Nothing sees val/test ref_gap in training.
"""

from __future__ import annotations

import argparse
import importlib.util
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesRegressor, RandomForestRegressor,
                              HistGradientBoostingRegressor,
                              GradientBoostingRegressor)
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

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


def scaled(model):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("scale", RobustScaler(quantile_range=(10, 90))),
                     ("model", model)])


def anchor_level_candidates(seed):
    """Learners for the ~1,275-row anchor problem."""
    c = {}
    for mf in [0.1, 0.2, 0.35, 0.6, 1.0]:
        for msl in [1, 2, 5]:
            c[f"et_mf{mf}_leaf{msl}"] = imp(ExtraTreesRegressor(
                n_estimators=800, max_features=mf, min_samples_leaf=msl,
                random_state=seed, n_jobs=-1))
    for mf in [0.2, 0.35, 0.6]:
        c[f"rf_mf{mf}"] = imp(RandomForestRegressor(
            n_estimators=600, max_features=mf, min_samples_leaf=1,
            random_state=seed, n_jobs=-1))
    for lr in [0.02, 0.05, 0.1]:
        for leaves in [7, 15, 31]:
            c[f"hgb_lr{lr}_lv{leaves}"] = imp(HistGradientBoostingRegressor(
                max_iter=800, learning_rate=lr, max_leaf_nodes=leaves,
                l2_regularization=0.1, early_stopping=True,
                validation_fraction=0.15, n_iter_no_change=60,
                loss="absolute_error", random_state=seed))
    c["gbr_huber"] = imp(GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=3,
        loss="huber", random_state=seed))
    c["ridge"] = scaled(RidgeCV(alphas=np.logspace(-4, 4, 40)))
    for a in [0.1, 1.0]:
        for g in [1e-4, 1e-3]:
            c[f"krr_a{a}_g{g}"] = scaled(KernelRidge(alpha=a, kernel="rbf", gamma=g))
    return c


def metrics(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    return (float(mean_absolute_error(y, p)),
            float(np.sqrt(np.mean((p - y) ** 2))),
            float(r2_score(y, p)),
            float(np.mean(p - y)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="runs/rceg_improvement_search")
    ap.add_argument("--base-cache", default=None)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    m = load_model_module()

    with open(SCRATCH / "cache.pkl", "rb") as fh:
        cache = pickle.load(fh)
    pqr, audit, ref = cache["pqr"], cache["audit"], cache["ref"]

    ext = m.load_reference(
        "external_validation/qmugs/qmugs_external_reference.csv", "q")
    extset = set(pqr.merge(ext, on="smiles", how="inner")["smiles"])
    pqr = pqr[~pqr.smiles.isin(extset)].reset_index(drop=True)
    audit = audit[~audit.smiles.isin(extset)].reset_index(drop=True)
    ref = ref[~ref.smiles.isin(extset)].reset_index(drop=True)

    feat = m.leakage_filter(pqr, audit, 0.94, outdir)
    ref_pqr = pqr.merge(ref, on="smiles", how="inner")
    groups = m.assign_split_groups(list(ref_pqr["smiles"]), "random", seed=args.seed)

    base_cache_path = SCRATCH / "base_preds.pkl"
    base_cache = {}
    if base_cache_path.exists():
        base_cache = pickle.load(open(base_cache_path, "rb"))
        print(f"loaded base-prediction cache with {len(base_cache)} repeats")

    rows = []

    for rep in range(args.repeats):
        rseed = args.seed + 1000 * rep
        cal_i, val_i, test_i = m.grouped_three_way_split(groups, rseed)
        ref_cal, ref_val, ref_test = (ref_pqr.iloc[cal_i].copy(),
                                      ref_pqr.iloc[val_i].copy(),
                                      ref_pqr.iloc[test_i].copy())
        ya = ref_cal["ref_gap"].to_numpy()
        yv = ref_val["ref_gap"].to_numpy()
        yt = ref_test["ref_gap"].to_numpy()
        Xa = ref_cal[feat].to_numpy()
        Xv = ref_val[feat].to_numpy()
        Xt = ref_test[feat].to_numpy()

        print(f"\n=== repeat {rep} seed={rseed} ===")

        if rep not in base_cache:
            t0 = time.time()
            cal_predict, *_ = m.fit_calibration_ensemble(ref_cal, ref_val, feat, rseed)
            pseudo = cal_predict(pqr)
            holdout = set(ref_val["smiles"]) | set(ref_test["smiles"])
            keep = ~pqr["smiles"].isin(holdout).to_numpy()
            train_pool = pqr.loc[keep]
            X = train_pool[feat].to_numpy()
            base = imp(ExtraTreesRegressor(n_estimators=700, max_features=0.35,
                                           min_samples_leaf=1,
                                           random_state=rseed + 310, n_jobs=-1))
            base.fit(X, pseudo[keep])
            base_cache[rep] = {"a": base.predict(Xa), "v": base.predict(Xv),
                               "t": base.predict(Xt)}
            pickle.dump(base_cache, open(base_cache_path, "wb"))
            print(f"  base model fitted and cached ({time.time()-t0:.0f}s)")

        ba, bv, bt = (base_cache[rep]["a"], base_cache[rep]["v"], base_cache[rep]["t"])
        res_a = ya - ba

        def rec(name, pv, pt):
            mv = metrics(yv, pv); mt = metrics(yt, pt)
            rows.append({"config": name, "repeat": rep,
                         "val_mae": mv[0], "test_mae": mt[0],
                         "test_rmse": mt[1], "test_r2": mt[2], "bias": mt[3]})

        cands = anchor_level_candidates(rseed + 410)
        t0 = time.time()
        for name, mdl in cands.items():
            # delta-learning form
            mdl.fit(Xa, res_a)
            rec(f"residual::{name}", bv + mdl.predict(Xv), bt + mdl.predict(Xt))
            # direct anchor-only form
            mdl2 = anchor_level_candidates(rseed + 410)[name]
            mdl2.fit(Xa, ya)
            rec(f"direct::{name}", mdl2.predict(Xv), mdl2.predict(Xt))
        print(f"  swept {2*len(cands)} anchor-level candidates in {time.time()-t0:.0f}s")

        # baseline for reference
        rec("baseline_pseudo_only", bv, bt)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "improvement_round2_raw.csv", index=False)

    summ = (df.groupby("config")
              .agg(val_mae=("val_mae", "mean"), val_sd=("val_mae", "std"),
                   test_mae=("test_mae", "mean"), test_sd=("test_mae", "std"),
                   test_r2=("test_r2", "mean"), bias=("bias", "mean"),
                   n=("val_mae", "size"))
              .reset_index().sort_values("val_mae"))
    summ.to_csv(outdir / "improvement_round2_summary.csv", index=False)

    print("\n" + "=" * 86)
    print("TOP 20 BY VALIDATION MAE (selection metric)")
    print("=" * 86)
    print(summ.head(20).to_string(index=False))

    bl = summ[summ.config == "baseline_pseudo_only"]
    if len(bl):
        bl = bl.iloc[0]; best = summ.iloc[0]
        print(f"\nbaseline (pseudo-only base): val={bl.val_mae:.4f} test={bl.test_mae:.4f}")
        print(f"best     ({best.config}): val={best.val_mae:.4f} test={best.test_mae:.4f}")
        print(f"test improvement: {100*(bl.test_mae-best.test_mae)/bl.test_mae:+.1f}%")


if __name__ == "__main__":
    main()
