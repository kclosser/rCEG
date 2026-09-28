#!/usr/bin/env python3
"""
improve_rceg_error.py

Search for genuine reductions in held-out reference MAE.

Discipline
----------
Every candidate is evaluated on the SAME repeated three-way splits of the
reference pool. Selection decisions are made on the VALIDATION partition;
the TEST partition is reported but never used to choose a configuration.
No candidate is allowed to see ref_gap for validation or test molecules at
any point during training.

Diagnosis this is built from
----------------------------
  qm9_overlap subset              MAE 0.229  (n=232)
  recomputed_pqr_reference subset MAE 0.580  (n=191)
  charged_or_radical mean signed error  +0.614 eV

The Psi4-recomputed reference sits on a systematically lower gap scale than
QM9, and the model over-predicts it. The cause is structural: 73k pseudo
labels distilled from pqr_gap outvote 1,275 real anchors by 58:1, so the
model barely fits the real reference at all.

Candidates
----------
  baseline            current architecture, unweighted
  anchor_weight_W     real anchors upweighted W-fold via sample_weight
  residual            base model on pseudo-labels + residual model fitted on
                      anchors only (delta-learning). The base model is trained
                      with PSEUDO labels for anchors too, so anchor residuals
                      are honest rather than memorised.
  residual_weight_W   both
  anchor_only         trained on the 1,275 anchors alone (diagnostic: are the
                      pseudo-labels helping or hurting?)
"""

from __future__ import annotations

import argparse
import importlib.util
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
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


def et_raw(seed, n=700):
    return ExtraTreesRegressor(n_estimators=n, max_features=0.35,
                               min_samples_leaf=1, random_state=seed, n_jobs=-1)


def make_et(seed, n=700):
    """Imputer + ExtraTrees, sample_weight-capable."""
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("model", et_raw(seed, n))])


def fit_weighted(pipe, X, y, w=None):
    if w is None:
        pipe.fit(X, y)
    else:
        pipe.fit(X, y, model__sample_weight=w)
    return pipe


def metrics(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    return {"mae": float(mean_absolute_error(y, p)),
            "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
            "r2": float(r2_score(y, p)),
            "mean_signed_error": float(np.mean(p - y))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-trees", type=int, default=700)
    ap.add_argument("--outdir", default="runs/rceg_improvement_search")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    m = load_model_module()

    with open(SCRATCH / "cache.pkl", "rb") as fh:
        cache = pickle.load(fh)
    pqr, audit, ref = cache["pqr"], cache["audit"], cache["ref"]

    # external excision, identical to the live pipeline
    ext = m.load_reference(
        "external_validation/qmugs/qmugs_external_reference.csv", "q")
    extset = set(pqr.merge(ext, on="smiles", how="inner")["smiles"])
    pqr = pqr[~pqr.smiles.isin(extset)].reset_index(drop=True)
    audit = audit[~audit.smiles.isin(extset)].reset_index(drop=True)
    ref = ref[~ref.smiles.isin(extset)].reset_index(drop=True)

    feat = m.leakage_filter(pqr, audit, 0.94, outdir)
    ref_pqr = pqr.merge(ref, on="smiles", how="inner")
    print(f"\ncorpus={len(pqr):,}  reference pool={len(ref_pqr):,}  "
          f"features={len(feat)}")

    groups = m.assign_split_groups(list(ref_pqr["smiles"]), "random",
                                   seed=args.seed)
    rows = []

    for rep in range(args.repeats):
        t0 = time.time()
        rseed = args.seed + 1000 * rep
        cal_i, val_i, test_i = m.grouped_three_way_split(groups, rseed)
        ref_cal = ref_pqr.iloc[cal_i].copy()
        ref_val = ref_pqr.iloc[val_i].copy()
        ref_test = ref_pqr.iloc[test_i].copy()
        print(f"\n=== repeat {rep}  seed={rseed}  "
              f"cal={len(ref_cal)} val={len(ref_val)} test={len(ref_test)} ===")

        cal_predict, *_ = m.fit_calibration_ensemble(
            ref_cal, ref_val, feat, rseed)

        pseudo = cal_predict(pqr)

        holdout = set(ref_val["smiles"]) | set(ref_test["smiles"])
        keep = ~pqr["smiles"].isin(holdout).to_numpy()
        train_pool = pqr.loc[keep].copy()

        anchors = dict(zip(ref_cal["smiles"], ref_cal["ref_gap"]))
        is_anchor = train_pool["smiles"].isin(anchors).to_numpy()

        # y_mixed: real label where we have one (the current architecture)
        y_mixed = pseudo[keep].copy()
        y_mixed[is_anchor] = train_pool.loc[is_anchor, "smiles"].map(anchors).to_numpy()
        # y_pseudo_only: pseudo everywhere, so anchor residuals stay honest
        y_pseudo_only = pseudo[keep].copy()

        X = train_pool[feat].to_numpy()
        Xv = ref_val[feat].to_numpy()
        Xt = ref_test[feat].to_numpy()
        yv = ref_val["ref_gap"].to_numpy()
        yt = ref_test["ref_gap"].to_numpy()

        Xa = train_pool.loc[is_anchor, feat].to_numpy()
        ya = train_pool.loc[is_anchor, "smiles"].map(anchors).to_numpy()
        print(f"  anchors in training pool: {is_anchor.sum():,} of "
              f"{len(train_pool):,} ({100*is_anchor.mean():.2f}%)")

        def record(name, pv, pt, extra=""):
            mv, mt = metrics(yv, pv), metrics(yt, pt)
            rows.append({"config": name, "repeat": rep, "seed": rseed,
                         "val_mae": mv["mae"], "test_mae": mt["mae"],
                         "test_rmse": mt["rmse"], "test_r2": mt["r2"],
                         "test_mse_signed": mt["mean_signed_error"],
                         "notes": extra})
            print(f"    {name:24s} val={mv['mae']:.4f}  test={mt['mae']:.4f}"
                  f"  r2={mt['r2']:.4f}  bias={mt['mean_signed_error']:+.3f}")

        # ---------- baseline ----------
        base = fit_weighted(make_et(rseed + 110, args.n_trees), X, y_mixed)
        record("baseline", base.predict(Xv), base.predict(Xt))

        # ---------- anchor upweighting ----------
        for W in [5, 20, 50, 200]:
            w = np.ones(len(y_mixed)); w[is_anchor] = W
            mdl = fit_weighted(make_et(rseed + 110, args.n_trees), X, y_mixed, w)
            record(f"anchor_weight_{W}", mdl.predict(Xv), mdl.predict(Xt))

        # ---------- delta-learning ----------
        base_p = fit_weighted(make_et(rseed + 310, args.n_trees), X, y_pseudo_only)
        res_a = ya - base_p.predict(Xa)
        print(f"    anchor residual: mean={res_a.mean():+.3f} "
              f"sd={res_a.std():.3f} eV")

        for rn in [300, 700]:
            rm = fit_weighted(make_et(rseed + 410, rn), Xa, res_a)
            record(f"residual_et{rn}",
                   base_p.predict(Xv) + rm.predict(Xv),
                   base_p.predict(Xt) + rm.predict(Xt))

        rm_h = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("model", HistGradientBoostingRegressor(
                             max_iter=400, learning_rate=0.05,
                             loss="absolute_error", random_state=rseed + 420))])
        rm_h.fit(Xa, res_a)
        record("residual_hgb",
               base_p.predict(Xv) + rm_h.predict(Xv),
               base_p.predict(Xt) + rm_h.predict(Xt))

        # ---------- residual on top of the weighted model ----------
        w = np.ones(len(y_mixed)); w[is_anchor] = 20
        bw = fit_weighted(make_et(rseed + 510, args.n_trees), X, y_pseudo_only, w)
        res_bw = ya - bw.predict(Xa)
        rmw = fit_weighted(make_et(rseed + 520, 700), Xa, res_bw)
        record("residual_plus_weight20",
               bw.predict(Xv) + rmw.predict(Xv),
               bw.predict(Xt) + rmw.predict(Xt))

        # ---------- anchors only ----------
        ao = fit_weighted(make_et(rseed + 610, args.n_trees), Xa, ya)
        record("anchor_only", ao.predict(Xv), ao.predict(Xt))

        print(f"  repeat took {time.time()-t0:.0f}s")

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "improvement_search_raw.csv", index=False)

    summ = (df.groupby("config")
              .agg(val_mae_mean=("val_mae", "mean"),
                   test_mae_mean=("test_mae", "mean"),
                   test_mae_sd=("test_mae", "std"),
                   test_r2_mean=("test_r2", "mean"),
                   bias=("test_mse_signed", "mean"),
                   n=("test_mae", "size"))
              .reset_index()
              .sort_values("val_mae_mean"))
    summ.to_csv(outdir / "improvement_search_summary.csv", index=False)

    print("\n" + "=" * 78)
    print("RANKED BY VALIDATION MAE (selection metric; test shown but not used)")
    print("=" * 78)
    print(summ.to_string(index=False))

    best = summ.iloc[0]
    bl = summ[summ.config == "baseline"].iloc[0]
    print(f"\nbest by val: {best.config}")
    print(f"  val  {bl.val_mae_mean:.4f} -> {best.val_mae_mean:.4f} "
          f"({100*(bl.val_mae_mean-best.val_mae_mean)/bl.val_mae_mean:+.1f}%)")
    print(f"  test {bl.test_mae_mean:.4f} -> {best.test_mae_mean:.4f} "
          f"({100*(bl.test_mae_mean-best.test_mae_mean)/bl.test_mae_mean:+.1f}%)")


if __name__ == "__main__":
    main()
