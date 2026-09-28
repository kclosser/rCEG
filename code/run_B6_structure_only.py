#!/usr/bin/env python3
"""
run_B6_structure_only.py   (revision-2 task B6)

Reviewer 4: "The model is not structure-only. Four scalar inputs come directly
from PQR ... Therefore rCEG cannot be applied to an arbitrary new molecule from
SMILES alone."

That objection is correct as stated. This measures its size.

Three configurations, evaluated on the SAME ten repeated splits:

  full             all 427 features (the reported rCEG)
  drop_x0_x4       x0-x4 removed, which is what the spec asks for
  drop_x2_x4       only x2, x3, x4 removed

Why the third configuration matters
-----------------------------------
x0 and x1 are molecular mass and exact mass. Both are computable from a SMILES
string with no quantum calculation at all, and both are ALREADY DUPLICATED in
the RDKit block at x380 (MolWt) and x381 (ExactMolWt). Dropping them therefore
removes nothing that is not still present. The only inputs that genuinely
require a semiempirical calculation are x2 (dipole moment), x3 (heat of
formation) and x4 (polarizability). `drop_x2_x4` is the honest test of
Reviewer 4's objection; `drop_x0_x4` is the spec's literal request.

A caveat that must be stated either way: the TRAINING LABELS are calibrated
pseudo-labels derived from pqr_gap, itself a semiempirical quantity. So even a
structure-only variant depends on PM7 during training. The claim a successful
result supports is narrow and specific: INFERENCE on a new molecule needs only
its SMILES.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from rceg_inputs import add_input_args, build_inputs, load_model_module

OUT = Path("runs/revision2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    add_input_args(ap)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    m = load_model_module()

    pqr, audit, ref = build_inputs(m, args)
    ext = m.load_reference(args.external_ref, "q")
    es = set(pqr.merge(ext, on="smiles", how="inner")["smiles"])
    pqr = pqr[~pqr.smiles.isin(es)].reset_index(drop=True)
    audit = audit[~audit.smiles.isin(es)].reset_index(drop=True)
    ref = ref[~ref.smiles.isin(es)].reset_index(drop=True)

    feat_all = m.leakage_filter(pqr, audit, 0.94, OUT)
    CONFIGS = {
        "full": feat_all,
        "drop_x0_x4": [c for c in feat_all if c not in
                       {"x0", "x1", "x2", "x3", "x4"}],
        "drop_x2_x4": [c for c in feat_all if c not in {"x2", "x3", "x4"}],
    }
    for k, v in CONFIGS.items():
        print(f"  {k:12s} {len(v)} features")

    ref_pqr = pqr.merge(ref, on="smiles", how="inner")
    groups = m.assign_split_groups(list(ref_pqr["smiles"]), "random",
                                   seed=args.seed)

    rows, per_mol = [], []
    for rep in range(args.repeats):
        rseed = args.seed + 1000 * rep
        ci, vi, ti = m.grouped_three_way_split(groups, rseed)
        cal, val, tst = (ref_pqr.iloc[ci].copy(), ref_pqr.iloc[vi].copy(),
                         ref_pqr.iloc[ti].copy())
        yt = tst["ref_gap"].to_numpy()

        # Calibration always uses the full feature set: it is a training-time
        # step, and restricting it would conflate two separate questions.
        cal_predict, *_ = m.fit_calibration_ensemble(cal, val, feat_all, rseed)
        pseudo = cal_predict(pqr)
        holdout = set(val["smiles"]) | set(tst["smiles"])
        keep = ~pqr["smiles"].isin(holdout).to_numpy()
        pool = pqr.loc[keep]
        y = pseudo[keep].copy()
        anch = dict(zip(cal["smiles"], cal["ref_gap"]))
        ia = pool["smiles"].isin(anch).to_numpy()
        y[ia] = pool.loc[ia, "smiles"].map(anch).to_numpy()

        for name, feats in CONFIGS.items():
            Xtr = pool[feats].to_numpy()
            Xt = tst[feats].to_numpy()
            dom_t, piece_t = m._train_domain_and_piecewise(
                Xtr, y, pool["domain"].to_numpy(), pool, feats,
                Xt, tst["domain"].to_numpy(), tst,
                np.full(len(tst), float(np.median(y))), rseed)
            pred = piece_t
            rows.append({"config": name, "repeat": rep, "seed": rseed,
                         "n_features": len(feats),
                         "test_mae": round(float(mean_absolute_error(yt, pred)), 10),
                         "test_rmse": round(float(np.sqrt(np.mean((pred - yt) ** 2))), 10),
                         "test_r2": round(float(r2_score(yt, pred)), 10)})
            d = tst[["smiles", "domain", "ref_gap"]].copy()
            d["config"] = name; d["repeat"] = rep
            d["abs_err"] = np.abs(pred - yt)
            per_mol.append(d)
        print(f"  repeat {rep}: " + "  ".join(
            f"{r['config']}={r['test_mae']:.4f}" for r in rows[-3:]), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "B6_structure_only.csv", index=False)
    pm = pd.concat(per_mol, ignore_index=True)

    summ = (df.groupby("config")
              .agg(n_features=("n_features", "first"),
                   mae=("test_mae", "mean"), mae_sd=("test_mae", "std"),
                   rmse=("test_rmse", "mean"), r2=("test_r2", "mean"))
              .reset_index())
    base = summ[summ.config == "full"].iloc[0]
    summ["delta_vs_full"] = summ.mae - base.mae
    summ["pct_worse"] = 100 * summ.delta_vs_full / base.mae
    summ.to_csv(OUT / "B6_summary.csv", index=False)

    print("\n" + "=" * 78)
    print(summ.to_string(index=False))

    # paired test against the full model
    p = df.pivot(index="repeat", columns="config", values="test_mae")
    from scipy import stats
    lines = ["## B6 — structure-only ablation\n"]
    lines.append("| configuration | features | MAE ± SD | Δ vs full | paired p |")
    lines.append("|---|---|---|---|---|")
    for _, r in summ.iterrows():
        if r.config == "full":
            pv = "—"
        else:
            pv = f"{stats.ttest_rel(p[r.config], p['full']).pvalue:.4f}"
        lines.append(f"| `{r.config}` | {int(r.n_features)} | "
                     f"{r.mae:.3f} ± {r.mae_sd:.3f} | "
                     f"{r.delta_vs_full:+.3f} ({r.pct_worse:+.1f}%) | {pv} |")
    dom = (pm.groupby(["config", "domain"]).abs_err.mean().unstack(0))
    lines.append("\n**Per-domain MAE**\n")
    lines.append(dom.round(4).to_markdown())
    (OUT / "paste").mkdir(exist_ok=True)
    (OUT / "paste" / "B6_structure_only.md").write_text("\n".join(lines) + "\n")
    print("\nper-domain:")
    print(dom.round(4).to_string())
    print(f"\nwrote {OUT/'B6_structure_only.csv'} and paste/B6_structure_only.md")


if __name__ == "__main__":
    main()
