#!/usr/bin/env python3
"""
run_C3_inference.py   (revision-2 task T3 / closes DISCREPANCIES §D2)

Measures rCEG inference latency for the manuscript's Table 9, and does it in a
way that can be regenerated from a clean checkout.

Why this script exists
----------------------
Table 9 reported 9.496 ms of inference for every one of its five rows. The
script that produced that figure was never preserved, so the headline speedup
could not be reproduced. Two other inference numbers also circulate --
0.40 ms and 38.3 ms in `timing_benchmark.csv` -- and they differ from each
other by a factor of ~95 because they measure different things. Any speedup
quoted without saying which was used is unreproducible.

This measures both definitions explicitly, on the same machine, using the
production architecture:

  single-molecule latency   one molecule at a time, model already warm.
                            This is the honest cost of answering a query.
  batched throughput        all molecules in one predict() call, divided by
                            count. This is the honest cost of screening a
                            library.

Descriptor generation is EXCLUDED from both and reported separately, because
it is a different stage with a different cost driver, and Table 9 already has
its own descriptor column.

Why the model is retrained here
-------------------------------
The frozen run was not invoked with --save-models, and that flag would not
have helped: it persists `sorted(domain_models.keys())`, the expert NAMES, not
the fitted estimators. So there is no serialized production model to load.
This script therefore rebuilds the Stage-2 experts through the model module's
own constructors (`et()`), on the same split and the same pseudo-labels the
production run used, so the timed object has the production structure --
500-tree ExtraTrees over the same 427-column matrix.

Usage
-----
    PYTHONHASHSEED=0 python3 run_C3_inference.py --cache runs/.../inputs_cache.pkl
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rceg_inputs import (add_input_args, build_inputs, excise_external,
                         load_model_module)

OUT = Path("runs/revision2")
TABLE9 = OUT / "C3_timings.csv"
REPS = OUT / "C3_representatives.csv"


def hardware():
    """Record the machine, so a reader can judge the absolute numbers."""
    info = {"platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "python": platform.python_version()}
    try:  # macOS
        info["cpu_model"] = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True, stderr=subprocess.DEVNULL).strip()
        info["logical_cores"] = int(subprocess.check_output(
            ["sysctl", "-n", "hw.logicalcpu"], text=True).strip())
        info["memory_gb"] = round(int(subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True).strip()) / 1e9, 1)
    except Exception:
        import os
        info["cpu_model"] = info["processor"]
        info["logical_cores"] = os.cpu_count()
    return info


def median_ms(fn, n, warmup=3):
    """Median wall time of fn() over n calls, after discarding warm-ups."""
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        t.append(time.perf_counter() - t0)
    a = np.array(t) * 1000.0
    return {"median_ms": float(np.median(a)), "mean_ms": float(a.mean()),
            "p10_ms": float(np.percentile(a, 10)),
            "p90_ms": float(np.percentile(a, 90)),
            "min_ms": float(a.min()), "n": int(n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeats", type=int, default=50,
                    help="timing repeats per molecule (spec minimum 20)")
    ap.add_argument("--threads", type=int, default=1,
                    help="n_jobs at predict time; 1 matches Table 9")
    add_input_args(ap)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    hw = hardware()
    print("hardware:", json.dumps(hw, indent=2))

    m = load_model_module()
    pqr, audit, ref = build_inputs(m, args)
    pqr, audit, ref, _ = excise_external(m, pqr, audit, ref, args.external_ref)

    feat = m.leakage_filter(pqr, audit, 0.94, OUT)
    ref_pqr = pqr.merge(ref, on="smiles", how="inner")
    groups = m.assign_split_groups(list(ref_pqr["smiles"]), "random",
                                   seed=args.seed)
    ci, vi, _ti = m.grouped_three_way_split(groups, args.seed)
    cal, val = ref_pqr.iloc[ci].copy(), ref_pqr.iloc[vi].copy()

    print(f"\ntraining Stage-2 experts on the production split "
          f"({len(feat)} features)...", flush=True)
    cal_predict, *_ = m.fit_calibration_ensemble(cal, val, feat, args.seed)
    pseudo = cal_predict(pqr)
    holdout = set(val["smiles"])
    keep = ~pqr["smiles"].isin(holdout).to_numpy()
    pool = pqr.loc[keep]
    y = pseudo[keep].copy()
    anch = dict(zip(cal["smiles"], cal["ref_gap"]))
    ia = pool["smiles"].isin(anch).to_numpy()
    y[ia] = pool.loc[ia, "smiles"].map(anch).to_numpy()

    # Production Stage 2: one ExtraTrees expert per domain with >= min_domain
    # training molecules. Same constructor and seeding as the model script.
    experts = {}
    doms = pool["domain"].to_numpy()
    Xpool = pool[feat].to_numpy()
    for domain in sorted(pd.unique(doms)):
        mask = doms == domain
        if int(mask.sum()) < 500:
            continue
        mod = m.et(args.seed + 200 + m.stable_seed(str(domain), 500), n=500)
        mod.fit(Xpool[mask], y[mask])
        if hasattr(mod.named_steps.get("model"), "n_jobs"):
            mod.named_steps["model"].n_jobs = args.threads
        experts[domain] = mod
        print(f"  {domain:26s} trained on {int(mask.sum()):,}", flush=True)

    # ---------------- the representatives Table 9 uses ----------------
    if not REPS.exists():
        raise SystemExit(f"missing {REPS}; run run_C3_timings.py first")
    reps = pd.read_csv(REPS)
    feats_by_smiles = pqr.set_index("smiles")

    rows = []
    for _, r in reps.iterrows():
        dom = r["domain"]
        if dom not in experts:
            print(f"  SKIP {dom}: no expert (below min_domain)")
            continue
        sub = pqr[pqr["domain"] == dom]
        if sub.empty:
            continue
        X1 = sub[feat].to_numpy()[:1]
        mod = experts[dom]
        single = median_ms(lambda: mod.predict(X1), args.repeats)
        rows.append({"domain": dom, "mol_id": r["mol_id"],
                     "n_heavy_atoms": int(r["n_heavy"]),
                     "mode": "single_molecule", **single})
        print(f"  {dom:26s} single {single['median_ms']:7.3f} ms", flush=True)

    # ---------------- batched: all five at once ----------------
    batch_rows = []
    for _, r in reps.iterrows():
        dom = r["domain"]
        if dom not in experts:
            continue
        sub = pqr[pqr["domain"] == dom]
        nb = min(len(sub), 1000)
        Xb = sub[feat].to_numpy()[:nb]
        mod = experts[dom]
        b = median_ms(lambda: mod.predict(Xb), max(5, args.repeats // 5))
        per = {k: (v / nb if k.endswith("_ms") else v) for k, v in b.items()}
        batch_rows.append({"domain": dom, "mol_id": r["mol_id"],
                           "n_heavy_atoms": int(r["n_heavy"]),
                           "mode": f"batched_{nb}", **per})
        print(f"  {dom:26s} batched/mol {per['median_ms']:7.4f} ms "
              f"(n={nb})", flush=True)

    df = pd.DataFrame(rows + batch_rows)
    df["threads"] = args.threads
    df["cpu_model"] = hw.get("cpu_model")
    df["logical_cores"] = hw.get("logical_cores")
    df.to_csv(OUT / "C3_inference_timings.csv", index=False)
    (OUT / "C3_inference_hardware.json").write_text(json.dumps(hw, indent=2))
    print(f"\nwrote {OUT/'C3_inference_timings.csv'}")

    # ---------------- reconcile against the tabulated 9.496 ms --------
    single_med = df[df["mode"] == "single_molecule"]["median_ms"]
    print("\n" + "=" * 70)
    print(f"single-molecule latency: median {single_med.median():.3f} ms "
          f"(range {single_med.min():.3f}-{single_med.max():.3f})")
    bt = df[df["mode"].str.startswith("batched")]["median_ms"]
    print(f"batched per molecule   : median {bt.median():.4f} ms "
          f"(range {bt.min():.4f}-{bt.max():.4f})")
    print(f"Table 9 currently says : 9.496 ms for every row")

    # ---------------- speedups under both definitions -----------------
    if TABLE9.exists():
        t9 = pd.read_csv(TABLE9).set_index("domain")
        out = []
        for _, r in df[df["mode"] == "single_molecule"].iterrows():
            d = r["domain"]
            if d not in t9.index:
                continue
            desc = float(t9.loc[d, "descriptors_ms"])
            b3 = float(t9.loc[d, "b3lyp_s"]) * 1000.0
            bmatch = df[(df.domain == d) & df["mode"].str.startswith("batched")]
            bms = float(bmatch["median_ms"].iloc[0]) if len(bmatch) else np.nan
            out.append({
                "domain": d, "b3lyp_s": b3 / 1000.0,
                "descriptors_ms": desc,
                "single_infer_ms": r["median_ms"],
                "batched_infer_ms": bms,
                "e2e_single_ms": desc + r["median_ms"],
                "e2e_batched_ms": desc + bms,
                "speedup_single": b3 / (desc + r["median_ms"]),
                "speedup_batched": b3 / (desc + bms),
            })
        sp = pd.DataFrame(out)
        sp.to_csv(OUT / "C3_speedups_both_definitions.csv", index=False)
        print("\n--- speedup under each definition ---")
        print(sp[["domain", "speedup_single", "speedup_batched"]]
              .round(0).to_string(index=False))
        print(f"\nsingle-molecule range : {sp.speedup_single.min():,.0f}x - "
              f"{sp.speedup_single.max():,.0f}x")
        print(f"batched range         : {sp.speedup_batched.min():,.0f}x - "
              f"{sp.speedup_batched.max():,.0f}x")
        print(f"wrote {OUT/'C3_speedups_both_definitions.csv'}")


if __name__ == "__main__":
    main()
