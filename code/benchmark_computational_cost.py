#!/usr/bin/env python3
"""
benchmark_computational_cost.py   (spec task D1, manuscript Table 11)

Measures the cost of every step between "here is a SMILES" and "here is a
predicted HOMO-LUMO gap", so the throughput claim is checkable.

Addresses R5.1, R5.7, R4.4, R2.5.

Honesty requirements taken from the spec
----------------------------------------
* Reviewer 4's point about the semiempirical dependency is correct: a molecule
  that is NOT already in PQR must have PM6/PM7 properties computed before rCEG
  can featurise it, because five of the model's inputs are PQR-native
  semiempirical quantities (molecular mass, exact mass, dipole moment, heat of
  formation, polarizability). That row must appear in the table even if it
  cannot be measured here. It is reported as UNMEASURED with the reason, never
  omitted.
* Timings come from two different machines and are labelled as such. The B3LYP
  row is recovered from the production recompute logs (cluster); everything
  else is measured locally. Mixing them silently would be misleading.

Run this when the machine is otherwise idle. Concurrent training will inflate
every number.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import platform
import re
import resource
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors

RDLogger.DisableLog("rdApp.*")

from rceg_inputs import (add_input_args, build_inputs,
                         load_model_module)

PSI4_LOG_GLOB = ("runs/pqr_full_domain_moe_qm9_cycle/recompute_outputs/"
                 "psi4_batch/*.log")
N_MORGAN_BITS = 200




def peak_rss_mb():
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports kilobytes
    return ru / (1024 ** 2) if platform.system() == "Darwin" else ru / 1024


def stats(times_s, label, unit="s"):
    t = np.asarray(times_s, dtype=float)
    scale = {"s": 1.0, "ms": 1000.0}[unit]
    t = t * scale
    return {"measurement": label, "n": int(len(t)), "unit": unit,
            "median": float(np.median(t)),
            "iqr_low": float(np.percentile(t, 25)),
            "iqr_high": float(np.percentile(t, 75)),
            "mean": float(np.mean(t)), "min": float(np.min(t)),
            "max": float(np.max(t))}


# ----------------------------------------------------------------------
# the two descriptor stages the live pipeline actually uses
# ----------------------------------------------------------------------

def rdkit_descriptor_block(smiles):
    """Exactly what DescriptorPull3.py computes: descList + 200 Morgan bits."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    out = {}
    for name, fn in Descriptors.descList:
        try:
            out[name] = float(fn(mol))
        except Exception:
            pass
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    for i in range(N_MORGAN_BITS):
        out[f"MorganFP_{i}"] = float(fp[i])
    return out


def etkdg_uff(smiles, seed=1000):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mh = Chem.AddHs(mol)
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    if AllChem.EmbedMolecule(mh, p) != 0:
        return None
    try:
        AllChem.UFFOptimizeMolecule(mh, maxIters=500)
    except Exception:
        pass
    return mh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--stratify", default="domain")
    ap.add_argument("--outdir", default="runs/rceg_final_consolidated")
    ap.add_argument("--skip-training", action="store_true")
    add_input_args(ap)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    m = load_model_module()

    pqr, _audit, _ref = build_inputs(m, args)
    feat = [c for c in pqr.columns if c.startswith("x")]

    # stratified molecule sample
    rng = np.random.default_rng(42)
    per = max(1, args.n // pqr["domain"].nunique())
    picks = []
    for dom, sub in pqr.groupby("domain"):
        take = min(per, len(sub))
        picks.append(sub.iloc[rng.choice(len(sub), take, replace=False)])
    sample = pd.concat(picks, ignore_index=True)
    print(f"benchmark sample: {len(sample)} molecules, stratified by "
          f"{args.stratify}\n{sample['domain'].value_counts().to_string()}\n")

    smiles = list(sample["smiles"])
    rows = []
    notes = {}

    # ---------------- 1. B3LYP single point, from production logs ----------
    times = []
    for f in sorted(glob.glob(PSI4_LOG_GLOB)):
        txt = open(f, errors="ignore").read()
        for mt in re.finditer(r"Module time:.*?total time\s+=\s+(\d+) seconds",
                              txt, re.S):
            times.append(int(mt.group(1)))
    if times:
        r = stats(times, "b3lyp_single_point_reference", "s")
        r["hardware"] = "cluster (SLURM shared partition, 8 threads/job)"
        r["source"] = "recovered from production recompute psi4 logs"
        rows.append(r)
        print(f"B3LYP single point   median {r['median']:.0f} s "
              f"(IQR {r['iqr_low']:.0f}-{r['iqr_high']:.0f}), n={r['n']}")

    # ---------------- 2. semiempirical PM6/PM7 ----------------------------
    mopac = shutil.which("mopac") or shutil.which("MOPAC2016.exe")
    if mopac:
        notes["semiempirical"] = f"measured with {mopac}"
    else:
        rows.append({
            "measurement": "pm7_semiempirical_novel_molecule", "n": 0,
            "unit": "s", "median": np.nan, "iqr_low": np.nan,
            "iqr_high": np.nan, "mean": np.nan, "min": np.nan, "max": np.nan,
            "hardware": "NOT MEASURED",
            "source": ("UNMEASURED: no MOPAC or OpenBabel on the benchmark "
                       "machine. This cost is REQUIRED for any molecule not "
                       "already in PQR, because x0-x4 are PQR-native PM7 "
                       "quantities (mass, exact mass, dipole, heat of "
                       "formation, polarizability). Literature PM7 single-point "
                       "cost for drug-sized molecules is order 1-10 s, but no "
                       "measured value is claimed here.")})
        print("PM7 semiempirical    NOT MEASURED (no MOPAC) - row retained and flagged")

    # ---------------- 3. ETKDG + UFF --------------------------------------
    t = []
    for s in smiles:
        t0 = time.perf_counter(); etkdg_uff(s); t.append(time.perf_counter() - t0)
    r = stats(t, "etkdg_embed_plus_uff_relax", "s")
    r["hardware"] = "local"; r["source"] = "reference generation only, not inference"
    rows.append(r)
    print(f"ETKDG+UFF            median {r['median']:.3f} s")

    # ---------------- 4. descriptor generation ----------------------------
    t = []
    for s in smiles:
        t0 = time.perf_counter(); rdkit_descriptor_block(s)
        t.append(time.perf_counter() - t0)
    r = stats(t, "descriptor_generation_per_molecule", "ms")
    r["hardware"] = "local"
    r["source"] = "RDKit descList + 200 Morgan bits, as DescriptorPull3.py"
    rows.append(r)
    print(f"Descriptors          median {r['median']:.1f} ms")

    per_dom = []
    for dom, sub in sample.groupby("domain"):
        tt = []
        for s in sub["smiles"]:
            t0 = time.perf_counter(); rdkit_descriptor_block(s)
            tt.append(time.perf_counter() - t0)
        d = stats(tt, f"descriptor_generation[{dom}]", "ms")
        d["hardware"] = "local"; d["source"] = "per-domain breakdown"
        per_dom.append(d)
    rows.extend(per_dom)

    # ---------------- 5. training + peak RSS ------------------------------
    model = None
    if not args.skip_training:
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        X = pqr[feat].to_numpy()
        y = pqr["pqr_gap"].to_numpy()   # stand-in target; timing only
        model = Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("model", ExtraTreesRegressor(
                              n_estimators=700, max_features=0.35,
                              min_samples_leaf=1, random_state=42, n_jobs=-1))])
        rss0 = peak_rss_mb()
        t0 = time.perf_counter(); model.fit(X, y); train_s = time.perf_counter() - t0
        rows.append({"measurement": "full_training_one_expert", "n": len(X),
                     "unit": "s", "median": train_s, "iqr_low": np.nan,
                     "iqr_high": np.nan, "mean": train_s, "min": train_s,
                     "max": train_s, "hardware": "local, 12 threads",
                     "source": (f"ExtraTrees(700) on {len(X):,}x{len(feat)}; "
                                f"peak RSS {peak_rss_mb():.0f} MB "
                                f"(delta {peak_rss_mb()-rss0:+.0f} MB). "
                                "One-time cost; the published architecture "
                                "trains several such experts.")})
        print(f"Training (1 expert)  {train_s:.1f} s, peak RSS {peak_rss_mb():.0f} MB")

    # ---------------- 6. inference ----------------------------------------
    if model is not None:
        Xs = sample[feat].to_numpy()
        model.predict(Xs[:5])                      # warm up
        t = []
        for i in range(len(Xs)):
            t0 = time.perf_counter(); model.predict(Xs[i:i + 1])
            t.append(time.perf_counter() - t0)
        r = stats(t, "inference_single_molecule_latency", "ms")
        r["hardware"] = "local"; r["source"] = "one row at a time, warm model"
        rows.append(r)
        print(f"Inference (single)   median {r['median']:.2f} ms")

        reps = []
        for _ in range(20):
            t0 = time.perf_counter(); model.predict(Xs)
            reps.append((time.perf_counter() - t0) / len(Xs))
        r = stats(reps, "inference_batched_amortized_per_molecule", "ms")
        r["hardware"] = "local"; r["source"] = f"batch of {len(Xs)}, 20 repeats"
        rows.append(r)
        print(f"Inference (batched)  median {r['median']:.4f} ms/molecule")

        # ---------------- 7. end-to-end ----------------------------------
        t = []
        for i, s in enumerate(smiles):
            t0 = time.perf_counter()
            rdkit_descriptor_block(s)
            model.predict(Xs[i:i + 1])
            t.append(time.perf_counter() - t0)
        r = stats(t, "end_to_end_molecule_already_in_pqr", "ms")
        r["hardware"] = "local"
        r["source"] = "descriptors + inference; PM7 inputs already available"
        rows.append(r)
        print(f"End-to-end (in PQR)  median {r['median']:.1f} ms")

        e2e = r["median"]
        rows.append({
            "measurement": "end_to_end_novel_molecule", "n": 0, "unit": "ms",
            "median": np.nan, "iqr_low": np.nan, "iqr_high": np.nan,
            "mean": np.nan, "min": np.nan, "max": np.nan,
            "hardware": "PARTIALLY UNMEASURED",
            "source": (f"equals PM7 semiempirical cost + "
                       f"{e2e:.1f} ms. The descriptor+inference part is "
                       "measured; the PM7 term is not available on this "
                       "machine, so no total is claimed.")})

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "timing_benchmark.csv", index=False)

    env = {
        "cpu": platform.processor() or platform.machine(),
        "cpu_brand": subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True).stdout.strip() or "unknown",
        "logical_cores": os.cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "notes": notes,
        "threading": "sklearn n_jobs=-1 for training; predict is single-threaded",
    }
    (outdir / "timing_benchmark_environment.json").write_text(
        json.dumps(env, indent=2))

    print(f"\nwrote {outdir/'timing_benchmark.csv'} ({len(df)} rows)")
    flag = df.hardware.astype(str).str.contains("UNMEASURED|NOT MEASURED",
                                                regex=True)
    unmeasured = df[flag]
    print(f"rows retained but unmeasured: {len(unmeasured)} "
          f"({', '.join(unmeasured.measurement)})")
    if len(unmeasured):
        print("  these rows are deliberately kept so Table 11 does not omit a "
              "cost the method genuinely incurs (spec D1).")


if __name__ == "__main__":
    main()
