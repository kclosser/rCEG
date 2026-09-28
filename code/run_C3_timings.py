#!/usr/bin/env python3
"""
run_C3_timings.py   (revision-2 task C3 / manuscript Table 9)

Per-domain representative-molecule timing. One molecule per chemical domain,
chosen at that domain's MEDIAN heavy-atom count, so the table reflects typical
rather than best or worst cases.

Measured per molecule:
  1. GFN2-xTB semiempirical single point  (proxy for the PM7 step; see note)
  2. RDKit descriptor generation           (descList + 200 Morgan bits)
  3. B3LYP single point in Psi4            (the reference calculation replaced)

rCEG inference is measured separately by run_C3_inference.py, because the psi4
conda environment has no scikit-learn. Both halves run on the same machine.

Repeats: 5 for the millisecond-scale measurements. **3 for B3LYP**, a documented
deviation from the spec's 5 — a single B3LYP call on the largest representative
takes minutes, its run-to-run variance is small, and T1 already provides n=20
for the same quantity.

Run with the psi4 environment:
    /Applications/anaconda3/envs/rceg_qc/bin/python run_C3_timings.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors

RDLogger.DisableLog("rdApp.*")

HARTREE_TO_EV = 27.211386245988
QM9_ATOMS = {"H", "C", "N", "O", "F"}
XYZ_DIR = Path("runs/pqr_full_domain_moe_qm9_cycle/recompute_inputs/xyz")
OUT = Path("runs/revision2")
N_FAST, N_DFT = 5, 3


def read_xyz(mol_id):
    p = XYZ_DIR / f"{mol_id}.xyz"
    if not p.exists():
        return None, None
    lines = p.read_text().splitlines()
    return lines[1], "\n".join(lines[2:]).rstrip()


def basis_for(block):
    els = {ln.split()[0] for ln in block.splitlines() if ln.strip()}
    return "6-31G(d,p)" if els <= QM9_ATOMS else "def2-SVP"


def rdkit_block(smiles):
    mol = Chem.MolFromSmiles(smiles)
    out = {}
    for name, fn in Descriptors.descList:
        try:
            out[name] = float(fn(mol))
        except Exception:
            pass
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    for i in range(200):
        out[f"MorganFP_{i}"] = float(fp[i])
    return out


def select_representatives(manifest_path, out_path):
    """
    Pick one molecule per chemical domain at that domain's MEDIAN heavy-atom
    count, so the table reflects typical rather than best or worst cases.

    IMPORTANT -- this is a RECONSTRUCTION of the selection criterion, not the
    original selection. The published Table 9 was produced from a file written
    ad hoc to /tmp, with no generating script preserved. This function applies
    the documented criterion but does NOT reproduce the original five
    molecules: it agrees on heavy-atom count for 4 of 5 domains but picks
    different molecules (many share the median count and the original
    tie-breaking is unknown), and for heteroatom_rich_non_qm9 it computes a
    median of 19 where the original used 18, implying the original median was
    taken over a different population.

    Consequently `C3_representatives.csv` -- the preserved record of the five
    molecules actually timed -- is authoritative, and main() prefers it when
    present. Use this function only to select a FRESH set, and if you do,
    re-time every row: the published timings belong to the old molecules. In
    particular the published charged_or_radical representative is an
    open-shell dication (charge +2, multiplicity 3) whose B3LYP cost of 256 s
    drives the headline 16,979x speedup; a neutral closed-shell pick at the
    same heavy-atom count is far cheaper and would lower that figure
    substantially.
    """
    man = pd.read_csv(manifest_path)
    rows = []
    for dom, sub in man.groupby("domain"):
        sub = sub.copy()
        heavy = []
        for smi in sub["smiles"].astype(str):
            mol = Chem.MolFromSmiles(smi)
            heavy.append(mol.GetNumHeavyAtoms() if mol else np.nan)
        sub["n_heavy"] = heavy
        sub = sub.dropna(subset=["n_heavy"])
        # only molecules whose geometry is available can be timed
        sub = sub[[(XYZ_DIR / f"{i}.xyz").exists() for i in sub["id"]]]
        if sub.empty:
            continue
        med = float(sub["n_heavy"].median())
        # closest to the median; ties broken by id so the choice is stable
        sub = sub.assign(_d=(sub["n_heavy"] - med).abs())
        pick = sub.sort_values(["_d", "id"]).iloc[0]
        rows.append({"domain": dom, "mol_id": pick["id"],
                     "n_heavy": int(pick["n_heavy"]),
                     "domain_median_heavy": med,
                     "charge": int(pick["charge"]),
                     "mult": int(pick["multiplicity"])})
    reps = pd.DataFrame(rows).sort_values("domain").reset_index(drop=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    reps.to_csv(out_path, index=False)
    print(f"selected {len(reps)} representatives -> {out_path}")
    return reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", default=str(OUT / "C3_representatives.csv"),
                    help="representative-molecule table; derived if absent")
    ap.add_argument("--manifest",
                    default="runs/pqr_full_domain_moe_qm9_cycle/"
                            "recompute_inputs/recompute_manifest.csv")
    args = ap.parse_args()

    if Path(args.reps).exists():
        reps = pd.read_csv(args.reps)
        print(f"using existing representatives from {args.reps}")
    else:
        reps = select_representatives(args.manifest, args.reps)
    xtb = shutil.which("xtb") or "/Applications/anaconda3/envs/rceg_qc/bin/xtb"
    ref = pd.read_csv("pqr_recomputed_reference.csv").set_index("id")
    rows = []

    for _, r in reps.iterrows():
        mid = r["mol_id"]
        smi = str(ref.loc[mid, "smiles"])
        comment, coords = read_xyz(mid)
        basis = basis_for(coords) if coords else "def2-SVP"
        charge, mult = int(r["charge"]), int(r["mult"])
        rec = {"domain": r["domain"], "mol_id": mid,
               "n_heavy_atoms": int(r["n_heavy"]), "charge": charge,
               "multiplicity": mult, "basis": basis}
        print(f"\n{r['domain']} — {mid} ({int(r['n_heavy'])} heavy, "
              f"charge {charge}, mult {mult}, {basis})", flush=True)

        # ---- 1. RDKit descriptors ----
        t = []
        for _ in range(N_FAST):
            t0 = time.perf_counter(); rdkit_block(smi); t.append(time.perf_counter() - t0)
        rec["descriptors_ms"] = round(float(np.median(t)) * 1000, 3)
        print(f"  descriptors      {rec['descriptors_ms']:.1f} ms", flush=True)

        # ---- 2. GFN2-xTB ----
        with tempfile.TemporaryDirectory() as td:
            mol = Chem.AddHs(Chem.MolFromSmiles(smi))
            ps = AllChem.ETKDGv3(); ps.randomSeed = 42
            AllChem.EmbedMolecule(mol, ps)
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
            conf = mol.GetConformer()
            xyzp = Path(td) / "m.xyz"
            xyzp.write_text(f"{mol.GetNumAtoms()}\n{mid}\n" + "\n".join(
                f"{a.GetSymbol()} {conf.GetAtomPosition(a.GetIdx()).x:.8f} "
                f"{conf.GetAtomPosition(a.GetIdx()).y:.8f} "
                f"{conf.GetAtomPosition(a.GetIdx()).z:.8f}"
                for a in mol.GetAtoms()) + "\n")
            t = []
            for _ in range(N_FAST):
                t0 = time.perf_counter()
                subprocess.run([xtb, str(xyzp), "--gfn", "2", "--chrg", str(charge),
                                "--sp"], cwd=td, capture_output=True, timeout=600,
                               env={**os.environ, "OMP_NUM_THREADS": "1"})
                t.append(time.perf_counter() - t0)
            rec["semiempirical_gfn2xtb_ms"] = round(float(np.median(t)) * 1000, 3)
        print(f"  GFN2-xTB         {rec['semiempirical_gfn2xtb_ms']:.0f} ms", flush=True)

        # ---- 3. B3LYP single point ----
        import psi4
        t, ok = [], True
        for k in range(N_DFT):
            sdir = Path(f"/tmp/c3_psi4_{mid}_{k}")
            shutil.rmtree(sdir, ignore_errors=True); sdir.mkdir(parents=True)
            try:
                psi4.core.clean_options()
                psi4.set_memory("8 GB"); psi4.set_num_threads(1)
                psi4.core.set_output_file(str(sdir / "p.log"), False)
                m = psi4.geometry(f"{charge} {mult}\n{coords}\nunits angstrom\n"
                                 f"no_reorient\nno_com\n")
                psi4.set_options({"basis": basis,
                                  "reference": "uks" if mult > 1 else "rks",
                                  "scf_type": "df", "df_scf_guess": True,
                                  "e_convergence": 1e-6, "d_convergence": 1e-6,
                                  "maxiter": 150})
                t0 = time.perf_counter()
                psi4.energy("b3lyp", molecule=m)
                t.append(time.perf_counter() - t0)
            except Exception as exc:
                ok = False
                rec["b3lyp_error"] = f"{type(exc).__name__}: {exc}"[:150]
                break
            finally:
                try: psi4.core.clean()
                except Exception: pass
                shutil.rmtree(sdir, ignore_errors=True)
        rec["b3lyp_s"] = round(float(np.median(t)), 2) if t else None
        rec["b3lyp_converged"] = ok and bool(t)
        print(f"  B3LYP            {rec['b3lyp_s']} s "
              f"({'ok' if rec['b3lyp_converged'] else 'FAILED'})", flush=True)
        rows.append(rec)

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "C3_timings_qc.csv", index=False)
    print(f"\nwrote {OUT/'C3_timings_qc.csv'}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
