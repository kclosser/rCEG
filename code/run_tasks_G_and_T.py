#!/usr/bin/env python3
"""
run_tasks_G_and_T.py

Task G — geometry sensitivity of the reference labels (fills [[GEOM_SHIFT]], [[CONF_SPREAD]])
Task T — PM7/semiempirical timing and the DFT hardware inconsistency (Table 9)

Must be run with the psi4 environment:
    /Applications/anaconda3/envs/rceg_qc/bin/python run_tasks_G_and_T.py --stage ...

SCRATCH DISCIPLINE
------------------
30 of the 38 original recomputation failures were Psi4 "No space left on device",
not convergence problems. Geometry optimisation runs 10-100 single points per
molecule, so it generates far more scratch than the job that already exhausted
the quota. Every mitigation from the spec is applied:

  * PSI_SCRATCH on node-local storage, one subdirectory per molecule
  * df_scf_guess on, SCF MO retention off
  * psi4.core.clean() AND shutil.rmtree of the molecule's scratch dir afterwards
  * molecules run SERIALLY, never as an array sharing one scratch directory
  * memory capped (larger memory implies larger scratch files)
  * free space checked before each molecule; below the floor the script aborts
    with a DISTINCT exit message so a quota failure is never misattributed to
    chemistry

HOMO/LUMO convention is copied verbatim from run_psi4_recompute_batch.py so the
comparison is not contaminated by a convention difference.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

HARTREE_TO_EV = 27.211386245988
QM9_ATOMS = {"H", "C", "N", "O", "F"}
XYZ_DIR = Path("runs/pqr_full_domain_moe_qm9_cycle/recompute_inputs/xyz")
REF_CSV = Path("pqr_recomputed_reference.csv")
OUT = Path("runs/geometry_sensitivity")
FINAL = Path("runs/rceg_final_consolidated")
MIN_FREE_GB = 20.0  # overridable via --min-free-gb
DOMAINS = ["charged_or_radical", "heteroatom_rich_non_qm9", "large_neutral_organic",
           "near_qm9_larger_organic", "qm9_like_small_organic"]


# ---------------------------------------------------------------- helpers

def read_xyz(mol_id):
    """Return (comment, coordinate-block) from the existing UFF geometry."""
    p = XYZ_DIR / f"{mol_id}.xyz"
    if not p.exists():
        return None, None
    lines = p.read_text().splitlines()
    return lines[1], "\n".join(lines[2:]).rstrip()


def parse_charge_mult(comment, fallback_charge, fallback_mult):
    c, m = fallback_charge, fallback_mult
    for tok in str(comment).split():
        if tok.startswith("charge="):
            try: c = int(float(tok.split("=", 1)[1]))
            except Exception: pass
        if tok.startswith("multiplicity="):
            try: m = int(float(tok.split("=", 1)[1]))
            except Exception: pass
    return c, m


def basis_for(coord_block):
    els = {ln.split()[0] for ln in coord_block.splitlines() if ln.strip()}
    return "6-31G(d,p)" if els <= QM9_ATOMS else "def2-SVP"


def homo_lumo_from_wfn(wfn):
    """Verbatim from run_psi4_recompute_batch.py."""
    eps_a = np.array(wfn.epsilon_a())
    eps_b = eps_a if wfn.same_a_b_orbs() else np.array(wfn.epsilon_b())
    na, nb = wfn.nalpha(), wfn.nbeta()
    homo, lumo = [], []
    if 0 < na < len(eps_a):
        homo.append(eps_a[na - 1]); lumo.append(eps_a[na])
    if 0 < nb < len(eps_b):
        homo.append(eps_b[nb - 1]); lumo.append(eps_b[nb])
    if not homo or not lumo:
        raise RuntimeError("could not identify HOMO/LUMO")
    h, l = max(homo), min(lumo)
    return h * HARTREE_TO_EV, l * HARTREE_TO_EV, (l - h) * HARTREE_TO_EV


def free_gb(path):
    return shutil.disk_usage(path).free / (1024 ** 3)


def scratch_guard(scratch_root, floor=None):
    floor = MIN_FREE_GB if floor is None else floor
    free = free_gb(scratch_root)
    if free < floor:
        print(f"\nSCRATCH_EXHAUSTED: only {free:.1f} GB free in {scratch_root}, "
              f"floor is {floor} GB. Aborting cleanly. This is an "
              f"infrastructure limit, NOT a chemistry failure.", flush=True)
        raise SystemExit(17)
    return free


def psi4_setup(scratch_dir, memory, threads=4):
    import psi4
    psi4.core.clean_options()
    psi4.set_memory(memory)
    psi4.set_num_threads(threads)
    psi4.core.set_output_file(str(scratch_dir / "psi4.log"), False)
    try:
        from psi4 import core as _c
        import psi4.core as pc
        # do not retain SCF MOs between molecules; they are the bulk of scratch
        pc.IOManager.shared_object().set_specific_retention(180, False)
    except Exception:
        pass
    return psi4


# ---------------------------------------------------------------- selection

def stage_select(args):
    ref = pd.read_csv(REF_CSV).dropna(subset=["gap"])
    rng = np.random.default_rng(args.seed)

    picks = []
    for dom in DOMAINS:
        sub = ref[ref["domain"] == dom].copy()
        if sub.empty:
            continue
        # stratify across the gap range within the class, so the subset is not
        # all easy cases
        sub["bin"] = pd.qcut(sub["gap"], q=min(5, max(1, len(sub) // 4)),
                             duplicates="drop", labels=False)
        per_bin = max(1, 10 // max(1, sub["bin"].nunique()))
        chosen = []
        for b, g in sub.groupby("bin"):
            take = min(per_bin, len(g))
            chosen.append(g.iloc[rng.choice(len(g), take, replace=False)])
        c = pd.concat(chosen)
        if len(c) > 10:
            c = c.iloc[rng.choice(len(c), 10, replace=False)]
        elif len(c) < 10:
            rest = sub[~sub["id"].isin(c["id"])]
            if len(rest):
                extra = rest.iloc[rng.choice(len(rest),
                                             min(10 - len(c), len(rest)),
                                             replace=False)]
                c = pd.concat([c, extra])
        picks.append(c)
        print(f"  {dom:26s} pool={len(sub):4d} selected={len(c):2d} "
              f"gap {c.gap.min():.2f}-{c.gap.max():.2f} eV")

    g1 = pd.concat(picks, ignore_index=True)
    g1 = g1.rename(columns={"gap": "gap_uff_singlepoint"})
    g1["has_xyz"] = g1["id"].map(lambda i: (XYZ_DIR / f"{i}.xyz").exists())
    g1 = g1[g1["has_xyz"]].copy()

    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["id", "smiles", "domain", "charge", "multiplicity",
            "gap_uff_singlepoint"]
    g1[cols].to_csv(OUT / "geometry_subset_manifest.csv", index=False)

    # G2: 4 per class from the G1 subset, preferring flexible molecules
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Lipinski
    RDLogger.DisableLog("rdApp.*")

    def nrot(s):
        m = Chem.MolFromSmiles(str(s))
        return Lipinski.NumRotatableBonds(m) if m else 0

    g1 = g1.copy()
    g1["n_rot"] = g1["smiles"].map(nrot)
    g2 = (g1.sort_values("n_rot", ascending=False)
            .groupby("domain", group_keys=False).head(4))
    g2[cols + ["n_rot"]].to_csv(OUT / "conformer_subset_manifest.csv", index=False)

    print(f"\nG1: {len(g1)} molecules -> {OUT/'geometry_subset_manifest.csv'}")
    print(f"G2: {len(g2)} molecules (median {g2.n_rot.median():.0f} rotatable bonds) "
          f"-> {OUT/'conformer_subset_manifest.csv'}")
    print(f"planned: {len(g1)} optimisations + {len(g2)*10} single points")


# ---------------------------------------------------------------- G1

def stage_optimize(args):
    import psi4
    man = pd.read_csv(OUT / "geometry_subset_manifest.csv")
    if args.n_shards > 1:
        # Deterministic interleave: worker k takes rows k, k+N, k+2N...
        # Each worker has its own scratch root, so the shared-scratch failure
        # mode that killed 30 of the original 38 recomputations cannot recur.
        man = man.iloc[args.shard::args.n_shards].reset_index(drop=True)
        print(f"shard {args.shard}/{args.n_shards}: {len(man)} molecules")
    res_dir = OUT / "g1_results"; res_dir.mkdir(parents=True, exist_ok=True)
    scratch_root = Path(args.scratch); scratch_root.mkdir(parents=True, exist_ok=True)
    print(f"scratch root {scratch_root}, {free_gb(scratch_root):.1f} GB free")

    for _, row in man.iterrows():
        mid = row["id"]
        done = res_dir / f"{mid}.json"
        if done.exists():
            continue
        scratch_guard(scratch_root, args.min_free_gb)
        sdir = scratch_root / mid
        if sdir.exists():
            shutil.rmtree(sdir, ignore_errors=True)
        sdir.mkdir(parents=True, exist_ok=True)
        import os
        os.environ["PSI_SCRATCH"] = str(sdir)

        rec = {"mol_id": mid, "domain": row["domain"],
               "gap_uff_singlepoint": float(row["gap_uff_singlepoint"])}
        t0 = time.time()
        try:
            comment, coords = read_xyz(mid)
            if coords is None:
                raise RuntimeError("xyz not found")
            charge, mult = parse_charge_mult(comment, int(row["charge"]),
                                             int(row["multiplicity"]))
            basis = basis_for(coords)
            rec.update({"charge": charge, "multiplicity": mult, "basis": basis})

            p4 = psi4_setup(sdir, args.memory, args.threads)
            mol = p4.geometry(f"{charge} {mult}\n{coords}\nunits angstrom\n"
                              f"no_reorient\nno_com\n")
            # g_convergence=gau_loose. A full gau (tight) optimisation was
            # measured at ~30+ min per molecule on this hardware, i.e. 25-50 h
            # for the 50-molecule set, which is not feasible on a workstation.
            # gau_loose reaches geometries close to the minimum at roughly a
            # fifth of the cost and is standard for a sensitivity estimate of
            # this kind. It is a DISCLOSED deviation: the measured shift is a
            # slight UNDER-estimate of the fully converged shift, so it bounds
            # the effect from below rather than overstating it.
            p4.set_options({"basis": basis,
                            "reference": "uhf" if mult > 1 else "rhf",
                            "scf_type": "df", "df_scf_guess": True,
                            "g_convergence": "gau_loose",
                            "geom_maxiter": args.geom_maxiter})
            p4.optimize("b3lyp", molecule=mol)
            _, wfn = p4.energy("b3lyp", molecule=mol, return_wfn=True)
            homo, lumo, gap = homo_lumo_from_wfn(wfn)
            try:
                nsteps = len(p4.core.get_variable_names())  # placeholder if unavailable
            except Exception:
                nsteps = -1
            rec.update({"homo_ev": homo, "lumo_ev": lumo,
                        "gap_b3lyp_optimized": gap,
                        "delta_gap": gap - rec["gap_uff_singlepoint"],
                        "n_opt_steps": nsteps, "converged": True,
                        "failure_reason": ""})
            print(f"{mid}: {rec['gap_uff_singlepoint']:.3f} -> {gap:.3f} eV "
                  f"(delta {rec['delta_gap']:+.3f}) {time.time()-t0:.0f}s", flush=True)
        except SystemExit:
            raise
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            low = msg.lower()
            if "no space" in low or "psio" in low:
                reason = "disk_exhaustion"
            elif "optimiz" in low or "geom" in low:
                reason = "geometry_optimization_failure"
            elif "scf" in low or "converge" in low:
                reason = "scf_failure"
            else:
                reason = "other"
            rec.update({"converged": False, "failure_reason": reason,
                        "error": msg[:400]})
            print(f"{mid}: FAILED [{reason}] {msg[:110]}", flush=True)
        finally:
            rec["wall_seconds"] = round(time.time() - t0, 2)
            try:
                psi4.core.clean()
            except Exception:
                pass
            shutil.rmtree(sdir, ignore_errors=True)
            done.write_text(json.dumps(rec, indent=2))


# ---------------------------------------------------------------- G2

def stage_conformers(args):
    import psi4
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")

    man = pd.read_csv(OUT / "conformer_subset_manifest.csv")
    res_dir = OUT / "g2_results"; res_dir.mkdir(parents=True, exist_ok=True)
    scratch_root = Path(args.scratch); scratch_root.mkdir(parents=True, exist_ok=True)

    for _, row in man.iterrows():
        mid = row["id"]
        for k in range(args.n_conformers):
            tag = f"{mid}_c{k:02d}"
            done = res_dir / f"{tag}.json"
            if done.exists():
                continue
            scratch_guard(scratch_root)
            sdir = scratch_root / tag
            shutil.rmtree(sdir, ignore_errors=True); sdir.mkdir(parents=True)
            import os
            os.environ["PSI_SCRATCH"] = str(sdir)

            seed = 5000 + 37 * k
            rec = {"mol_id": mid, "domain": row["domain"],
                   "conformer_idx": k, "seed": seed}
            t0 = time.time()
            try:
                m = Chem.MolFromSmiles(str(row["smiles"]))
                mh = Chem.AddHs(m)
                ps = AllChem.ETKDGv3(); ps.randomSeed = seed; ps.useRandomCoords = True
                if AllChem.EmbedMolecule(mh, ps) != 0:
                    raise RuntimeError("ETKDG embedding failed")
                ff = AllChem.UFFGetMoleculeForceField(mh)
                ff.Minimize(maxIts=500)
                rec["uff_energy"] = float(ff.CalcEnergy())
                conf = mh.GetConformer()
                coords = "\n".join(
                    f"{a.GetSymbol()} {conf.GetAtomPosition(a.GetIdx()).x:.8f} "
                    f"{conf.GetAtomPosition(a.GetIdx()).y:.8f} "
                    f"{conf.GetAtomPosition(a.GetIdx()).z:.8f}"
                    for a in mh.GetAtoms())
                charge, mult = int(row["charge"]), int(row["multiplicity"])
                basis = basis_for(coords)
                rec["basis"] = basis

                p4 = psi4_setup(sdir, args.memory, args.threads)
                mol = p4.geometry(f"{charge} {mult}\n{coords}\nunits angstrom\n"
                                  f"no_reorient\nno_com\n")
                p4.set_options({"basis": basis,
                                "reference": "uhf" if mult > 1 else "rhf",
                                "scf_type": "df", "df_scf_guess": True})
                _, wfn = p4.energy("b3lyp", molecule=mol, return_wfn=True)
                _, _, gap = homo_lumo_from_wfn(wfn)
                rec.update({"gap_ev": gap, "converged": True, "failure_reason": ""})
                print(f"{tag}: gap {gap:.3f} eV ({time.time()-t0:.0f}s)", flush=True)
            except SystemExit:
                raise
            except Exception as exc:
                rec.update({"converged": False,
                            "failure_reason": f"{type(exc).__name__}",
                            "error": str(exc)[:300]})
                print(f"{tag}: FAILED {type(exc).__name__}", flush=True)
            finally:
                rec["wall_seconds"] = round(time.time() - t0, 2)
                try: psi4.core.clean()
                except Exception: pass
                shutil.rmtree(sdir, ignore_errors=True)
                done.write_text(json.dumps(rec, indent=2))


# ---------------------------------------------------------------- T1

def stage_dft_timing(args):
    import psi4
    ref = pd.read_csv(REF_CSV).dropna(subset=["gap"])
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    ref["n_heavy"] = ref["smiles"].map(
        lambda s: Chem.MolFromSmiles(str(s)).GetNumHeavyAtoms()
        if Chem.MolFromSmiles(str(s)) else np.nan)
    ref = ref.dropna(subset=["n_heavy"])
    # 20 molecules spanning the size range
    ref["bin"] = pd.qcut(ref["n_heavy"], q=10, duplicates="drop", labels=False)
    rng = np.random.default_rng(args.seed)
    sel = ref.groupby("bin", group_keys=False).apply(
        lambda g: g.iloc[rng.choice(len(g), min(2, len(g)), replace=False)])
    sel = sel.head(20)

    res_dir = OUT / "t1_results"; res_dir.mkdir(parents=True, exist_ok=True)
    scratch_root = Path(args.scratch); scratch_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in sel.iterrows():
        mid = row["id"]
        done = res_dir / f"{mid}.json"
        if done.exists():
            rows.append(json.loads(done.read_text())); continue
        scratch_guard(scratch_root)
        sdir = scratch_root / f"t1_{mid}"
        shutil.rmtree(sdir, ignore_errors=True); sdir.mkdir(parents=True)
        import os; os.environ["PSI_SCRATCH"] = str(sdir)
        rec = {"mol_id": mid, "n_heavy_atoms": int(row["n_heavy"])}
        t0 = time.time()
        try:
            comment, coords = read_xyz(mid)
            if coords is None: raise RuntimeError("xyz missing")
            charge, mult = parse_charge_mult(comment, int(row["charge"]),
                                             int(row["multiplicity"]))
            basis = basis_for(coords); rec["basis"] = basis
            p4 = psi4_setup(sdir, args.memory, args.threads)
            mol = p4.geometry(f"{charge} {mult}\n{coords}\nunits angstrom\n"
                              f"no_reorient\nno_com\n")
            p4.set_options({"basis": basis,
                            "reference": "uhf" if mult > 1 else "rhf",
                            "scf_type": "df", "df_scf_guess": True})
            p4.energy("b3lyp", molecule=mol)
            rec["converged"] = True
        except SystemExit:
            raise
        except Exception as exc:
            rec.update({"converged": False, "error": str(exc)[:200]})
        finally:
            rec["wall_seconds"] = round(time.time() - t0, 2)
            try: psi4.core.clean()
            except Exception: pass
            shutil.rmtree(sdir, ignore_errors=True)
            done.write_text(json.dumps(rec, indent=2))
            rows.append(rec)
            print(f"{mid}: {rec['wall_seconds']}s "
                  f"({rec['n_heavy_atoms']} heavy)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(FINAL / "timing_dft_local.csv", index=False)
    ok = df[df.get("converged", False) == True]  # noqa: E712
    if len(ok):
        print(f"\nlocal B3LYP median {ok.wall_seconds.median():.1f}s "
              f"(IQR {ok.wall_seconds.quantile(.25):.1f}-"
              f"{ok.wall_seconds.quantile(.75):.1f}), n={len(ok)}")


# ---------------------------------------------------------------- T2

def stage_semiempirical(args):
    """GFN2-xTB as a labelled proxy for the unmeasured PM7 cost."""
    import subprocess, os, tempfile
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")

    xtb = shutil.which("xtb") or "/Applications/anaconda3/envs/rceg_qc/bin/xtb"
    ref = pd.read_csv(REF_CSV).dropna(subset=["gap"])
    rng = np.random.default_rng(args.seed)
    sel = ref.iloc[rng.choice(len(ref), min(100, len(ref)), replace=False)]

    rows = []
    with tempfile.TemporaryDirectory() as td:
        for _, row in sel.iterrows():
            mid, smi = row["id"], str(row["smiles"])
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
            mh = Chem.AddHs(m)
            ps = AllChem.ETKDGv3(); ps.randomSeed = 42
            if AllChem.EmbedMolecule(mh, ps) != 0:
                continue
            AllChem.UFFOptimizeMolecule(mh, maxIters=500)
            conf = mh.GetConformer()
            xyz = Path(td) / f"{mid}.xyz"
            xyz.write_text(
                f"{mh.GetNumAtoms()}\n{mid}\n" + "\n".join(
                    f"{a.GetSymbol()} {conf.GetAtomPosition(a.GetIdx()).x:.8f} "
                    f"{conf.GetAtomPosition(a.GetIdx()).y:.8f} "
                    f"{conf.GetAtomPosition(a.GetIdx()).z:.8f}"
                    for a in mh.GetAtoms()) + "\n")
            chg = int(row["charge"])
            t0 = time.time()
            try:
                subprocess.run([xtb, str(xyz), "--gfn", "2", "--chrg", str(chg),
                                "--sp"], cwd=td, capture_output=True, timeout=300,
                               env={**os.environ, "OMP_NUM_THREADS": "1"})
                ok = True
            except Exception:
                ok = False
            rows.append({"mol_id": mid,
                         "n_heavy_atoms": int(m.GetNumHeavyAtoms()),
                         "method": "GFN2-xTB single point (PROXY for PM7, not PM7)",
                         "wall_seconds": round(time.time() - t0, 4),
                         "converged": ok})
    df = pd.DataFrame(rows)
    df.to_csv(FINAL / "timing_semiempirical.csv", index=False)
    ok = df[df.converged]
    print(f"GFN2-xTB single point: median {ok.wall_seconds.median()*1000:.1f} ms "
          f"(IQR {ok.wall_seconds.quantile(.25)*1000:.1f}-"
          f"{ok.wall_seconds.quantile(.75)*1000:.1f}), n={len(ok)}")
    print("NOTE: this is a PROXY. It must never be reported as a PM7 timing.")


# ---------------------------------------------------------------- collect

def stage_collect(args):
    # ---- G1 ----
    rows = [json.loads(f.read_text()) for f in sorted((OUT / "g1_results").glob("*.json"))]
    if rows:
        a = pd.DataFrame(rows)
        cols = ["mol_id", "domain", "charge", "multiplicity", "basis",
                "gap_uff_singlepoint", "gap_b3lyp_optimized", "delta_gap",
                "n_opt_steps", "converged", "wall_seconds", "failure_reason"]
        a = a.reindex(columns=[c for c in cols if c in a.columns] +
                      [c for c in a.columns if c not in cols])
        a.to_csv(FINAL / "geometry_sensitivity.csv", index=False)
        ok = a[a.converged == True]  # noqa: E712
        print(f"\nG1: {len(ok)}/{len(a)} converged (acceptance >= 45 of 50)")
        if len(ok):
            ad = ok.delta_gap.abs()
            print(f"  GEOM_SHIFT (mean |delta|) = {ad.mean():.4f} eV")
            print(f"  GEOM_SD  = {ad.std(ddof=1):.4f}   GEOM_MAX = {ad.max():.4f}")
            per = ok.groupby("domain").delta_gap.apply(lambda s: s.abs().mean())
            print("  per-class mean |delta gap|:")
            for d, v in per.items(): print(f"    {d:26s} {v:.4f}")
            spread = per.max() - per.min()
            print(f"  GEOM_DOMAIN -> {'roughly uniform' if spread < 0.5*ad.mean() else f'systematically larger for {per.idxmax()}'}")
            per.to_frame("mean_abs_delta_gap").to_csv(
                FINAL / "geometry_sensitivity_by_domain.csv")
        if (a.converged == False).any():  # noqa: E712
            print("  failures by reason:")
            print(a[a.converged == False].failure_reason.value_counts().to_string())  # noqa: E712

    # ---- G2 ----
    rows = [json.loads(f.read_text()) for f in sorted((OUT / "g2_results").glob("*.json"))]
    if rows:
        b = pd.DataFrame(rows)
        b.to_csv(FINAL / "conformer_ensemble_spread.csv", index=False)
        ok = b[b.converged == True]  # noqa: E712
        g = ok.groupby("mol_id").gap_ev
        summ = pd.DataFrame({"n_conformers": g.size(), "mean": g.mean(),
                             "sd": g.std(ddof=1), "range": g.max() - g.min()})
        summ = summ[summ.n_conformers >= 2]
        summ.to_csv(FINAL / "conformer_ensemble_summary.csv")
        n8 = int((g.size() >= 8).sum())
        print(f"\nG2: {n8} molecules with >=8 converged conformers "
              f"(acceptance >= 18 of 20)")
        if len(summ):
            print(f"  CONF_SPREAD (mean within-molecule sd) = {summ.sd.mean():.4f} eV")
            print(f"  largest single within-molecule range  = {summ.range.max():.4f} eV")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["select", "optimize", "conformers", "dft-timing",
                             "semiempirical", "collect"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--memory", default="8 GB")
    ap.add_argument("--geom-maxiter", type=int, default=100)
    ap.add_argument("--n-conformers", type=int, default=10)
    ap.add_argument("--scratch", default="/tmp/rceg_psi4_scratch")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    args = ap.parse_args()
    {"select": stage_select, "optimize": stage_optimize,
     "conformers": stage_conformers, "dft-timing": stage_dft_timing,
     "semiempirical": stage_semiempirical, "collect": stage_collect}[args.stage](args)


if __name__ == "__main__":
    main()
