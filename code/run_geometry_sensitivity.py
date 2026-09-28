#!/usr/bin/env python3
"""
run_geometry_sensitivity.py   (spec task A5)

Quantifies how much of the reported HOMO-LUMO gap depends on the geometry the
single-point was run on. This is the only task in the revision that requires
new quantum chemistry.

Reviewer context: the 909 reference recomputations were single-point B3LYP on
ETKDGv3 + UFF geometries, while the QM9 reference labels sit on B3LYP-optimised
geometries. Reviewers R3.1, R4.6 and R5.2 all ask what that costs.

Two measurements
----------------
A5a  50 molecules stratified across the five chemical domains. Re-optimise at
     B3LYP starting from the existing UFF geometry, then recompute the gap.
     delta_gap = gap_b3lyp_optimized - gap_uff_singlepoint.

A5b  20 molecules. Generate a 10-conformer ETKDGv3 ensemble with distinct
     seeds, UFF-relax each, run the same B3LYP single point on all ten, and
     report the within-molecule spread.

Basis-set assignment follows the rule already used by
run_psi4_recompute_batch.py: 6-31G(d,p) when every atom is in {H,C,N,O,F},
def2-SVP otherwise.

Stages
------
  --stage select      pick molecules, write manifests.       NO psi4 needed
  --stage optimize    A5a geometry optimisations.            psi4 required
  --stage conformers  A5b conformer ensembles.               psi4 required
  --stage collect     aggregate per-molecule results.        NO psi4 needed

select and collect are runnable anywhere, so the sampling and the aggregation
can be verified without a psi4 install. Both psi4 stages are resumable: a
molecule whose output file already exists is skipped, matching the behaviour of
run_psi4_recompute_batch.py.

Non-convergence is recorded, never silently dropped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

HARTREE_TO_EV = 27.211386245988
QM9_ATOMS = {1, 6, 7, 8, 9}

N_OPT_TOTAL = 50          # A5a
N_CONF_MOLECULES = 20     # A5b
N_CONFORMERS = 10         # A5b conformers per molecule
DOMAINS = ["charged_or_radical", "heteroatom_rich_non_qm9",
           "large_neutral_organic", "near_qm9_larger_organic",
           "qm9_like_small_organic"]


# ----------------------------------------------------------------------
# shared helpers
# ----------------------------------------------------------------------

def choose_basis(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    atoms = {a.GetAtomicNum() for a in mol.GetAtoms()} if mol is not None else set()
    return "6-31G(d,p)" if atoms and atoms.issubset(QM9_ATOMS) else "def2-SVP"


def charge_and_multiplicity(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None, None
    charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    radicals = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    return charge, (1 if radicals == 0 else radicals + 1)


def embed_uff(smiles, seed):
    """ETKDGv3 embed + UFF relax. Returns (symbols, coords) or None."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    mh = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.useRandomCoords = True
    if AllChem.EmbedMolecule(mh, params) != 0:
        return None
    try:
        AllChem.UFFOptimizeMolecule(mh, maxIters=500)
    except Exception:
        pass
    conf = mh.GetConformer()
    out = []
    for atom in mh.GetAtoms():
        p = conf.GetAtomPosition(atom.GetIdx())
        out.append((atom.GetSymbol(), p.x, p.y, p.z))
    return out


def xyz_block(coords):
    return "\n".join(f"{s} {x:.8f} {y:.8f} {z:.8f}" for s, x, y, z in coords)


def homo_lumo_from_wfn(wfn):
    """
    Frontier orbital energies in eV. For open-shell systems this takes
    homo = max(alpha_HOMO, beta_HOMO) and lumo = min(alpha_LUMO, beta_LUMO),
    identical to run_psi4_recompute_batch.py so the two sets stay comparable.
    """
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


# ----------------------------------------------------------------------
# stage: select
# ----------------------------------------------------------------------

def stage_select(args):
    ref = pd.read_csv(args.recomputed)
    ref = ref.dropna(subset=["smiles", "gap"])
    print(f"converged reference recomputations available: {len(ref):,}")

    rng = np.random.default_rng(args.seed)
    per_domain = N_OPT_TOTAL // len(DOMAINS)

    picks = []
    for dom in DOMAINS:
        sub = ref[ref["domain"] == dom]
        if len(sub) == 0:
            print(f"  WARNING: no molecules in domain {dom}")
            continue
        take = min(per_domain, len(sub))
        idx = rng.choice(sub.index.to_numpy(), size=take, replace=False)
        picks.append(sub.loc[idx])
        print(f"  {dom:26s} pool={len(sub):4d} selected={take}")

    opt = pd.concat(picks, ignore_index=True)

    # top up from the overall pool if a domain was short
    if len(opt) < N_OPT_TOTAL:
        remaining = ref[~ref["id"].isin(set(opt["id"]))]
        extra = remaining.sample(
            min(N_OPT_TOTAL - len(opt), len(remaining)), random_state=args.seed)
        opt = pd.concat([opt, extra], ignore_index=True)

    opt = opt.rename(columns={"gap": "gap_uff_singlepoint"})
    keep = ["id", "smiles", "domain", "charge", "multiplicity",
            "gap_uff_singlepoint"]
    opt = opt[[c for c in keep if c in opt.columns]]
    opt["basis"] = opt["smiles"].map(choose_basis)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    opt.to_csv(outdir / "a5a_optimize_manifest.csv", index=False)

    # A5b: stratified subset of the A5a selection
    per_dom_conf = max(1, N_CONF_MOLECULES // len(DOMAINS))
    cpicks = []
    for dom in DOMAINS:
        sub = opt[opt["domain"] == dom]
        if len(sub):
            cpicks.append(sub.head(per_dom_conf))
    conf = pd.concat(cpicks, ignore_index=True).head(N_CONF_MOLECULES)
    conf.to_csv(outdir / "a5b_conformer_manifest.csv", index=False)

    print(f"\nwrote {outdir/'a5a_optimize_manifest.csv'} ({len(opt)} molecules)")
    print(f"wrote {outdir/'a5b_conformer_manifest.csv'} ({len(conf)} molecules)")
    print(f"basis split: {opt['basis'].value_counts().to_dict()}")
    print(f"planned psi4 jobs: {len(opt)} optimisations + "
          f"{len(conf)*N_CONFORMERS} single points")


# ----------------------------------------------------------------------
# stage: optimize  (A5a)
# ----------------------------------------------------------------------

def stage_optimize(args):
    import psi4

    outdir = Path(args.outdir)
    resdir = outdir / "a5a_results"; resdir.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(outdir / "a5a_optimize_manifest.csv")
    man = man.iloc[args.start:args.end]

    psi4.set_memory(args.memory)
    psi4.set_num_threads(args.threads)

    for _, row in man.iterrows():
        mol_id = row["id"]
        ok_file = resdir / f"{mol_id}.json"
        bad_file = resdir / f"{mol_id}.failed.json"
        if ok_file.exists() or bad_file.exists():
            print(f"skip {mol_id} (already attempted)")
            continue

        psi4.core.set_output_file(str(resdir / f"{mol_id}.psi4.log"), False)
        rec = {"mol_id": mol_id, "smiles": row["smiles"],
               "domain": row["domain"], "basis": row["basis"],
               "gap_uff_singlepoint": float(row["gap_uff_singlepoint"])}
        try:
            coords = embed_uff(row["smiles"], seed=1000)
            if coords is None:
                raise RuntimeError("ETKDG embedding failed")

            charge, mult = charge_and_multiplicity(row["smiles"])
            rec.update({"charge": charge, "multiplicity": mult})

            mol = psi4.geometry(
                f"{charge} {mult}\n{xyz_block(coords)}\nunits angstrom\n"
                "no_reorient\nno_com\n")
            psi4.set_options({"basis": row["basis"], "reference":
                              "uks" if mult > 1 else "rks",
                              "scf_type": "df", "g_convergence": "gau_loose",
                              "geom_maxiter": args.geom_maxiter})

            psi4.optimize("b3lyp", molecule=mol)
            _, wfn = psi4.energy("b3lyp", molecule=mol, return_wfn=True)
            homo, lumo, gap = homo_lumo_from_wfn(wfn)

            rec.update({"homo_ev": homo, "lumo_ev": lumo,
                        "gap_b3lyp_optimized": gap,
                        "delta_gap": gap - rec["gap_uff_singlepoint"],
                        "converged": True})
            ok_file.write_text(json.dumps(rec, indent=2))
            print(f"{mol_id}: gap {rec['gap_uff_singlepoint']:.3f} -> "
                  f"{gap:.3f} eV  (delta {rec['delta_gap']:+.3f})")
        except Exception as exc:
            rec.update({"converged": False, "error": f"{type(exc).__name__}: {exc}"})
            bad_file.write_text(json.dumps(rec, indent=2))
            print(f"{mol_id}: FAILED {type(exc).__name__}: {exc}")
        finally:
            psi4.core.clean()


# ----------------------------------------------------------------------
# stage: conformers  (A5b)
# ----------------------------------------------------------------------

def stage_conformers(args):
    import psi4

    outdir = Path(args.outdir)
    resdir = outdir / "a5b_results"; resdir.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(outdir / "a5b_conformer_manifest.csv")

    psi4.set_memory(args.memory)
    psi4.set_num_threads(args.threads)

    for _, row in man.iterrows():
        mol_id = row["id"]
        for k in range(N_CONFORMERS):
            tag = f"{mol_id}_conf{k:02d}"
            ok_file = resdir / f"{tag}.json"
            bad_file = resdir / f"{tag}.failed.json"
            if ok_file.exists() or bad_file.exists():
                continue

            psi4.core.set_output_file(str(resdir / f"{tag}.psi4.log"), False)
            rec = {"mol_id": mol_id, "conformer_idx": k,
                   "smiles": row["smiles"], "domain": row["domain"],
                   "basis": row["basis"]}
            try:
                coords = embed_uff(row["smiles"], seed=5000 + 37 * k)
                if coords is None:
                    raise RuntimeError("ETKDG embedding failed")
                charge, mult = charge_and_multiplicity(row["smiles"])
                mol = psi4.geometry(
                    f"{charge} {mult}\n{xyz_block(coords)}\nunits angstrom\n"
                    "no_reorient\nno_com\n")
                psi4.set_options({"basis": row["basis"], "reference":
                                  "uks" if mult > 1 else "rks",
                                  "scf_type": "df"})
                _, wfn = psi4.energy("b3lyp", molecule=mol, return_wfn=True)
                homo, lumo, gap = homo_lumo_from_wfn(wfn)
                rec.update({"homo_ev": homo, "lumo_ev": lumo,
                            "gap_ev": gap, "converged": True})
                ok_file.write_text(json.dumps(rec, indent=2))
                print(f"{tag}: gap {gap:.3f} eV")
            except Exception as exc:
                rec.update({"converged": False,
                            "error": f"{type(exc).__name__}: {exc}"})
                bad_file.write_text(json.dumps(rec, indent=2))
                print(f"{tag}: FAILED {type(exc).__name__}: {exc}")
            finally:
                psi4.core.clean()


# ----------------------------------------------------------------------
# stage: collect
# ----------------------------------------------------------------------

def stage_collect(args):
    outdir = Path(args.outdir)
    final = Path(args.final_outdir); final.mkdir(parents=True, exist_ok=True)

    # ---- A5a ----
    rows = []
    for f in sorted((outdir / "a5a_results").glob("*.json")):
        if f.name.endswith(".psi4.log"):
            continue
        rows.append(json.loads(f.read_text()))
    if rows:
        a = pd.DataFrame(rows)
        cols = ["mol_id", "domain", "charge", "multiplicity", "basis",
                "gap_uff_singlepoint", "gap_b3lyp_optimized", "delta_gap",
                "converged", "error"]
        a = a[[c for c in cols if c in a.columns]]
        a.to_csv(final / "geometry_sensitivity.csv", index=False)

        ok = a[a["converged"] == True]  # noqa: E712
        print(f"\nA5a: {len(ok)}/{len(a)} converged "
              f"(acceptance needs >= 45 of {N_OPT_TOTAL})")
        if len(ok):
            ad = ok["delta_gap"].abs()
            print(f"  GEOM_SHIFT (mean |delta gap|) = {ad.mean():.4f} eV")
            print(f"  GEOM_SD                       = {ad.std(ddof=1):.4f} eV")
            print(f"  GEOM_MAX                      = {ad.max():.4f} eV")
            per = ok.groupby("domain")["delta_gap"].apply(lambda s: s.abs().mean())
            print("  per-domain mean |delta gap|:")
            for d, v in per.items():
                print(f"    {d:26s} {v:.4f}")
            spread = per.max() - per.min()
            verdict = ("roughly uniform" if spread < 0.5 * ad.mean()
                       else f"systematically larger for {per.idxmax()}")
            print(f"  GEOM_DOMAIN -> {verdict}")
            per.to_frame("mean_abs_delta_gap").to_csv(
                final / "geometry_sensitivity_by_domain.csv")
    else:
        print("\nA5a: no results yet")

    # ---- A5b ----
    rows = []
    for f in sorted((outdir / "a5b_results").glob("*.json")):
        if f.name.endswith(".psi4.log"):
            continue
        rows.append(json.loads(f.read_text()))
    if rows:
        b = pd.DataFrame(rows)
        cols = ["mol_id", "domain", "conformer_idx", "basis", "gap_ev",
                "converged", "error"]
        b = b[[c for c in cols if c in b.columns]]
        b.to_csv(final / "conformer_ensemble_spread.csv", index=False)

        ok = b[b["converged"] == True]  # noqa: E712
        nmol = ok["mol_id"].nunique()
        print(f"\nA5b: {nmol} molecules with >=1 converged conformer "
              f"(acceptance needs >= 18 of {N_CONF_MOLECULES})")
        if len(ok):
            g = ok.groupby("mol_id")["gap_ev"]
            summ = pd.DataFrame({"n_conformers": g.size(), "mean": g.mean(),
                                 "sd": g.std(ddof=1),
                                 "range": g.max() - g.min()})
            summ.to_csv(final / "conformer_ensemble_summary.csv")
            print(f"  CONF_SPREAD (mean within-molecule sd) = "
                  f"{summ['sd'].mean():.4f} eV")
            print(f"  mean within-molecule range            = "
                  f"{summ['range'].mean():.4f} eV")
    else:
        print("\nA5b: no results yet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["select", "optimize", "conformers", "collect"])
    ap.add_argument("--recomputed", default="pqr_recomputed_reference.csv")
    ap.add_argument("--outdir", default="runs/geometry_sensitivity")
    ap.add_argument("--final-outdir", default="runs/rceg_final_consolidated")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=10_000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--memory", default="8 GB")
    ap.add_argument("--geom-maxiter", type=int, default=60)
    args = ap.parse_args()

    {"select": stage_select, "optimize": stage_optimize,
     "conformers": stage_conformers, "collect": stage_collect}[args.stage](args)


if __name__ == "__main__":
    main()
