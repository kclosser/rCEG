from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import psi4
from rdkit import Chem

HARTREE_TO_EV = 27.211386245988

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=50)
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--memory", default="4 GB")
args = parser.parse_args()

manifest_path = Path("runs/pqr_full_domain_moe_qm9_cycle/recompute_inputs/recompute_manifest.csv")
outdir = Path("runs/pqr_full_domain_moe_qm9_cycle/recompute_outputs/psi4_batch")
outdir.mkdir(parents=True, exist_ok=True)

manifest = pd.read_csv(manifest_path).iloc[args.start:args.end].copy()

psi4.set_memory(args.memory)
psi4.set_num_threads(args.threads)
psi4.core.set_output_file(str(outdir / f"psi4_batch_{args.start}_{args.end}.log"), False)

def choose_basis(smiles):
    mol = Chem.MolFromSmiles(smiles)
    atoms = {a.GetAtomicNum() for a in mol.GetAtoms()} if mol is not None else set()
    qm9_atoms = {1, 6, 7, 8, 9}
    if atoms and atoms.issubset(qm9_atoms):
        return "6-31G(d,p)"
    return "def2-SVP"

def read_xyz_coords(xyz_path):
    lines = Path(xyz_path).read_text().splitlines()
    return "\n".join(lines[2:])

def homo_lumo_from_wfn(wfn):
    eps_a = np.array(wfn.epsilon_a())
    eps_b = eps_a if wfn.same_a_b_orbs() else np.array(wfn.epsilon_b())

    nalpha = wfn.nalpha()
    nbeta = wfn.nbeta()

    homo_candidates = []
    lumo_candidates = []

    if nalpha > 0 and nalpha < len(eps_a):
        homo_candidates.append(eps_a[nalpha - 1])
        lumo_candidates.append(eps_a[nalpha])

    if nbeta > 0 and nbeta < len(eps_b):
        homo_candidates.append(eps_b[nbeta - 1])
        lumo_candidates.append(eps_b[nbeta])

    if not homo_candidates or not lumo_candidates:
        raise RuntimeError("Could not identify HOMO/LUMO.")

    homo = max(homo_candidates)
    lumo = min(lumo_candidates)
    return homo * HARTREE_TO_EV, lumo * HARTREE_TO_EV, (lumo - homo) * HARTREE_TO_EV

for _, row in manifest.iterrows():
    mol_id = row["id"]
    outfile = outdir / f"{mol_id}.csv"
    badfile = outdir / f"{mol_id}.bad.csv"

    if outfile.exists() or badfile.exists():
        print(f"Skipping existing {mol_id}")
        continue

    smiles = row["smiles"]
    basis = choose_basis(smiles)
    charge = int(row["charge"])
    mult = int(row["multiplicity"])

    print(f"\nRunning {mol_id} index={row.name} charge={charge} mult={mult} basis={basis}")

    try:
        psi4.core.clean()

        psi4.set_options({
            "basis": basis,
            "scf_type": "df",
            "e_convergence": 1e-6,
            "d_convergence": 1e-6,
            "maxiter": 150,
        })

        if mult > 1:
            psi4.set_options({"reference": "uks"})
        else:
            psi4.set_options({"reference": "rks"})

        geom = f"""
{charge} {mult}
{read_xyz_coords(row["xyz_file"])}
no_com
no_reorient
"""
        mol = psi4.geometry(geom)

        energy, wfn = psi4.energy("b3lyp", molecule=mol, return_wfn=True)
        homo_ev, lumo_ev, gap_ev = homo_lumo_from_wfn(wfn)

        pd.DataFrame([{
            "id": mol_id,
            "smiles": smiles,
            "domain": row["domain"],
            "charge": charge,
            "multiplicity": mult,
            "basis": basis,
            "homo_ev": homo_ev,
            "lumo_ev": lumo_ev,
            "gap": gap_ev,
            "pqr_gap": row["pqr_gap"],
            "pred_qm9_aligned_gap": row["pred_qm9_aligned_gap"],
            "cycle_error": row["cycle_error"],
            "final_confidence": row["final_confidence"],
        }]).to_csv(outfile, index=False)

        print(f"  SUCCESS gap={gap_ev:.4f} eV")

    except Exception as exc:
        pd.DataFrame([{
            "id": mol_id,
            "smiles": smiles,
            "domain": row["domain"],
            "reason": repr(exc),
        }]).to_csv(badfile, index=False)

        print(f"  FAILED: {exc}")

print("\nBatch complete.")
