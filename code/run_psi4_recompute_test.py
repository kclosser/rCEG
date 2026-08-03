from pathlib import Path
import pandas as pd
import numpy as np
import psi4
from rdkit import Chem

HARTREE_TO_EV = 27.211386245988

manifest_path = Path("runs/pqr_full_domain_moe_qm9_cycle/recompute_inputs/recompute_manifest.csv")
outdir = Path("runs/pqr_full_domain_moe_qm9_cycle/recompute_outputs/psi4_test")
outdir.mkdir(parents=True, exist_ok=True)

manifest = pd.read_csv(manifest_path).head(5)

psi4.set_memory("4 GB")
psi4.set_num_threads(4)
psi4.core.set_output_file(str(outdir / "psi4_test_master.log"), False)

def choose_basis(smiles):
    """
    QM9-like molecules use a Pople basis available in Psi4.
    Broad/heavy molecules use def2-SVP for wider element coverage.
    """
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
    same = wfn.same_a_b_orbs()
    eps_b = eps_a if same else np.array(wfn.epsilon_b())

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
        raise RuntimeError("Could not identify HOMO/LUMO from wavefunction orbital arrays.")

    homo = max(homo_candidates)
    lumo = min(lumo_candidates)

    return homo * HARTREE_TO_EV, lumo * HARTREE_TO_EV, (lumo - homo) * HARTREE_TO_EV

rows = []
bad = []

for _, row in manifest.iterrows():
    mol_id = row["id"]
    smiles = row["smiles"]
    basis = choose_basis(smiles)

    print(f"\nRunning {mol_id} with basis {basis}")

    charge = int(row["charge"])
    mult = int(row["multiplicity"])

    geom = f"""
{charge} {mult}
{read_xyz_coords(row["xyz_file"])}
no_com
no_reorient
"""

    try:
        psi4.core.clean()
        psi4.set_options({
            "basis": basis,
            "scf_type": "df",
            "e_convergence": 1e-6,
            "d_convergence": 1e-6,
            "maxiter": 100,
        })

        mol = psi4.geometry(geom)

        if mult > 1:
            psi4.set_options({"reference": "uks"})
        else:
            psi4.set_options({"reference": "rks"})

        e, wfn = psi4.energy("b3lyp", molecule=mol, return_wfn=True)

        homo_ev, lumo_ev, gap_ev = homo_lumo_from_wfn(wfn)

        rows.append({
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
        })

        print(f"  HOMO={homo_ev:.4f} eV  LUMO={lumo_ev:.4f} eV  GAP={gap_ev:.4f} eV")

    except Exception as exc:
        bad.append({
            "id": mol_id,
            "smiles": smiles,
            "basis": basis,
            "reason": repr(exc),
        })
        print(f"  FAILED: {exc}")

pd.DataFrame(rows).to_csv("pqr_recomputed_reference_TEST.csv", index=False)
pd.DataFrame(bad).to_csv("pqr_recomputed_reference_TEST_bad.csv", index=False)

print("\nGood:", len(rows))
print("Bad:", len(bad))
print("Wrote pqr_recomputed_reference_TEST.csv")
