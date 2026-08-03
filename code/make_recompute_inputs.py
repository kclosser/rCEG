from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

inp = Path("runs/pqr_full_domain_moe_qm9_cycle/pqr_next_reference_recompute_set.csv")
outdir = Path("runs/pqr_full_domain_moe_qm9_cycle/recompute_inputs")
xyz_dir = outdir / "xyz"
gjf_dir = outdir / "gaussian_gjf"
orca_dir = outdir / "orca_inp"

for d in [outdir, xyz_dir, gjf_dir, orca_dir]:
    d.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(inp)

manifest = []
bad = []

def get_charge_mult(mol):
    charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    radical_e = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    mult = 1 if radical_e == 0 else radical_e + 1
    return charge, mult

for i, row in df.reset_index(drop=True).iterrows():
    smi = row["smiles"]
    mol = Chem.MolFromSmiles(smi)

    if mol is None:
        bad.append({"idx": i, "smiles": smi, "reason": "bad_smiles"})
        continue

    charge, mult = get_charge_mult(mol)

    m = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 1000 + i
    params.useRandomCoords = True

    ok = AllChem.EmbedMolecule(m, params)
    if ok != 0:
        bad.append({"idx": i, "smiles": smi, "reason": "embed_failed"})
        continue

    try:
        AllChem.UFFOptimizeMolecule(m, maxIters=500)
    except Exception:
        pass

    conf = m.GetConformer()
    mol_id = f"pqr_{i:04d}"
    xyz_path = xyz_dir / f"{mol_id}.xyz"
    gjf_path = gjf_dir / f"{mol_id}.gjf"
    orca_path = orca_dir / f"{mol_id}.inp"

    coords = []
    for atom in m.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append((atom.GetSymbol(), pos.x, pos.y, pos.z))

    with open(xyz_path, "w") as f:
        f.write(f"{len(coords)}\n")
        f.write(f"{mol_id} smiles={smi} charge={charge} multiplicity={mult}\n")
        for sym, x, y, z in coords:
            f.write(f"{sym:2s} {x: .8f} {y: .8f} {z: .8f}\n")

    # Gaussian-style QM9-like input.
    # QM9 is commonly associated with B3LYP/6-31G(2df,p)-level targets.
    with open(gjf_path, "w") as f:
        f.write(f"%nprocshared=4\n")
        f.write(f"%mem=8GB\n")
        f.write(f"#p B3LYP/6-31G(2df,p) Opt SCF=Tight Pop=Full\n\n")
        f.write(f"{mol_id} QM9-like HOMO-LUMO recomputation\n\n")
        f.write(f"{charge} {mult}\n")
        for sym, x, y, z in coords:
            f.write(f"{sym:2s} {x: .8f} {y: .8f} {z: .8f}\n")
        f.write("\n")

    # ORCA input. This is a practical B3LYP input; exact basis support depends on your ORCA install.
    with open(orca_path, "w") as f:
        f.write("! B3LYP 6-31G(d,p) Opt TightSCF\n")
        f.write("%pal nprocs 4 end\n\n")
        f.write(f"* xyz {charge} {mult}\n")
        for sym, x, y, z in coords:
            f.write(f"{sym:2s} {x: .8f} {y: .8f} {z: .8f}\n")
        f.write("*\n")

    manifest.append({
        "id": mol_id,
        "smiles": smi,
        "domain": row.get("domain", ""),
        "pqr_gap": row.get("pqr_gap", ""),
        "pred_qm9_aligned_gap": row.get("pred_qm9_aligned_gap", ""),
        "cycle_error": row.get("cycle_error", ""),
        "final_confidence": row.get("final_confidence", ""),
        "charge": charge,
        "multiplicity": mult,
        "xyz_file": str(xyz_path),
        "gaussian_input": str(gjf_path),
        "orca_input": str(orca_path),
    })

pd.DataFrame(manifest).to_csv(outdir / "recompute_manifest.csv", index=False)
pd.DataFrame(bad).to_csv(outdir / "recompute_bad_rows.csv", index=False)

print("Wrote:")
print(outdir / "recompute_manifest.csv")
print(xyz_dir)
print(gjf_dir)
print(orca_dir)
print("\nGood inputs:", len(manifest))
print("Bad rows:", len(bad))

print("\nBy domain:")
print(pd.DataFrame(manifest)["domain"].value_counts())
