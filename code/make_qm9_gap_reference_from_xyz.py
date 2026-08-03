from pathlib import Path
import pandas as pd
from rdkit import Chem

xyz_dir = Path("dsgdb9nsd.xyz")
out = Path("qm9_gap_reference.csv")

HARTREE_TO_EV = 27.211386245988

if not xyz_dir.exists():
    raise FileNotFoundError(f"Missing folder: {xyz_dir.resolve()}")

rows = []
bad = 0

for path in sorted(xyz_dir.glob("dsgdb9nsd_*.xyz")):
    try:
        lines = path.read_text(errors="ignore").splitlines()
        if len(lines) < 4:
            bad += 1
            continue

        # QM9 property line format:
        # gdb_id, A, B, C, mu, alpha, homo, lumo, gap, ...
        props = lines[1].split()
        qm9_id = props[1] if len(props) > 1 else path.stem

        gap_hartree = float(props[9])
        gap_ev = gap_hartree * HARTREE_TO_EV

        # In QM9 .xyz files, one of the final lines contains SMILES.
        # We search from the bottom for a line RDKit can parse.
        smiles = None
        for line in reversed(lines):
            parts = line.strip().split()
            for token in parts:
                mol = Chem.MolFromSmiles(token)
                if mol is not None and mol.GetNumAtoms() > 0:
                    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
                    break
            if smiles is not None:
                break

        if smiles is None:
            bad += 1
            continue

        rows.append({
            "qm9_id": qm9_id,
            "smiles": smiles,
            "gap": gap_ev,
            "gap_hartree": gap_hartree,
            "source_file": str(path),
        })

    except Exception:
        bad += 1

df = pd.DataFrame(rows)

# Median in case duplicate canonical SMILES exist.
df = df.groupby("smiles", as_index=False).agg({
    "gap": "median",
    "gap_hartree": "median",
    "qm9_id": "first",
    "source_file": "first",
})

df.to_csv(out, index=False)

print(f"Wrote {out.resolve()}")
print(f"Rows: {len(df):,}")
print(f"Bad/skipped files: {bad:,}")
print(df.head())
print()
print("Gap summary in eV:")
print(df["gap"].describe())
