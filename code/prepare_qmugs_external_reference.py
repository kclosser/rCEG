#!/usr/bin/env python3

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

PQR_PATH = Path("enhanced_dataset_lasso_STRICT.jsonl")
QMUGS_PATH = Path("external_validation/qmugs/summary.csv")
OUTDIR = Path("external_validation/qmugs")
OUTDIR.mkdir(parents=True, exist_ok=True)

HARTREE_TO_EV = 27.211386245988


def canon(smiles):
    try:
        mol = Chem.MolFromSmiles(str(smiles), sanitize=True)
        if mol is None:
            return None
        return Chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=True,
        )
    except Exception:
        return None


print("Reading canonical PQR molecules...")

pqr_smiles = set()

with PQR_PATH.open("r", encoding="utf-8", errors="ignore") as handle:
    for line in handle:
        try:
            record = json.loads(line)
            if (
                isinstance(record, list)
                and len(record) >= 5
                and isinstance(record[1], str)
            ):
                smi = canon(record[1])
                if smi is not None:
                    pqr_smiles.add(smi)
        except Exception:
            continue

print(f"Unique PQR molecules: {len(pqr_smiles):,}")

header = pd.read_csv(QMUGS_PATH, nrows=0)
columns = list(header.columns)

lower_map = {str(c).lower(): c for c in columns}

smiles_col = lower_map.get("smiles")
if smiles_col is None:
    raise RuntimeError(
        f"Could not identify the QMugs SMILES column. Columns begin: {columns[:30]}"
    )

gap_candidates = [
    c for c in columns
    if (
        "dft" in str(c).lower()
        and "homo" in str(c).lower()
        and "lumo" in str(c).lower()
        and "gap" in str(c).lower()
    )
]

if not gap_candidates:
    raise RuntimeError(
        "Could not identify DFT_HOMO_LUMO_GAP in QMugs summary.csv."
    )

gap_col = gap_candidates[0]

print("QMugs SMILES column:", smiles_col)
print("QMugs gap column:", gap_col)

matches = []
total_rows = 0

for chunk_number, chunk in enumerate(
    pd.read_csv(
        QMUGS_PATH,
        usecols=[smiles_col, gap_col],
        chunksize=100_000,
    ),
    start=1,
):
    total_rows += len(chunk)

    chunk = chunk.rename(
        columns={
            smiles_col: "qmugs_smiles",
            gap_col: "qmugs_gap_raw",
        }
    )

    chunk["qmugs_gap_raw"] = pd.to_numeric(
        chunk["qmugs_gap_raw"],
        errors="coerce",
    )
    chunk = chunk.dropna(subset=["qmugs_smiles", "qmugs_gap_raw"])

    # First test exact strings; canonicalize only the remainder.
    chunk["smiles"] = chunk["qmugs_smiles"].astype(str)
    exact = chunk["smiles"].isin(pqr_smiles)

    if (~exact).any():
        chunk.loc[~exact, "smiles"] = (
            chunk.loc[~exact, "qmugs_smiles"]
            .astype(str)
            .map(canon)
        )

    overlap = chunk[chunk["smiles"].isin(pqr_smiles)].copy()

    if len(overlap):
        matches.append(
            overlap[["smiles", "qmugs_gap_raw"]]
        )

    if chunk_number % 5 == 0:
        found = sum(len(x) for x in matches)
        print(
            f"Processed {total_rows:,} conformers; "
            f"overlap conformers found={found:,}"
        )

if not matches:
    raise RuntimeError("No canonical PQR/QMugs overlap was found.")

overlap = pd.concat(matches, ignore_index=True)

# QMugs stores orbital energies and gaps in Hartree.
# Automatically verify before conversion.
raw_median = float(overlap["qmugs_gap_raw"].median())

if raw_median < 1.5:
    overlap["gap"] = overlap["qmugs_gap_raw"] * HARTREE_TO_EV
    units_detected = "Hartree; converted to eV"
else:
    overlap["gap"] = overlap["qmugs_gap_raw"]
    units_detected = "Already appears to be eV"

external = (
    overlap.groupby("smiles", as_index=False)
    .agg(
        gap=("gap", "median"),
        qmugs_conformer_count=("gap", "size"),
        qmugs_gap_conformer_sd=("gap", "std"),
    )
)

external["ref_source"] = "qmugs_dft_external"
external["qmugs_level_of_theory"] = "omegaB97X-D/def2-SVP"
external["qmugs_units_handling"] = units_detected

# Remove molecules previously used as real reference anchors.
anchor_smiles = set()

for anchor_path in [
    Path("qm9_gap_reference.csv"),
    Path("pqr_recomputed_reference.csv"),
]:
    if not anchor_path.exists():
        continue

    anchor = pd.read_csv(anchor_path)
    if "smiles" not in anchor.columns:
        continue

    anchor_smiles.update(
        x
        for x in anchor["smiles"].map(canon)
        if x is not None
    )

before_anchor_removal = len(external)
external = external[
    ~external["smiles"].isin(anchor_smiles)
].copy()

external = external[
    np.isfinite(external["gap"])
    & external["gap"].between(0, 20)
].drop_duplicates("smiles")

external.to_csv(
    OUTDIR / "qmugs_external_reference.csv",
    index=False,
)

summary = pd.DataFrame([{
    "qmugs_rows_scanned": total_rows,
    "pqr_unique_molecules": len(pqr_smiles),
    "overlap_before_anchor_removal": before_anchor_removal,
    "previous_reference_anchors_removed":
        before_anchor_removal - len(external),
    "final_external_test_molecules": len(external),
    "median_external_gap_ev": external["gap"].median(),
    "mean_external_gap_ev": external["gap"].mean(),
    "median_conformers_per_molecule":
        external["qmugs_conformer_count"].median(),
    "units_handling": units_detected,
}])

summary.to_csv(
    OUTDIR / "qmugs_external_preparation_summary.csv",
    index=False,
)

print("\nExternal benchmark prepared.")
print(summary.to_string(index=False))
print("\nGap summary:")
print(external["gap"].describe().to_string())
print("\nWrote:")
print(OUTDIR / "qmugs_external_reference.csv")
print(OUTDIR / "qmugs_external_preparation_summary.csv")
