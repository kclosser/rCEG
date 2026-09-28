#!/usr/bin/env python3

"""
Generate revised paper figures addressing PI comments:

1. Full PQR descriptor plots with reference subsets highlighted.
2. Separate descriptor plots containing only real-reference molecules.
3. Piecewise split plots with interpretable x-axis names.
4. Validation error versus boosting iteration.
5. QM9/PQR matched-gap calibration figure.

No HOMO, LUMO, or HOMO-LUMO-derived quantity is used as an input
descriptor in any model-related figure.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT = Path(".")
PQR_JSONL = ROOT / "enhanced_dataset_lasso_STRICT.jsonl"
QM9_REFERENCE = ROOT / "qm9_gap_reference.csv"
RECOMPUTED_REFERENCE = ROOT / "pqr_recomputed_reference.csv"

# Use the most recent model run containing the piecewise results.
RUN_CANDIDATES = [
    ROOT / "runs/pqr_full_domain_moe_scaled_regularized_LEAK094",
    ROOT / "runs/pqr_full_domain_moe_piecewise_split_LEAK094",
    ROOT / "runs/pqr_qmugs_strict_external_LEAK094",
]

RUN = next((p for p in RUN_CANDIDATES if p.exists()), None)

if RUN is None:
    raise FileNotFoundError(
        "Could not find a model output folder. Checked:\n"
        + "\n".join(str(p) for p in RUN_CANDIDATES)
    )

OUT = ROOT / "paper_figures_revised"
OUT.mkdir(parents=True, exist_ok=True)

# Use the actual current model implementation to reconstruct x-features.
MODEL_CANDIDATES = [
    ROOT / "pqr_full_domain_moe_scaled_regularized.py",
    ROOT / "pqr_full_domain_moe_piecewise_split.py",
    ROOT / "pqr_full_domain_moe.py",
]

MODEL_PATH = next((p for p in MODEL_CANDIDATES if p.exists()), None)

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "figure.titlesize": 15,
    "savefig.dpi": 300,
})


# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------

def canon(smiles: object) -> str | None:
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        return Chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=True,
        )
    except Exception:
        return None


def numeric(value: object) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else np.nan
    except Exception:
        return np.nan


def pick_gap_column(df: pd.DataFrame) -> str:
    candidates = [
        "ref_gap",
        "gap",
        "gap_ev",
        "homo_lumo_gap",
        "HOMO_LUMO_gap",
        "DeltaEHL",
        "deltaE",
    ]

    lower = {str(c).lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        if candidate.lower() in lower:
            return lower[candidate.lower()]

    raise RuntimeError(
        f"Could not identify a gap column. Columns: {list(df.columns)}"
    )


def load_reference(path: Path, source: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["smiles", "ref_gap", "source"])

    df = pd.read_csv(path)

    if "smiles" not in df.columns:
        raise RuntimeError(f"{path} does not contain a smiles column.")

    gap_col = pick_gap_column(df)

    out = pd.DataFrame({
        "smiles": df["smiles"].map(canon),
        "ref_gap": pd.to_numeric(df[gap_col], errors="coerce"),
        "source": source,
    })

    return (
        out.dropna(subset=["smiles", "ref_gap"])
        .groupby("smiles", as_index=False)
        .agg({
            "ref_gap": "median",
            "source": "first",
        })
    )


def load_pqr_properties() -> tuple[pd.DataFrame, int]:
    rows: list[dict[str, object]] = []
    lasso_length = None

    with PQR_JSONL.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                entry = json.loads(line)
            except Exception:
                continue

            if not isinstance(entry, list) or len(entry) < 5:
                continue

            smiles = canon(entry[1])
            pqr = entry[2] if isinstance(entry[2], list) else []
            lasso = entry[3] if isinstance(entry[3], list) else []

            if smiles is None or len(pqr) < 5:
                continue

            if lasso_length is None:
                lasso_length = len(lasso)

            rows.append({
                "smiles": smiles,
                "pqr_gap": numeric(entry[4]),
                "mol_weight": numeric(pqr[0]),
                "exact_mass": numeric(pqr[1]),
                "dipole_moment": numeric(pqr[2]),
                "heat_formation": numeric(pqr[3]),
                "polarizability": numeric(pqr[4]),
                "_pqr": pqr,
                "_lasso": lasso,
            })

    df = pd.DataFrame(rows).drop_duplicates("smiles")

    return df, int(lasso_length or 0)


def locate_file(filename: str) -> Path | None:
    direct = RUN / filename
    if direct.exists():
        return direct

    matches = sorted(
        ROOT.glob(f"runs/**/{filename}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    return matches[0] if matches else None


def quantile_limits(values: pd.Series) -> tuple[float, float, int]:
    clean = pd.to_numeric(values, errors="coerce").dropna()

    if len(clean) == 0:
        return 0.0, 1.0, 0

    low = float(clean.quantile(0.01))
    high = float(clean.quantile(0.99))
    outside = int(((clean < low) | (clean > high)).sum())

    if low == high:
        low = float(clean.min())
        high = float(clean.max())

    margin = 0.04 * max(high - low, 1e-9)

    return low - margin, high + margin, outside


# ---------------------------------------------------------------------
# Load core datasets
# ---------------------------------------------------------------------

print("Loading PQR properties...")
pqr, lasso_length = load_pqr_properties()

qm9 = load_reference(QM9_REFERENCE, "QM9 overlap")
recomputed = load_reference(
    RECOMPUTED_REFERENCE,
    "Psi4-recomputed PQR",
)

qm9_set = set(qm9["smiles"])
recomputed_set = set(recomputed["smiles"])

pqr["is_qm9_reference"] = pqr["smiles"].isin(qm9_set)
pqr["is_recomputed_reference"] = pqr["smiles"].isin(recomputed_set)
pqr["is_real_reference"] = (
    pqr["is_qm9_reference"]
    | pqr["is_recomputed_reference"]
)

print(f"PQR molecules: {len(pqr):,}")
print(f"QM9-overlap references: {pqr['is_qm9_reference'].sum():,}")
print(
    "Psi4-recomputed references: "
    f"{pqr['is_recomputed_reference'].sum():,}"
)


# ---------------------------------------------------------------------
# Figure 1: Full database with highlighted reference subsets
# ---------------------------------------------------------------------

descriptor_columns = [
    ("mol_weight", "Molecular weight"),
    ("exact_mass", "Exact mass"),
    ("polarizability", "PM7 polarizability"),
    ("heat_formation", "PM7 heat of formation"),
]

fig, axes = plt.subplots(2, 2, figsize=(11, 8.3))
axes = axes.ravel()

for ax, (column, label) in zip(axes, descriptor_columns):
    data = pqr[[column, "pqr_gap"]].dropna()
    r = data[column].corr(data["pqr_gap"])

    # Background: complete PQR database.
    ax.scatter(
        pqr[column],
        pqr["pqr_gap"],
        s=5,
        alpha=0.10,
        linewidths=0,
        label=f"Complete PQR dataset (n={len(pqr):,})",
    )

    # QM9-overlap reference points.
    qm9_points = pqr[pqr["is_qm9_reference"]]
    ax.scatter(
        qm9_points[column],
        qm9_points["pqr_gap"],
        s=11,
        alpha=0.50,
        linewidths=0,
        label=f"QM9-overlap references (n={len(qm9_points):,})",
    )

    # Recomputed molecules as outlined points so overlap remains visible.
    recomputed_points = pqr[pqr["is_recomputed_reference"]]
    ax.scatter(
        recomputed_points[column],
        recomputed_points["pqr_gap"],
        s=20,
        facecolors="none",
        edgecolors="black",
        linewidths=0.45,
        alpha=0.75,
        label=f"Psi4-recomputed references (n={len(recomputed_points):,})",
    )

    x_min, x_max, outside = quantile_limits(pqr[column])
    ax.set_xlim(x_min, x_max)

    # Linear trend is descriptive only.
    fit = data[
        data[column].between(x_min, x_max)
        & data["pqr_gap"].notna()
    ]

    if len(fit) > 10 and fit[column].std() > 0:
        slope, intercept = np.polyfit(
            fit[column],
            fit["pqr_gap"],
            1,
        )
        grid = np.linspace(x_min, x_max, 200)
        ax.plot(
            grid,
            slope * grid + intercept,
            linewidth=1.4,
        )

    ax.set_title(f"{label} versus raw PQR gap (r={r:.2f})")
    ax.set_xlabel(label)
    ax.set_ylabel("Raw PQR HOMO–LUMO gap (eV)")
    ax.grid(alpha=0.22)

    ax.text(
        0.02,
        0.02,
        (
            "Axes show the 1st–99th percentile.\n"
            f"{outside:,} x-axis outliers remain in analysis."
        ),
        transform=ax.transAxes,
        va="bottom",
        fontsize=7.5,
    )

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=3,
    frameon=False,
)

fig.suptitle(
    "Molecular Descriptors Across the PQR Dataset and Reference Subsets",
    y=0.98,
)

fig.tight_layout(rect=[0, 0.08, 1, 0.95])

fig.savefig(
    OUT / "figure_descriptor_relationships_highlighted.png",
    bbox_inches="tight",
)
plt.close(fig)

pqr[
    [
        "smiles",
        "pqr_gap",
        "mol_weight",
        "exact_mass",
        "polarizability",
        "heat_formation",
        "is_qm9_reference",
        "is_recomputed_reference",
    ]
].to_csv(
    OUT / "figure_descriptor_relationships_highlighted_data.csv",
    index=False,
)


# ---------------------------------------------------------------------
# Figure 2: Reference molecules only
# ---------------------------------------------------------------------

references_only = pqr[pqr["is_real_reference"]].copy()

fig, axes = plt.subplots(2, 2, figsize=(11, 8.3))
axes = axes.ravel()

for ax, (column, label) in zip(axes, descriptor_columns):
    for source_label, mask, marker in [
        (
            "QM9-overlap",
            references_only["is_qm9_reference"],
            "o",
        ),
        (
            "Psi4-recomputed PQR",
            references_only["is_recomputed_reference"],
            "^",
        ),
    ]:
        group = references_only[mask]

        ax.scatter(
            group[column],
            group["pqr_gap"],
            s=15,
            alpha=0.45,
            marker=marker,
            linewidths=0,
            label=f"{source_label} (n={len(group):,})",
        )

    ax.set_xlabel(label)
    ax.set_ylabel("Raw PQR HOMO–LUMO gap (eV)")
    ax.set_title(label)
    ax.grid(alpha=0.22)

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=2,
    frameon=False,
)

fig.suptitle(
    "Descriptor Coverage of the Real-Reference Molecular Subset",
    y=0.98,
)

fig.tight_layout(rect=[0, 0.08, 1, 0.95])

fig.savefig(
    OUT / "figure_descriptor_reference_subset.png",
    bbox_inches="tight",
)
plt.close(fig)


# ---------------------------------------------------------------------
# Piecewise feature reconstruction and interpretable names
# ---------------------------------------------------------------------

def load_model_module():
    if MODEL_PATH is None:
        raise FileNotFoundError(
            "No model script was found for reconstructing x-features."
        )

    spec = importlib.util.spec_from_file_location(
        "rceg_model",
        MODEL_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {MODEL_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def feature_index(feature: str) -> int:
    match = re.fullmatch(r"x(\d+)", str(feature))

    if not match:
        raise ValueError(f"Unexpected feature name: {feature}")

    return int(match.group(1))


def infer_feature_name(
    idx: int,
    rdkit_length: int,
) -> str:
    base_names = [
        "Molecular weight",
        "Exact mass",
        "Dipole moment",
        "Heat of formation",
        "Polarizability",
    ]

    if idx < 5:
        return f"{base_names[idx]} (x{idx})"

    if idx < 5 + lasso_length:
        lasso_idx = idx - 5 + 1
        return f"LASSO-selected descriptor {lasso_idx} (x{idx})"

    rdkit_start = 5 + lasso_length
    rdkit_idx = idx - rdkit_start

    rdkit_names = [name for name, _ in Descriptors.descList]

    if 0 <= rdkit_idx < rdkit_length:
        if rdkit_length == len(rdkit_names):
            return f"RDKit {rdkit_names[rdkit_idx]} (x{idx})"

        return f"RDKit descriptor {rdkit_idx + 1} (x{idx})"

    bond_idx = rdkit_idx - rdkit_length + 1
    return f"Bond-step descriptor {bond_idx} (x{idx})"


branch_path = locate_file(
    "piecewise_split_architecture_branches.csv"
)

predictions_path = locate_file(
    "all_pqr_predictions_with_label_source.csv"
)

if branch_path is not None and predictions_path is not None:
    print("Generating named piecewise split figure...")

    branches = pd.read_csv(branch_path)
    predictions = pd.read_csv(predictions_path)

    if "smiles" not in predictions.columns:
        raise RuntimeError(
            f"{predictions_path} lacks a smiles column."
        )

    predictions["smiles"] = predictions["smiles"].map(canon)

    target_candidates = [
        "training_label",
        "calibrated_gap",
        "pred_qm9_aligned_gap",
        "final_prediction",
    ]

    target_col = next(
        (c for c in target_candidates if c in predictions.columns),
        None,
    )

    if target_col is None:
        raise RuntimeError(
            "Could not identify a training-label column in "
            f"{predictions_path}. Columns: {list(predictions.columns)}"
        )

    selected_features = list(
        dict.fromkeys(branches["feature"].astype(str))
    )
    selected_indices = {
        feature: feature_index(feature)
        for feature in selected_features
    }

    model = load_model_module()

    if not hasattr(model, "rdkit_features"):
        raise RuntimeError(
            f"{MODEL_PATH} has no rdkit_features() function."
        )

    if not hasattr(model, "bond_step_features"):
        raise RuntimeError(
            f"{MODEL_PATH} has no bond_step_features() function."
        )

    feature_rows = []
    rdkit_length = None

    print(
        "Reconstructing selected x-features for piecewise plots. "
        "This may take several minutes..."
    )

    with PQR_JSONL.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as handle:
        for row_number, line in enumerate(handle, start=1):
            try:
                entry = json.loads(line)
            except Exception:
                continue

            if not isinstance(entry, list) or len(entry) < 5:
                continue

            smiles = canon(entry[1])
            pqr_values = (
                entry[2] if isinstance(entry[2], list) else []
            )
            lasso_values = (
                entry[3] if isinstance(entry[3], list) else []
            )

            if smiles is None:
                continue

            mol = Chem.MolFromSmiles(smiles)

            if mol is None:
                continue

            rdkit_values = list(model.rdkit_features(mol))
            bond_values = list(model.bond_step_features(mol))

            if rdkit_length is None:
                rdkit_length = len(rdkit_values)

            full_vector = (
                [numeric(v) for v in pqr_values[:5]]
                + [numeric(v) for v in lasso_values]
                + [numeric(v) for v in rdkit_values]
                + [numeric(v) for v in bond_values]
            )

            out_row = {"smiles": smiles}

            for feature, idx in selected_indices.items():
                out_row[feature] = (
                    full_vector[idx]
                    if idx < len(full_vector)
                    else np.nan
                )

            feature_rows.append(out_row)

            if row_number % 10000 == 0:
                print(f"  processed {row_number:,} JSONL rows")

    feature_table = pd.DataFrame(
        feature_rows
    ).drop_duplicates("smiles")

    plot_data = predictions.merge(
        feature_table,
        on="smiles",
        how="inner",
    )

    n_panels = len(branches)
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(9, max(3.1 * n_panels, 6)),
    )

    if n_panels == 1:
        axes = [axes]

    name_rows = []

    for ax, (_, branch) in zip(axes, branches.iterrows()):
        domain = str(branch["domain"])
        feature = str(branch["feature"])
        cut = float(branch["split_value"])

        feature_label = infer_feature_name(
            feature_index(feature),
            int(rdkit_length or 0),
        )

        name_rows.append({
            "feature": feature,
            "interpretable_label": feature_label,
            "domain": domain,
            "split_value": cut,
        })

        domain_data = plot_data.copy()

        if "domain" in domain_data.columns:
            domain_data = domain_data[
                domain_data["domain"].astype(str) == domain
            ]

        domain_data[feature] = pd.to_numeric(
            domain_data[feature],
            errors="coerce",
        )

        domain_data[target_col] = pd.to_numeric(
            domain_data[target_col],
            errors="coerce",
        )

        domain_data = domain_data.dropna(
            subset=[feature, target_col]
        )

        # Use a reproducible sample for visualization only.
        if len(domain_data) > 7000:
            domain_data = domain_data.sample(
                7000,
                random_state=42,
            )

        ax.scatter(
            domain_data[feature],
            domain_data[target_col],
            s=5,
            alpha=0.15,
            linewidths=0,
        )

        ax.axvline(
            cut,
            linestyle="--",
            linewidth=1.5,
            label=f"Selected breakpoint = {cut:.4g}",
        )

        # Separate descriptive lines on the two sides.
        for side in [
            domain_data[domain_data[feature] <= cut],
            domain_data[domain_data[feature] > cut],
        ]:
            if len(side) < 20 or side[feature].std() == 0:
                continue

            slope, intercept = np.polyfit(
                side[feature],
                side[target_col],
                1,
            )

            grid = np.linspace(
                side[feature].min(),
                side[feature].max(),
                150,
            )

            ax.plot(
                grid,
                slope * grid + intercept,
                linewidth=1.5,
            )

        domain_title = domain.replace("_", " ").title()

        ax.set_title(domain_title)
        ax.set_xlabel(feature_label)
        ax.set_ylabel("Reference-aligned training target (eV)")
        ax.grid(alpha=0.20)
        ax.legend(loc="best")

    fig.suptitle(
        "Descriptor-Adaptive Piecewise Splits by Chemical Domain",
        y=0.995,
    )

    fig.tight_layout()

    fig.savefig(
        OUT / "figure_piecewise_splits_named_axes.png",
        bbox_inches="tight",
    )
    plt.close(fig)

    pd.DataFrame(name_rows).to_csv(
        OUT / "piecewise_feature_name_map.csv",
        index=False,
    )

    plot_data.to_csv(
        OUT / "figure_piecewise_splits_named_axes_data.csv",
        index=False,
    )

else:
    print(
        "Skipping piecewise figure because the branch or prediction "
        "CSV could not be found."
    )


# ---------------------------------------------------------------------
# Error versus boosting iteration
# ---------------------------------------------------------------------

history_path = locate_file(
    "global_hgb_convergence_history.csv"
)

metrics_path = locate_file(
    "FINAL_reference_holdout_metrics.csv"
)

if history_path is not None:
    history = pd.read_csv(history_path)

    required = {"iteration", "val_mae"}
    missing = required - set(history.columns)

    if missing:
        raise RuntimeError(
            f"{history_path} is missing {sorted(missing)}"
        )

    history = history.sort_values("iteration")
    best_row = history.loc[history["val_mae"].idxmin()]

    fig, ax = plt.subplots(figsize=(8.5, 5.3))

    ax.plot(
        history["iteration"],
        history["val_mae"],
        marker="o",
        markersize=2.5,
        linewidth=1.25,
        label="Validation MAE",
    )

    ax.scatter(
        [best_row["iteration"]],
        [best_row["val_mae"]],
        marker="*",
        s=150,
        zorder=5,
        label=(
            f"Minimum validation MAE = "
            f"{best_row['val_mae']:.3f} eV"
        ),
    )

    # Show the final untouched test MAE as context, not as a curve used
    # for model selection.
    if metrics_path is not None:
        metrics = pd.read_csv(metrics_path)
        match = metrics[
            metrics["name"] == "TEST global_hgb"
        ]

        if len(match):
            final_test_mae = float(match.iloc[0]["mae"])
            ax.axhline(
                final_test_mae,
                linestyle="--",
                linewidth=1.2,
                label=(
                    "Final held-out test MAE "
                    f"= {final_test_mae:.3f} eV"
                ),
            )

    ax.set_xlabel("Boosting iteration")
    ax.set_ylabel("Mean absolute error (eV)")
    ax.set_title(
        "Validation Error Across Global HGB Boosting Iterations"
    )
    ax.grid(alpha=0.24)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUT / "figure_error_vs_boosting_iteration.png",
        bbox_inches="tight",
    )
    plt.close(fig)

    history.to_csv(
        OUT / "figure_error_vs_boosting_iteration_data.csv",
        index=False,
    )


# ---------------------------------------------------------------------
# QMugs external-validation figure
# ---------------------------------------------------------------------

QMUGS_RUN = ROOT / "runs/pqr_qmugs_strict_external_LEAK094"
QMUGS_PREDICTIONS = QMUGS_RUN / "qmugs_external_predictions.csv"

if not QMUGS_PREDICTIONS.exists():
    raise FileNotFoundError(
        f"Missing QMugs prediction file: {QMUGS_PREDICTIONS}"
    )

qmugs_pred = pd.read_csv(QMUGS_PREDICTIONS)

required_columns = {
    "ref_gap",
    "pred_domain_expert",
    "pred_domain_expert_crossfit_offset_aligned",
}

missing_columns = required_columns - set(qmugs_pred.columns)

if missing_columns:
    raise RuntimeError(
        "QMugs prediction file is missing required columns: "
        f"{sorted(missing_columns)}\n"
        f"Available columns: {list(qmugs_pred.columns)}"
    )

qmugs_pred["ref_gap"] = pd.to_numeric(
    qmugs_pred["ref_gap"],
    errors="coerce",
)

qmugs_pred["pred_domain_expert"] = pd.to_numeric(
    qmugs_pred["pred_domain_expert"],
    errors="coerce",
)

qmugs_pred["pred_domain_expert_crossfit_offset_aligned"] = (
    pd.to_numeric(
        qmugs_pred[
            "pred_domain_expert_crossfit_offset_aligned"
        ],
        errors="coerce",
    )
)

qmugs_pred = qmugs_pred.dropna(
    subset=[
        "ref_gap",
        "pred_domain_expert",
        "pred_domain_expert_crossfit_offset_aligned",
    ]
).copy()

qmugs_pred["signed_residual_ev"] = (
    qmugs_pred["ref_gap"]
    - qmugs_pred["pred_domain_expert"]
)

qmugs_pred["raw_abs_error_ev"] = np.abs(
    qmugs_pred["ref_gap"]
    - qmugs_pred["pred_domain_expert"]
)

qmugs_pred["aligned_abs_error_ev"] = np.abs(
    qmugs_pred["ref_gap"]
    - qmugs_pred[
        "pred_domain_expert_crossfit_offset_aligned"
    ]
)

raw_mae = float(
    qmugs_pred["raw_abs_error_ev"].mean()
)

aligned_mae = float(
    qmugs_pred["aligned_abs_error_ev"].mean()
)

raw_r2 = float(
    1
    - np.sum(
        (
            qmugs_pred["ref_gap"]
            - qmugs_pred["pred_domain_expert"]
        ) ** 2
    )
    / np.sum(
        (
            qmugs_pred["ref_gap"]
            - qmugs_pred["ref_gap"].mean()
        ) ** 2
    )
)

aligned_r2 = float(
    1
    - np.sum(
        (
            qmugs_pred["ref_gap"]
            - qmugs_pred[
                "pred_domain_expert_crossfit_offset_aligned"
            ]
        ) ** 2
    )
    / np.sum(
        (
            qmugs_pred["ref_gap"]
            - qmugs_pred["ref_gap"].mean()
        ) ** 2
    )
)

pearson_r = float(
    qmugs_pred[
        ["ref_gap", "pred_domain_expert"]
    ].corr().iloc[0, 1]
)

mean_residual = float(
    qmugs_pred["signed_residual_ev"].mean()
)

median_residual = float(
    qmugs_pred["signed_residual_ev"].median()
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5.3),
)

# -------------------------------------------------------------
# Panel A: raw and aligned predictions versus QMugs references
# -------------------------------------------------------------

ax = axes[0]

# Reproducible display sample to reduce overplotting.
if len(qmugs_pred) > 6000:
    display_data = qmugs_pred.sample(
        6000,
        random_state=42,
    )
else:
    display_data = qmugs_pred

ax.scatter(
    display_data["ref_gap"],
    display_data["pred_domain_expert"],
    s=8,
    alpha=0.12,
    linewidths=0,
    label=(
        f"Frozen transfer\n"
        f"MAE={raw_mae:.3f} eV"
    ),
)

ax.scatter(
    display_data["ref_gap"],
    display_data[
        "pred_domain_expert_crossfit_offset_aligned"
    ],
    s=8,
    alpha=0.18,
    linewidths=0,
    label=(
        f"Cross-fitted aligned\n"
        f"MAE={aligned_mae:.3f} eV"
    ),
)

all_values = np.concatenate([
    qmugs_pred["ref_gap"].to_numpy(),
    qmugs_pred["pred_domain_expert"].to_numpy(),
    qmugs_pred[
        "pred_domain_expert_crossfit_offset_aligned"
    ].to_numpy(),
])

low = float(np.nanpercentile(all_values, 0.5))
high = float(np.nanpercentile(all_values, 99.5))

ax.plot(
    [low, high],
    [low, high],
    linestyle="--",
    linewidth=1.2,
    label="Ideal agreement",
)

ax.set_xlim(low, high)
ax.set_ylim(low, high)

ax.set_xlabel("QMugs DFT HOMO–LUMO gap (eV)")
ax.set_ylabel("rCEG prediction (eV)")

ax.set_title(
    "A. Frozen and reference-aligned transfer"
)

ax.grid(alpha=0.22)
ax.legend(loc="best")

ax.text(
    0.03,
    0.03,
    (
        f"n={len(qmugs_pred):,}\n"
        f"Frozen Pearson r={pearson_r:.3f}\n"
        f"Aligned R²={aligned_r2:.3f}"
    ),
    transform=ax.transAxes,
    va="bottom",
    fontsize=8.5,
)

# -------------------------------------------------------------
# Panel B: signed residual distribution
# -------------------------------------------------------------

ax = axes[1]

ax.hist(
    qmugs_pred["signed_residual_ev"],
    bins=65,
    alpha=0.82,
)

ax.axvline(
    median_residual,
    linewidth=2,
    label=f"Median offset={median_residual:.3f} eV",
)

ax.axvline(
    mean_residual,
    linestyle="--",
    linewidth=2,
    label=f"Mean residual={mean_residual:.3f} eV",
)

# Add the five exact fold offsets if the saved file exists.
fold_offset_path = (
    QMUGS_RUN / "qmugs_crossfit_fold_offsets.csv"
)

if fold_offset_path.exists():
    fold_offsets = pd.read_csv(fold_offset_path)

    if "median_offset_ev" in fold_offsets.columns:
        for fold_number, offset in enumerate(
            fold_offsets["median_offset_ev"],
            start=1,
        ):
            ax.axvline(
                float(offset),
                linestyle=":",
                linewidth=0.9,
                alpha=0.7,
            )

ax.set_xlabel(
    "Signed residual: QMugs reference − frozen prediction (eV)"
)

ax.set_ylabel("Molecule count")

ax.set_title(
    "B. Systematic target-scale residual"
)

ax.grid(axis="y", alpha=0.22)
ax.legend(loc="best")

fig.suptitle(
    "External Validation of rCEG on the Published QMugs Dataset",
    y=1.01,
)

fig.text(
    0.5,
    -0.01,
    (
        "Each aligned prediction used a median offset estimated "
        "from the other four folds; a molecule's own QMugs label "
        "was not used to calculate its applied offset."
    ),
    ha="center",
    fontsize=9,
)

fig.tight_layout(rect=[0, 0.045, 1, 0.97])

fig.savefig(
    OUT / "figure_qmugs_external_validation.png",
    bbox_inches="tight",
)

plt.close(fig)

# Save exact figure data.
qmugs_pred[
    [
        "smiles",
        "domain",
        "ref_gap",
        "pred_domain_expert",
        "pred_domain_expert_crossfit_offset_aligned",
        "signed_residual_ev",
        "raw_abs_error_ev",
        "aligned_abs_error_ev",
    ]
].to_csv(
    OUT / "figure_qmugs_external_validation_data.csv",
    index=False,
)

qmugs_summary = pd.DataFrame([
    {
        "evaluation": "Frozen external transfer",
        "n": len(qmugs_pred),
        "mae_ev": raw_mae,
        "r2": raw_r2,
        "pearson_r": pearson_r,
        "median_signed_residual_ev": median_residual,
        "mean_signed_residual_ev": mean_residual,
    },
    {
        "evaluation": "Five-fold cross-fitted offset aligned",
        "n": len(qmugs_pred),
        "mae_ev": aligned_mae,
        "r2": aligned_r2,
        "pearson_r": np.nan,
        "median_signed_residual_ev": np.nan,
        "mean_signed_residual_ev": np.nan,
    },
])

qmugs_summary.to_csv(
    OUT / "figure_qmugs_external_validation_summary.csv",
    index=False,
)

print(
    "Generated QMugs external-validation figure:"
)
print(
    OUT / "figure_qmugs_external_validation.png"
)



# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

print("\nGenerated revised figures in:")
print(OUT.resolve())

for path in sorted(OUT.glob("*")):
    print(" ", path.name)
