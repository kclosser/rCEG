from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, MolSurf

ROOT = Path(".")
OUT = ROOT / "paper_figures_revised"
OUT.mkdir(exist_ok=True)

BASE = ROOT / "enhanced_dataset_lasso.csv"
if not BASE.exists():
    raise FileNotFoundError(f"Missing file: {BASE}")

# Try to locate the run table automatically
candidate_runs = [
    ROOT / "runs" / "pqr_qmugs_strict_external_LEAK094",
    ROOT / "runs" / "pqr_full_domain_moe_scaled_regularized_LEAK094",
    ROOT / "runs" / "pqr_full_domain_moe_piecewise_split_LEAK094",
]
pred_file = None
for run in candidate_runs:
    f = run / "all_pqr_predictions_with_label_source.csv"
    if f.exists():
        pred_file = f
        break

if pred_file is None:
    # broader search fallback
    hits = list(ROOT.rglob("all_pqr_predictions_with_label_source.csv"))
    if hits:
        pred_file = hits[0]

if pred_file is None:
    raise FileNotFoundError("Could not find all_pqr_predictions_with_label_source.csv")

print("Using prediction/label file:", pred_file)

base = pd.read_csv(BASE)
pred = pd.read_csv(pred_file)

if "smiles" not in base.columns or "smiles" not in pred.columns:
    raise RuntimeError("Both tables need a 'smiles' column.")

# Choose the best available y-column automatically
y_candidates = [
    "reference_aligned_gap",
    "final_training_label",
    "training_label",
    "training_gap_label",
    "calibrated_gap",
    "pseudo_label",
    "target_gap",
    "gap",
]
y_col = None
for c in y_candidates:
    if c in pred.columns:
        y_col = c
        break

if y_col is None:
    print("Columns available in prediction file:")
    for c in pred.columns:
        print(" ", c)
    raise RuntimeError("Could not find a usable training-label / aligned-gap column.")

print("Using y-axis column:", y_col)

keep_pred_cols = ["smiles", y_col]
for extra in ["label_source", "final_confidence", "cycle_confidence", "calibration_confidence"]:
    if extra in pred.columns:
        keep_pred_cols.append(extra)

pred = pred[keep_pred_cols].copy()
df = base.merge(pred, on="smiles", how="inner")

# compute interpretable descriptors
def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return {
            "TPSA": Descriptors.TPSA(mol),                         # Å²
            "LabuteASA": MolSurf.LabuteASA(mol),                  # Å²
            "HeavyAtomMolWt": Descriptors.HeavyAtomMolWt(mol),    # Da
            "MolMR": Crippen.MolMR(mol),                          # molar refractivity
            "NumValenceElectrons": Descriptors.NumValenceElectrons(mol),  # electrons
            "NumHAcceptors": Lipinski.NumHAcceptors(mol),         # count
        }
    except Exception:
        return None

rows = []
for smi in df["smiles"]:
    vals = featurize(smi)
    rows.append(vals)

feat = pd.DataFrame(rows)
df = pd.concat([df.reset_index(drop=True), feat], axis=1)

df = df.dropna(subset=[y_col]).copy()
for col in ["TPSA", "LabuteASA", "HeavyAtomMolWt", "MolMR", "NumValenceElectrons", "NumHAcceptors"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Mark real-reference anchors if possible
if "label_source" in df.columns:
    df["is_real_anchor"] = df["label_source"].astype(str).str.contains("reference", case=False, na=False)
else:
    df["is_real_anchor"] = False

descriptor_specs = [
    ("TPSA", "Topological polar surface area", "Å²"),
    ("LabuteASA", "Labute approximate surface area", "Å²"),
    ("HeavyAtomMolWt", "Heavy-atom molecular weight", "Da"),
    ("MolMR", "Molar refractivity", "cm³/mol"),
    ("NumValenceElectrons", "Number of valence electrons", "electrons"),
    ("NumHAcceptors", "H-bond acceptor count", "count"),
]

# Optional downsample for visual clarity
main_df = df.copy()
if len(main_df) > 25000:
    main_df = main_df.sample(25000, random_state=1)

fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
axes = axes.ravel()

for ax, (col, title, unit) in zip(axes, descriptor_specs):
    plot_df = main_df[[col, y_col, "is_real_anchor"]].dropna().copy()

    xlo = plot_df[col].quantile(0.01)
    xhi = plot_df[col].quantile(0.99)
    ylo = plot_df[y_col].quantile(0.01)
    yhi = plot_df[y_col].quantile(0.99)

    vis = plot_df[
        (plot_df[col] >= xlo) & (plot_df[col] <= xhi) &
        (plot_df[y_col] >= ylo) & (plot_df[y_col] <= yhi)
    ].copy()

    bg = vis[~vis["is_real_anchor"]]
    anchors = vis[vis["is_real_anchor"]]

    # density-ish background via small alpha points
    ax.scatter(
        bg[col], bg[y_col],
        s=5, alpha=0.12, rasterized=True
    )

    if len(anchors):
        ax.scatter(
            anchors[col], anchors[y_col],
            s=18, alpha=0.70, marker="o", edgecolors="black", linewidths=0.25,
            label="real-reference anchors"
        )

    if len(vis) > 10 and vis[col].std() > 0:
        m, b = np.polyfit(vis[col], vis[y_col], 1)
        xs = np.linspace(vis[col].min(), vis[col].max(), 200)
        ax.plot(xs, m * xs + b, linewidth=1.5)

    r = vis[col].corr(vis[y_col])
    ax.set_title(f"{title}\nr = {r:.2f}")
    ax.set_xlabel(f"{title} ({unit})")
    ax.set_ylabel("Training label / reference-aligned gap (eV)")
    ax.grid(alpha=0.2)

handles, labels = axes[0].get_legend_handles_labels()
if handles:
    fig.legend(handles, labels, loc="lower center", ncol=1, frameon=False, bbox_to_anchor=(0.5, 0.01))

fig.suptitle("Training-label / reference-aligned gap versus interpretable molecular descriptors", fontsize=16, y=0.98)
fig.tight_layout(rect=[0, 0.05, 1, 0.95])

png = OUT / "figure_training_label_vs_interpretable_descriptors.png"
pdf = OUT / "figure_training_label_vs_interpretable_descriptors.pdf"
csv = OUT / "figure_training_label_vs_interpretable_descriptors_data.csv"

fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

export_cols = ["smiles", y_col, "is_real_anchor"] + [x[0] for x in descriptor_specs]
df[export_cols].to_csv(csv, index=False)

print("Saved:")
print(png)
print(pdf)
print(csv)
