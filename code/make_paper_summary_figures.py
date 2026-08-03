from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

OUTDIR = Path("paper_figures_updated")
OUTDIR.mkdir(exist_ok=True)

FINAL_DIR = Path("runs/pqr_full_domain_moe_qm9_plus_recomputed_full_benchmark")
METRICS_CSV = FINAL_DIR / "FINAL_reference_holdout_metrics.csv"
DOMAIN_CSV = FINAL_DIR / "mae_by_domain_and_source.csv"

def savefig(name):
    path = OUTDIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

def make_final_model_mae_figure():
    metrics = pd.read_csv(METRICS_CSV)
    test = metrics[metrics["name"].str.startswith("TEST")].copy()
    test["model"] = test["name"].str.replace("TEST ", "", regex=False)

    plt.figure(figsize=(10, 5))
    x = np.arange(len(test))
    plt.bar(x, test["mae"])
    plt.xticks(x, test["model"], rotation=45, ha="right")
    plt.ylabel("MAE (eV)")
    plt.title("Final Held-Out Real-Reference Benchmark Performance")
    plt.grid(axis="y", alpha=0.25)

    for i, v in enumerate(test["mae"]):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)

    savefig("figure_final_model_mae.png")

def make_domain_mae_figure():
    df = pd.read_csv(DOMAIN_CSV)
    dom = df[df["group_type"] == "domain"].copy()
    dom = dom.sort_values("mae_stack", ascending=True)

    plt.figure(figsize=(10, 5))
    x = np.arange(len(dom))
    plt.bar(x, dom["mae_stack"])
    plt.xticks(x, dom["group"], rotation=35, ha="right")
    plt.ylabel("MAE (eV)")
    plt.title("Held-Out MAE by Molecular Domain")
    plt.grid(axis="y", alpha=0.25)

    for i, v in enumerate(dom["mae_stack"]):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)

    savefig("figure_domain_mae.png")

def make_reference_source_figure():
    df = pd.read_csv(DOMAIN_CSV)
    ref = df[df["group_type"] == "ref_source"].copy()
    ref = ref.sort_values("mae_stack", ascending=True)

    plt.figure(figsize=(8, 5))
    x = np.arange(len(ref))
    plt.bar(x, ref["mae_stack"])
    plt.xticks(x, ref["group"], rotation=30, ha="right")
    plt.ylabel("MAE (eV)")
    plt.title("Held-Out MAE by Reference Label Source")
    plt.grid(axis="y", alpha=0.25)

    for i, v in enumerate(ref["mae_stack"]):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)

    savefig("figure_reference_source_mae.png")

def add_box(ax, xy, w, h, text, fontsize=9):
    x, y = xy
    rect = Rectangle((x, y), w, h, linewidth=1.4, edgecolor="black", facecolor="#d9e8f5")
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)
    return rect

def add_arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start, end,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.4,
            color="black"
        )
    )

def make_flowchart():
    fig, ax = plt.subplots(figsize=(10, 13))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis("off")

    add_box(ax, (3.0, 14.4), 4.0, 1.1,
            "Input Datasets\n\nPQR molecules: 84,143\nQM9 reference molecules: 133,798\nRecomputed PQR references: 909",
            fontsize=9)

    add_box(ax, (3.0, 12.6), 4.0, 1.2,
            "Feature Construction and Leakage Audit\n\nSMILES-standardized molecules\nx-descriptor feature matrix only\nHOMO/LUMO/gap-derived quantities excluded from final predictors",
            fontsize=8.5)

    add_box(ax, (3.0, 10.7), 4.0, 1.2,
            "Reference Calibration Stage\n\nPQR/QM9 overlap used to map raw PQR gap + descriptors to QM9-aligned reference gap",
            fontsize=8.5)

    add_box(ax, (0.8, 8.6), 3.4, 1.3,
            "Real Reference Anchor Branch\n\nQM9-overlap labels\n+ recomputed Psi4 PQR labels\nUsed for held-out validation",
            fontsize=8.5)

    add_box(ax, (5.8, 8.6), 3.4, 1.3,
            "Calibrated Pseudo-Label Branch\n\nCalibrator applied across PQR\nCycle consistency and applicability-domain confidence computed",
            fontsize=8.5)

    add_box(ax, (3.0, 6.5), 4.0, 1.4,
            "Domain-Aware Mixture of Experts\n\nGlobal models\nChemical-domain experts\nGap-regime experts\nReference-validated stack",
            fontsize=8.5)

    add_box(ax, (3.0, 4.4), 4.0, 1.2,
            "Held-Out Evaluation\n\nMetrics calculated only on real reference labels\nMAE and R² reported by model, domain, and reference source",
            fontsize=8.5)

    add_box(ax, (3.0, 2.4), 4.0, 1.2,
            "Final Output\n\nReference-aligned HOMO-LUMO gap predictions\n0.347 eV overall held-out MAE\n0.466 eV on recomputed PQR subset",
            fontsize=8.5)

    add_arrow(ax, (5.0, 14.4), (5.0, 13.8))
    add_arrow(ax, (5.0, 12.6), (5.0, 11.9))
    add_arrow(ax, (5.0, 10.7), (2.5, 9.9))
    add_arrow(ax, (5.0, 10.7), (7.5, 9.9))
    add_arrow(ax, (2.5, 8.6), (4.5, 7.9))
    add_arrow(ax, (7.5, 8.6), (5.5, 7.9))
    add_arrow(ax, (5.0, 6.5), (5.0, 5.6))
    add_arrow(ax, (5.0, 4.4), (5.0, 3.6))

    ax.set_title(
        "Updated rCEG Pipeline for Reference-Aligned HOMO-LUMO Gap Prediction",
        fontsize=14,
        weight="bold",
        pad=20
    )

    savefig("figure_flowchart_updated.png")

def make_summary_panel():
    df = pd.read_csv(DOMAIN_CSV)

    source = df[df["group_type"] == "ref_source"].copy()
    domain = df[df["group_type"] == "domain"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    source = source.sort_values("mae_stack")
    x1 = np.arange(len(source))
    axes[0].bar(x1, source["mae_stack"])
    axes[0].set_xticks(x1)
    axes[0].set_xticklabels(source["group"], rotation=30, ha="right")
    axes[0].set_ylabel("MAE (eV)")
    axes[0].set_title("MAE by Reference Source")
    axes[0].grid(axis="y", alpha=0.25)
    for i, v in enumerate(source["mae_stack"]):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)

    domain = domain.sort_values("mae_stack")
    x2 = np.arange(len(domain))
    axes[1].bar(x2, domain["mae_stack"])
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(domain["group"], rotation=35, ha="right")
    axes[1].set_ylabel("MAE (eV)")
    axes[1].set_title("MAE by Molecular Domain")
    axes[1].grid(axis="y", alpha=0.25)
    for i, v in enumerate(domain["mae_stack"]):
        axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)

    fig.suptitle("Reference-Validated Generalization Across PQR Molecular Domains", fontsize=14, weight="bold")
    savefig("figure_reference_validation_summary_panel.png")

def main():
    make_final_model_mae_figure()
    make_domain_mae_figure()
    make_reference_source_figure()
    make_summary_panel()
    make_flowchart()
    print("\nDone. Figures are in:", OUTDIR.resolve())

if __name__ == "__main__":
    main()
