from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

OUTDIR = Path("paper_figures")
OUTDIR.mkdir(exist_ok=True)

PQR_JSONL = Path("enhanced_dataset_lasso_STRICT.jsonl")
FINAL_DIR = Path("runs/pqr_full_domain_moe_qm9_plus_recomputed_full_benchmark")

METRICS_CSV = FINAL_DIR / "FINAL_reference_holdout_metrics.csv"
DOMAIN_CSV = FINAL_DIR / "mae_by_domain_and_source.csv"


def savefig(name):
    path = OUTDIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote {path}")


def load_pqr_list_jsonl(path):
    """
    enhanced_dataset_lasso_STRICT.jsonl appears to have rows like:
    [
      id_or_none,
      smiles,
      [mol_weight, exact_mass, dipole_moment, heat_formation, polarizability, homo, lumo],
      [descriptor_0, descriptor_1, ...],
      gap
    ]
    """
    rows = []

    basic_names = [
        "mol_weight",
        "exact_mass",
        "dipole_moment",
        "heat_formation",
        "polarizability",
        "homo",
        "lumo",
    ]

    with open(path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            try:
                rec = json.loads(line)
            except Exception:
                continue

            if not isinstance(rec, list) or len(rec) < 5:
                continue

            row = {
                "raw_id": rec[0],
                "smiles": rec[1],
                "gap": rec[4],
            }

            props = rec[2]
            if isinstance(props, list):
                for i, val in enumerate(props):
                    if i < len(basic_names):
                        row[basic_names[i]] = val
                    else:
                        row[f"basic_{i}"] = val

            desc = rec[3]
            if isinstance(desc, list):
                for i, val in enumerate(desc):
                    row[f"x{i}"] = val

            rows.append(row)

    df = pd.DataFrame(rows)

    # Convert numeric columns.
    for c in df.columns:
        if c not in ["raw_id", "smiles"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def scatter_descriptor_grid(df):
    candidates = [
        "mol_weight",
        "exact_mass",
        "polarizability",
        "dipole_moment",
        "heat_formation",
        "homo",
        "lumo",
    ]

    # Keep columns that exist and have enough data.
    cols = []
    for c in candidates:
        if c in df.columns and df[c].notna().sum() > 100:
            cols.append(c)

    # Add strongest descriptor columns by correlation with gap.
    desc_cols = [c for c in df.columns if c.startswith("x")]
    corrs = []
    for c in desc_cols:
        valid = df[c].notna() & df["gap"].notna()
        if valid.sum() > 500 and df[c].nunique(dropna=True) > 10:
            r = df.loc[valid, c].corr(df.loc[valid, "gap"])
            if pd.notna(r):
                corrs.append((c, abs(r), r))

    for c, _, _ in sorted(corrs, key=lambda t: t[1], reverse=True)[:3]:
        if c not in cols:
            cols.append(c)

    cols = cols[:8]

    if not cols:
        print("No scatter columns found.")
        return

    n = len(cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axes = np.array(axes).ravel()

    for ax, c in zip(axes, cols):
        valid = df[c].notna() & df["gap"].notna()
        plot_df = df.loc[valid, [c, "gap"]].copy()

        if len(plot_df) > 7000:
            plot_df = plot_df.sample(7000, random_state=42)

        r = plot_df[c].corr(plot_df["gap"])

        ax.scatter(plot_df[c], plot_df["gap"], s=4, alpha=0.35)

        # Regression line.
        try:
            x = plot_df[c].to_numpy()
            y = plot_df["gap"].to_numpy()
            if np.nanstd(x) > 0:
                m, b = np.polyfit(x, y, 1)
                xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
                ax.plot(xs, m * xs + b, linewidth=1.5)
        except Exception:
            pass

        ax.set_title(f"{c} vs gap (r={r:.2f})", fontsize=10)
        ax.set_xlabel(c)
        ax.set_ylabel("HOMO-LUMO gap (eV)")
        ax.grid(True, alpha=0.25)

    for ax in axes[len(cols):]:
        ax.axis("off")

    fig.suptitle("Molecular properties and descriptors vs HOMO-LUMO gap", fontsize=14)
    savefig("figure_descriptor_scatter_grid.png")


def correlation_heatmap(df):
    base_cols = [
        "gap",
        "mol_weight",
        "exact_mass",
        "dipole_moment",
        "heat_formation",
        "polarizability",
        "homo",
        "lumo",
    ]

    cols = [c for c in base_cols if c in df.columns]

    # Add strongest descriptor columns, but keep figure readable.
    desc_cols = [c for c in df.columns if c.startswith("x")]
    corrs = []
    for c in desc_cols:
        valid = df[c].notna() & df["gap"].notna()
        if valid.sum() > 500 and df[c].nunique(dropna=True) > 10:
            r = df.loc[valid, c].corr(df.loc[valid, "gap"])
            if pd.notna(r):
                corrs.append((c, abs(r)))

    for c, _ in sorted(corrs, key=lambda t: t[1], reverse=True)[:4]:
        if c not in cols:
            cols.append(c)

    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr.values, vmin=-1, vmax=1)

    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)

    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr.values[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)

    ax.set_title("Correlation matrix of molecular features", fontsize=13)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation")
    savefig("figure_correlation_matrix.png")


def model_metric_barplot():
    if not METRICS_CSV.exists():
        print(f"Missing {METRICS_CSV}")
        return

    df = pd.read_csv(METRICS_CSV)
    test = df[df["name"].str.startswith("TEST")].copy()
    test["model"] = test["name"].str.replace("TEST ", "", regex=False)

    plt.figure(figsize=(10, 5))
    x = np.arange(len(test))
    plt.bar(x, test["mae"])
    plt.xticks(x, test["model"], rotation=45, ha="right")
    plt.ylabel("MAE (eV)")
    plt.title("Final held-out real-reference benchmark performance")
    plt.grid(axis="y", alpha=0.25)

    for i, v in enumerate(test["mae"]):
        plt.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    savefig("figure_final_model_mae.png")


def domain_mae_plot():
    if not DOMAIN_CSV.exists():
        print(f"Missing {DOMAIN_CSV}")
        return

    df = pd.read_csv(DOMAIN_CSV)

    dom = df[df["group_type"] == "domain"].copy()
    dom = dom.sort_values("mae_stack", ascending=True)

    plt.figure(figsize=(10, 5))
    x = np.arange(len(dom))
    plt.bar(x, dom["mae_stack"])
    plt.xticks(x, dom["group"], rotation=35, ha="right")
    plt.ylabel("MAE (eV)")
    plt.title("Held-out MAE by molecular domain")
    plt.grid(axis="y", alpha=0.25)

    for i, v in enumerate(dom["mae_stack"]):
        plt.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    savefig("figure_domain_mae.png")

    ref = df[df["group_type"] == "ref_source"].copy()
    ref = ref.sort_values("mae_stack", ascending=True)

    plt.figure(figsize=(8, 5))
    x = np.arange(len(ref))
    plt.bar(x, ref["mae_stack"])
    plt.xticks(x, ref["group"], rotation=30, ha="right")
    plt.ylabel("MAE (eV)")
    plt.title("Held-out MAE by reference-label source")
    plt.grid(axis="y", alpha=0.25)

    for i, v in enumerate(ref["mae_stack"]):
        plt.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    savefig("figure_reference_source_mae.png")


def gap_distribution_plot(df):
    plt.figure(figsize=(8, 5))
    plt.hist(df["gap"].dropna(), bins=60)
    plt.xlabel("HOMO-LUMO gap (eV)")
    plt.ylabel("Molecule count")
    plt.title("Distribution of PQR HOMO-LUMO gaps")
    plt.grid(axis="y", alpha=0.25)
    savefig("figure_gap_distribution.png")


def homo_lumo_gap_check(df):
    if "homo" not in df.columns or "lumo" not in df.columns:
        return

    check = df[["homo", "lumo", "gap"]].dropna().copy()
    check["lumo_minus_homo"] = check["lumo"] - check["homo"]
    check["difference"] = check["gap"] - check["lumo_minus_homo"]

    sample = check.sample(min(len(check), 7000), random_state=42)

    plt.figure(figsize=(6, 6))
    plt.scatter(sample["lumo_minus_homo"], sample["gap"], s=5, alpha=0.35)

    lo = min(sample["lumo_minus_homo"].min(), sample["gap"].min())
    hi = max(sample["lumo_minus_homo"].max(), sample["gap"].max())
    plt.plot([lo, hi], [lo, hi], linewidth=1.5)

    mae = np.mean(np.abs(sample["difference"]))
    plt.xlabel("LUMO - HOMO from orbital columns (eV)")
    plt.ylabel("Stored gap label (eV)")
    plt.title(f"Internal consistency check: gap vs LUMO-HOMO\nMAE difference={mae:.4f} eV")
    plt.grid(True, alpha=0.25)
    savefig("figure_gap_internal_consistency.png")


def main():
    print("Loading PQR list-format JSONL...")
    df = load_pqr_list_jsonl(PQR_JSONL)

    print(f"Loaded rows: {len(df):,}")
    print(f"Loaded columns: {len(df.columns):,}")
    print("First columns:", list(df.columns[:20]))

    if "gap" not in df.columns:
        raise RuntimeError("Still could not find gap. The parser needs another adjustment.")

    print("\nGap summary:")
    print(df["gap"].describe())

    scatter_descriptor_grid(df)
    correlation_heatmap(df)
    gap_distribution_plot(df)
    homo_lumo_gap_check(df)

    model_metric_barplot()
    domain_mae_plot()

    print("\nDone. Figures are in:", OUTDIR.resolve())


if __name__ == "__main__":
    main()
