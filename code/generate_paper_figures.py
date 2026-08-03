from pathlib import Path
import json
import math
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
PREDICTIONS_CSV = FINAL_DIR / "all_pqr_predictions_with_label_source.csv"


def savefig(name):
    path = OUTDIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote {path}")


def flatten_record(record, parent_key="", sep="_"):
    """Flatten nested dictionaries/lists from JSONL into one row."""
    items = {}

    if isinstance(record, dict):
        iterator = record.items()
    else:
        return {str(parent_key or "value"): record}

    for k, v in iterator:
        k = str(k)
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            items.update(flatten_record(v, new_key, sep=sep))
        elif isinstance(v, list):
            # If list is numeric, expand it as descriptor-like columns.
            if all(isinstance(x, (int, float, type(None))) for x in v):
                for j, x in enumerate(v):
                    items[f"{new_key}_{j}"] = x
            else:
                items[new_key] = json.dumps(v)
        else:
            items[new_key] = v

    return items


def load_jsonl_sample(path, max_rows=None):
    rows = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            try:
                rec = json.loads(line)
                rows.append(flatten_record(rec))
            except Exception:
                continue

    df = pd.DataFrame(rows)
    df.columns = [str(c) for c in df.columns]
    return df


def pick_column(df, candidates):
    df.columns = [str(c) for c in df.columns]
    cols_lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        c = str(c)
        if c in df.columns:
            return c
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    return None


def numeric_clean(s):
    return pd.to_numeric(s, errors="coerce")


def find_gap_column(df):
    candidates = [
        "gap", "pqr_gap", "homo_lumo_gap", "lumo_minus_homo",
        "target", "y", "HOMO-LUMO gap", "HOMO_LUMO_gap",
        "properties_gap", "props_gap", "data_gap",
        "properties_lumo_minus_homo", "props_lumo_minus_homo",
    ]

    direct = pick_column(df, candidates)
    if direct is not None:
        return direct

    # Fallback: find any column containing gap or lumo_minus_homo.
    possible = []
    for c in df.columns:
        cl = str(c).lower()
        if "lumo_minus_homo" in cl or cl.endswith("_gap") or cl == "gap":
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.notna().sum() >= 100:
                possible.append(c)

    if possible:
        print("Auto-selected gap-like column:", possible[0])
        return possible[0]

    return None


def find_descriptor_columns(df):
    # Try chemically meaningful columns first.
    desired = [
        "mol_weight", "exact_mass", "polarizability", "dipole_moment",
        "heat_formation", "homo", "lumo", "heavy_atoms", "num_atoms",
        "num_rings", "tpsa", "logp", "qed"
    ]

    found = []
    for c in desired:
        col = pick_column(df, [c])
        if col and col not in found:
            found.append(col)

    # Add descriptor columns if available.
    desc_cols = [
        c for c in df.columns
        if str(c).startswith("desc_")
        or str(c).startswith("x")
        or str(c).startswith("d")
        or "descriptor" in str(c).lower()
        or "mordred" in str(c).lower()
        or "padel" in str(c).lower()
        or "features_" in str(c).lower()
        or "descriptors_" in str(c).lower()
    ]

    # Keep only numeric columns with enough non-null values.
    candidates = found + desc_cols
    good = []
    for c in candidates:
        vals = numeric_clean(df[c])
        if vals.notna().sum() >= 100 and vals.nunique(dropna=True) > 5:
            good.append(c)

    # Remove duplicates while preserving order.
    seen = set()
    out = []
    for c in good:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def scatter_descriptor_grid(df, gap_col, descriptor_cols, name="figure_descriptor_scatter_grid.png"):
    # Use the first 5 strongest correlated descriptors.
    records = []
    y = numeric_clean(df[gap_col])
    for c in descriptor_cols:
        x = numeric_clean(df[c])
        valid = x.notna() & y.notna()
        if valid.sum() < 100:
            continue
        corr = x[valid].corr(y[valid])
        if pd.notna(corr):
            records.append((c, corr, abs(corr), valid.sum()))

    if not records:
        print("No descriptor scatter columns found.")
        return

    records = sorted(records, key=lambda t: t[2], reverse=True)[:5]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.ravel()

    for ax, (c, corr, _, n) in zip(axes, records):
        x = numeric_clean(df[c])
        y = numeric_clean(df[gap_col])
        valid = x.notna() & y.notna()

        # Downsample for readability.
        plot_df = pd.DataFrame({"x": x[valid], "y": y[valid]})
        if len(plot_df) > 6000:
            plot_df = plot_df.sample(6000, random_state=42)

        ax.scatter(plot_df["x"], plot_df["y"], s=4, alpha=0.35)

        # Regression line.
        try:
            xvals = plot_df["x"].to_numpy()
            yvals = plot_df["y"].to_numpy()
            if np.nanstd(xvals) > 0:
                m, b = np.polyfit(xvals, yvals, 1)
                xs = np.linspace(np.nanmin(xvals), np.nanmax(xvals), 100)
                ax.plot(xs, m * xs + b, linewidth=1.5)
        except Exception:
            pass

        ax.set_title(f"{c} vs {gap_col} (r={corr:.2f})", fontsize=9)
        ax.set_xlabel(c, fontsize=8)
        ax.set_ylabel(gap_col, fontsize=8)
        ax.grid(True, alpha=0.25)

    # Hide unused subplot.
    for ax in axes[len(records):]:
        ax.axis("off")

    fig.suptitle("Descriptor relationships with HOMO-LUMO gap", fontsize=14)
    savefig(name)


def correlation_heatmap(df, gap_col, descriptor_cols, name="figure_correlation_matrix.png"):
    # Pick gap plus top correlated descriptors, and include homo/lumo if present.
    y = numeric_clean(df[gap_col])

    records = []
    for c in descriptor_cols:
        x = numeric_clean(df[c])
        valid = x.notna() & y.notna()
        if valid.sum() < 100:
            continue
        corr = x[valid].corr(y[valid])
        if pd.notna(corr):
            records.append((c, abs(corr)))

    top = [c for c, _ in sorted(records, key=lambda t: t[1], reverse=True)[:7]]

    cols = [gap_col] + top
    # Keep unique columns.
    seen = set()
    cols = [c for c in cols if not (c in seen or seen.add(c))]

    corr_df = pd.DataFrame({c: numeric_clean(df[c]) for c in cols}).corr()

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr_df.values, vmin=-1, vmax=1)

    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)

    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr_df.values[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)

    ax.set_title("Correlation matrix of numeric molecular features", fontsize=13)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation")
    savefig(name)


def metrics_barplot(metrics_path, name="figure_final_model_mae.png"):
    if not metrics_path.exists():
        print(f"Missing {metrics_path}")
        return

    df = pd.read_csv(metrics_path)
    test = df[df["name"].str.startswith("TEST")].copy()
    if test.empty:
        test = df.copy()

    test["model"] = test["name"].str.replace("TEST ", "", regex=False)

    plt.figure(figsize=(10, 5))
    x = np.arange(len(test))
    plt.bar(x, test["mae"])
    plt.xticks(x, test["model"], rotation=45, ha="right")
    plt.ylabel("MAE (eV)")
    plt.title("Final held-out reference benchmark performance")
    plt.grid(axis="y", alpha=0.25)

    for i, v in enumerate(test["mae"]):
        plt.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    savefig(name)


def domain_mae_plot(domain_path, name="figure_domain_mae.png"):
    if not domain_path.exists():
        print(f"Missing {domain_path}")
        return

    df = pd.read_csv(domain_path)

    dom = df[df["group_type"] == "domain"].copy()
    if not dom.empty:
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

        savefig(name)

    ref = df[df["group_type"] == "ref_source"].copy()
    if not ref.empty:
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


def predicted_vs_reference_plot(pred_path, name="figure_predicted_vs_reference.png"):
    if not pred_path.exists():
        print(f"Missing {pred_path}")
        return

    df = pd.read_csv(pred_path)

    # Try common reference/prediction columns.
    ref_col = pick_column(df, [
        "reference_gap", "ref_gap", "gap", "target_gap", "y_true",
        "true_gap", "label"
    ])
    pred_col = pick_column(df, [
        "stack_reference_validated", "pred_stack", "prediction",
        "pred_gap", "calibrated_gap", "final_pred"
    ])

    if ref_col is None or pred_col is None:
        print("Could not find reference/prediction columns in predictions file.")
        print("Columns:", list(df.columns)[:80])
        return

    x = numeric_clean(df[ref_col])
    y = numeric_clean(df[pred_col])
    valid = x.notna() & y.notna()
    if valid.sum() < 10:
        print("Not enough valid prediction/reference rows.")
        return

    plot_df = pd.DataFrame({"Reference": x[valid], "Predicted": y[valid]})
    if len(plot_df) > 8000:
        plot_df = plot_df.sample(8000, random_state=42)

    plt.figure(figsize=(6, 6))
    plt.scatter(plot_df["Reference"], plot_df["Predicted"], s=8, alpha=0.35)

    lo = min(plot_df["Reference"].min(), plot_df["Predicted"].min())
    hi = max(plot_df["Reference"].max(), plot_df["Predicted"].max())
    plt.plot([lo, hi], [lo, hi], linewidth=1.5)

    mae = np.mean(np.abs(plot_df["Predicted"] - plot_df["Reference"]))
    plt.xlabel(f"Reference HOMO-LUMO gap ({ref_col})")
    plt.ylabel(f"Predicted HOMO-LUMO gap ({pred_col})")
    plt.title(f"Predicted vs reference HOMO-LUMO gap\nMAE={mae:.3f} eV")
    plt.grid(True, alpha=0.25)
    savefig(name)


def main():
    print("Loading PQR data...")
    df = load_jsonl_sample(PQR_JSONL)

    print(f"Loaded rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    gap_col = find_gap_column(df)
    if gap_col is None:
        raise RuntimeError("Could not find gap column. Check column names.")

    descriptor_cols = find_descriptor_columns(df)
    print("Gap column:", gap_col)
    print("Descriptor columns selected:", descriptor_cols[:20])

    scatter_descriptor_grid(df, gap_col, descriptor_cols)
    correlation_heatmap(df, gap_col, descriptor_cols)

    metrics_barplot(METRICS_CSV)
    domain_mae_plot(DOMAIN_CSV)
    predicted_vs_reference_plot(PREDICTIONS_CSV)

    print("\nDone. Figures are in:", OUTDIR.resolve())


if __name__ == "__main__":
    main()
