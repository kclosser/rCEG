from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RUN = Path("runs/pqr_qmugs_strict_external_LEAK094")
OUT = RUN / "figures"
OUT.mkdir(parents=True, exist_ok=True)

pred = pd.read_csv(RUN / "qmugs_external_predictions.csv")
ext_metrics = pd.read_csv(RUN / "qmugs_external_metrics.csv")
domain = pd.read_csv(RUN / "qmugs_external_mae_by_domain.csv")
internal = pd.read_csv(RUN / "FINAL_reference_holdout_metrics.csv")

# ------------------------------------------------------------
# Comparison data file
# ------------------------------------------------------------
internal_domain = internal[
    internal["name"] == "TEST domain_expert"
].iloc[0]

external_raw = ext_metrics[
    (ext_metrics["evaluation"] == "strict_frozen_external")
    & (ext_metrics["model"] == "domain_expert")
].iloc[0]

external_aligned = ext_metrics[
    ext_metrics["evaluation"]
    == "five_fold_crossfit_offset_aligned"
].iloc[0]

comparison = pd.DataFrame([
    {
        "benchmark": "Internal held-out real-reference test",
        "model": "rCEG chemical-domain expert",
        "mae_ev": internal_domain["mae"],
        "n": int(internal_domain["n"]),
        "comparison_type": "Internal reference holdout",
        "directly_comparable": True,
    },
    {
        "benchmark": "QMugs strict frozen external transfer",
        "model": "rCEG chemical-domain expert",
        "mae_ev": external_raw["mae"],
        "n": int(external_raw["n"]),
        "comparison_type": (
            "Unseen molecules; external omegaB97X-D/def2-SVP labels"
        ),
        "directly_comparable": False,
    },
    {
        "benchmark": "QMugs cross-fitted offset-aligned transfer",
        "model": "rCEG chemical-domain expert",
        "mae_ev": external_aligned["mae"],
        "n": int(external_aligned["n"]),
        "comparison_type": (
            "Secondary diagnostic correcting computational-method offset"
        ),
        "directly_comparable": False,
    },
    {
        "benchmark": "Published QMugs test",
        "model": "DelFTa direct learning",
        "mae_ev": 0.0529,
        "n": 88000,
        "comparison_type": (
            "Published model trained directly on QMugs conformers"
        ),
        "directly_comparable": False,
    },
    {
        "benchmark": "Published QMugs test",
        "model": "DelFTa delta learning",
        "mae_ev": 0.0473,
        "n": 88000,
        "comparison_type": (
            "Published model trained on QMugs with GFN2-xTB baseline"
        ),
        "directly_comparable": False,
    },
])

comparison.to_csv(
    RUN / "qmugs_external_published_comparison.csv",
    index=False,
)

# ------------------------------------------------------------
# Figure 1: predicted versus reference
# ------------------------------------------------------------
x = pred["ref_gap"].to_numpy()
y = pred["pred_domain_expert"].to_numpy()

lo = min(x.min(), y.min())
hi = max(x.max(), y.max())

plt.figure(figsize=(6.5, 6.5))
plt.scatter(x, y, s=10, alpha=0.35)
plt.plot([lo, hi], [lo, hi], linewidth=1.5)

raw_mae = np.mean(np.abs(x - y))

plt.xlabel("QMugs DFT HOMO-LUMO gap (eV)")
plt.ylabel("Frozen rCEG prediction (eV)")
plt.title(
    "Strict External Validation on QMugs\n"
    f"MAE = {raw_mae:.3f} eV; n = {len(pred):,}"
)
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(
    OUT / "figure_qmugs_predicted_vs_reference.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# ------------------------------------------------------------
# Figure 2: MAE by chemical domain
# ------------------------------------------------------------
domain = domain.sort_values("mae_domain_expert")

plt.figure(figsize=(9, 5))
positions = np.arange(len(domain))

plt.bar(
    positions,
    domain["mae_domain_expert"],
)

plt.xticks(
    positions,
    domain["domain"],
    rotation=35,
    ha="right",
)
plt.ylabel("External MAE (eV)")
plt.title("QMugs External MAE by Chemical Domain")
plt.grid(axis="y", alpha=0.25)

for i, (_, row) in enumerate(domain.iterrows()):
    plt.text(
        i,
        row["mae_domain_expert"] + 0.01,
        f'{row["mae_domain_expert"]:.3f}\nn={int(row["n"])}',
        ha="center",
        fontsize=8,
    )

plt.tight_layout()
plt.savefig(
    OUT / "figure_qmugs_mae_by_domain.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# ------------------------------------------------------------
# Figure 3: contextual benchmark comparison
# ------------------------------------------------------------
plot_data = comparison.copy()

plt.figure(figsize=(10, 5.5))
positions = np.arange(len(plot_data))

plt.bar(
    positions,
    plot_data["mae_ev"],
)

labels = [
    "Internal\nrCEG",
    "External QMugs\nrCEG raw",
    "External QMugs\nrCEG aligned",
    "DelFTa\ndirect",
    "DelFTa\ndelta",
]

plt.xticks(positions, labels)
plt.ylabel("MAE (eV)")
plt.title(
    "HOMO-LUMO Gap Performance Context\n"
    "Published DelFTa values are not matched transfer benchmarks"
)
plt.grid(axis="y", alpha=0.25)

for i, value in enumerate(plot_data["mae_ev"]):
    plt.text(
        i,
        value + max(plot_data["mae_ev"]) * 0.02,
        f"{value:.3f}",
        ha="center",
        fontsize=9,
    )

plt.tight_layout()
plt.savefig(
    OUT / "figure_qmugs_published_context_comparison.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

print(comparison.to_string(index=False))
print("\nWrote:")
print(RUN / "qmugs_external_published_comparison.csv")
print(OUT / "figure_qmugs_predicted_vs_reference.png")
print(OUT / "figure_qmugs_mae_by_domain.png")
print(OUT / "figure_qmugs_published_context_comparison.png")
