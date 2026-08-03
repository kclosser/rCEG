from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RUN = Path("runs/pqr_qmugs_strict_external_LEAK094")
OUT = RUN / "figures"
OUT.mkdir(parents=True, exist_ok=True)

input_path = RUN / "qmugs_external_mae_by_domain.csv"

if not input_path.exists():
    raise FileNotFoundError(f"Missing input file: {input_path}")

domain = pd.read_csv(input_path)

required = {
    "domain",
    "n",
    "mae_domain_expert",
    "mae_offset_aligned",
}

missing = required - set(domain.columns)
if missing:
    raise RuntimeError(
        f"Missing required columns: {sorted(missing)}\n"
        f"Available columns: {list(domain.columns)}"
    )

domain = domain.sort_values(
    "mae_offset_aligned",
    ascending=True,
).reset_index(drop=True)

# Cleaner labels for publication.
label_map = {
    "qm9_like_small_organic": "QM9-like\nsmall organic",
    "near_qm9_larger_organic": "Near-QM9\nlarger organic",
    "large_neutral_organic": "Large neutral\norganic",
    "heteroatom_rich_non_qm9": "Heteroatom-rich\nnon-QM9",
    "charged_or_radical": "Charged or\nradical",
}

labels = [
    label_map.get(
        value,
        str(value).replace("_", " ").title(),
    )
    for value in domain["domain"]
]

positions = np.arange(len(domain))

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(
    positions,
    domain["mae_offset_aligned"],
    width=0.68,
)

ax.set_xticks(positions)
ax.set_xticklabels(labels)

ax.set_ylabel("Cross-fitted offset-aligned MAE (eV)")
ax.set_title(
    "Externally Calibrated Performance on QMugs by Chemical Domain",
    pad=16,
)

ax.grid(
    axis="y",
    alpha=0.25,
)

ax.set_axisbelow(True)

upper = float(domain["mae_offset_aligned"].max())
ax.set_ylim(0, upper * 1.23)

for bar, (_, row) in zip(bars, domain.iterrows()):
    value = float(row["mae_offset_aligned"])
    count = int(row["n"])

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + upper * 0.025,
        f"{value:.3f} eV\nn={count:,}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

fig.text(
    0.5,
    0.01,
    (
        "Predictions were recalibrated using a five-fold cross-fitted "
        "constant offset; each molecule was evaluated using an offset "
        "estimated without its own label."
    ),
    ha="center",
    fontsize=9,
)

fig.tight_layout(rect=[0, 0.055, 1, 1])

output_path = (
    OUT /
    "figure_qmugs_offset_aligned_mae_by_domain.png"
)

fig.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

# Save the exact plotted values as figure data.
figure_data = domain[
    [
        "domain",
        "n",
        "mae_domain_expert",
        "mae_offset_aligned",
        "mean_reference_gap_ev",
    ]
].copy()

figure_data.to_csv(
    OUT / "figure_qmugs_offset_aligned_mae_by_domain_data.csv",
    index=False,
)

print("Saved figure:")
print(output_path)

print("\nSaved figure data:")
print(
    OUT /
    "figure_qmugs_offset_aligned_mae_by_domain_data.csv"
)

print("\nPlotted values:")
print(
    domain[
        ["domain", "n", "mae_offset_aligned"]
    ].to_string(index=False)
)
