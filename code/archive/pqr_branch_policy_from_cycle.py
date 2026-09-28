import pandas as pd
from pathlib import Path

inp = Path("runs/pqr_full_domain_moe_qm9_cycle/all_pqr_predictions_with_label_source.csv")
outdir = Path("runs/pqr_full_domain_moe_qm9_cycle")

df = pd.read_csv(inp)

def choose_branch(row):
    domain = row["domain"]
    cal = row["calibration_confidence"]
    cyc = row["cycle_confidence"]
    cycle_error = row["cycle_error"]

    if cal == "high_in_domain" and cyc in ["high_cycle_consistency", "medium_cycle_consistency"]:
        return "qm9_aligned_high_confidence_expert"

    if domain in ["qm9_like_small_organic", "near_qm9_larger_organic"] and cal != "low_out_of_domain" and cycle_error <= 1.0:
        return "qm9_aligned_medium_confidence_expert"

    if cycle_error > 0.50 or cal == "low_out_of_domain":
        return "raw_pqr_expert_pending_reference"

    return "hybrid_review_expert"

df["moe_branch_policy"] = df.apply(choose_branch, axis=1)

# Priority for recomputation: broad domains, high cycle error, low calibration confidence.
priority_domains = {
    "heteroatom_rich_non_qm9": 0,
    "charged_or_radical": 1,
    "large_neutral_organic": 2,
    "near_qm9_larger_organic": 3,
    "qm9_like_small_organic": 4,
}
df["recompute_priority_domain"] = df["domain"].map(priority_domains).fillna(9)

recompute = df[df["moe_branch_policy"] == "raw_pqr_expert_pending_reference"].copy()
recompute = recompute.sort_values(
    ["recompute_priority_domain", "cycle_error", "calibration_distance"],
    ascending=[True, False, False],
)

# Make a balanced recomputation subset.
samples = []
for domain, g in recompute.groupby("domain"):
    if domain in ["heteroatom_rich_non_qm9", "charged_or_radical", "large_neutral_organic"]:
        n = min(250, len(g))
    else:
        n = min(100, len(g))

    try:
        g["gap_bin"] = pd.qcut(g["pred_qm9_aligned_gap"], q=min(5, len(g)), duplicates="drop")
        s = g.groupby("gap_bin", group_keys=False).apply(
            lambda x: x.sample(min(len(x), max(1, n // max(1, g["gap_bin"].nunique()))), random_state=42)
        )
        samples.append(s.head(n))
    except Exception:
        samples.append(g.head(n))

cand = pd.concat(samples, ignore_index=True).drop_duplicates("smiles")

df.to_csv(outdir / "pqr_moe_branch_policy.csv", index=False)
cand.to_csv(outdir / "pqr_next_reference_recompute_set.csv", index=False)

summary = df.groupby(["domain", "calibration_confidence", "cycle_confidence", "moe_branch_policy"]).size().reset_index(name="n")
summary.to_csv(outdir / "branch_policy_summary.csv", index=False)

print("Wrote:")
print(outdir / "pqr_moe_branch_policy.csv")
print(outdir / "branch_policy_summary.csv")
print(outdir / "pqr_next_reference_recompute_set.csv")

print("\nBranch policy counts:")
print(df["moe_branch_policy"].value_counts())

print("\nBranch policy by domain:")
print(pd.crosstab(df["domain"], df["moe_branch_policy"]))

print("\nNext recomputation set:")
print(cand["domain"].value_counts())
print("\nRecompute candidate count:", len(cand))
