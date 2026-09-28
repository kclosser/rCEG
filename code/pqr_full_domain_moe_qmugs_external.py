#!/usr/bin/env python3
"""
pqr_full_domain_moe.py

Full PQR domain-aware calibrated mixture-of-experts model.

Purpose:
- Train on the full PQR dataset, not only QM9-overlap molecules.
- Use QM9 overlap and optional recomputed PQR reference labels as real anchors.
- Use calibrated pseudo-labels for the rest of PQR.
- Evaluate honestly only on held-out real reference labels.
- Report MAE by domain, confidence, and label source.

No HOMO/LUMO leakage:
- pqr[5] and pqr[6] are never used as features.
- pqr_gap is used only for calibration/teacher pseudo-label generation.
- Final predictors use molecular features only, not pqr_gap or calibrated_gap.
"""

from __future__ import annotations

import argparse
import json
import math
import zlib
from pathlib import Path
from collections import Counter

import joblib
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer


def stable_seed(s, modulo=10000):
    """
    Process-stable replacement for abs(hash(str)) % modulo.

    Python randomizes str hashes per process unless PYTHONHASHSEED is set,
    which made piecewise branch-expert seeds vary between otherwise identical
    runs. zlib.crc32 is deterministic on every platform and interpreter.
    """
    return zlib.crc32(s.encode("utf-8")) % modulo


def fnum(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else np.nan
    except Exception:
        return np.nan


def canon(smiles, isomeric=True):
    try:
        mol = Chem.MolFromSmiles(str(smiles), sanitize=True)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)
    except Exception:
        return None


def classify_domain(mol):
    atoms = [a.GetAtomicNum() for a in mol.GetAtoms()]
    allowed_qm9 = {1, 6, 7, 8, 9}
    allowed = all(a in allowed_qm9 for a in atoms)
    has_carbon = any(a == 6 for a in atoms)
    heavy = mol.GetNumHeavyAtoms()
    charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    radicals = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    single = len(Chem.GetMolFrags(mol)) == 1

    if allowed and has_carbon and single and charge == 0 and radicals == 0 and heavy <= 9:
        return "qm9_like_small_organic"
    if allowed and has_carbon and single and charge == 0 and radicals == 0 and heavy <= 20:
        return "near_qm9_larger_organic"
    if allowed and has_carbon and single and charge == 0:
        return "large_neutral_organic"
    if charge != 0 or radicals != 0:
        return "charged_or_radical"
    return "heteroatom_rich_non_qm9"


def rdkit_features(mol):
    atoms = list(mol.GetAtoms())
    bonds = list(mol.GetBonds())
    return [
        Descriptors.MolWt(mol),
        Descriptors.ExactMolWt(mol),
        Descriptors.HeavyAtomMolWt(mol),
        Descriptors.NumValenceElectrons(mol),
        mol.GetNumHeavyAtoms(),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumAliphaticRings(mol),
        rdMolDescriptors.CalcNumSaturatedRings(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcTPSA(mol),
        Crippen.MolLogP(mol),
        Crippen.MolMR(mol),
        Lipinski.NumRotatableBonds(mol),
        Lipinski.NumHeteroatoms(mol),
        Lipinski.FractionCSP3(mol),
        sum(a.GetIsAromatic() for a in atoms),
        sum(b.GetIsAromatic() for b in bonds),
        sum(b.GetBondTypeAsDouble() == 2 for b in bonds),
        sum(b.GetBondTypeAsDouble() == 3 for b in bonds),
        sum(a.GetAtomicNum() == 6 for a in atoms),
        sum(a.GetAtomicNum() == 7 for a in atoms),
        sum(a.GetAtomicNum() == 8 for a in atoms),
        sum(a.GetAtomicNum() == 9 for a in atoms),
        sum(a.GetAtomicNum() == 16 for a in atoms),
        sum(a.GetAtomicNum() == 17 for a in atoms),
        sum(a.GetAtomicNum() == 35 for a in atoms),
        sum(a.GetAtomicNum() == 53 for a in atoms),
        sum(a.GetAtomicNum() == 15 for a in atoms),
        sum(a.GetAtomicNum() == 5 for a in atoms),
    ]


def bond_step_features(mol):
    try:
        dm = Chem.GetDistanceMatrix(mol).astype(float)
        upper = dm[np.triu_indices_from(dm, k=1)]
        upper = upper[np.isfinite(upper)]
        if len(upper) == 0:
            return [0.0] * 16

        out = [
            np.max(upper),
            np.mean(upper),
            np.std(upper),
            np.median(upper),
            np.percentile(upper, 25),
            np.percentile(upper, 75),
        ]

        for d in range(1, 11):
            out.append(float(np.sum(upper == d)) / len(upper))

        return out
    except Exception:
        return [np.nan] * 16


def load_pqr(path):
    rows = []
    feats = []
    audit_rows = []
    drops = Counter()

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip().rstrip(",")
            if not line:
                continue

            e = json.loads(line)
            if not (isinstance(e, list) and len(e) >= 5 and isinstance(e[1], str)):
                drops["bad_shape"] += 1
                continue

            smiles_raw = e[1]
            pqr_gap = fnum(e[4])
            pqr = e[2] if isinstance(e[2], list) else []
            lasso = e[3] if isinstance(e[3], list) else []

            if not math.isfinite(pqr_gap):
                drops["bad_gap"] += 1
                continue

            mol = Chem.MolFromSmiles(smiles_raw, sanitize=True)
            if mol is None:
                drops["bad_smiles"] += 1
                continue

            smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

            # Non-leakage features only:
            # pqr[:5] excludes pqr[5]=HOMO and pqr[6]=LUMO.
            x = []
            x.extend([fnum(v) for v in pqr[:5]])
            x.extend([fnum(v) for v in lasso])
            x.extend(rdkit_features(mol))
            x.extend(bond_step_features(mol))

            arr = np.array(x, dtype=np.float64)
            arr[~np.isfinite(arr)] = np.nan
            arr = np.clip(arr, -1e6, 1e6)

            rows.append({
                "smiles": smiles,
                "pqr_gap": pqr_gap,
                "domain": classify_domain(mol),
                "heavy_atoms": mol.GetNumHeavyAtoms(),
            })
            feats.append(arr.astype(np.float32))

            if isinstance(pqr, list) and len(pqr) >= 7:
                audit_rows.append({
                    "smiles": smiles,
                    "gap": pqr_gap,
                    "homo": fnum(pqr[5]),
                    "lumo": fnum(pqr[6]),
                    "lumo_minus_homo": fnum(pqr[6]) - fnum(pqr[5]),
                })

    max_len = max(len(x) for x in feats)
    X = np.full((len(feats), max_len), np.nan, dtype=np.float32)
    for i, x in enumerate(feats):
        X[i, :len(x)] = x

    df = pd.DataFrame(rows)
    Xdf = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    df = pd.concat([df, Xdf], axis=1)

    feat_cols = [c for c in df.columns if c.startswith("x")]

    agg = {"pqr_gap": "median", "domain": "first", "heavy_atoms": "median"}
    for c in feat_cols:
        agg[c] = "median"

    df = df.groupby("smiles", as_index=False).agg(agg)
    audit = pd.DataFrame(audit_rows).drop_duplicates("smiles")

    print(f"PQR unique molecules: {len(df):,}")
    print(f"PQR drops: {dict(drops)}")
    print("PQR domains:")
    print(df["domain"].value_counts().to_string())

    return df, audit


def load_reference(path, source_name):
    ref = pd.read_csv(path)
    if "smiles" not in ref.columns:
        raise ValueError(f"{path} must contain a smiles column.")

    gap_col = None
    for c in ["gap", "ref_gap", "homo_lumo_gap", "HOMO_LUMO_gap", "deltaE", "DeltaEHL", "gap_ev"]:
        if c in ref.columns:
            gap_col = c
            break

    if gap_col is None:
        raise ValueError(f"{path} must contain a gap/ref_gap column.")

    rows = []
    for _, r in ref.iterrows():
        smi = canon(r["smiles"], isomeric=True)
        g = fnum(r[gap_col])
        if smi is not None and math.isfinite(g):
            rows.append({"smiles": smi, "ref_gap": g, "ref_source": source_name})

    out = pd.DataFrame(rows)
    out = out.groupby("smiles", as_index=False).agg({
        "ref_gap": "median",
        "ref_source": "first",
    })

    print(f"Reference {source_name}: {len(out):,} unique molecules")
    return out


def leakage_filter(df, audit, threshold, outdir):
    feat_cols = [c for c in df.columns if c.startswith("x")]
    merged = df[["smiles"] + feat_cols].merge(audit, on="smiles", how="inner")

    targets = ["gap", "homo", "lumo", "lumo_minus_homo"]
    bad = set()
    records = []

    for c in feat_cols:
        vals = pd.to_numeric(merged[c], errors="coerce")
        if vals.notna().sum() < 100:
            continue

        for t in targets:
            corr = vals.corr(merged[t])
            ac = abs(corr) if pd.notna(corr) else np.nan
            records.append({"feature": c, "target": t, "abs_corr": ac, "corr": corr})
            if pd.notna(ac) and ac >= threshold:
                bad.add(c)

    audit_df = pd.DataFrame(records).sort_values("abs_corr", ascending=False)
    audit_df.to_csv(outdir / "feature_leakage_audit.csv", index=False)

    keep = [c for c in feat_cols if c not in bad]
    print(f"Leakage filter dropped {len(bad)} features at abs_corr >= {threshold}")
    print(audit_df.head(20).to_string(index=False))
    return keep


class DeterministicPipeline(Pipeline):
    """
    Pipeline that switches its final estimator to single-threaded prediction
    after fitting.

    sklearn's forests accumulate per-tree predictions in parallel in whatever
    order the workers finish, so float non-associativity makes predict() vary
    by ~8e-15 between otherwise identical processes. Fitting is unaffected and
    stays parallel; only prediction is serialised, which is cheap. Without this
    the pipeline is reproducible to about 1e-9 but not bit-identical, because a
    value sitting on a rounding boundary can round either way.
    """

    def fit(self, X, y=None, **kw):
        super().fit(X, y, **kw)
        est = self.named_steps.get("model")
        if est is not None and hasattr(est, "n_jobs"):
            est.n_jobs = 1
        return self


def pipe(model, scale=True):
    steps = [
        ("imp", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(1e-12)),
    ]
    if scale:
        steps.append(("scale", RobustScaler(with_centering=False)))
    steps.append(("model", model))
    return DeterministicPipeline(steps)


def et(seed, n=500):
    return pipe(ExtraTreesRegressor(
        n_estimators=n,
        max_features=0.35,
        min_samples_leaf=1,
        random_state=seed,
        n_jobs=-1,
    ), scale=False)


def rf(seed, n=400):
    return pipe(RandomForestRegressor(
        n_estimators=n,
        max_features=0.35,
        min_samples_leaf=1,
        random_state=seed,
        n_jobs=-1,
    ), scale=False)


def hgb(seed):
    return DeterministicPipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(1e-12)),
        ("model", HistGradientBoostingRegressor(
            max_iter=700,
            learning_rate=0.035,
            max_leaf_nodes=31,
            l2_regularization=0.03,
            validation_fraction=0.15,
            n_iter_no_change=80,
            early_stopping=True,
            loss="absolute_error",
            random_state=seed,
        )),
    ])


def _continuous_scaler(kind):
    """
    Scaling is fitted inside each sklearn Pipeline, so validation/test
    statistics never influence training preprocessing.
    """
    if kind == "robust":
        # Wider quantile range avoids allowing a very narrow IQR to
        # exaggerate small differences in sparse/count descriptors.
        return RobustScaler(
            with_centering=True,
            with_scaling=True,
            quantile_range=(10.0, 90.0),
        )

    if kind == "standard":
        return StandardScaler(
            with_mean=True,
            with_std=True,
        )

    if kind == "power":
        # Yeo-Johnson supports zero and negative descriptor values.
        # It reduces skew and then standardizes to zero mean/unit variance.
        return PowerTransformer(
            method="yeo-johnson",
            standardize=True,
        )

    raise ValueError(f"Unknown scaler kind: {kind}")


def ridge_scaled(kind="robust"):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(1e-12)),
        ("scale", _continuous_scaler(kind)),
        ("model", RidgeCV(alphas=np.logspace(-6, 5, 60))),
    ])


def linear_scaled(kind="robust"):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(1e-12)),
        ("scale", _continuous_scaler(kind)),
        ("model", LinearRegression()),
    ])


# Preserve the original function names for inverse calibration and any
# other existing code paths.
def ridge():
    return ridge_scaled("robust")


def lin():
    return linear_scaled("robust")


def metric(name, y, pred):
    mae = mean_absolute_error(y, pred)
    r2 = r2_score(y, pred)
    print(f"{name:38s} MAE={mae:.4f} eV  R2={r2:.4f}")
    return {"name": name, "mae": float(mae), "r2": float(r2), "n": int(len(y))}


def add_confidence(pqr, ref_train, ref_val, ref_test, feat_cols):
    imp = SimpleImputer(strategy="median")
    scaler = RobustScaler()

    X_train_ref = scaler.fit_transform(imp.fit_transform(ref_train[feat_cols]))
    X_all = scaler.transform(imp.transform(pqr[feat_cols]))

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(X_train_ref)

    d_all, _ = nn.kneighbors(X_all)
    pqr["calibration_distance"] = d_all[:, 0]

    X_eval_ref = scaler.transform(imp.transform(pd.concat([ref_val, ref_test])[feat_cols]))
    d_eval, _ = nn.kneighbors(X_eval_ref)
    ref_dist = d_eval[:, 0]

    high_thr = float(np.quantile(ref_dist, 0.90))
    med_thr = float(np.quantile(ref_dist, 0.975))

    def label(d):
        if d <= high_thr:
            return "high_in_domain"
        if d <= med_thr:
            return "medium_near_domain"
        return "low_out_of_domain"

    pqr["calibration_confidence"] = [label(d) for d in pqr["calibration_distance"]]

    print("\nApplicability-domain thresholds:")
    print(f"  high <= {high_thr:.4f}")
    print(f"  medium <= {med_thr:.4f}")
    print(pqr["calibration_confidence"].value_counts().to_string())

    return pqr


def _line_mae(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(y) < 50 or np.nanstd(x) < 1e-12:
        return np.inf
    A = np.column_stack([x, np.ones(len(x))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    return float(np.mean(np.abs(pred - y)))


def _best_piecewise_split(Xdf, y, candidate_cols, min_leaf=300, max_features=40):
    """
    Find the (x-feature, breakpoint) pair whose two local linear fits beat a
    single global line by the largest margin. Uses only x-features and the
    training label -- no gap-derived quantity defines the split.
    """
    y = np.asarray(y, dtype=float)

    # Rank candidate x-features by absolute correlation with the training label.
    ranked = []
    for c in candidate_cols:
        x = pd.to_numeric(Xdf[c], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 2 * min_leaf:
            continue
        if np.nanstd(x[ok]) < 1e-12:
            continue
        corr = np.corrcoef(x[ok], y[ok])[0, 1]
        if np.isfinite(corr):
            ranked.append((c, abs(corr), corr))

    ranked = sorted(ranked, key=lambda z: z[1], reverse=True)[:max_features]

    best = None

    for c, abs_corr, corr in ranked:
        x = pd.to_numeric(Xdf[c], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        x_ok = x[ok]
        y_ok = y[ok]

        if len(y_ok) < 2 * min_leaf:
            continue

        base_mae = _line_mae(x_ok, y_ok)

        # Candidate breakpoints between the 20th and 80th percentiles.
        qs = np.linspace(0.20, 0.80, 25)
        cuts = np.unique(np.quantile(x_ok, qs))

        for cut in cuts:
            left = x_ok <= cut
            right = ~left

            if left.sum() < min_leaf or right.sum() < min_leaf:
                continue

            mae_left = _line_mae(x_ok[left], y_ok[left])
            mae_right = _line_mae(x_ok[right], y_ok[right])

            if not np.isfinite(mae_left) or not np.isfinite(mae_right):
                continue

            piece_mae = (left.sum() * mae_left + right.sum() * mae_right) / len(y_ok)
            improvement = base_mae - piece_mae

            rec = {
                "feature": c,
                "corr": float(corr),
                "abs_corr": float(abs_corr),
                "split_value": float(cut),
                "single_line_mae": float(base_mae),
                "piecewise_line_mae": float(piece_mae),
                "improvement": float(improvement),
                "left_n": int(left.sum()),
                "right_n": int(right.sum()),
            }

            if best is None or rec["improvement"] > best["improvement"]:
                best = rec

    return best


# ======================================================================
# Split protocols (spec C3)
# ======================================================================

def murcko_scaffold(smiles):
    """
    Bemis-Murcko scaffold SMILES. Acyclic molecules yield an empty scaffold;
    those are given per-molecule unique group ids rather than being collapsed
    into one giant group, which the spec calls out explicitly.
    """
    from rdkit.Chem.Scaffolds import MurckoScaffold
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return None


def assign_split_groups(smiles_list, mode, cutoff=0.6, seed=42):
    """
    Return an integer group id per molecule. Molecules sharing a group id must
    land in the same partition.

      random   -> every molecule its own group (ordinary random splitting)
      scaffold -> Bemis-Murcko scaffold; acyclic molecules stay singletons
      cluster  -> Butina clustering on Morgan-fingerprint Tanimoto

    Morgan fingerprints are used here purely as a SIMILARITY METRIC for
    grouping. Note that the descriptor block already contains 176 Morgan bits
    (see lasso_selected_feature_names.json), so this is not an independence
    claim about the feature set.
    """
    n = len(smiles_list)

    if mode == "random":
        return np.arange(n)

    if mode == "scaffold":
        groups = np.empty(n, dtype=np.int64)
        lookup = {}
        next_id = 0
        n_acyclic = 0
        for i, smi in enumerate(smiles_list):
            scaf = murcko_scaffold(smi)
            if not scaf:
                # Empty scaffold (acyclic). Keep as its own singleton group.
                groups[i] = -(i + 1)
                n_acyclic += 1
                continue
            if scaf not in lookup:
                lookup[scaf] = next_id
                next_id += 1
            groups[i] = lookup[scaf]
        print(f"    scaffold groups: {next_id:,} ring scaffolds, "
              f"{n_acyclic:,} acyclic singletons")
        return groups

    if mode == "cluster":
        from rdkit.Chem import AllChem
        from rdkit import DataStructs
        from rdkit.ML.Cluster import Butina

        fps, valid = [], []
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
            valid.append(i)

        dists = []
        for i in range(1, len(fps)):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
            dists.extend(1.0 - s for s in sims)

        clusters = Butina.ClusterData(dists, len(fps), 1.0 - cutoff,
                                      isDistData=True)
        groups = np.full(n, -1, dtype=np.int64)
        for cid, members in enumerate(clusters):
            for m in members:
                groups[valid[m]] = cid
        # Anything unassigned becomes its own singleton.
        for i in range(n):
            if groups[i] < 0:
                groups[i] = 10_000_000 + i
        print(f"    Butina clusters at Tanimoto {cutoff}: {len(clusters):,}")
        return groups

    raise ValueError(f"Unknown split mode: {mode}")


def grouped_three_way_split(groups, seed, frac_cal=0.60, frac_val=0.20):
    """
    Partition indices into calibration / validation / test so that no group
    straddles two partitions. Groups are shuffled, then greedily packed until
    each partition reaches its target size.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)

    members = {g: np.where(groups == g)[0] for g in uniq}
    n_total = len(groups)
    n_cal_target = int(round(frac_cal * n_total))
    n_val_target = int(round(frac_val * n_total))

    cal, val, test = [], [], []
    for g in uniq:
        idx = members[g]
        if len(cal) < n_cal_target:
            cal.extend(idx)
        elif len(val) < n_val_target:
            val.extend(idx)
        else:
            test.extend(idx)

    return np.array(cal, dtype=int), np.array(val, dtype=int), np.array(test, dtype=int)


def max_tanimoto_to_train(test_smiles, train_smiles, sample_train=4000, seed=42):
    """Mean/median nearest-neighbour Tanimoto from each test molecule to train."""
    from rdkit.Chem import AllChem
    from rdkit import DataStructs

    rng = np.random.default_rng(seed)
    if len(train_smiles) > sample_train:
        train_smiles = list(rng.choice(np.asarray(train_smiles, dtype=object),
                                       sample_train, replace=False))

    def fp(s):
        m = Chem.MolFromSmiles(s)
        return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) if m else None

    tr = [f for f in (fp(s) for s in train_smiles) if f is not None]
    if not tr:
        return np.nan, np.nan

    best = []
    for s in test_smiles:
        f = fp(s)
        if f is None:
            continue
        best.append(max(DataStructs.BulkTanimotoSimilarity(f, tr)))

    if not best:
        return np.nan, np.nan
    return float(np.mean(best)), float(np.median(best))


def permute_domains_preserving_sizes(domains, seed):
    """
    Shuffle domain labels across molecules while preserving the exact size of
    every domain (spec C4). A straight permutation of the label array does
    this by construction.
    """
    rng = np.random.default_rng(seed)
    out = np.array(domains, dtype=object).copy()
    rng.shuffle(out)
    return out


def full_metrics(y, pred):
    """
    Metrics, quantised to 10 decimal places.

    sklearn forests accumulate per-tree predictions in parallel in completion
    order, so repeated runs differ by ~8e-15 through float non-associativity.
    That is 13 orders of magnitude below anything reported, but it makes output
    files fail a byte-identical reproducibility check. Quantising here makes
    repeated runs at the same seed produce identical CSVs.
    """
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = pred - y
    q = 10
    return {
        "mae": round(float(mean_absolute_error(y, pred)), q),
        "rmse": round(float(np.sqrt(np.mean(err ** 2))), q),
        "r2": round(float(r2_score(y, pred)), q),
        "median_abs_error": round(float(np.median(np.abs(err))), q),
        "mean_signed_error": round(float(np.mean(err)), q),
        "n": int(len(y)),
    }


def fit_calibration_ensemble(ref_cal, ref_val, feat_cols, seed, top_k=3):
    """
    Fit the calibration candidates on ref_cal, rank by ref_val MAE, and return
    a 1/MAE-weighted top-k ensemble as a callable, plus its member metrics.
    """
    Xc_gap = ref_cal[["pqr_gap"]].to_numpy()
    Xv_gap = ref_val[["pqr_gap"]].to_numpy()
    Xc_full = ref_cal[["pqr_gap"] + feat_cols].to_numpy()
    Xv_full = ref_val[["pqr_gap"] + feat_cols].to_numpy()
    yc = ref_cal["ref_gap"].to_numpy()
    yv = ref_val["ref_gap"].to_numpy()

    cands = {
        "gap_linear_robust":  (linear_scaled("robust"),   Xc_gap,  Xv_gap,  "gap"),
        "gap_ridge_robust":   (ridge_scaled("robust"),    Xc_gap,  Xv_gap,  "gap"),
        "gap_ridge_standard": (ridge_scaled("standard"),  Xc_gap,  Xv_gap,  "gap"),
        "gap_ridge_power":    (ridge_scaled("power"),     Xc_gap,  Xv_gap,  "gap"),
        "desc_ridge_robust":  (ridge_scaled("robust"),    Xc_full, Xv_full, "full"),
        "desc_hgb":           (hgb(seed + 10),            Xc_full, Xv_full, "full"),
        "desc_et":            (et(seed + 20, n=700),      Xc_full, Xv_full, "full"),
        "desc_rf":            (rf(seed + 30, n=500),      Xc_full, Xv_full, "full"),
    }

    fitted, scores = {}, {}
    for name, (model, Xc, Xv, mode) in cands.items():
        try:
            model.fit(Xc, yc)
        except Exception as exc:
            finite = np.isfinite(Xc).sum(axis=0)
            with np.errstate(all="ignore"):
                variances = np.nanvar(Xc, axis=0)
            raise RuntimeError(
                f"calibration candidate '{name}' failed to fit.\n"
                f"  X shape        : {Xc.shape}\n"
                f"  y shape        : {np.shape(yc)}\n"
                f"  finite per col : {finite[:8]}{' ...' if Xc.shape[1] > 8 else ''}\n"
                f"  variance/col   : {np.round(variances[:8], 8)}"
                f"{' ...' if Xc.shape[1] > 8 else ''}\n"
                f"  n all-NaN cols : {int((finite == 0).sum())}\n"
                f"  original error : {type(exc).__name__}: {exc}"
            ) from exc
        scores[name] = mean_absolute_error(yv, model.predict(Xv))
        fitted[name] = (model, mode)

    # Determinism guard.
    #
    # Candidate validation MAEs differ between processes at the 1e-17 level,
    # because RandomForest with n_jobs=-1 accumulates tree predictions in a
    # non-fixed order. That noise flows into the 1/MAE ensemble weights, then
    # into the pseudo-labels for all 74k molecules. Tree learners then AMPLIFY
    # it: a 1e-15 shift in y can flip a near-tied split near the root of an
    # ExtraTrees tree, which changes an entire subtree. The measured end effect
    # was up to 0.096 eV on individual predictions from a 1e-15 input change.
    #
    # Quantising the scores to 12 decimal places removes the sensitivity at
    # source. 1e-12 eV is ~10 orders of magnitude below anything reported.
    scores = {k: round(float(v), 12) for k, v in scores.items()}
    top = sorted(scores, key=lambda k: (scores[k], k))[:top_k]
    maes = np.array([scores[n] for n in top])
    w = 1.0 / np.maximum(maes, 1e-6)
    w = np.round(w / w.sum(), 12)

    def predict(df):
        Xg = df[["pqr_gap"]].to_numpy()
        Xf = df[["pqr_gap"] + feat_cols].to_numpy()
        out = np.zeros(len(df), dtype=float)
        for name, wi in zip(top, w):
            model, mode = fitted[name]
            out += wi * model.predict(Xg if mode == "gap" else Xf)
        # Second guard: quantise the pseudo-labels themselves, so any residual
        # float noise upstream cannot reach the tree learners' split selection.
        return np.round(out, 9)

    return predict, top, w, scores


def _train_domain_and_piecewise(Xtrain, ytrain, train_domains, train_pool, feat_cols,
                                Xt, test_domains, ref_test, global_mean_test, seed,
                                min_domain=500, min_piecewise=1000, collect_splits=None):
    """
    Train per-domain experts and, where a useful within-domain breakpoint
    exists, left/right branch experts. Returns (domain_pred, piecewise_pred)
    on the test partition.
    """
    dom_t = np.array(global_mean_test, dtype=float).copy()
    domain_models = {}

    for domain in sorted(pd.unique(train_domains)):
        mtr = train_domains == domain
        mt = test_domains == domain
        if int(mtr.sum()) < min_domain:
            continue
        model = et(seed + 200 + stable_seed(str(domain), 500), n=500)
        model.fit(Xtrain[mtr], ytrain[mtr])
        domain_models[domain] = model
        if mt.any():
            dom_t[mt] = model.predict(Xt[mt])

    piece_t = dom_t.copy()

    for domain in sorted(pd.unique(train_domains)):
        mtr = train_domains == domain
        mt = test_domains == domain
        n_domain = int(mtr.sum())
        if n_domain < min_piecewise:
            continue

        split = _best_piecewise_split(
            train_pool.loc[mtr, feat_cols], ytrain[mtr], feat_cols,
            min_leaf=max(250, min(750, n_domain // 12)), max_features=50,
        )
        if split is None or split["improvement"] <= 0:
            continue

        feature, cut = split["feature"], split["split_value"]
        if collect_splits is not None:
            rec = dict(split)
            rec["domain"] = domain
            rec["domain_n"] = n_domain
            collect_splits.append(rec)

        xtr = pd.to_numeric(train_pool.loc[mtr, feature], errors="coerce").to_numpy(float)
        left_tr, right_tr = xtr <= cut, xtr > cut
        Xd = Xtrain[mtr]
        yd = ytrain[mtr]

        lm = et(seed + 700 + stable_seed(str(domain) + feature + "L"), n=500)
        rm = et(seed + 800 + stable_seed(str(domain) + feature + "R"), n=500)
        lm.fit(Xd[left_tr], yd[left_tr])
        rm.fit(Xd[right_tr], yd[right_tr])

        if mt.any():
            ti = np.where(mt)[0]
            xte = pd.to_numeric(ref_test.loc[mt, feature], errors="coerce").to_numpy(float)
            lmask, rmask = xte <= cut, xte > cut
            if lmask.any():
                piece_t[ti[lmask]] = lm.predict(Xt[ti[lmask]])
            if rmask.any():
                piece_t[ti[rmask]] = rm.predict(Xt[ti[rmask]])

    return dom_t, piece_t


def run_repeat(pqr, ref_pqr, feat_cols, seed, repeat_idx, split_mode,
               groups, want_permutation=True, collect_splits=None,
               calibration_group_rows=None):
    """
    One complete re-split / re-calibrate / re-train / re-evaluate cycle
    (spec C2). Pseudo-labels are regenerated inside the loop so no earlier
    split can leak through them.

    Returns {config_key: {metric: value}} evaluated on this repeat's test
    partition, plus the per-domain breakdown for the primary repeat.
    """
    cal_idx, val_idx, test_idx = grouped_three_way_split(groups, seed)

    ref_cal = ref_pqr.iloc[cal_idx].copy()
    ref_val = ref_pqr.iloc[val_idx].copy()
    ref_test = ref_pqr.iloc[test_idx].copy()

    print(f"  [repeat {repeat_idx}] seed={seed} "
          f"cal={len(ref_cal)} val={len(ref_val)} test={len(ref_test)}")

    # ---- calibration, refit from scratch every repeat ----
    cal_predict, top, w, cal_scores = fit_calibration_ensemble(
        ref_cal, ref_val, feat_cols, seed)

    # ---- A4: calibration quality on held-out reference only ----
    if calibration_group_rows is not None:
        heldout = pd.concat([ref_val, ref_test], ignore_index=True)
        heldout = heldout.copy()
        heldout["cal_pred"] = cal_predict(heldout)
        heldout["pqr_minus_ref"] = heldout["pqr_gap"] - heldout["ref_gap"]

        def _emit(group_type, group, sub):
            if len(sub) < 2:
                return
            m = full_metrics(sub["ref_gap"], sub["cal_pred"])
            calibration_group_rows.append({
                "repeat": repeat_idx, "group_type": group_type, "group": group,
                "n": len(sub), "mae": m["mae"], "rmse": m["rmse"], "r2": m["r2"],
                "mean_pqr_minus_ref": float(sub["pqr_minus_ref"].mean()),
                "sd_pqr_minus_ref": float(sub["pqr_minus_ref"].std(ddof=1)),
            })

        for d, sub in heldout.groupby("domain"):
            _emit("domain", d, sub)
        if "ref_source" in heldout.columns:
            for s, sub in heldout.groupby("ref_source"):
                _emit("source", s, sub)
        _emit("overall", "all", heldout)

    # ---- pseudo-labels for the whole corpus ----
    pseudo = cal_predict(pqr)
    labels = pseudo.copy()

    anchor = dict(zip(ref_cal["smiles"], ref_cal["ref_gap"]))
    is_anchor = pqr["smiles"].isin(anchor).to_numpy()
    labels[is_anchor] = pqr.loc[is_anchor, "smiles"].map(anchor).to_numpy()

    holdout = set(ref_val["smiles"]) | set(ref_test["smiles"])
    keep = ~pqr["smiles"].isin(holdout).to_numpy()
    train_pool = pqr.loc[keep].copy()
    ytrain = labels[keep]

    Xtrain = train_pool[feat_cols].to_numpy()
    Xt = ref_test[feat_cols].to_numpy()
    Xv = ref_val[feat_cols].to_numpy()
    ytest = ref_test["ref_gap"].to_numpy()
    yval = ref_val["ref_gap"].to_numpy()

    results = {}
    pred_t, pred_v = {}, {}

    # ---- global experts ----
    models = {
        "global_hgb":   hgb(seed + 100),
        "global_et":    et(seed + 110, n=700),
        "global_rf":    rf(seed + 120, n=500),
        "global_ridge": ridge_scaled("robust"),
    }
    for key, m in models.items():
        m.fit(Xtrain, ytrain)
        pred_t[key] = m.predict(Xt)
        pred_v[key] = m.predict(Xv)
        results[key] = full_metrics(ytest, pred_t[key])

    pred_t["global_mean"] = (pred_t["global_hgb"] + pred_t["global_et"]
                             + pred_t["global_rf"]) / 3.0
    pred_v["global_mean"] = (pred_v["global_hgb"] + pred_v["global_et"]
                             + pred_v["global_rf"]) / 3.0
    results["global_mean"] = full_metrics(ytest, pred_t["global_mean"])

    # ---- domain + piecewise on TRUE domain labels ----
    train_domains = train_pool["domain"].to_numpy()
    test_domains = ref_test["domain"].to_numpy()

    dom_t, piece_t = _train_domain_and_piecewise(
        Xtrain, ytrain, train_domains, train_pool, feat_cols,
        Xt, test_domains, ref_test, pred_t["global_mean"], seed,
        collect_splits=collect_splits)

    pred_t["domain_expert"] = dom_t
    pred_t["piecewise_split"] = piece_t
    results["domain_expert"] = full_metrics(ytest, dom_t)
    results["piecewise_split"] = full_metrics(ytest, piece_t)

    # ---- full_rceg: ridge meta-learner fitted on VAL, reported on TEST ----
    # The previous code fitted on Pv then predicted Pv, so its validation
    # number was in-sample. Only the test number is reported now.
    dom_v, piece_v = _train_domain_and_piecewise(
        Xtrain, ytrain, train_domains, train_pool, feat_cols,
        Xv, ref_val["domain"].to_numpy(), ref_val, pred_v["global_mean"], seed)
    pred_v["domain_expert"] = dom_v
    pred_v["piecewise_split"] = piece_v

    stack_keys = ["global_hgb", "global_et", "global_rf", "global_ridge",
                  "global_mean", "domain_expert", "piecewise_split"]
    Pv = np.column_stack([pred_v[k] for k in stack_keys])
    Pt = np.column_stack([pred_t[k] for k in stack_keys])

    stacker = Pipeline([("scale", StandardScaler()),
                        ("model", RidgeCV(alphas=np.logspace(-6, 5, 60)))])
    # ------------------------------------------------------------------
    # The stacker is FITTED on the validation partition, so a raw validation
    # score for it is in-sample and cannot be compared against the other
    # configurations. Architecture selection must happen on validation, never
    # on test, so an honest out-of-fold validation MAE is cross-fitted within
    # the validation partition here. The stacker used for the TEST prediction
    # is still fitted on all of validation, which is legitimate because test is
    # untouched.
    # ------------------------------------------------------------------
    from sklearn.model_selection import KFold

    oof = np.zeros(len(yval), dtype=float)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for fit_idx, hold_idx in kf.split(Pv):
        s_cv = Pipeline([("scale", StandardScaler()),
                         ("model", RidgeCV(alphas=np.logspace(-6, 5, 60)))])
        s_cv.fit(Pv[fit_idx], yval[fit_idx])
        oof[hold_idx] = s_cv.predict(Pv[hold_idx])
    val_mae_full_rceg = round(float(mean_absolute_error(yval, oof)), 10)

    stacker.fit(Pv, yval)
    pred_t["full_rceg"] = stacker.predict(Pt)
    results["full_rceg"] = full_metrics(ytest, pred_t["full_rceg"])
    results["full_rceg"]["val_mae_crossfit"] = val_mae_full_rceg
    results["full_rceg"]["val_mae_in_sample_DO_NOT_REPORT"] = float(
        mean_absolute_error(yval, stacker.predict(Pv)))

    # ---- no_exact_mass ablation: drop x1 ----
    feat_nx1 = [c for c in feat_cols if c != "x1"]
    Xtr2 = train_pool[feat_nx1].to_numpy()
    Xv2 = ref_val[feat_nx1].to_numpy()
    Xt2 = ref_test[feat_nx1].to_numpy()
    m2 = et(seed + 110, n=700)
    m2.fit(Xtr2, ytrain)
    results["no_exact_mass"] = full_metrics(ytest, m2.predict(Xt2))
    pred_v["no_exact_mass"] = m2.predict(Xv2)

    # ---- domain permutation control (spec C4) ----
    if want_permutation:
        perm_train = permute_domains_preserving_sizes(train_domains, seed)
        perm_test = permute_domains_preserving_sizes(test_domains, seed + 7)
        pdom_t, ppiece_t = _train_domain_and_piecewise(
            Xtrain, ytrain, perm_train, train_pool, feat_cols,
            Xt, perm_test, ref_test, pred_t["global_mean"], seed)
        results["domain_permuted"] = full_metrics(ytest, pdom_t)
        results["domain_permuted_piecewise"] = full_metrics(ytest, ppiece_t)

    for k in results:
        if k == "full_rceg":
            # honest out-of-fold value; the in-sample one is never used
            results[k]["val_mae"] = round(float(val_mae_full_rceg), 10)
        elif k in pred_v:
            results[k]["val_mae"] = round(
                float(mean_absolute_error(yval, pred_v[k])), 10)
        else:
            results[k]["val_mae"] = np.nan

    return results, ref_test, pred_t


def qedges(y, k):
    edges = np.percentile(y, np.linspace(0, 100, k + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def assign(y, edges):
    return np.digitize(y, edges[1:-1], right=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pqr", required=True)
    ap.add_argument("--qm9-ref", required=True)
    ap.add_argument("--extra-ref", default=None, help="Optional recomputed PQR reference CSV with smiles,gap columns.")
    ap.add_argument("--outdir", default="runs/pqr_full_domain_moe")
    ap.add_argument("--regimes", type=int, default=6)
    ap.add_argument("--leakage-corr-threshold", type=float, default=0.98)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-models", action="store_true")
    # ---- ACS Omega revision flags (spec C1-C5, A4) ----
    ap.add_argument("--n-repeats", type=int, default=10,
                    help="Repeated reference splits for dispersion (spec C2). 1 disables.")
    ap.add_argument("--split-mode", choices=["random", "scaffold", "cluster"],
                    default="random",
                    help="Partition protocol for the reference pool (spec C3).")
    ap.add_argument("--cluster-cutoff", type=float, default=0.6,
                    help="Butina Tanimoto cutoff when --split-mode cluster.")
    ap.add_argument("--emit-calibration-by-group", action="store_true",
                    help="Emit calibration_by_domain_and_source.csv (spec A4).")
    ap.add_argument("--emit-permutation-control", action="store_true",
                    help="Emit domain_permutation_control.csv (spec C4).")
    ap.add_argument("--skip-legacy-single-run", action="store_true",
                    help="Only run the repeated-splits analysis, skip the "
                         "original single-run artifact path.")
    ap.add_argument(
        "--external-ref",
        default=None,
        help=(
            "External reference CSV containing smiles and gap. "
            "Matching molecules are excluded completely from training."
        ),
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pqr, audit = load_pqr(args.pqr)
    qm9 = load_reference(args.qm9_ref, "qm9_overlap")

    refs = [qm9]
    if args.extra_ref:
        extra = load_reference(args.extra_ref, "recomputed_pqr_reference")
        refs.append(extra)

    ref = pd.concat(refs, ignore_index=True)
    ref = ref.groupby("smiles", as_index=False).agg({
        "ref_gap": "median",
        "ref_source": lambda x: "+".join(sorted(set(map(str, x)))),
    })

    # --------------------------------------------------------
    # Strict published external benchmark.
    #
    # External molecules are captured with their existing PQR
    # feature vectors and then removed from:
    #   - leakage filtering
    #   - calibration
    #   - pseudo-label generation
    #   - final expert training
    #
    # The external labels are never accessed during training.
    # --------------------------------------------------------
    external_test = None

    if args.external_ref:
        external_ref = load_reference(
            args.external_ref,
            "qmugs_dft_external",
        )

        external_test = pqr.merge(
            external_ref,
            on="smiles",
            how="inner",
        )

        external_smiles = set(external_test["smiles"])

        if len(external_test) < 50:
            raise RuntimeError(
                f"Only {len(external_test)} external overlap molecules found."
            )

        pqr = pqr[
            ~pqr["smiles"].isin(external_smiles)
        ].reset_index(drop=True)

        audit = audit[
            ~audit["smiles"].isin(external_smiles)
        ].reset_index(drop=True)

        ref = ref[
            ~ref["smiles"].isin(external_smiles)
        ].reset_index(drop=True)

        print(
            f"\nSTRICT EXTERNAL HOLDOUT: "
            f"{len(external_test):,} QMugs molecules removed "
            f"from all model-development stages."
        )

        # ------------------------------------------------------------------
        # Spec C5: make the excision checkable rather than merely asserted in
        # prose. Fail loudly if any external SMILES survives into the training
        # corpus, the reference pool, or the leakage-audit input.
        # ------------------------------------------------------------------
        contamination = {
            "training_corpus": set(pqr["smiles"]) & external_smiles,
            "reference_pool": set(ref["smiles"]) & external_smiles,
            "leakage_audit_input": set(audit["smiles"]) & external_smiles,
        }
        offenders = {k: v for k, v in contamination.items() if v}
        if offenders:
            raise RuntimeError(
                "EXTERNAL HOLDOUT VIOLATION - QMugs molecules leaked into: "
                + "; ".join(f"{k} ({len(v)} molecules, e.g. {sorted(v)[:3]})"
                            for k, v in offenders.items())
            )
        print(
            f"  [C5 assertion PASSED] 0 of {len(external_smiles):,} external "
            f"SMILES present in training corpus, reference pool, or leakage audit."
        )

    feat_cols = leakage_filter(pqr, audit, args.leakage_corr_threshold, outdir)

    # Hard no-leakage assertion.
    # Final model input features must be x-columns only.
    banned_tokens = [
        "homo",
        "lumo",
        "lumo_minus_homo",
        "homo_lumo",
        "gap",
        "ref_gap",
        "pqr_gap",
        "training_label",
        "pred_qm9_aligned_gap",
        "inverse_reconstructed_pqr_gap",
        "cycle_error",
    ]

    bad_features = [
        c for c in feat_cols
        if any(tok in str(c).lower() for tok in banned_tokens)
    ]

    non_x_features = [c for c in feat_cols if not str(c).startswith("x")]

    if bad_features or non_x_features:
        raise RuntimeError(
            f"Leakage/non-x features found. bad_features={bad_features}, non_x_features={non_x_features}"
        )

    print(f"No-leakage feature check passed: {len(feat_cols)} x-features used.")

    ref_pqr = pqr.merge(ref, on="smiles", how="inner")
    print(f"\nReference-labeled PQR molecules available: {len(ref_pqr):,}")
    print(ref_pqr["ref_source"].value_counts().to_string())

    if len(ref_pqr) < 300:
        raise RuntimeError("Too few reference-labeled molecules for honest validation.")

    # ==================================================================
    # Repeated-split analysis (spec C1-C4). Emits every dispersion table
    # the revision needs. Runs before the legacy single-run path so that
    # a crash later still leaves these artifacts on disk.
    # ==================================================================
    if args.n_repeats and args.n_repeats >= 1:
        print("\n" + "=" * 70)
        print(f"REPEATED-SPLIT ANALYSIS  n_repeats={args.n_repeats} "
              f"split_mode={args.split_mode}")
        print("=" * 70)

        print("  assigning split groups ...")
        groups = assign_split_groups(
            list(ref_pqr["smiles"]), args.split_mode,
            cutoff=args.cluster_cutoff, seed=args.seed)

        ablation_rows = []
        calib_rows = [] if args.emit_calibration_by_group else None
        split_records = []
        primary = None

        for i in range(args.n_repeats):
            rseed = args.seed + 1000 * i
            res, rtest, rpred = run_repeat(
                pqr, ref_pqr, feat_cols, rseed, i, args.split_mode, groups,
                want_permutation=args.emit_permutation_control,
                collect_splits=split_records if i == 0 else None,
                calibration_group_rows=calib_rows,
            )
            if i == 0:
                primary = (rtest, rpred)

            for cfg, m in res.items():
                ablation_rows.append({
                    "config": cfg, "repeat": i, "seed": rseed,
                    "split_mode": args.split_mode,
                    "test_mae": m["mae"], "test_rmse": m["rmse"],
                    "test_r2": m["r2"],
                    "test_median_abs_error": m["median_abs_error"],
                    "test_mean_signed_error": m["mean_signed_error"],
                    "val_mae": m.get("val_mae", np.nan),
                    "n_test": m["n"],
                })

        ab = pd.DataFrame(ablation_rows)
        ab.to_csv(outdir / "ablation_repeated_splits.csv", index=False)

        summary = (ab.groupby("config")
                     .agg(mae_mean=("test_mae", "mean"),
                          mae_sd=("test_mae", "std"),
                          mae_min=("test_mae", "min"),
                          mae_max=("test_mae", "max"),
                          rmse_mean=("test_rmse", "mean"),
                          r2_mean=("test_r2", "mean"),
                          r2_sd=("test_r2", "std"),
                          n_repeats=("test_mae", "size"))
                     .reset_index()
                     .sort_values("mae_mean"))
        summary["split_mode"] = args.split_mode
        summary.to_csv(outdir / "repeated_split_summary.csv", index=False)

        print("\nRepeated-split summary (sorted by mean test MAE):")
        print(summary.to_string(index=False))

        # ---- spec 8.1 check: does domain_expert beat global_et by >1 SD? ----
        try:
            de = summary[summary.config == "domain_expert"].iloc[0]
            ge = summary[summary.config == "global_et"].iloc[0]
            delta = ge.mae_mean - de.mae_mean
            pooled = float(np.sqrt((de.mae_sd ** 2 + ge.mae_sd ** 2) / 2))
            verdict = "EXCEEDS" if delta > pooled else "DOES NOT EXCEED"
            print(f"\n[spec 8.1] domain_expert vs global_et: "
                  f"delta={delta:.4f} eV, pooled_sd={pooled:.4f} -> {verdict} 1 SD")
        except Exception as exc:
            print(f"[spec 8.1] check skipped: {exc}")

        # ---- C4: permutation control ----
        if args.emit_permutation_control and "domain_permuted" in set(ab.config):
            t = ab[ab.config == "domain_expert"][["repeat", "test_mae"]]
            p = ab[ab.config == "domain_permuted"][["repeat", "test_mae"]]
            perm = t.merge(p, on="repeat", suffixes=("_true", "_perm"))
            perm.columns = ["repeat", "mae_true_domains", "mae_permuted_domains"]
            perm["delta"] = perm.mae_permuted_domains - perm.mae_true_domains
            perm.to_csv(outdir / "domain_permutation_control.csv", index=False)
            md = float(perm.delta.mean())
            print(f"\n[spec 8.2] permutation control: mean delta={md:+.4f} eV "
                  f"({'degrades as expected' if md > 0 else 'DOES NOT DEGRADE'})")

        # ---- A4: calibration by domain and source ----
        if calib_rows:
            cg = pd.DataFrame(calib_rows)
            agg = (cg.groupby(["group_type", "group"])
                     .agg(n=("n", "mean"), mae=("mae", "mean"), rmse=("rmse", "mean"),
                          r2=("r2", "mean"),
                          mean_pqr_minus_ref=("mean_pqr_minus_ref", "mean"),
                          sd_pqr_minus_ref=("sd_pqr_minus_ref", "mean"))
                     .reset_index())
            agg.to_csv(outdir / "calibration_by_domain_and_source.csv", index=False)
            dom = agg[agg.group_type == "domain"]["mean_pqr_minus_ref"]
            if len(dom):
                off = {"OFF_MIN": float(dom.min()), "OFF_MAX": float(dom.max()),
                       "OFF_MEAN": float(dom.mean())}
                off["OFF_SPREAD"] = off["OFF_MAX"] - off["OFF_MIN"]
                pd.DataFrame([off]).to_csv(outdir / "calibration_offsets.csv", index=False)
                print(f"\n[spec 8.3] OFF_MIN={off['OFF_MIN']:.3f} "
                      f"OFF_MAX={off['OFF_MAX']:.3f} "
                      f"OFF_SPREAD={off['OFF_SPREAD']:.3f} "
                      f"OFF_MEAN={off['OFF_MEAN']:.3f} eV")
                if off["OFF_SPREAD"] > 1.0:
                    print("  WARNING: OFF_SPREAD exceeds 1 eV -> calibration is "
                          "strongly domain-dependent (spec A4 requires the "
                          "manuscript paragraph be rewritten as a concession).")

        # ---- C1: per-domain / per-source metrics on the primary repeat ----
        if primary is not None:
            rtest, rpred = primary
            best = "piecewise_split" if "piecewise_split" in rpred else "domain_expert"
            rep = rtest[["smiles", "ref_gap", "domain"]].copy()
            if "ref_source" in rtest.columns:
                rep["ref_source"] = rtest["ref_source"].values
            # quantised for the same reason as full_metrics
            rep["pred"] = np.round(rpred[best], 9)
            rows = []
            for gt, col in [("domain", "domain"), ("source", "ref_source")]:
                if col not in rep.columns:
                    continue
                for g, sub in rep.groupby(col):
                    m = full_metrics(sub["ref_gap"], sub["pred"])
                    rows.append({"group_type": gt, "group": g, **m})
            m = full_metrics(rep["ref_gap"], rep["pred"])
            rows.append({"group_type": "overall", "group": "all", **m})
            pd.DataFrame(rows).to_csv(outdir / "mae_by_domain.csv", index=False)
            rep.to_csv(outdir / "figure6_data.csv", index=False)
            print(f"\nWrote mae_by_domain.csv and figure6_data.csv "
                  f"(model={best})")

        if split_records:
            pd.DataFrame(split_records).to_csv(
                outdir / "piecewise_split_architecture_branches.csv", index=False)

        # ---- C3: split-protocol comparison row ----
        try:
            cal_i, val_i, test_i = grouped_three_way_split(groups, args.seed)
            tr_smi = list(ref_pqr.iloc[np.concatenate([cal_i, val_i])]["smiles"])
            te_smi = list(ref_pqr.iloc[test_i]["smiles"])
            mean_t, med_t = max_tanimoto_to_train(te_smi, tr_smi, seed=args.seed)
        except Exception as exc:
            mean_t = med_t = np.nan
            print(f"  tanimoto summary skipped: {exc}")

        best_cfg = summary.iloc[0]
        pd.DataFrame([{
            "split_mode": args.split_mode,
            "best_config": best_cfg.config,
            "test_mae": best_cfg.mae_mean, "test_mae_sd": best_cfg.mae_sd,
            "test_rmse": best_cfg.rmse_mean, "test_r2": best_cfg.r2_mean,
            "n_test": int(ab[ab.config == best_cfg.config]["n_test"].mean()),
            "n_repeats": int(best_cfg.n_repeats),
            "mean_max_tanimoto_test_to_train": mean_t,
            "median_max_tanimoto_test_to_train": med_t,
        }]).to_csv(outdir / "split_protocol_comparison.csv", index=False)

        print(f"\nRepeated-split artifacts written to {outdir.resolve()}")

        if args.skip_legacy_single_run:
            print("\n--skip-legacy-single-run set; stopping before the "
                  "original single-run artifact path.")
            return

    # Split real-reference molecules into calibration train, stack validation, final test.
    idx = np.arange(len(ref_pqr))
    cal_idx, temp_idx = train_test_split(idx, test_size=0.40, random_state=args.seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=args.seed + 1)

    ref_cal = ref_pqr.iloc[cal_idx].copy()
    ref_val = ref_pqr.iloc[val_idx].copy()
    ref_test = ref_pqr.iloc[test_idx].copy()

    print(f"\nReference split: cal_train={len(ref_cal):,}, stack_val={len(ref_val):,}, final_test={len(ref_test):,}")

    # Calibration: PQR gap + features -> real reference gap.
    Xcal_gap = ref_cal[["pqr_gap"]].to_numpy()
    Xval_gap = ref_val[["pqr_gap"]].to_numpy()
    Xtest_gap = ref_test[["pqr_gap"]].to_numpy()

    Xcal_full = ref_cal[["pqr_gap"] + feat_cols].to_numpy()
    Xval_full = ref_val[["pqr_gap"] + feat_cols].to_numpy()
    Xtest_full = ref_test[["pqr_gap"] + feat_cols].to_numpy()

    ycal = ref_cal["ref_gap"].to_numpy()
    yval = ref_val["ref_gap"].to_numpy()
    ytest = ref_test["ref_gap"].to_numpy()

    calibrators = {
        "gap_linear_robust": (
            linear_scaled("robust"),
            Xcal_gap, Xval_gap, Xtest_gap, "gap"
        ),
        "gap_ridge_robust": (
            ridge_scaled("robust"),
            Xcal_gap, Xval_gap, Xtest_gap, "gap"
        ),
        "gap_ridge_standard": (
            ridge_scaled("standard"),
            Xcal_gap, Xval_gap, Xtest_gap, "gap"
        ),
        "gap_ridge_power": (
            ridge_scaled("power"),
            Xcal_gap, Xval_gap, Xtest_gap, "gap"
        ),

        # Descriptor-assisted regularized calibration baselines.
        "desc_ridge_robust": (
            ridge_scaled("robust"),
            Xcal_full, Xval_full, Xtest_full, "full"
        ),
        "desc_ridge_standard": (
            ridge_scaled("standard"),
            Xcal_full, Xval_full, Xtest_full, "full"
        ),
        "desc_ridge_power": (
            ridge_scaled("power"),
            Xcal_full, Xval_full, Xtest_full, "full"
        ),

        "desc_hgb": (hgb(args.seed + 10), Xcal_full, Xval_full, Xtest_full, "full"),
        "desc_et": (et(args.seed + 20, n=700), Xcal_full, Xval_full, Xtest_full, "full"),
        "desc_rf": (rf(args.seed + 30, n=500), Xcal_full, Xval_full, Xtest_full, "full"),
    }

    cal_models = {}
    cal_rows = []
    val_preds = {}
    test_preds = {}

    print("\nCalibration models:")
    for name, (model, Xc, Xv, Xt, mode) in calibrators.items():
        model.fit(Xc, ycal)
        pv = model.predict(Xv)
        pt = model.predict(Xt)

        row_v = metric("VAL cal_" + name, yval, pv)
        row_t = metric("TEST cal_" + name, ytest, pt)
        row_v["test_mae"] = row_t["mae"]
        row_v["test_r2"] = row_t["r2"]
        row_v["mode"] = mode
        cal_rows.append(row_v)

        cal_models[name] = (model, mode)
        val_preds[name] = pv
        test_preds[name] = pt

    cal_df = pd.DataFrame(cal_rows).sort_values("mae")
    cal_df.to_csv(outdir / "calibration_metrics.csv", index=False)

    top = list(cal_df.head(3)["name"].str.replace("VAL cal_", "", regex=False))
    maes = np.array([cal_df[cal_df["name"] == "VAL cal_" + n]["mae"].iloc[0] for n in top])
    weights = 1 / np.maximum(maes, 1e-6)
    weights = weights / weights.sum()

    print("\nCalibration ensemble:")
    for n, w in zip(top, weights):
        print(f"  {n}: weight={w:.3f}")

    ens_val = sum(w * val_preds[n] for n, w in zip(top, weights))
    ens_test = sum(w * test_preds[n] for n, w in zip(top, weights))
    ens_rows = [
        metric("VAL calibration_ensemble", yval, ens_val),
        metric("TEST calibration_ensemble", ytest, ens_test),
    ]
    pd.DataFrame(ens_rows).to_csv(outdir / "calibration_ensemble_metrics.csv", index=False)

    # ------------------------------------------------------------------
    # Inverse calibration / cycle-consistency model:
    #   real reference gap + features -> PQR-style gap
    #
    # This lets us test:
    #   PQR gap -> forward calibration -> QM9-aligned gap
    #           -> inverse calibration -> reconstructed PQR gap
    #
    # cycle_error = |reconstructed_pqr_gap - original_pqr_gap|
    #
    # This is a confidence diagnostic, not an external proof of correctness.
    # ------------------------------------------------------------------
    print("\nInverse calibration models for cycle-consistency:")

    ycal_pqr = ref_cal["pqr_gap"].to_numpy()
    yval_pqr = ref_val["pqr_gap"].to_numpy()
    ytest_pqr = ref_test["pqr_gap"].to_numpy()

    Xinv_cal_gap = ycal.reshape(-1, 1)
    Xinv_val_gap = yval.reshape(-1, 1)
    Xinv_test_gap = ytest.reshape(-1, 1)

    Xinv_cal_full = np.column_stack([ycal, ref_cal[feat_cols].to_numpy()])
    Xinv_val_full = np.column_stack([yval, ref_val[feat_cols].to_numpy()])
    Xinv_test_full = np.column_stack([ytest, ref_test[feat_cols].to_numpy()])

    inverse_candidates = {
        "inv_gap_linear": (lin(), Xinv_cal_gap, Xinv_val_gap, Xinv_test_gap, "gap"),
        "inv_gap_ridge": (ridge(), Xinv_cal_gap, Xinv_val_gap, Xinv_test_gap, "gap"),
        "inv_desc_hgb": (hgb(args.seed + 410), Xinv_cal_full, Xinv_val_full, Xinv_test_full, "full"),
        "inv_desc_et": (et(args.seed + 420, n=700), Xinv_cal_full, Xinv_val_full, Xinv_test_full, "full"),
        "inv_desc_rf": (rf(args.seed + 430, n=500), Xinv_cal_full, Xinv_val_full, Xinv_test_full, "full"),
    }

    inv_models = {}
    inv_val_preds = {}
    inv_test_preds = {}
    inv_rows = []

    for name, (model, Xc, Xv, Xt, mode) in inverse_candidates.items():
        model.fit(Xc, ycal_pqr)
        pv = model.predict(Xv)
        pt = model.predict(Xt)

        row_v = metric("VAL " + name, yval_pqr, pv)
        row_t = metric("TEST " + name, ytest_pqr, pt)

        row_v["test_mae"] = row_t["mae"]
        row_v["test_r2"] = row_t["r2"]
        row_v["mode"] = mode

        inv_rows.append(row_v)
        inv_models[name] = (model, mode)
        inv_val_preds[name] = pv
        inv_test_preds[name] = pt

    inv_df = pd.DataFrame(inv_rows).sort_values("mae")
    inv_df.to_csv(outdir / "inverse_calibration_metrics.csv", index=False)

    inv_top = list(inv_df.head(3)["name"].str.replace("VAL ", "", regex=False))
    inv_maes = np.array([inv_df[inv_df["name"] == "VAL " + n]["mae"].iloc[0] for n in inv_top])
    inv_weights = 1 / np.maximum(inv_maes, 1e-6)
    inv_weights = inv_weights / inv_weights.sum()

    print("\nInverse calibration ensemble:")
    for n, w in zip(inv_top, inv_weights):
        print(f"  {n}: weight={w:.3f}")

    inv_ens_val = sum(w * inv_val_preds[n] for n, w in zip(inv_top, inv_weights))
    inv_ens_test = sum(w * inv_test_preds[n] for n, w in zip(inv_top, inv_weights))

    inv_ens_rows = [
        metric("VAL inverse_ensemble", yval_pqr, inv_ens_val),
        metric("TEST inverse_ensemble", ytest_pqr, inv_ens_test),
    ]
    pd.DataFrame(inv_ens_rows).to_csv(outdir / "inverse_calibration_ensemble_metrics.csv", index=False)

    # Generate calibrated labels for all PQR.
    Xall_gap = pqr[["pqr_gap"]].to_numpy()
    Xall_full = pqr[["pqr_gap"] + feat_cols].to_numpy()

    pseudo = np.zeros(len(pqr), dtype=float)
    for n, w in zip(top, weights):
        model, mode = cal_models[n]
        if mode == "gap":
            pseudo += w * model.predict(Xall_gap)
        else:
            pseudo += w * model.predict(Xall_full)

    # Forward calibrated label: PQR-style -> QM9/reference-aligned.
    pqr["pred_qm9_aligned_gap"] = pseudo
    pqr["training_label"] = pseudo
    pqr["label_source"] = "calibrated_pseudo_label"

    # Inverse reconstruction: QM9/reference-aligned -> reconstructed PQR-style.
    Xinv_all_gap = pseudo.reshape(-1, 1)
    Xinv_all_full = np.column_stack([pseudo, pqr[feat_cols].to_numpy()])

    inv_recon = np.zeros(len(pqr), dtype=float)
    for n, w in zip(inv_top, inv_weights):
        model, mode = inv_models[n]
        if mode == "gap":
            inv_recon += w * model.predict(Xinv_all_gap)
        else:
            inv_recon += w * model.predict(Xinv_all_full)

    pqr["inverse_reconstructed_pqr_gap"] = inv_recon
    pqr["cycle_error"] = np.abs(pqr["inverse_reconstructed_pqr_gap"] - pqr["pqr_gap"])

    def _cycle_conf(e):
        if e <= 0.20:
            return "high_cycle_consistency"
        if e <= 0.50:
            return "medium_cycle_consistency"
        return "low_cycle_consistency"

    pqr["cycle_confidence"] = [_cycle_conf(e) for e in pqr["cycle_error"].to_numpy()]

    print("\nForward-inverse cycle consistency across all PQR:")
    print(pqr["cycle_error"].describe().to_string())
    print("\nCycle confidence counts:")
    print(pqr["cycle_confidence"].value_counts().to_string())

    # Use real reference labels for calibration-training anchors only.
    ref_map = dict(ref_cal[["smiles", "ref_gap"]].values)
    pqr.loc[pqr["smiles"].isin(ref_map), "training_label"] = pqr.loc[pqr["smiles"].isin(ref_map), "smiles"].map(ref_map)
    pqr.loc[pqr["smiles"].isin(ref_map), "label_source"] = "real_reference_anchor"

    # Do not train on validation/test reference molecules.
    holdout = set(ref_val["smiles"]) | set(ref_test["smiles"])
    train_pool = pqr[~pqr["smiles"].isin(holdout)].copy()

    pqr = add_confidence(pqr, ref_cal, ref_val, ref_test, feat_cols)

    def _combine_conf(row):
        cal = row["calibration_confidence"]
        cyc = row["cycle_confidence"]

        if cal == "low_out_of_domain" or cyc == "low_cycle_consistency":
            return "low_confidence"
        if cal == "medium_near_domain" or cyc == "medium_cycle_consistency":
            return "medium_confidence"
        return "high_confidence"

    pqr["final_confidence"] = pqr.apply(_combine_conf, axis=1)

    print("\nFinal combined confidence counts:")
    print(pqr["final_confidence"].value_counts().to_string())

    Xtrain = train_pool[feat_cols].to_numpy()
    ytrain = train_pool["training_label"].to_numpy()

    Xv_final = ref_val[feat_cols].to_numpy()
    Xt_final = ref_test[feat_cols].to_numpy()

    print(f"\nFull final training pool: {len(train_pool):,}")
    print("Training label sources:")
    print(train_pool["label_source"].value_counts().to_string())

    # Final global experts.
    print("\nTraining final full-PQR experts...")
    g_hgb = hgb(args.seed + 100)
    g_et = et(args.seed + 110, n=700)
    g_rf = rf(args.seed + 120, n=500)

    g_hgb.fit(Xtrain, ytrain)
    g_et.fit(Xtrain, ytrain)
    g_rf.fit(Xtrain, ytrain)

    # Regularized scaled experts. Because preprocessing is inside each
    # pipeline, every scaler is fitted on Xtrain only.
    g_ridge_robust = ridge_scaled("robust")
    g_ridge_standard = ridge_scaled("standard")

    print("Training scaled regularized global experts...")
    g_ridge_robust.fit(Xtrain, ytrain)
    g_ridge_standard.fit(Xtrain, ytrain)

    pred_v = {}
    pred_t = {}

    pred_v["global_hgb"] = g_hgb.predict(Xv_final)
    pred_t["global_hgb"] = g_hgb.predict(Xt_final)

    pred_v["global_et"] = g_et.predict(Xv_final)
    pred_t["global_et"] = g_et.predict(Xt_final)

    pred_v["global_rf"] = g_rf.predict(Xv_final)
    pred_t["global_rf"] = g_rf.predict(Xt_final)

    pred_v["global_ridge_robust"] = g_ridge_robust.predict(Xv_final)
    pred_t["global_ridge_robust"] = g_ridge_robust.predict(Xt_final)

    pred_v["global_ridge_standard"] = g_ridge_standard.predict(Xv_final)
    pred_t["global_ridge_standard"] = g_ridge_standard.predict(Xt_final)

    # Preserve the original tree-only mean for routing and direct
    # comparison with the prior model.
    pred_v["global_mean"] = (
        pred_v["global_hgb"]
        + pred_v["global_et"]
        + pred_v["global_rf"]
    ) / 3
    pred_t["global_mean"] = (
        pred_t["global_hgb"]
        + pred_t["global_et"]
        + pred_t["global_rf"]
    ) / 3

    # Domain experts.
    print("\nTraining chemical-domain experts...")
    dom_v = np.zeros(len(ref_val))
    dom_t = np.zeros(len(ref_test))
    domain_models = {}

    train_domains = train_pool["domain"].to_numpy()
    val_domains = ref_val["domain"].to_numpy()
    test_domains = ref_test["domain"].to_numpy()

    for domain in sorted(train_pool["domain"].unique()):
        mtr = train_domains == domain
        mv = val_domains == domain
        mt = test_domains == domain
        n = int(mtr.sum())

        if n >= 500:
            print(f"  {domain}: {n:,} rows")
            model = et(args.seed + 200 + len(domain), n=500)
            model.fit(Xtrain[mtr], ytrain[mtr])
            domain_models[domain] = model

            if mv.any():
                dom_v[mv] = model.predict(Xv_final[mv])
            if mt.any():
                dom_t[mt] = model.predict(Xt_final[mt])
        else:
            if mv.any():
                dom_v[mv] = pred_v["global_mean"][mv]
            if mt.any():
                dom_t[mt] = pred_t["global_mean"][mt]

    pred_v["domain_expert"] = dom_v
    pred_t["domain_expert"] = dom_t

    # --------------------------------------------------------
    # Frozen-model external QMugs evaluation.
    # --------------------------------------------------------
    if external_test is not None:
        from sklearn.model_selection import KFold

        Xext = external_test[feat_cols].to_numpy()
        yext = external_test["ref_gap"].to_numpy()

        ext_hgb = g_hgb.predict(Xext)
        ext_et = g_et.predict(Xext)
        ext_rf = g_rf.predict(Xext)
        ext_global_mean = (ext_hgb + ext_et + ext_rf) / 3.0

        ext_domain = ext_global_mean.copy()
        ext_domains = external_test["domain"].to_numpy()

        for domain, model in domain_models.items():
            mask = ext_domains == domain
            if mask.any():
                ext_domain[mask] = model.predict(Xext[mask])

        external_report = external_test[
            ["smiles", "ref_gap", "domain", "heavy_atoms"]
        ].copy()

        external_report["pred_global_hgb"] = ext_hgb
        external_report["pred_global_et"] = ext_et
        external_report["pred_global_rf"] = ext_rf
        external_report["pred_global_mean"] = ext_global_mean
        external_report["pred_domain_expert"] = ext_domain

        for name in [
            "global_hgb",
            "global_et",
            "global_rf",
            "global_mean",
            "domain_expert",
        ]:
            external_report["abs_err_" + name] = np.abs(
                external_report["pred_" + name]
                - external_report["ref_gap"]
            )

        # Secondary diagnostic:
        # five-fold cross-fitted constant offset correction. This estimates
        # performance after correcting only the computational-method offset.
        aligned = np.zeros(len(external_report), dtype=float)

        kf = KFold(
            n_splits=5,
            shuffle=True,
            random_state=args.seed,
        )

        raw_pred = external_report[
            "pred_domain_expert"
        ].to_numpy()

        for fit_idx, eval_idx in kf.split(raw_pred):
            offset = np.median(
                yext[fit_idx] - raw_pred[fit_idx]
            )
            aligned[eval_idx] = raw_pred[eval_idx] + offset

        external_report[
            "pred_domain_expert_crossfit_offset_aligned"
        ] = aligned

        external_report[
            "abs_err_domain_expert_crossfit_offset_aligned"
        ] = np.abs(aligned - yext)

        external_report.to_csv(
            outdir / "qmugs_external_predictions.csv",
            index=False,
        )

        external_metrics = []

        for name in [
            "global_hgb",
            "global_et",
            "global_rf",
            "global_mean",
            "domain_expert",
        ]:
            pred = external_report["pred_" + name].to_numpy()
            external_metrics.append({
                "evaluation": "strict_frozen_external",
                "model": name,
                "mae": float(mean_absolute_error(yext, pred)),
                "rmse": float(
                    np.sqrt(np.mean((yext - pred) ** 2))
                ),
                "r2": float(r2_score(yext, pred)),
                "n": int(len(yext)),
            })

        external_metrics.append({
            "evaluation": "five_fold_crossfit_offset_aligned",
            "model": "domain_expert",
            "mae": float(mean_absolute_error(yext, aligned)),
            "rmse": float(
                np.sqrt(np.mean((yext - aligned) ** 2))
            ),
            "r2": float(r2_score(yext, aligned)),
            "n": int(len(yext)),
        })

        pd.DataFrame(external_metrics).to_csv(
            outdir / "qmugs_external_metrics.csv",
            index=False,
        )

        domain_rows = []

        for domain, group in external_report.groupby("domain"):
            domain_rows.append({
                "domain": domain,
                "n": len(group),
                "mae_domain_expert": float(
                    group["abs_err_domain_expert"].mean()
                ),
                "mae_offset_aligned": float(
                    group[
                        "abs_err_domain_expert_crossfit_offset_aligned"
                    ].mean()
                ),
                "mean_reference_gap_ev": float(
                    group["ref_gap"].mean()
                ),
            })

        pd.DataFrame(domain_rows).to_csv(
            outdir / "qmugs_external_mae_by_domain.csv",
            index=False,
        )

        print("\nSTRICT QMugs external performance:")
        print(
            pd.DataFrame(external_metrics)
            .sort_values("mae")
            .to_string(index=False)
        )

    # ------------------------------------------------------------------
    # Piecewise split architecture:
    #   For each chemical domain, identify an x-feature whose relationship
    #   with the training label is better approximated by two local lines
    #   than by one global line. The best breakpoint is selected using only
    #   x-features and training labels.
    #
    #   Then each domain is split into left/right branches at that breakpoint,
    #   and a separate local expert is trained for each branch.
    #
    #   This replaces the old calibrated-gap regime split. It is designed to
    #   capture nonlinear descriptor-target relationships without using HOMO,
    #   LUMO, pqr_gap, ref_gap, or any direct gap-derived feature as input.
    # ------------------------------------------------------------------
    print("\nTraining piecewise descriptor-split architecture experts...")

    piece_v = np.zeros(len(ref_val))
    piece_t = np.zeros(len(ref_test))

    # Fallback to domain expert if a split branch is too small or not helpful.
    piece_v[:] = pred_v["domain_expert"]
    piece_t[:] = pred_t["domain_expert"]

    piece_rows = []
    piecewise_models = {}

    for domain in sorted(train_pool["domain"].unique()):
        mtr = train_domains == domain
        mv = val_domains == domain
        mt = test_domains == domain

        n_domain = int(mtr.sum())
        if n_domain < 1000:
            print(f"  {domain}: n={n_domain:,}; using domain-expert fallback")
            continue

        X_domain_df = train_pool.loc[mtr, feat_cols]
        y_domain = ytrain[mtr]

        split = _best_piecewise_split(
            X_domain_df,
            y_domain,
            feat_cols,
            min_leaf=max(250, min(750, n_domain // 12)),
            max_features=50,
        )

        if split is None or split["improvement"] <= 0:
            print(f"  {domain}: no useful piecewise split found; using fallback")
            continue

        feature = split["feature"]
        cut = split["split_value"]

        x_train = pd.to_numeric(train_pool.loc[mtr, feature], errors="coerce").to_numpy(dtype=float)
        left_train = x_train <= cut
        right_train = x_train > cut

        print(
            f"  {domain}: split {feature} <= {cut:.5g}; "
            f"linear MAE {split['single_line_mae']:.3f} -> {split['piecewise_line_mae']:.3f}; "
            f"left={left_train.sum():,}, right={right_train.sum():,}"
        )

        split["domain"] = domain
        split["domain_n"] = n_domain
        piece_rows.append(split)

        # Train branch experts.
        # Deterministic across processes. Python salts str hashes per process
        # unless PYTHONHASHSEED is fixed, so builtin hash() made branch experts
        # irreproducible between runs. crc32 is stable everywhere.
        left_model = et(args.seed + 700 + stable_seed(domain + feature + "L"), n=500)
        right_model = et(args.seed + 800 + stable_seed(domain + feature + "R"), n=500)

        X_domain = Xtrain[mtr]
        left_model.fit(X_domain[left_train], y_domain[left_train])
        right_model.fit(X_domain[right_train], y_domain[right_train])

        # Retained so the external QMugs evaluation can route through the
        # piecewise branches too. The external block runs earlier in the
        # script, when these models do not yet exist, so the piecewise
        # external numbers are computed after this loop.
        piecewise_models[domain] = (feature, cut, left_model, right_model)

        # Route validation molecules in this domain.
        if mv.any():
            val_indices = np.where(mv)[0]
            x_val = pd.to_numeric(ref_val.loc[mv, feature], errors="coerce").to_numpy(dtype=float)
            left_val = x_val <= cut
            right_val = x_val > cut

            if left_val.any():
                piece_v[val_indices[left_val]] = left_model.predict(Xv_final[val_indices[left_val]])
            if right_val.any():
                piece_v[val_indices[right_val]] = right_model.predict(Xv_final[val_indices[right_val]])

        # Route test molecules in this domain.
        if mt.any():
            test_indices = np.where(mt)[0]
            x_test = pd.to_numeric(ref_test.loc[mt, feature], errors="coerce").to_numpy(dtype=float)
            left_test = x_test <= cut
            right_test = x_test > cut

            if left_test.any():
                piece_t[test_indices[left_test]] = left_model.predict(Xt_final[test_indices[left_test]])
            if right_test.any():
                piece_t[test_indices[right_test]] = right_model.predict(Xt_final[test_indices[right_test]])

    piece_df = pd.DataFrame(piece_rows)
    piece_df.to_csv(outdir / "piecewise_split_architecture_branches.csv", index=False)

    pred_v["piecewise_split_expert"] = piece_v
    pred_t["piecewise_split_expert"] = piece_t

    # Visualization of learned piecewise splits.
    try:
        import matplotlib.pyplot as plt

        if len(piece_df):
            nplot = min(6, len(piece_df))
            plot_df = piece_df.sort_values("improvement", ascending=False).head(nplot)

            fig, axes = plt.subplots(nplot, 1, figsize=(8, 3.2 * nplot))
            if nplot == 1:
                axes = [axes]

            for ax, (_, row) in zip(axes, plot_df.iterrows()):
                domain = row["domain"]
                feature = row["feature"]
                cut = row["split_value"]

                mask = train_pool["domain"].to_numpy() == domain
                d = train_pool.loc[mask, [feature, "training_label"]].copy()
                d = d.replace([np.inf, -np.inf], np.nan).dropna()

                if len(d) > 5000:
                    d = d.sample(5000, random_state=args.seed)

                ax.scatter(d[feature], d["training_label"], s=4, alpha=0.25)
                ax.axvline(cut, linestyle="--", linewidth=1.5)

                # Draw separate local linear fits for left and right.
                for side_mask in [d[feature] <= cut, d[feature] > cut]:
                    sub = d[side_mask]
                    if len(sub) > 50 and sub[feature].std() > 1e-12:
                        xs = sub[feature].to_numpy(dtype=float)
                        ys = sub["training_label"].to_numpy(dtype=float)
                        A = np.column_stack([xs, np.ones(len(xs))])
                        coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
                        grid = np.linspace(xs.min(), xs.max(), 100)
                        ax.plot(grid, coef[0] * grid + coef[1], linewidth=1.5)

                ax.set_title(
                    f"{domain}: {feature} split at {cut:.4g} "
                    f"(linear MAE improvement={row['improvement']:.3f} eV)"
                )
                ax.set_xlabel(feature)
                ax.set_ylabel("Training label / reference-aligned gap (eV)")
                ax.grid(True, alpha=0.25)

            fig.suptitle("Learned Piecewise Descriptor Splits", fontsize=14, weight="bold")
            fig.tight_layout()
            fig.savefig(outdir / "figure_piecewise_descriptor_splits.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved piecewise split figure to: {outdir / 'figure_piecewise_descriptor_splits.png'}")
    except Exception as exc:
        print(f"WARNING: Could not generate piecewise split figure: {exc}")

    # ------------------------------------------------------------------
    # External QMugs evaluation for the piecewise architecture.
    #
    # The main external block runs before the piecewise experts exist, so it
    # can only score the global and domain models. rCEG is the piecewise
    # architecture, so its external numbers are computed here and merged into
    # the same metrics file.
    # ------------------------------------------------------------------
    if external_test is not None and piecewise_models:
        from sklearn.model_selection import KFold as _KFold

        ext_piece = ext_domain.copy()
        ext_domains_p = external_test["domain"].to_numpy()
        for domain, (feature, cut, lm, rm) in piecewise_models.items():
            mask = ext_domains_p == domain
            if not mask.any():
                continue
            idx = np.where(mask)[0]
            xv = pd.to_numeric(external_test.loc[mask, feature],
                               errors="coerce").to_numpy(dtype=float)
            lsel, rsel = xv <= cut, xv > cut
            if lsel.any():
                ext_piece[idx[lsel]] = lm.predict(Xext[idx[lsel]])
            if rsel.any():
                ext_piece[idx[rsel]] = rm.predict(Xext[idx[rsel]])

        aligned_p = np.zeros(len(ext_piece), dtype=float)
        for fit_idx, eval_idx in _KFold(n_splits=5, shuffle=True,
                                        random_state=args.seed).split(ext_piece):
            off = np.median(yext[fit_idx] - ext_piece[fit_idx])
            aligned_p[eval_idx] = ext_piece[eval_idx] + off

        rows_p = [
            {"evaluation": "strict_frozen_external", "model": "piecewise_split",
             "mae": float(mean_absolute_error(yext, ext_piece)),
             "rmse": float(np.sqrt(np.mean((yext - ext_piece) ** 2))),
             "r2": float(r2_score(yext, ext_piece)), "n": int(len(yext))},
            {"evaluation": "five_fold_crossfit_offset_aligned",
             "model": "piecewise_split",
             "mae": float(mean_absolute_error(yext, aligned_p)),
             "rmse": float(np.sqrt(np.mean((yext - aligned_p) ** 2))),
             "r2": float(r2_score(yext, aligned_p)), "n": int(len(yext))},
        ]
        mpath = outdir / "qmugs_external_metrics.csv"
        prev = pd.read_csv(mpath) if mpath.exists() else pd.DataFrame()
        prev = prev[prev["model"] != "piecewise_split"] if len(prev) else prev
        pd.concat([prev, pd.DataFrame(rows_p)], ignore_index=True).to_csv(
            mpath, index=False)

        per_dom = []
        for domain in sorted(set(ext_domains_p)):
            m = ext_domains_p == domain
            per_dom.append({
                "domain": domain, "n": int(m.sum()),
                "mae_piecewise_split": float(
                    np.mean(np.abs(ext_piece[m] - yext[m]))),
                "mae_piecewise_offset_aligned": float(
                    np.mean(np.abs(aligned_p[m] - yext[m]))),
            })
        pd.DataFrame(per_dom).to_csv(
            outdir / "qmugs_external_mae_by_domain_piecewise.csv", index=False)

        print(f"\nQMugs external, piecewise_split (rCEG): "
              f"raw MAE {rows_p[0]['mae']:.4f}, "
              f"offset-aligned {rows_p[1]['mae']:.4f} "
              f"(R2 {rows_p[1]['r2']:.4f})")

    # Stacker trained only on stack-validation real reference labels.
    names = list(pred_v.keys())
    Pv = np.column_stack([pred_v[n] for n in names])
    Pt = np.column_stack([pred_t[n] for n in names])

    stacker = Pipeline([
        ("scale", StandardScaler()),
        ("model", RidgeCV(alphas=np.logspace(-6, 5, 60))),
    ])
    stacker.fit(Pv, yval)

    pred_v["stack_reference_validated"] = stacker.predict(Pv)
    pred_t["stack_reference_validated"] = stacker.predict(Pt)

    # Final metrics.
    print("\nFINAL metrics on real held-out reference labels:")
    final_rows = []
    for n in pred_v:
        final_rows.append(metric("VAL " + n, yval, pred_v[n]))
        final_rows.append(metric("TEST " + n, ytest, pred_t[n]))

    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(outdir / "FINAL_reference_holdout_metrics.csv", index=False)

    preprocessing_rows = [
        {
            "component": "ExtraTrees and RandomForest",
            "transformation": "Median imputation + variance filtering; no scaling",
            "reason": "Tree split thresholds are scale invariant",
        },
        {
            "component": "HistGradientBoosting",
            "transformation": "Median imputation + variance filtering; no scaling",
            "reason": "Tree-based boosting does not require feature scaling",
        },
        {
            "component": "Linear/Ridge calibration",
            "transformation": "Robust, Standard, and Yeo-Johnson variants compared",
            "reason": "Scaling and skew correction can affect regularized linear models",
        },
        {
            "component": "Global Ridge experts",
            "transformation": "RobustScaler and StandardScaler variants",
            "reason": "Provides scaled regularized alternatives to tree experts",
        },
        {
            "component": "Reference-validated stacker",
            "transformation": "StandardScaler fitted on validation predictions",
            "reason": "Makes Ridge penalties comparable between model outputs",
        },
        {
            "component": "Applicability-domain distance",
            "transformation": "Median imputation + RobustScaler",
            "reason": "Euclidean distances require comparable feature scales",
        },
    ]
    pd.DataFrame(preprocessing_rows).to_csv(
        outdir / "preprocessing_and_scaling_summary.csv",
        index=False,
    )

    # Metrics by domain / source / confidence on final test.
    test_report = ref_test[["smiles", "ref_gap", "pqr_gap", "domain", "ref_source"]].copy()
    for n, p in pred_t.items():
        test_report["pred_" + n] = p
        test_report["abs_err_" + n] = np.abs(p - test_report["ref_gap"])

    test_report.to_csv(outdir / "final_test_predictions.csv", index=False)

    best_col = "abs_err_stack_reference_validated"
    group_rows = []
    for group_col in ["domain", "ref_source"]:
        for key, g in test_report.groupby(group_col):
            group_rows.append({
                "group_type": group_col,
                "group": key,
                "n": len(g),
                "mae_stack": float(g[best_col].mean()),
                "mae_domain_expert": float(g["abs_err_domain_expert"].mean()),
                "mae_global_et": float(g["abs_err_global_et"].mean()),
            })

    pd.DataFrame(group_rows).to_csv(outdir / "mae_by_domain_and_source.csv", index=False)

    # Save all-PQR predictions.
    all_preds = pqr[[
        "smiles",
        "domain",
        "pqr_gap",
        "pred_qm9_aligned_gap",
        "inverse_reconstructed_pqr_gap",
        "cycle_error",
        "cycle_confidence",
        "training_label",
        "label_source",
        "heavy_atoms",
        "calibration_distance",
        "calibration_confidence",
        "final_confidence",
    ]].copy()

    all_preds.to_csv(outdir / "all_pqr_predictions_with_label_source.csv", index=False)

    # Recompute candidates: low-confidence / broad domains / high cycle error.
    cand = all_preds[
        (all_preds["final_confidence"].isin(["low_confidence", "medium_confidence"]))
        | (all_preds["cycle_error"] > 0.50)
    ].copy()

    cand = cand.sort_values(
        ["final_confidence", "cycle_error", "calibration_distance"],
        ascending=[True, False, False],
    )

    cand.to_csv(outdir / "recommended_next_recompute_candidates.csv", index=False)

    if args.save_models:
        joblib.dump({
            "calibration_models": cal_models,
            "calibration_top": top,
            "calibration_weights": weights,
            "global_hgb": g_hgb,
            "global_et": g_et,
            "global_rf": g_rf,
            "stacker": stacker,
            "feature_columns": feat_cols,
            # "regime_edges" was dropped when the calibrated-gap regime split
            # was replaced by the piecewise descriptor split; `edges` no longer
            # exists and referencing it raised NameError under --save-models.
            "piecewise_branches": piece_rows,
            "domain_models": sorted(domain_models.keys()),
        }, outdir / "full_domain_moe_model_bundle.joblib")

    print(f"\nSaved outputs to: {outdir.resolve()}")
    print("Key files:")
    print("  FINAL_reference_holdout_metrics.csv")
    print("  mae_by_domain_and_source.csv")
    print("  all_pqr_predictions_with_label_source.csv")
    print("  recommended_next_recompute_candidates.csv")


if __name__ == "__main__":
    main()
