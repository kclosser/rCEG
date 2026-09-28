#!/usr/bin/env python3
"""
pqr_final_hybrid_regime.py

Final hybrid electronic-regime model for HOMO-LUMO gap prediction.

Goal:
- Treat number of regimes K as a validation-selected hyperparameter.
- Use descriptor/RDKit/Morgan features.
- Optionally add RDKit 3D conformer descriptors.
- Use a SMILES Transformer-style encoder.
- Use a graph neural network over RDKit molecular graphs.
- Train regime-specific experts.
- Evaluate realistic routed predictions, not only oracle predictions.

Important:
This is a real non-oracle model. It may or may not reach <0.1 eV.
The model will tell us whether the realistic router is strong enough.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

try:
    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors, AllChem, Descriptors3D
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    from rdkit.DataStructs import ConvertToNumpyArray
    RDLogger.DisableLog("rdApp.*")
    RDKIT = True
except Exception:
    RDKIT = False


# -----------------------------
# utilities
# -----------------------------

class PercentileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lo=0.1, hi=99.9):
        self.lo = lo
        self.hi = hi

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        self.lo_ = np.nanpercentile(X, self.lo, axis=0)
        self.hi_ = np.nanpercentile(X, self.hi, axis=0)
        self.lo_ = np.where(np.isfinite(self.lo_), self.lo_, -1e6)
        self.hi_ = np.where(np.isfinite(self.hi_), self.hi_, 1e6)
        same = self.hi_ <= self.lo_
        self.hi_[same] = self.lo_[same] + 1e-6
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        return np.clip(X, self.lo_, self.hi_)


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def fnum(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else np.nan
    except Exception:
        return np.nan


def load_json_or_jsonl(path):
    path = Path(path)
    raw = path.read_text(encoding="utf-8", errors="ignore").strip()

    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            if len(obj) >= 5 and isinstance(obj[1], str):
                return [obj]
            return obj
        return [obj]
    except json.JSONDecodeError:
        pass

    out = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip().rstrip(",")
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, list) and len(obj) >= 5 and isinstance(obj[1], str):
                    out.append(obj)
                elif isinstance(obj, list) and obj and isinstance(obj[0], list):
                    out.extend(obj)
            except Exception:
                pass
    return out


# -----------------------------
# chemistry features
# -----------------------------

SMARTS = {
    "aromatic": "a",
    "hetero_aromatic": "[a;!#6]",
    "carbonyl": "[CX3]=[OX1]",
    "amide": "C(=O)N",
    "ester": "C(=O)O",
    "acid": "C(=O)[OX2H1]",
    "nitrile": "C#N",
    "alkene": "C=C",
    "alkyne": "C#C",
    "amine": "[NX3;H2,H1,H0;!$(NC=O)]",
    "ether": "[OD2]([#6])[#6]",
    "alcohol": "[OX2H][#6]",
    "fluoro": "[F]",
    "chloro": "[Cl]",
    "sulfur": "[S]",
    "nitro": "[NX3](=O)=O",
    "phenol_like": "aO",
    "aniline_like": "aN",
}


def smarts_flags(mol):
    vals = []
    for smarts in SMARTS.values():
        patt = Chem.MolFromSmarts(smarts)
        vals.append(float(patt is not None and mol.HasSubstructMatch(patt)))
    return vals


def rdkit_2d_features(mol):
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
    ]


def rdkit_3d_features(mol):
    """
    Adds true 3D conformer-derived features.
    If conformer generation fails, returns NaNs that are later imputed.
    """
    try:
        m = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        ok = AllChem.EmbedMolecule(m, params)
        if ok != 0:
            return [np.nan] * 11

        try:
            AllChem.UFFOptimizeMolecule(m, maxIters=100)
        except Exception:
            pass

        return [
            Descriptors3D.Asphericity(m),
            Descriptors3D.Eccentricity(m),
            Descriptors3D.InertialShapeFactor(m),
            Descriptors3D.NPR1(m),
            Descriptors3D.NPR2(m),
            Descriptors3D.PMI1(m),
            Descriptors3D.PMI2(m),
            Descriptors3D.PMI3(m),
            Descriptors3D.RadiusOfGyration(m),
            Descriptors3D.SpherocityIndex(m),
            Descriptors3D.PBF(m),
        ]
    except Exception:
        return [np.nan] * 11


def morgan_arr(mol, gen, n_bits):
    fp = gen.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.float32)
    ConvertToNumpyArray(fp, arr)
    return arr


# -----------------------------
# SMILES tokenization
# -----------------------------

TOKENS = [
    "<PAD>", "<CLS>", "<UNK>",
    "Br", "Cl", "Si", "Na", "Li", "Mg", "Al", "Ca",
    "C", "N", "O", "S", "P", "F", "I", "B",
    "c", "n", "o", "s", "p",
    "[", "]", "(", ")", "=", "#", "-", "+", "@", "/", "\\", ".",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "%",
    "H", "h",
]
TOK2IDX = {t: i for i, t in enumerate(TOKENS)}
PAD_IDX = TOK2IDX["<PAD>"]
CLS_IDX = TOK2IDX["<CLS>"]
UNK_IDX = TOK2IDX["<UNK>"]


def tokenize(smiles, max_len):
    out = [CLS_IDX]
    i = 0
    while i < len(smiles) and len(out) < max_len:
        if i + 1 < len(smiles) and smiles[i:i+2] in TOK2IDX:
            out.append(TOK2IDX[smiles[i:i+2]])
            i += 2
        elif smiles[i] in TOK2IDX:
            out.append(TOK2IDX[smiles[i]])
            i += 1
        else:
            out.append(UNK_IDX)
            i += 1
    while len(out) < max_len:
        out.append(PAD_IDX)
    return np.array(out[:max_len], dtype=np.int64)


# -----------------------------
# graph features
# -----------------------------

ATOM_LIST = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]


def atom_features(atom):
    z = atom.GetAtomicNum()
    onehot = [float(z == x) for x in ATOM_LIST]
    other = float(z not in ATOM_LIST)
    return onehot + [
        other,
        atom.GetTotalDegree() / 4.0,
        atom.GetFormalCharge(),
        float(atom.GetIsAromatic()),
        atom.GetTotalNumHs() / 4.0,
        atom.GetMass() / 200.0,
    ]


def graph_arrays(mol, max_atoms):
    nfeat = len(ATOM_LIST) + 6
    X = np.zeros((max_atoms, nfeat), dtype=np.float32)
    A = np.zeros((max_atoms, max_atoms), dtype=np.float32)

    atoms = list(mol.GetAtoms())
    n = min(len(atoms), max_atoms)

    for i in range(n):
        X[i] = np.array(atom_features(atoms[i]), dtype=np.float32)

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        if i < max_atoms and j < max_atoms:
            order = bond.GetBondTypeAsDouble()
            A[i, j] = order
            A[j, i] = order

    for i in range(n):
        A[i, i] = 1.0

    deg = A.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1.0
    A = A / deg
    return X, A


# -----------------------------
# dataset construction
# -----------------------------

def build_dataset(entries, args):
    gen2 = GetMorganGenerator(radius=2, fpSize=args.n_bits)
    gen3 = GetMorganGenerator(radius=3, fpSize=args.n_bits)

    rng = np.random.default_rng(args.seed)
    if args.max_rows is not None and len(entries) > args.max_rows:
        idx = rng.choice(len(entries), args.max_rows, replace=False)
        entries = [entries[i] for i in idx]
        print(f"Subsampled to {len(entries):,} rows")

    rows, Xs, tokens, atom_Xs, adjs = [], [], [], [], []
    drops = Counter()

    for i, e in enumerate(entries):
        if not (isinstance(e, list) and len(e) >= 5 and isinstance(e[1], str)):
            drops["bad_shape"] += 1
            continue

        gap = fnum(e[4])
        if not math.isfinite(gap) or gap < 0.10 or gap > 25:
            drops["bad_gap"] += 1
            continue

        mol = Chem.MolFromSmiles(e[1], sanitize=True)
        if mol is None:
            drops["bad_smiles"] += 1
            continue
        if len(Chem.GetMolFrags(mol)) > 1:
            drops["fragment"] += 1
            continue

        canon = Chem.MolToSmiles(mol, canonical=True)
        pqr = e[2] if isinstance(e[2], list) else []
        lasso = e[3] if isinstance(e[3], list) else []

        pqr5 = [fnum(v) for v in pqr[:5]]
        lasso_vals = [fnum(v) for v in lasso]

        feats = []
        feats.extend(pqr5)
        feats.extend(lasso_vals)
        feats.extend(rdkit_2d_features(mol))
        feats.extend(smarts_flags(mol))
        if args.use_3d:
            feats.extend(rdkit_3d_features(mol))
        feats.extend(morgan_arr(mol, gen2, args.n_bits))
        feats.extend(morgan_arr(mol, gen3, args.n_bits))

        gx, ga = graph_arrays(mol, args.max_atoms)

        rows.append({
            "idx": i,
            "smiles": canon,
            "gap": gap,
            "heavy_atoms": mol.GetNumHeavyAtoms(),
            "rings": rdMolDescriptors.CalcNumRings(mol),
            "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        })
        Xs.append(np.array(feats, dtype=np.float32))
        tokens.append(tokenize(canon, args.max_smiles_len))
        atom_Xs.append(gx)
        adjs.append(ga)

    max_len = max(len(x) for x in Xs)
    X = np.full((len(Xs), max_len), np.nan, dtype=np.float32)
    for i, x in enumerate(Xs):
        X[i, :len(x)] = x

    df = pd.DataFrame(rows)
    y = df["gap"].to_numpy(np.float32)
    groups = df["smiles"].to_numpy()

    print(f"Usable rows: {len(df):,}; drops={dict(drops)}")
    print(f"Target mean/sd/min/max: {y.mean():.3f}/{y.std():.3f}/{y.min():.3f}/{y.max():.3f}")
    print(f"Feature matrix: {X.shape}")
    print(f"Token matrix:   {np.vstack(tokens).shape}")
    print(f"Graph tensor:   {np.stack(atom_Xs).shape}")

    return df, X, np.vstack(tokens), np.stack(atom_Xs), np.stack(adjs), y, groups


def split(groups, y, seed):
    idx = np.arange(len(y))
    trainval, test = next(
        GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed).split(idx, y, groups)
    )
    train_rel, val_rel = next(
        GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=seed + 1).split(
            trainval, y[trainval], groups[trainval]
        )
    )
    return trainval[train_rel], trainval[val_rel], test


def make_regimes(y_train, k):
    edges = np.percentile(y_train, np.linspace(0, 100, k + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def apply_regimes(y, edges):
    return np.digitize(y, edges[1:-1], right=False)


# -----------------------------
# sklearn models
# -----------------------------

def make_tree_reg(seed):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clip", PercentileClipper()),
        ("var", VarianceThreshold(1e-12)),
        ("model", ExtraTreesRegressor(
            n_estimators=500,
            max_features=0.35,
            min_samples_leaf=1,
            n_jobs=1,
            random_state=seed,
        )),
    ])


# -----------------------------
# torch model
# -----------------------------

class HybridDS(Dataset):
    def __init__(self, Xtab, tokens, gx, adj, y, reg, indices):
        self.Xtab = torch.tensor(Xtab[indices], dtype=torch.float32)
        self.tokens = torch.tensor(tokens[indices], dtype=torch.long)
        self.gx = torch.tensor(gx[indices], dtype=torch.float32)
        self.adj = torch.tensor(adj[indices], dtype=torch.float32)
        self.y = torch.tensor(y[indices], dtype=torch.float32)
        self.reg = torch.tensor(reg[indices], dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {
            "Xtab": self.Xtab[i],
            "tokens": self.tokens[i],
            "gx": self.gx[i],
            "adj": self.adj[i],
            "y": self.y[i],
            "reg": self.reg[i],
        }


class HybridRouter(nn.Module):
    def __init__(self, n_tab, n_atom, k, vocab=len(TOKENS), d=96):
        super().__init__()

        self.tok_emb = nn.Embedding(vocab, d, padding_idx=PAD_IDX)
        self.pos_emb = nn.Embedding(256, d)
        enc = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.smiles_enc = nn.TransformerEncoder(enc, num_layers=2)

        self.g1 = nn.Linear(n_atom, d)
        self.g2 = nn.Linear(d, d)
        self.g3 = nn.Linear(d, d)

        self.tab = nn.Sequential(
            nn.Linear(n_tab, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(256, d),
            nn.LayerNorm(d),
            nn.SiLU(),
        )

        self.fuse = nn.Sequential(
            nn.Linear(d * 3, 192),
            nn.LayerNorm(192),
            nn.SiLU(),
            nn.Dropout(0.1),
        )

        self.regime = nn.Linear(192, k)
        self.gap = nn.Linear(192, 1)

    def forward(self, Xtab, tokens, gx, adj):
        B, L = tokens.shape
        pos = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, -1)
        pad = tokens == PAD_IDX
        s = self.tok_emb(tokens) + self.pos_emb(pos)
        s = self.smiles_enc(s, src_key_padding_mask=pad)
        s = s[:, 0, :]

        h = F.silu(self.g1(gx))
        h = torch.bmm(adj, h)
        h = F.silu(self.g2(h))
        h = torch.bmm(adj, h)
        h = F.silu(self.g3(h))
        mask = gx.abs().sum(dim=-1) > 0
        denom = mask.sum(dim=1).clamp(min=1).unsqueeze(-1).float()
        g = (h * mask.unsqueeze(-1).float()).sum(dim=1) / denom

        t = self.tab(Xtab)

        z = self.fuse(torch.cat([s, g, t], dim=1))
        return self.regime(z), self.gap(z).squeeze(-1), z


def preprocess_tab(X_train, X_all):
    imp = SimpleImputer(strategy="median")
    clip = PercentileClipper()
    X1 = imp.fit_transform(X_train)
    clip.fit(X1)
    Xtr = clip.transform(X1)
    Xa = clip.transform(imp.transform(X_all))

    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0)
    std[std < 1e-8] = 1.0

    Xa = ((Xa - mean) / std).astype(np.float32)
    return Xa, (imp, clip, mean, std)


def train_router_model(model, train_ds, val_ds, device, args):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    vloader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    y_mean = float(train_ds.y.mean())
    y_std = float(train_ds.y.std().clamp(min=1e-6))

    best_state = None
    best_score = -1e9
    patience = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        for b in loader:
            Xtab = b["Xtab"].to(device)
            tok = b["tokens"].to(device)
            gx = b["gx"].to(device)
            adj = b["adj"].to(device)
            y = b["y"].to(device)
            reg = b["reg"].to(device)

            logits, pred, _ = model(Xtab, tok, gx, adj)
            loss_regime = F.cross_entropy(logits, reg)
            loss_gap = F.smooth_l1_loss((pred - y_mean) / y_std, (y - y_mean) / y_std)
            loss = loss_regime + args.gap_loss_weight * loss_gap

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        preds, true = [], []
        gp, gt = [], []
        with torch.no_grad():
            for b in vloader:
                logits, pred, _ = model(
                    b["Xtab"].to(device),
                    b["tokens"].to(device),
                    b["gx"].to(device),
                    b["adj"].to(device),
                )
                preds.extend(logits.argmax(1).cpu().numpy())
                true.extend(b["reg"].numpy())
                gp.extend(pred.cpu().numpy())
                gt.extend(b["y"].numpy())

        bal = balanced_accuracy_score(true, preds)
        mae = mean_absolute_error(gt, gp)
        score = bal - 0.02 * mae

        if ep == 1 or ep % 5 == 0:
            print(f"    epoch {ep:03d}: val_bal_acc={bal:.4f}, val_gap_mae={mae:.4f}")

        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break

    model.load_state_dict(best_state)
    return model


def predict_router(model, ds, device, k, batch_size):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    probs, gaps = [], []
    with torch.no_grad():
        for b in loader:
            logits, pred, _ = model(
                b["Xtab"].to(device),
                b["tokens"].to(device),
                b["gx"].to(device),
                b["adj"].to(device),
            )
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            gaps.append(pred.cpu().numpy())
    return np.vstack(probs), np.concatenate(gaps)


# -----------------------------
# experiment per K
# -----------------------------

def evaluate_prediction(name, y_true, pred):
    mae = mean_absolute_error(y_true, pred)
    r2 = r2_score(y_true, pred)
    print(f"{name:30s} MAE={mae:.4f} eV  R2={r2:.4f}")
    return {"name": name, "mae": float(mae), "r2": float(r2)}


def run_for_k(k, df, X, tokens, gx, adj, y, groups, train_idx, val_idx, test_idx, Xscaled, device, args, outdir):
    print("\n" + "=" * 80)
    print(f"Running K={k} regimes")
    print("=" * 80)

    edges = make_regimes(y[train_idx], k)
    reg = apply_regimes(y, edges)

    for i in range(k):
        print(f"  regime {i}: {edges[i]:8.3f} to {edges[i+1]:8.3f}, n_train={(reg[train_idx] == i).sum():,}")

    # global model
    global_model = make_tree_reg(args.seed + k)
    global_model.fit(X[train_idx], y[train_idx])
    val_global = global_model.predict(X[val_idx])
    test_global = global_model.predict(X[test_idx])

    # train neural router
    tr_ds = HybridDS(Xscaled, tokens, gx, adj, y, reg, train_idx)
    va_ds = HybridDS(Xscaled, tokens, gx, adj, y, reg, val_idx)
    te_ds = HybridDS(Xscaled, tokens, gx, adj, y, reg, test_idx)

    model = HybridRouter(n_tab=Xscaled.shape[1], n_atom=gx.shape[2], k=k, d=args.hidden_dim)
    model = train_router_model(model, tr_ds, va_ds, device, args)

    val_probs, val_nn_gap = predict_router(model, va_ds, device, k, args.batch_size)
    test_probs, test_nn_gap = predict_router(model, te_ds, device, k, args.batch_size)

    val_reg_pred = val_probs.argmax(axis=1)
    test_reg_pred = test_probs.argmax(axis=1)

    val_acc = accuracy_score(reg[val_idx], val_reg_pred)
    test_acc = accuracy_score(reg[test_idx], test_reg_pred)
    val_bal = balanced_accuracy_score(reg[val_idx], val_reg_pred)
    test_bal = balanced_accuracy_score(reg[test_idx], test_reg_pred)
    val_adj = np.mean(np.abs(val_reg_pred - reg[val_idx]) <= 1)
    test_adj = np.mean(np.abs(test_reg_pred - reg[test_idx]) <= 1)

    print(f"Router VAL acc={val_acc:.4f}, bal={val_bal:.4f}, adjacent={val_adj:.4f}")
    print(f"Router TEST acc={test_acc:.4f}, bal={test_bal:.4f}, adjacent={test_adj:.4f}")

    # regime experts
    val_expert = np.zeros((len(val_idx), k), dtype=np.float32)
    test_expert = np.zeros((len(test_idx), k), dtype=np.float32)

    for r in range(k):
        mask = reg[train_idx] == r
        expert = make_tree_reg(args.seed + 1000 + k * 10 + r)
        expert.fit(X[train_idx][mask], y[train_idx][mask])
        val_expert[:, r] = expert.predict(X[val_idx])
        test_expert[:, r] = expert.predict(X[test_idx])

    val_soft = (val_probs * val_expert).sum(axis=1)
    test_soft = (test_probs * test_expert).sum(axis=1)

    val_hard = val_expert[np.arange(len(val_idx)), val_reg_pred]
    test_hard = test_expert[np.arange(len(test_idx)), test_reg_pred]

    val_oracle = val_expert[np.arange(len(val_idx)), reg[val_idx]]
    test_oracle = test_expert[np.arange(len(test_idx)), reg[test_idx]]

    # stacker trained on validation only, evaluated on test
    P_val = np.vstack([val_global, val_nn_gap, val_soft, val_hard]).T
    P_test = np.vstack([test_global, test_nn_gap, test_soft, test_hard]).T
    stacker = RidgeCV(alphas=np.logspace(-6, 3, 30))
    stacker.fit(P_val, y[val_idx])
    val_stack = stacker.predict(P_val)
    test_stack = stacker.predict(P_test)

    results = []
    results.append(evaluate_prediction("VAL global", y[val_idx], val_global))
    results.append(evaluate_prediction("TEST global", y[test_idx], test_global))
    results.append(evaluate_prediction("VAL neural_gap", y[val_idx], val_nn_gap))
    results.append(evaluate_prediction("TEST neural_gap", y[test_idx], test_nn_gap))
    results.append(evaluate_prediction("VAL soft_moe", y[val_idx], val_soft))
    results.append(evaluate_prediction("TEST soft_moe", y[test_idx], test_soft))
    results.append(evaluate_prediction("VAL hard_moe", y[val_idx], val_hard))
    results.append(evaluate_prediction("TEST hard_moe", y[test_idx], test_hard))
    results.append(evaluate_prediction("VAL stack_realistic", y[val_idx], val_stack))
    results.append(evaluate_prediction("TEST stack_realistic", y[test_idx], test_stack))
    results.append(evaluate_prediction("VAL ORACLE", y[val_idx], val_oracle))
    results.append(evaluate_prediction("TEST ORACLE", y[test_idx], test_oracle))

    kdir = outdir / f"K{k}"
    kdir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(results).to_csv(kdir / "metrics.csv", index=False)
    pd.DataFrame([{
        "K": k,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "val_bal": val_bal,
        "test_bal": test_bal,
        "val_adjacent": val_adj,
        "test_adjacent": test_adj,
    }]).to_csv(kdir / "router_report.csv", index=False)

    test_report = df.iloc[test_idx].copy()
    test_report["true_regime"] = reg[test_idx]
    test_report["pred_regime"] = test_reg_pred
    test_report["pred_stack"] = test_stack
    test_report["pred_oracle"] = test_oracle
    test_report["abs_error_stack"] = np.abs(test_stack - y[test_idx])
    test_report["abs_error_oracle"] = np.abs(test_oracle - y[test_idx])
    test_report.sort_values("abs_error_stack", ascending=False).to_csv(kdir / "test_error_report.csv", index=False)

    torch.save(model.state_dict(), kdir / "hybrid_router.pt")
    joblib.dump({"global": global_model, "stacker": stacker, "edges": edges}, kdir / "sklearn_parts.joblib")

    return {
        "K": k,
        "val_stack_mae": mean_absolute_error(y[val_idx], val_stack),
        "test_stack_mae": mean_absolute_error(y[test_idx], test_stack),
        "val_soft_mae": mean_absolute_error(y[val_idx], val_soft),
        "test_soft_mae": mean_absolute_error(y[test_idx], test_soft),
        "val_oracle_mae": mean_absolute_error(y[val_idx], val_oracle),
        "test_oracle_mae": mean_absolute_error(y[test_idx], test_oracle),
        "val_router_bal": val_bal,
        "test_router_bal": test_bal,
        "val_adjacent": val_adj,
        "test_adjacent": test_adj,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", default="runs/pqr_final_hybrid")
    ap.add_argument("--max-rows", type=int, default=30000)
    ap.add_argument("--k-list", type=int, nargs="+", default=[2, 4, 6, 8])
    ap.add_argument("--n-bits", type=int, default=512)
    ap.add_argument("--max-atoms", type=int, default=75)
    ap.add_argument("--max-smiles-len", type=int, default=160)
    ap.add_argument("--use-3d", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--hidden-dim", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gap-loss-weight", type=float, default=0.10)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not RDKIT:
        raise RuntimeError("RDKit is required.")

    seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Final hybrid electronic-regime model")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"3D features: {args.use_3d}")

    entries = load_json_or_jsonl(args.data)
    print(f"Loaded raw entries: {len(entries):,}")

    df, X, tokens, gx, adj, y, groups = build_dataset(entries, args)

    train_idx, val_idx, test_idx = split(groups, y, args.seed)
    print(f"Split sizes: train={len(train_idx):,}, val={len(val_idx):,}, test={len(test_idx):,}")

    Xscaled, scaler = preprocess_tab(X[train_idx], X)

    all_rows = []
    for k in args.k_list:
        row = run_for_k(k, df, X, tokens, gx, adj, y, groups, train_idx, val_idx, test_idx, Xscaled, device, args, outdir)
        all_rows.append(row)

    summary = pd.DataFrame(all_rows).sort_values("val_stack_mae")
    summary.to_csv(outdir / "K_sweep_summary.csv", index=False)

    print("\n" + "=" * 80)
    print("K sweep summary, sorted by validation realistic MAE")
    print("=" * 80)
    print(summary.to_string(index=False))
    print(f"\nSaved summary: {outdir / 'K_sweep_summary.csv'}")
    print("\nFor the paper, choose K by validation realistic MAE, then report test once.")


if __name__ == "__main__":
    main()
