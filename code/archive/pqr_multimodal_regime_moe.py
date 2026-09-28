#!/usr/bin/env python3
"""
pqr_multimodal_regime_moe.py

A multimodal electronic-regime mixture-of-experts for HOMO-LUMO gap prediction.

This is the non-oracle version of the regime idea:
1. Define electronic regimes from TRAIN-ONLY gap quantiles.
2. Train base learners: HGB/ExtraTrees, SMILES-BERT router, bond-step SchNet router.
3. Use out-of-fold predictions/probabilities as meta-features, so the regime router
   does not see leaked labels.
4. Train regime-specific gap experts.
5. Predict by probability-weighted mixture of experts.

Quick run:
  python pqr_multimodal_regime_moe.py --data enhanced_dataset_lasso_STRICT.jsonl \
    --outdir runs/pqr_multimodal_quick --max-rows 30000 --n-regimes 8 --epochs 12 --oof-folds 3

Stronger run:
  python pqr_multimodal_regime_moe.py --data enhanced_dataset_lasso_STRICT.jsonl \
    --outdir runs/pqr_multimodal_full --n-regimes 8 --epochs 35 --oof-folds 5
"""

from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline

try:
    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors, rdmolops
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    from rdkit.DataStructs import ConvertToNumpyArray
    RDLogger.DisableLog("rdApp.*")
    RDKIT = True
except Exception:
    RDKIT = False


class PercentileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lo=0.1, hi=99.9):
        self.lo = lo; self.hi = hi
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
        return np.clip(np.asarray(X, dtype=np.float32), self.lo_, self.hi_)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


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
            if len(obj) >= 5 and isinstance(obj[1], str): return [obj]
            return obj
        return [obj]
    except json.JSONDecodeError:
        pass
    out = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip().rstrip(",")
            if not line: continue
            try:
                obj = json.loads(line)
                if isinstance(obj, list) and len(obj) >= 5 and isinstance(obj[1], str): out.append(obj)
                elif isinstance(obj, list) and obj and isinstance(obj[0], list): out.extend(obj)
            except Exception:
                pass
    return out


SMARTS = {
    "aromatic": "a", "hetero_aromatic": "[a;!#6]", "carbonyl": "[CX3]=[OX1]",
    "amide": "C(=O)N", "ester": "C(=O)O", "acid": "C(=O)[OX2H1]",
    "nitrile": "C#N", "alkene": "C=C", "alkyne": "C#C", "amine": "[NX3;H2,H1,H0;!$(NC=O)]",
    "ether": "[OD2]([#6])[#6]", "alcohol": "[OX2H][#6]", "fluoro": "[F]", "chloro": "[Cl]",
    "sulfur": "[S]", "nitro": "[NX3](=O)=O", "phenol_like": "aO", "aniline_like": "aN",
}


def smarts_flags(mol):
    vals = []
    for s in SMARTS.values():
        p = Chem.MolFromSmarts(s)
        vals.append(float(p is not None and mol.HasSubstructMatch(p)))
    return vals


def rdkit_features(mol):
    atoms, bonds = list(mol.GetAtoms()), list(mol.GetBonds())
    return [
        Descriptors.MolWt(mol), Descriptors.ExactMolWt(mol), Descriptors.HeavyAtomMolWt(mol),
        Descriptors.NumValenceElectrons(mol), Descriptors.NumRadicalElectrons(mol), mol.GetNumHeavyAtoms(),
        rdMolDescriptors.CalcNumRings(mol), rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumAliphaticRings(mol), rdMolDescriptors.CalcNumSaturatedRings(mol),
        rdMolDescriptors.CalcNumHBA(mol), rdMolDescriptors.CalcNumHBD(mol), rdMolDescriptors.CalcTPSA(mol),
        Crippen.MolLogP(mol), Crippen.MolMR(mol), Lipinski.NumRotatableBonds(mol),
        Lipinski.NumHeteroatoms(mol), Lipinski.FractionCSP3(mol),
        sum(a.GetIsAromatic() for a in atoms), sum(b.GetIsAromatic() for b in bonds),
        sum(b.GetBondTypeAsDouble() == 2 for b in bonds), sum(b.GetBondTypeAsDouble() == 3 for b in bonds),
        sum(a.GetAtomicNum() == 6 for a in atoms), sum(a.GetAtomicNum() == 7 for a in atoms),
        sum(a.GetAtomicNum() == 8 for a in atoms), sum(a.GetAtomicNum() == 9 for a in atoms),
        sum(a.GetAtomicNum() == 16 for a in atoms), sum(a.GetAtomicNum() == 17 for a in atoms),
    ]


def morgan_arr(mol, gen, n_bits):
    fp = gen.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.float32)
    ConvertToNumpyArray(fp, arr)
    return arr


def bond_matrix(smiles, max_atoms):
    mol = Chem.MolFromSmiles(smiles)
    out = np.zeros((max_atoms, max_atoms), dtype=np.float32)
    if mol is None: return out
    dm = rdmolops.GetDistanceMatrix(mol).astype(np.float32)
    n = min(dm.shape[0], max_atoms)
    out[:n, :n] = dm[:n, :n]
    return out


TOKENS = ["<PAD>","<CLS>","<UNK>","Br","Cl","Si","Na","Li","Mg","Al","Ca","C","N","O","S","P","F","I","B","c","n","o","s","p","[","]","(",")","=","#","-","+","@","/","\\",".","1","2","3","4","5","6","7","8","9","0","%","H","h"]
TOK2IDX = {t:i for i,t in enumerate(TOKENS)}
PAD, CLS, UNK = TOK2IDX["<PAD>"], TOK2IDX["<CLS>"], TOK2IDX["<UNK>"]


def tokenize(smiles, max_len):
    ids = [CLS]
    i = 0
    while i < len(smiles) and len(ids) < max_len:
        if i+1 < len(smiles) and smiles[i:i+2] in TOK2IDX:
            ids.append(TOK2IDX[smiles[i:i+2]]); i += 2
        elif smiles[i] in TOK2IDX:
            ids.append(TOK2IDX[smiles[i]]); i += 1
        else:
            ids.append(UNK); i += 1
    ids += [PAD] * (max_len - len(ids))
    return np.array(ids[:max_len], dtype=np.int64)


def build_dataset(entries, args):
    rng = np.random.default_rng(args.seed)
    if args.max_rows and len(entries) > args.max_rows:
        entries = [entries[i] for i in rng.choice(len(entries), args.max_rows, replace=False)]
        print(f"Subsampled to {len(entries):,} rows")
    gen2, gen3 = GetMorganGenerator(radius=2, fpSize=args.n_bits), GetMorganGenerator(radius=3, fpSize=args.n_bits)
    rows, Xs, Ds, toks, bms, drops = [], [], [], [], [], Counter()
    for i,e in enumerate(entries):
        if not (isinstance(e, list) and len(e) >= 5 and isinstance(e[1], str)):
            drops["bad_shape"] += 1; continue
        gap = fnum(e[4])
        if not math.isfinite(gap) or gap < 0.1 or gap > 25:
            drops["bad_gap"] += 1; continue
        mol = Chem.MolFromSmiles(e[1], sanitize=True)
        if mol is None:
            drops["invalid_smiles"] += 1; continue
        if any(a.GetAtomicNum()==1 and a.GetDegree()==0 for a in mol.GetAtoms()):
            drops["isolated_h"] += 1; continue
        if len(Chem.GetMolFrags(mol)) > 1:
            drops["multi_fragment"] += 1; continue
        smi = Chem.MolToSmiles(mol, canonical=True)
        pqr = e[2] if isinstance(e[2], list) else []
        desc = e[3] if isinstance(e[3], list) else []
        pqr5 = [fnum(v) for v in pqr[:5]]
        lasso = [fnum(v) for v in desc]
        rdk = rdkit_features(mol)
        flags = smarts_flags(mol)
        fp2, fp3 = morgan_arr(mol, gen2, args.n_bits), morgan_arr(mol, gen3, args.n_bits)
        Xs.append(np.concatenate([np.array(pqr5, np.float32), np.array(lasso, np.float32), np.array(rdk, np.float32), np.array(flags, np.float32), fp2, fp3]))
        Ds.append(np.concatenate([np.array(pqr5, np.float32), np.array(lasso, np.float32), np.array(rdk, np.float32), np.array(flags, np.float32)]))
        toks.append(tokenize(smi, args.max_smiles_len))
        bms.append(bond_matrix(smi, args.max_atoms))
        rows.append({"idx": i, "smiles": smi, "gap": gap, "heavy_atoms": mol.GetNumHeavyAtoms(), "rings": rdMolDescriptors.CalcNumRings(mol)})
    if not rows: raise RuntimeError("No usable molecules")
    X = np.full((len(Xs), max(len(x) for x in Xs)), np.nan, dtype=np.float32)
    D = np.full((len(Ds), max(len(x) for x in Ds)), np.nan, dtype=np.float32)
    for i,x in enumerate(Xs): X[i,:len(x)] = x
    for i,d in enumerate(Ds): D[i,:len(d)] = d
    df = pd.DataFrame(rows)
    y = df.gap.to_numpy(np.float32)
    print(f"Usable rows: {len(df):,}; drops={dict(drops)}")
    print(f"Target mean/sd/min/max: {y.mean():.3f}/{y.std():.3f}/{y.min():.3f}/{y.max():.3f}")
    print(f"X={X.shape}, D={D.shape}, tokens={np.vstack(toks).shape}, BM={np.stack(bms).shape}")
    return df, X, D, np.vstack(toks), np.stack(bms), y, df.smiles.to_numpy()


def split(groups, y, seed):
    idx = np.arange(len(y))
    trainval, test = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed).split(idx, y, groups))
    tr_rel, va_rel = next(GroupShuffleSplit(n_splits=1, test_size=0.15/0.85, random_state=seed+1).split(trainval, y[trainval], groups[trainval]))
    return trainval[tr_rel], trainval[va_rel], test


def regimes(y_train, k):
    edges = np.percentile(y_train, np.linspace(0,100,k+1)); edges[0] = -np.inf; edges[-1] = np.inf
    return edges

def apply_regimes(y, edges): return np.digitize(y, edges[1:-1], right=False)


def hgb_reg(seed, kbest):
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("clip", PercentileClipper()), ("var", VarianceThreshold(1e-12)), ("sel", SelectKBest(f_regression, k=kbest)), ("m", HistGradientBoostingRegressor(max_iter=850, learning_rate=0.035, max_leaf_nodes=31, l2_regularization=0.03, early_stopping=True, validation_fraction=0.12, n_iter_no_change=60, random_state=seed, loss="absolute_error"))])

def et_reg(seed):
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("clip", PercentileClipper()), ("var", VarianceThreshold(1e-12)), ("m", ExtraTreesRegressor(n_estimators=450, max_features=0.3, min_samples_leaf=1, n_jobs=-1, random_state=seed))])

def router(kind, seed, kbest):
    if kind == "hgb":
        return Pipeline([("imp", SimpleImputer(strategy="median")), ("clip", PercentileClipper()), ("var", VarianceThreshold(1e-12)), ("sel", SelectKBest(f_classif, k=kbest)), ("m", HistGradientBoostingClassifier(max_iter=700, learning_rate=0.035, max_leaf_nodes=31, l2_regularization=0.03, early_stopping=True, validation_fraction=0.12, n_iter_no_change=55, random_state=seed))])
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("clip", PercentileClipper()), ("var", VarianceThreshold(1e-12)), ("m", ExtraTreesClassifier(n_estimators=600, max_features=0.35, min_samples_leaf=1, n_jobs=-1, random_state=seed, class_weight="balanced"))])


class MolDS(Dataset):
    def __init__(self, D, T, B, y, r, idx):
        self.D=torch.tensor(D[idx], dtype=torch.float32); self.T=torch.tensor(T[idx], dtype=torch.long); self.B=torch.tensor(B[idx], dtype=torch.float32)
        self.y=torch.tensor(y[idx], dtype=torch.float32); self.r=torch.tensor(r[idx], dtype=torch.long); self.idx=np.array(idx)
    def __len__(self): return len(self.idx)
    def __getitem__(self,i): return {"D":self.D[i],"T":self.T[i],"B":self.B[i],"y":self.y[i],"r":self.r[i],"idx":int(self.idx[i])}


class BertRouter(nn.Module):
    def __init__(self, k, max_len, d=96):
        super().__init__(); self.tok=nn.Embedding(len(TOKENS), d, padding_idx=PAD); self.pos=nn.Embedding(max_len,d)
        enc=nn.TransformerEncoderLayer(d_model=d,nhead=4,dim_feedforward=d*3,dropout=0.1,batch_first=True,norm_first=True)
        self.enc=nn.TransformerEncoder(enc, num_layers=3); self.norm=nn.LayerNorm(d); self.cls=nn.Linear(d,k); self.gap=nn.Sequential(nn.Linear(d,64),nn.SiLU(),nn.Linear(64,1))
    def forward(self,T,B=None,D=None):
        b,l=T.shape; p=torch.arange(l,device=T.device).unsqueeze(0).expand(b,-1); z=self.enc(self.tok(T)+self.pos(p), src_key_padding_mask=(T==PAD)); e=self.norm(z[:,0,:]); return self.cls(e), self.gap(e).squeeze(-1), e

class RBF(nn.Module):
    def __init__(self,n=20,cut=10.): super().__init__(); self.register_buffer("c", torch.linspace(0,cut,n)); self.w=(cut/n)**2
    def forward(self,B): return torch.exp(-((B.unsqueeze(-1)-self.c)**2)/self.w)
class SchLayer(nn.Module):
    def __init__(self,d,n): super().__init__(); self.W=nn.Sequential(nn.Linear(n,d),nn.SiLU(),nn.Linear(d,d)); self.a=nn.Sequential(nn.Linear(d,d),nn.SiLU(),nn.Linear(d,d)); self.s=nn.Parameter(torch.tensor(0.1))
    def forward(self,x,r,m): return (x+self.s*self.a((self.W(r)*x.unsqueeze(2).expand_as(self.W(r))).sum(2)))*m.unsqueeze(-1).float()
class SchNetRouter(nn.Module):
    def __init__(self, ndesc, k, max_atoms, d=72):
        super().__init__(); self.emb=nn.Embedding(max_atoms+2,d,padding_idx=0); self.rbf=RBF(); self.layers=nn.ModuleList([SchLayer(d,20) for _ in range(3)])
        self.dm=nn.Sequential(nn.Linear(ndesc,192),nn.LayerNorm(192),nn.SiLU(),nn.Linear(192,96),nn.LayerNorm(96),nn.SiLU())
        self.fuse=nn.Sequential(nn.Linear(d+96,128),nn.LayerNorm(128),nn.SiLU()); self.cls=nn.Linear(128,k); self.gap=nn.Linear(128,1)
    def forward(self,T,B,D):
        bs,n,_=B.shape; m=B.abs().sum(-1)>0; pos=torch.arange(1,n+1,device=B.device).unsqueeze(0).expand(bs,-1); x=self.emb(pos*m.long()); r=self.rbf(B)
        for L in self.layers: x=L(x,r,m)
        g=(x*m.unsqueeze(-1).float()).sum(1)/m.sum(1).clamp(min=1).unsqueeze(-1).float(); e=self.fuse(torch.cat([g,self.dm(D)],1)); return self.cls(e), self.gap(e).squeeze(-1), e


def scale_desc(Dtr, Dall):
    imp=SimpleImputer(strategy="median"); clip=PercentileClipper(); A=imp.fit_transform(Dtr); A=clip.fit_transform(A); mu=A.mean(0); sd=A.std(0); sd[sd<1e-8]=1
    return ((clip.transform(imp.transform(Dall))-mu)/sd).astype(np.float32)


def train_nn(model, train_ds, val_ds, device, args, label):
    model.to(device); opt=torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs,1))
    tr=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True); va=DataLoader(val_ds,batch_size=args.batch_size*2)
    ym=train_ds.y.numpy().mean(); ys=max(train_ds.y.numpy().std(),1e-6); best=None; bestscore=-1e9; pat=0
    for ep in range(1,args.epochs+1):
        model.train()
        for b in tr:
            T=b["T"].to(device); B=b["B"].to(device); D=b["D"].to(device); y=b["y"].to(device); r=b["r"].to(device)
            logits,pred,_=model(T,B,D); loss=F.cross_entropy(logits,r)+0.1*F.smooth_l1_loss((pred-ym)/ys,(y-ym)/ys)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        sched.step(); model.eval(); pr=[]; trr=[]; pg=[]; yg=[]
        with torch.no_grad():
            for b in va:
                logits,pred,_=model(b["T"].to(device), b["B"].to(device), b["D"].to(device)); pr+=logits.argmax(1).cpu().numpy().tolist(); trr+=b["r"].numpy().tolist(); pg+=pred.cpu().numpy().tolist(); yg+=b["y"].numpy().tolist()
        bal=balanced_accuracy_score(trr,pr); mae=mean_absolute_error(yg,pg); score=bal-0.02*mae
        if ep==1 or ep%5==0: print(f"    {label} ep{ep}: bal={bal:.3f}, gap_mae={mae:.3f}")
        if score>bestscore: bestscore=score; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; pat=0
        else:
            pat+=1
            if pat>=7: break
    if best: model.load_state_dict(best)
    return model


def nn_meta(model, ds, device, k):
    loader=DataLoader(ds,batch_size=256); model.to(device); model.eval(); out=[]; idx=[]
    with torch.no_grad():
        for b in loader:
            logits,pred,e=model(b["T"].to(device), b["B"].to(device), b["D"].to(device)); out.append(np.hstack([torch.softmax(logits,1).cpu().numpy(), pred.cpu().numpy().reshape(-1,1), e[:,:32].cpu().numpy()]).astype(np.float32)); idx += b["idx"].numpy().tolist()
    return np.array(idx), np.vstack(out)


def evalp(name,y,p):
    mae=mean_absolute_error(y,p); r2=r2_score(y,p); print(f"{name:38s} MAE={mae:.4f} eV R2={r2:.4f}"); return {"name":name,"mae":mae,"r2":r2}

def adjacc(a,b): return float(np.mean(np.abs(np.asarray(a)-np.asarray(b))<=1))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",required=True); ap.add_argument("--outdir",default="runs/pqr_multimodal")
    ap.add_argument("--max-rows",type=int); ap.add_argument("--n-regimes",type=int,default=8); ap.add_argument("--n-bits",type=int,default=1024); ap.add_argument("--max-atoms",type=int,default=75); ap.add_argument("--max-smiles-len",type=int,default=160)
    ap.add_argument("--oof-folds",type=int,default=3); ap.add_argument("--epochs",type=int,default=12); ap.add_argument("--batch-size",type=int,default=128); ap.add_argument("--lr",type=float,default=1e-3); ap.add_argument("--k-best",type=int,default=2500); ap.add_argument("--expert-k-best",type=int,default=2000); ap.add_argument("--router-k-best",type=int,default=2500); ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args(); seed_all(args.seed); out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True); device="cuda" if torch.cuda.is_available() else "cpu"
    if not RDKIT: raise RuntimeError("RDKit required")
    print("="*80); print("PQR multimodal regime MOE"); print(f"Device: {device}"); print("="*80)
    df,X,D,T,B,y,groups=build_dataset(load_json_or_jsonl(args.data),args); tr,va,te=split(groups,y,args.seed)
    edges=regimes(y[tr],args.n_regimes); r=apply_regimes(y,edges); k=len(edges)-1
    print("Regime edges:"); [print(f"  {i}: {edges[i]:.3f} to {edges[i+1]:.3f}, n={(r[tr]==i).sum()}") for i in range(k)]
    Xtr,Xva,Xte=X[tr],X[va],X[te]; ytr,yva,yte=y[tr],y[va],y[te]; rtr,rva,rte=r[tr],r[va],r[te]; gtr=groups[tr]
    # sklearn oof
    sk_oof=np.zeros((len(tr),5),np.float32); gkf=GroupKFold(n_splits=args.oof_folds); print("Building sklearn OOF...")
    for fold,(a,b) in enumerate(gkf.split(Xtr,ytr,gtr),1):
        print(f"  fold {fold}"); h=hgb_reg(args.seed+fold,min(args.k_best,Xtr.shape[1])); e=et_reg(args.seed+fold); h.fit(Xtr[a],ytr[a]); e.fit(Xtr[a],ytr[a]); ph=h.predict(Xtr[b]); pe=e.predict(Xtr[b]); sk_oof[b]=np.vstack([ph,pe,(ph+pe)/2,np.abs(ph-pe),ph-pe]).T
    hf=hgb_reg(args.seed,min(args.k_best,Xtr.shape[1])); ef=et_reg(args.seed); hf.fit(Xtr,ytr); ef.fit(Xtr,ytr)
    def skmeta(A):
        ph=hf.predict(A); pe=ef.predict(A); return np.vstack([ph,pe,(ph+pe)/2,np.abs(ph-pe),ph-pe]).T.astype(np.float32)
    sk_va, sk_te=skmeta(Xva), skmeta(Xte); results=[]; results += [evalp("VAL base_hgb",yva,sk_va[:,0]),evalp("TEST base_hgb",yte,sk_te[:,0]),evalp("VAL base_mean",yva,sk_va[:,2]),evalp("TEST base_mean",yte,sk_te[:,2])]
    # neural oof
    Dtr_raw,Dva_raw,Dte_raw=D[tr],D[va],D[te]; Ttr,Tva,Tte=T[tr],T[va],T[te]; Btr,Bva,Bte=B[tr],B[va],B[te]
    bert_oof=np.zeros((len(tr),k+33),np.float32); sch_oof=np.zeros_like(bert_oof); print("Building neural OOF...")
    for fold,(a,b) in enumerate(gkf.split(Dtr_raw,ytr,gtr),1):
        print(f"Neural fold {fold}"); Dscaled=scale_desc(Dtr_raw[a],Dtr_raw); ds_a=MolDS(Dscaled,Ttr,Btr,ytr,rtr,a); ds_b=MolDS(Dscaled,Ttr,Btr,ytr,rtr,b)
        bm=BertRouter(k,args.max_smiles_len); bm=train_nn(bm,ds_a,ds_b,device,args,f"BERT {fold}"); idx,meta=nn_meta(bm,ds_b,device,k); pos={int(x):i for i,x in enumerate(idx)}; bert_oof[b]=np.vstack([meta[pos[int(i)]] for i in b])
        sm=SchNetRouter(Dscaled.shape[1],k,args.max_atoms); sm=train_nn(sm,ds_a,ds_b,device,args,f"SchNet {fold}"); idx,meta=nn_meta(sm,ds_b,device,k); pos={int(x):i for i,x in enumerate(idx)}; sch_oof[b]=np.vstack([meta[pos[int(i)]] for i in b])
    # full neural for val/test
    print("Training full neural models..."); Dall=np.vstack([Dtr_raw,Dva_raw,Dte_raw]); Dscaled_all=scale_desc(Dtr_raw,Dall); Dtr_s,Dva_s,Dte_s=np.split(Dscaled_all,[len(Dtr_raw),len(Dtr_raw)+len(Dva_raw)])
    train_ids=np.arange(len(ytr)); sub_tr,sub_va=next(GroupShuffleSplit(n_splits=1,test_size=0.15,random_state=args.seed+77).split(train_ids,ytr,gtr))
    bfull=train_nn(BertRouter(k,args.max_smiles_len), MolDS(Dtr_s,Ttr,Btr,ytr,rtr,sub_tr), MolDS(Dtr_s,Ttr,Btr,ytr,rtr,sub_va), device,args,"BERT full")
    sfull=train_nn(SchNetRouter(Dtr_s.shape[1],k,args.max_atoms), MolDS(Dtr_s,Ttr,Btr,ytr,rtr,sub_tr), MolDS(Dtr_s,Ttr,Btr,ytr,rtr,sub_va), device,args,"SchNet full")
    _,bert_va=nn_meta(bfull,MolDS(Dva_s,Tva,Bva,yva,rva,np.arange(len(yva))),device,k); _,bert_te=nn_meta(bfull,MolDS(Dte_s,Tte,Bte,yte,rte,np.arange(len(yte))),device,k)
    _,sch_va=nn_meta(sfull,MolDS(Dva_s,Tva,Bva,yva,rva,np.arange(len(yva))),device,k); _,sch_te=nn_meta(sfull,MolDS(Dte_s,Tte,Bte,yte,rte,np.arange(len(yte))),device,k)
    router_rows=[]
    for name,vmeta,tmeta in [("bert",bert_va,bert_te),("schnet",sch_va,sch_te)]:
        pv=vmeta[:,:k].argmax(1); pt=tmeta[:,:k].argmax(1); router_rows.append({"router":name,"val_accuracy":accuracy_score(rva,pv),"val_balanced_accuracy":balanced_accuracy_score(rva,pv),"val_adjacent_accuracy":adjacc(rva,pv),"test_accuracy":accuracy_score(rte,pt),"test_balanced_accuracy":balanced_accuracy_score(rte,pt),"test_adjacent_accuracy":adjacc(rte,pt)}); print(router_rows[-1])
    # meta router
    Mtr=np.hstack([Xtr,sk_oof,bert_oof,sch_oof]); Mva=np.hstack([Xva,sk_va,bert_va,sch_va]); Mte=np.hstack([Xte,sk_te,bert_te,sch_te])
    probs_va={}; probs_te={}
    for nm,kind in [("hgb_multimodal","hgb"),("et_multimodal","et")]:
        print(f"Training router {nm}"); ro=router(kind,args.seed,min(args.router_k_best,Mtr.shape[1])); ro.fit(Mtr,rtr); pv=ro.predict(Mva); pt=ro.predict(Mte); router_rows.append({"router":nm,"val_accuracy":accuracy_score(rva,pv),"val_balanced_accuracy":balanced_accuracy_score(rva,pv),"val_adjacent_accuracy":adjacc(rva,pv),"test_accuracy":accuracy_score(rte,pt),"test_balanced_accuracy":balanced_accuracy_score(rte,pt),"test_adjacent_accuracy":adjacc(rte,pt)}); print(router_rows[-1])
        def P(A):
            outp=np.zeros((A.shape[0],k),np.float32); pp=ro.predict_proba(A); cls=ro.named_steps["m"].classes_
            for j,c in enumerate(cls): outp[:,int(c)]=pp[:,j]
            return outp
        probs_va[nm]=P(Mva); probs_te[nm]=P(Mte)
    probs_va["avg_bert_schnet_hgb"]=(bert_va[:,:k]+sch_va[:,:k]+probs_va["hgb_multimodal"])/3; probs_te["avg_bert_schnet_hgb"]=(bert_te[:,:k]+sch_te[:,:k]+probs_te["hgb_multimodal"])/3
    # experts
    val_exp=np.zeros((len(yva),k),np.float32); test_exp=np.zeros((len(yte),k),np.float32); exrows=[]
    print("Training regime experts...")
    for i in range(k):
        mask=rtr==i; ex=hgb_reg(args.seed+100+i,min(args.expert_k_best,Xtr.shape[1])); ex.fit(Xtr[mask],ytr[mask]); val_exp[:,i]=ex.predict(Xva); test_exp[:,i]=ex.predict(Xte); exrows.append({"regime":i,"n_train":int(mask.sum()),"val_oracle_mae":mean_absolute_error(yva[rva==i],val_exp[rva==i,i]),"test_oracle_mae":mean_absolute_error(yte[rte==i],test_exp[rte==i,i])})
    Pva=[sk_va[:,0],sk_va[:,2],bert_va[:,k],sch_va[:,k]]; Pte=[sk_te[:,0],sk_te[:,2],bert_te[:,k],sch_te[:,k]]
    results += [evalp("VAL bert_gap",yva,bert_va[:,k]),evalp("TEST bert_gap",yte,bert_te[:,k]),evalp("VAL schnet_gap",yva,sch_va[:,k]),evalp("TEST schnet_gap",yte,sch_te[:,k])]
    for nm in probs_va:
        vm=(probs_va[nm]*val_exp).sum(1); tm=(probs_te[nm]*test_exp).sum(1); results += [evalp("VAL mix_"+nm,yva,vm),evalp("TEST mix_"+nm,yte,tm)]; Pva.append(vm); Pte.append(tm)
    oracle_va=val_exp[np.arange(len(yva)),rva]; oracle_te=test_exp[np.arange(len(yte)),rte]; results += [evalp("VAL ORACLE_ROUTE",yva,oracle_va),evalp("TEST ORACLE_ROUTE",yte,oracle_te)]
    stack=RidgeCV(alphas=np.logspace(-6,3,30)); stack.fit(np.vstack(Pva).T,yva); stv=stack.predict(np.vstack(Pva).T); stt=stack.predict(np.vstack(Pte).T); results += [evalp("VAL stacked_realistic",yva,stv),evalp("TEST stacked_realistic",yte,stt)]
    pd.DataFrame(results).to_csv(out/"metrics.csv",index=False); pd.DataFrame(router_rows).to_csv(out/"router_report.csv",index=False); pd.DataFrame(exrows).to_csv(out/"regime_expert_report.csv",index=False)
    test_report=df.iloc[te].copy(); test_report["true_regime"]=rte; test_report["pred_stack"]=stt; test_report["pred_oracle"]=oracle_te; test_report["abs_error_stack"]=np.abs(stt-yte); test_report["abs_error_oracle"]=np.abs(oracle_te-yte); test_report.sort_values("abs_error_stack",ascending=False).to_csv(out/"test_error_report.csv",index=False)
    joblib.dump({"edges":edges,"stacker":stack,"args":vars(args)},out/"multimodal_regime_moe.joblib"); torch.save({"bert":bfull.state_dict(),"schnet":sfull.state_dict()},out/"neural_models.pt")
    print("Saved reports to",out)
    print("Most important files: metrics.csv and router_report.csv")

if __name__=="__main__": main()
