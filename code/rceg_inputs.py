#!/usr/bin/env python3
"""
rceg_inputs.py

Shared input loading for the analysis scripts that sit alongside the main
model (`run_B6_structure_only.py`, `emit_hgb_convergence_history.py`,
`benchmark_computational_cost.py`).

Why this exists
---------------
Each of those scripts used to open a pickled cache from a hardcoded path
inside a session scratch directory:

    SCRATCH = Path("/private/tmp/claude-501/.../scratchpad")
    cache = pickle.load(open(SCRATCH / "cache.pkl", "rb"))

That path exists only on the machine the analysis was first run on. Anyone
reproducing from the deposit would hit `FileNotFoundError` at run_all.sh
steps 8 and 10 — the steps that produce Figure 3's convergence trace and the
Table 9 cost benchmark. The cache was only ever a speed-up, never a source of
truth.

`build_inputs` rebuilds the corpus and merged reference pool from the source
files using the model module's own loaders, so the inputs are identical to
those `pqr_full_domain_moe_qmugs_external.py::main()` constructs. Passing
`--cache PATH` restores the speed-up: the file is read if it exists and
written if it does not.
"""

from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path

import pandas as pd

# Defaults match run_all.sh.
DEF_PQR = "enhanced_dataset_lasso_STRICT.jsonl"
DEF_QM9 = "qm9_gap_reference.csv"
DEF_EXTRA = "pqr_recomputed_reference.csv"
DEF_EXT = "external_validation/qmugs/qmugs_external_reference.csv"
MODEL_SCRIPT = "pqr_full_domain_moe_qmugs_external.py"


def load_model_module(path: str = MODEL_SCRIPT):
    """Import the model script as a module without executing its main()."""
    spec = importlib.util.spec_from_file_location("rceg", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def add_input_args(ap):
    """Attach the standard input flags to an ArgumentParser."""
    ap.add_argument("--pqr", default=DEF_PQR)
    ap.add_argument("--qm9-ref", default=DEF_QM9)
    ap.add_argument("--extra-ref", default=DEF_EXTRA)
    ap.add_argument("--external-ref", default=DEF_EXT)
    ap.add_argument("--cache", default=None,
                    help="optional pickle cache of (pqr, audit, ref); "
                         "read if present, written if absent")
    return ap


def build_inputs(m, args):
    """
    Return (pqr, audit, ref), merging the reference pool exactly as the
    model's main() does: QM9 overlap plus the Psi4 recomputed set, grouped by
    SMILES with a median gap and a joined source label.
    """
    if getattr(args, "cache", None) and Path(args.cache).exists():
        with open(args.cache, "rb") as fh:
            c = pickle.load(fh)
        return c["pqr"], c["audit"], c["ref"]

    missing = [p for p in (args.pqr, args.qm9_ref) if not Path(p).exists()]
    if missing:
        raise SystemExit(
            "Required input(s) not found: " + ", ".join(missing) +
            "\nRun the earlier steps of run_all.sh first, or pass explicit "
            "--pqr / --qm9-ref paths."
        )

    pqr, audit = m.load_pqr(args.pqr)
    refs = [m.load_reference(args.qm9_ref, "qm9_overlap")]
    if getattr(args, "extra_ref", None) and Path(args.extra_ref).exists():
        refs.append(m.load_reference(args.extra_ref,
                                     "recomputed_pqr_reference"))

    ref = pd.concat(refs, ignore_index=True)
    ref = ref.groupby("smiles", as_index=False).agg({
        "ref_gap": "median",
        "ref_source": lambda x: "+".join(sorted(set(map(str, x)))),
    })

    if getattr(args, "cache", None):
        with open(args.cache, "wb") as fh:
            pickle.dump({"pqr": pqr, "audit": audit, "ref": ref}, fh)
    return pqr, audit, ref


def excise_external(m, pqr, audit, ref, external_ref):
    """
    Remove the external benchmark molecules by SMILES from all three frames.
    Mirrors the excision the model performs before leakage filtering,
    calibration, pseudo-labelling and expert training.
    """
    extref = m.load_reference(external_ref, "qmugs_dft_external")
    ext = set(pqr.merge(extref, on="smiles", how="inner")["smiles"])
    pqr = pqr[~pqr.smiles.isin(ext)].reset_index(drop=True)
    audit = audit[~audit.smiles.isin(ext)].reset_index(drop=True)
    ref = ref[~ref.smiles.isin(ext)].reset_index(drop=True)
    return pqr, audit, ref, ext
