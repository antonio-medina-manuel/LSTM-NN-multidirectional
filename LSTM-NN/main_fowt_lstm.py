#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict FOWT responses with a trained early fusion LSTM (compiled-model only).

Key points:
- **Compiled-only model**: loads `nn_models` from a *sourceless* .pyc located in `private_bin/` by default.
- Loads inputs from HDF5 (waves; optional multi-direction).
- Applies min–max normalization from JSON (same stats used at training).
- Builds sliding windows for sequence modeling (seq2seq or seq2one).
- Loads weights from a checkpoint and runs inference.
- Denormalizes predictions to physical units and saves to HDF5 (+ optional CSV).
- Optional quick evaluation vs targets: MSE/MAE/R2 on covered region.

Expected HDF5 layout (adapt as needed):
- inputs.h5:
    /time_series_excitations/WAVES/<case_k>        -> (1,T) or (T,)
    /time_series_excitations/TIMES/<case_k>        -> (1,T) or (T,)
    /time_series_excitations/DIRECTIONALWAVES/WaveDir{d}_XXXX (if --multi-dir)
- outputs.h5 (optional for evaluation):
    /time_series_motionDOFs/{SURGE,HEAVE,...}/<case_k>  -> (1,T) or (T,)
    /time_series_motionDOFs/TIMES/<case_k>              -> (1,T) or (T,)
    /time_series_outputs/TENSIONSFAIRLEAD{1,2,3}/<case_k> (for fairleads)

Author: Antonio Medina (original code)
License: Apache-2.0 (recommended for the repo)
"""

from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import importlib.util
from importlib.machinery import SourcelessFileLoader

# ---------------------------------------------------------------------------------
# Compiled-only model import
# ---------------------------------------------------------------------------------
DEFAULT_PYC = Path(__file__).resolve().parent / "private_bin" / "nn_models.cpython-310.pyc"
DEFAULT_MODNAME = "nn_models"

def _import_compiled_model(pyc_path: Path, module_name: str = DEFAULT_MODNAME):
    """
    Import a compiled (sourceless) Python module from a .pyc file.
    Raises if the file does not exist or cannot be imported.
    """
    if not pyc_path.exists():
        raise FileNotFoundError(f"Compiled model .pyc not found at: {pyc_path}")

    spec = importlib.util.spec_from_loader(module_name, SourcelessFileLoader(module_name, str(pyc_path)))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for: {pyc_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ---------------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------------
def _as_1d(a: np.ndarray) -> np.ndarray:
    """Ensure array shape (T,) from (1,T) or (T,1) or (T,)."""
    a = np.asarray(a)
    if a.ndim == 2:
        if 1 in a.shape:
            return a.reshape(-1)
        # default: take first row if looks like (1, T)
        if a.shape[0] == 1:
            return a[0]
        if a.shape[1] == 1:
            return a[:, 0]
    return a.reshape(-1)

def load_h5_cases(fp: Path, waves_key: str, times_key: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load list of wave signals + times from HDF5.

    Returns:
        waves: list of arrays (T,)
        times: list of arrays (T,)
    """
    waves: List[np.ndarray] = []
    times: List[np.ndarray] = []
    with h5py.File(fp, "r") as f:
        wg = f[waves_key]
        tg = f[times_key]
        case_keys = sorted(list(wg.keys()))
        for k in case_keys:
            waves.append(_as_1d(wg[k][...]))
            times.append(_as_1d(tg[k][...]))
    return waves, times

def load_multidir_waves(fp: Path, dir_num: int,
                        base_group: str = "time_series_excitations/DIRECTIONALWAVES"
                        ) -> Tuple[List[np.ndarray], List[str]]:
    """
    Load directional waves as a list of (T, D) arrays, where D = dir_num.
    It expects datasets named WaveDir1_XXX, WaveDir2_XXX, ..., WaveDirD_XXX
    with the same set of case suffixes XXX across directions.
    """
    waves_cd: List[np.ndarray] = []
    with h5py.File(fp, "r") as f:
        g = f[base_group]
        dir_keys: List[List[str]] = []
        for d in range(1, dir_num + 1):
            prefix = f"WaveDir{d}"
            dir_keys.append(sorted([k for k in g.keys() if k.startswith(prefix)]))
        num_cases = len(dir_keys[0])
        if not all(len(lst) == num_cases for lst in dir_keys):
            raise RuntimeError("Direction lists not aligned across WaveDir* datasets.")
        for i in range(num_cases):
            cols = []
            for d in range(dir_num):
                arr = _as_1d(g[dir_keys[d][i]][...])
                cols.append(arr)
            T = min(len(a) for a in cols)
            cols = [a[:T] for a in cols]
            waves_cd.append(np.stack(cols, axis=-1))  # (T,D)
        suffixes = [k.split("_", 1)[1] if "_" in k else f"{i:04d}" for i, k in enumerate(dir_keys[0])]
    return waves_cd, suffixes

def load_h5_targets(fp: Path, dof_group: str, times_key: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load list of target DoF signals + (optional) times from HDF5.
    `dof_group` is something like 'time_series_motionDOFs/RELATIVE_WAVE'.
    """
    ys: List[np.ndarray] = []
    ts: List[np.ndarray] = []
    with h5py.File(fp, "r") as f:
        yg = f[dof_group]
        case_keys = sorted(list(yg.keys()))
        for k in case_keys:
            ys.append(_as_1d(yg[k][...]))
        if times_key in f:
            tg = f[times_key]
            for k in sorted(list(tg.keys())):
                ts.append(_as_1d(tg[k][...]))
        else:
            ts = [np.arange(len(y)) for y in ys]
    return ys, ts

def load_targets_by_dof(fp: Path, dof_name: str
                        ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Map DoF names to the proper HDF5 group + dataset name.
    - Motions: time_series_motionDOFs/{SURGE,SWAY,HEAVE,ROLL,PITCH,YAW}
    - Fairleads: time_series_outputs/{TENSIONSFAIRLEAD1,2,3}
    Returns (y_cases, t_cases).
    """
    motions = {"surge":"SURGE","sway":"SWAY","heave":"HEAVE",
               "roll":"ROLL","pitch":"PITCH","yaw":"YAW"}
    fl = {"fairlead1":"TENSIONSFAIRLEAD1",
          "fairlead2":"TENSIONSFAIRLEAD2",
          "fairlead3":"TENSIONSFAIRLEAD3"}

    dn = dof_name.lower()
    if dn in motions:
        group = f"time_series_motionDOFs/{motions[dn]}"
        tkey  = "time_series_motionDOFs/TIMES"
    elif dn in fl:
        group = f"time_series_outputs/{fl[dn]}"
        tkey  = "time_series_motionDOFs/TIMES"  # timing still lives here in your files
    else:
        raise ValueError(f"Unknown DoF '{dof_name}'.")
    return load_h5_targets(fp, group, tkey)

def clip_cases_to_common_length(*lists_of_cases: List[List[np.ndarray]]) -> int:
    """
    Clip all case arrays to the minimum length across all provided case-lists.
    Returns the common length used.
    """
    min_len = math.inf
    for cases in lists_of_cases:
        for arr in cases:
            min_len = min(min_len, len(arr))
    min_len = int(min_len)
    for cases in lists_of_cases:
        for i in range(len(cases)):
            cases[i] = cases[i][:min_len]
    return min_len

def decimate_cases(cases: List[np.ndarray], stride: int) -> List[np.ndarray]:
    if stride <= 1:
        return cases
    return [c[::stride] for c in cases]

def stack_cases_2d(cases: List[np.ndarray]) -> np.ndarray:
    """
    Stack list of (T,) arrays or (T,D) arrays into (C, T, I) float32.
    If input is (T,), I=1; if (T,D), I=D.
    """
    arrs = []
    for c in cases:
        c = np.asarray(c)
        if c.ndim == 1:
            c = c[:, None]
        arrs.append(c.astype(np.float32))
    return np.stack(arrs, axis=0).astype(np.float32)

# ---------------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------------
def load_norm_json(fp: Path) -> Dict:
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)

def _ordered_minmax(a: float, b: float) -> tuple[float, float]:
    return (a, b) if b >= a else (b, a)

def pick_output_stat(stats: dict, dof_name: str) -> dict:
    """
    Return {'min':..., 'max':..., 'name':...} for the requested DoF.
    Looks inside stats['outputs'] and matches by item['name'] case-insensitively.
    Also accepts FAST-like aliases for fairleads.
    """
    outs = stats.get("outputs")
    if not outs:
        raise ValueError("Normalization JSON must contain an 'outputs' key.")

    dn = dof_name.strip().lower()
    aliases = {
        "surge": "surge", "sway": "sway", "heave": "heave",
        "roll": "roll", "pitch": "pitch", "yaw": "yaw",
        "fairlead1": "fairlead1", "fairlead2": "fairlead2", "fairlead3": "fairlead3",
        # Accept FAST-like names if ever used
        "tensionsfairlead1": "fairlead1",
        "tensionsfairlead2": "fairlead2",
        "tensionsfairlead3": "fairlead3",
    }
    key = aliases.get(dn, dn)

    for s in outs:
        nm = str(s.get("name", "")).strip().lower()
        if nm == key:
            mn, mx = _ordered_minmax(float(s["min"]), float(s["max"]))
            return {"min": mn, "max": mx, "name": s.get("name", key)}

    raise ValueError(f"No output normalization entry named '{dof_name}' found in JSON.")

def apply_minmax(x: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    eps = 1e-12
    return (x - x_min) / max(x_max - x_min, eps)

def invert_minmax(xn: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    return xn * (x_max - x_min) + x_min

# ---------------------------------------------------------------------------------
# Sequence dataset
# ---------------------------------------------------------------------------------
class SeqDataset(Dataset):
    """
    Build sliding windows for sequence learning.

    If seq2seq=True, target is a sequence (predict each next step in window).
    If seq2seq=False, target is only the *next* step after the window (seq-to-one).
    """
    def __init__(self,
                 X: np.ndarray,  # (C, T, I)
                 Y: Optional[np.ndarray],  # (C, T, O) or None for inference-only
                 seq_len: int = 50,
                 seq2seq: bool = True):
        self.seq2seq = seq2seq
        self.inputs: List[np.ndarray] = []
        self.targets: Optional[List[np.ndarray]] = [] if Y is not None else None

        C, T, I = X.shape
        O = 0 if Y is None else Y.shape[-1]
        horizon_shift = 1  # predict one-step ahead

        for c in range(C):
            for t0 in range(0, T - seq_len - horizon_shift + 1):
                t1 = t0 + seq_len
                self.inputs.append(X[c, t0:t1, :])  # (seq_len, I)
                if Y is not None:
                    if seq2seq:
                        # predict the next-step for every position in the window
                        self.targets.append(Y[c, t0 + 1:t1 + 1, :])  # (seq_len, O)
                    else:
                        # predict only the next step after the window
                        self.targets.append(Y[c, t1:t1 + 1, :])  # (1, O)

        self.inputs = [a.astype(np.float32) for a in self.inputs]
        if self.targets is not None:
            self.targets = [a.astype(np.float32) for a in self.targets]

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.inputs[idx])  # (S, I)
        if self.targets is None:
            return x
        y = torch.from_numpy(self.targets[idx])  # (S, O) or (1, O)
        return x, y

# ---------------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------------
def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / max(ss_tot, 1e-12)

def evaluate_seq(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"MSE": mse, "MAE": mae, "R2": r2}

# ---------------------------------------------------------------------------------
# Inference runner
# ---------------------------------------------------------------------------------
def run_inference(
    inputs_h5: Path,
    outputs_h5: Optional[Path],
    dof_name: str,
    normalization_json: Path,
    checkpoint: Path,
    out_h5: Path,
    out_csv_dir: Optional[Path],
    sequence_length: int,
    batch_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    use_cuda: bool,
    seq2seq: bool,
    stride: int,
    multi_dir: bool,
    dir_num: int,
    nn_pyc: Path,
) -> None:

    device = torch.device("cuda:0" if (use_cuda and torch.cuda.is_available()) else "cpu")
    print(f"[+] Device: {device}")

    # --- Load inputs ---
    if multi_dir:
        X_cases, _ = load_multidir_waves(inputs_h5, dir_num=dir_num)  # list of (T, D)
        # times per case (re-use TIMES group)
        _, t_cases_in = load_h5_cases(inputs_h5, "time_series_excitations/TIMES",
                                      "time_series_excitations/TIMES")
    else:
        X_cases, t_cases_in = load_h5_cases(inputs_h5,
                                            "time_series_excitations/WAVES",
                                            "time_series_excitations/TIMES")
    print(f"[+] Loaded {len(X_cases)} wave cases from {inputs_h5}")

    # --- Load targets (optional) ---
    Y_cases = None
    if outputs_h5 is not None:
        Y_cases, _ = load_targets_by_dof(outputs_h5, dof_name)
        print(f"[+] Loaded targets {dof_name} for {len(Y_cases)} cases from {outputs_h5}")
        if len(Y_cases) != len(X_cases):
            raise RuntimeError("Inputs/targets case count mismatch!")

    # --- Ensure equal length across all cases ---
    if Y_cases is None:
        L = clip_cases_to_common_length(X_cases, t_cases_in)
    else:
        L = clip_cases_to_common_length(X_cases, t_cases_in, Y_cases)
    print(f"[+] Using common length T={L} for all cases")

    # --- Optional decimation ---
    if stride > 1:
        X_cases = decimate_cases(X_cases, stride)
        t_cases_in = decimate_cases(t_cases_in, stride)
        if Y_cases is not None:
            Y_cases = decimate_cases(Y_cases, stride)
        print(f"[+] Decimated signals by stride={stride}; new T={len(X_cases[0])}")

    # --- Stack (C, T, I) / (C, T, O) ---
    X = stack_cases_2d(X_cases)  # I = 1 (mono) or I = D (multi-dir)
    O_dim = 0
    if Y_cases is not None:
        Y = stack_cases_2d(Y_cases)  # O = 1 typically
        O_dim = Y.shape[-1]
    else:
        Y = None

    # --- Normalization (min–max) ---
    stats = load_norm_json(normalization_json)
    X = np.asarray(X, dtype=np.float32)

    # Require inputs and outputs present; we need outputs even in inference-only to denorm predictions
    assert "inputs" in stats and "outputs" in stats, \
        "Normalization JSON must contain both 'inputs' and 'outputs' keys."

    # inputs
    if len(stats["inputs"]) != X.shape[-1]:
        raise ValueError(f"Normalization 'inputs' length ({len(stats['inputs'])}) "
                         f"does not match feature count ({X.shape[-1]}).")
    for i, s in enumerate(stats["inputs"]):
        X[..., i] = apply_minmax(X[..., i], s["min"], s["max"])

    # outputs (only the selected DoF)
    out_stat = pick_output_stat(stats, dof_name)
    if Y is not None:
        if Y.shape[-1] != 1:
            raise ValueError(f"Expect single-target Y for DoF '{dof_name}', got O={Y.shape[-1]}.")
        Y[..., 0] = apply_minmax(Y[..., 0], out_stat["min"], out_stat["max"])

    # --- Build sequences + DataLoader ---
    dataset = SeqDataset(
        X=X,  # (C, T, I)
        Y=Y,  # (C, T, O) or None
        seq_len=sequence_length,  # --seq-len
        seq2seq=seq2seq  # --seq2seq (default: False -> seq-to-one)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # --- Import compiled model and build early-fused LSTM ---
    NN_MODELS = _import_compiled_model(nn_pyc)
    ctor = None
    if hasattr(NN_MODELS, "early_fused_LSTM"):
        ctor = NN_MODELS.early_fused_LSTM
    elif hasattr(NN_MODELS, "LSTM"):
        ctor = NN_MODELS.LSTM
    else:
        raise AttributeError("Compiled module does not expose 'early_fused_LSTM' or 'LSTM'.")
    print(f"[+] Using compiled LSTM from: {nn_pyc}")

    output_size = O_dim if O_dim > 0 else 1  # infer if no targets
    if multi_dir:
        expected_dirs = dir_num  # e.g., 5
        if X.shape[-1] != expected_dirs:
            print(f"[WARN] Got {X.shape[-1]} input channels; trimming to {expected_dirs} to match checkpoint.")
            X = X[..., :expected_dirs]
        input_size = 1  # <-- 1 feature per direction
        num_dirs = expected_dirs  # <-- 5 directions
    else:
        input_size = X.shape[-1]  # mono: this will be 1
        num_dirs = 1

    model = ctor(
        model="LSTM",
        input_size=input_size,
        output_size=output_size,
        hidden_size=hidden_size,
        n_layers=num_layers,
        act_function=nn.Tanh,
        device=device,
        initialize=False,
        pretrained_model_path=None,
        first_linear_layer=True,
        additional_linear_layer=False,
        dropout_layer=(dropout > 0),
        dropout_rate=dropout,
        num_directions=num_dirs,  # <-- use num_dirs defined above
        directionality_linear=False,
        sequential=False,
    )
    model.to(device)

    # --- Load weights ---
    print(f"[+] Loading checkpoint: {checkpoint}")
    sd = torch.load(checkpoint, map_location=device)
    state_dict = sd.get("model_state_dict", sd)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # --- Inference ---
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            if Y is None:
                xb = batch.to(device)  # (B,S,I)
            else:
                xb, _ = batch
                xb = xb.to(device)  # (B,S,I)

            # The compiled PINN expects [B, S, D, I]. We currently have [B, S, I].
            if num_dirs > 1:  # multi-direction inputs
                xb = xb.unsqueeze(-1)  # (B,S,D,1)
            else:  # mono: features-only
                xb = xb.unsqueeze(-2)  # (B,S,1,F)

            yb = model(xb)  # (B,S,O) or (B,1,O)
            preds.append(yb.detach().cpu().numpy())

    Yhat = np.concatenate(preds, axis=0)   # (#windows, S or 1, O)

    # --- Stitch back to time series per case ---
    C = len(X_cases)
    T = X.shape[1]
    O = output_size

    sums = np.zeros((C, T, O), dtype=np.float32)
    cnts = np.zeros((C, T, 1), dtype=np.float32)

    ptr = 0
    windows_per_case = T - sequence_length  # with one-step-ahead horizon
    for c in range(C):
        for w in range(windows_per_case):
            ph = Yhat[ptr]  # (S,O) or (1,O)
            if seq2seq:
                t_slice = slice(w + 1, w + sequence_length + 1)  # next-steps within window
                sums[c, t_slice, :] += ph
                cnts[c, t_slice, :] += 1.0
            else:
                t_idx = w + sequence_length  # only next step
                sums[c, t_idx, :] += ph[0]
                cnts[c, t_idx, :] += 1.0
            ptr += 1

    cnts[cnts == 0] = 1.0
    Ypred_norm = sums / cnts  # (C,T,O)

    # --- Denormalize outputs ---
    Ypred_denorm = Ypred_norm.copy()
    Ypred_denorm[..., 0] = invert_minmax(
        Ypred_denorm[..., 0], out_stat["min"], out_stat["max"])

    # --- Evaluate (if targets exist) ---
    if Y is not None:
        Y_denorm = Y.copy()
        Y_denorm[..., 0] = invert_minmax(Y_denorm[..., 0],
                                         out_stat["min"], out_stat["max"])
        cov_mask = (cnts[..., 0] > 0)
        y_true = Y_denorm[cov_mask]
        y_pred = Ypred_denorm[cov_mask]
        metrics = evaluate_seq(y_true, y_pred)
        print(f"[+] Metrics on covered region: {metrics}")

    # --- Save predictions ---
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_h5, "w") as f:
        # stamp version + metadata
        f.attrs["tool"] = "main_fowt_lstm"
        f.attrs["checkpoint"] = str(checkpoint)
        f.attrs["multi_dir"] = bool(multi_dir)
        f.attrs["dir_num"] = int(dir_num)

        grp = f.create_group(f"predictions/{dof_name}")
        for c, y in enumerate(Ypred_norm):
            grp.create_dataset(f"case_{c:04d}", data=y.squeeze(-1))
        tg = f.create_group("times")
        for c, t in enumerate(t_cases_in):
            tg.create_dataset(f"case_{c:04d}", data=t)
    print(f"[+] Saved HDF5 predictions to: {out_h5}")

    # Optional CSVs (one per case)
    if out_csv_dir is not None:
        out_csv_dir.mkdir(parents=True, exist_ok=True)
        for c, (t, y) in enumerate(zip(t_cases_in, Ypred_norm)):
            arr = np.column_stack([t, y.squeeze(-1)])
            header = f"time,prediction"
            np.savetxt(out_csv_dir / f"{dof_name}_pred_case_{c:04d}.csv",
                       arr, delimiter=",", header=header, comments="")
        print(f"[+] Also wrote CSVs to: {out_csv_dir}")

# ---------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Predict FOWT responses with a trained LSTM (compiled-model only).")
    p.add_argument("--inputs-h5", type=Path, required=True, help="Path to inputs HDF5 (waves & times).")
    p.add_argument("--outputs-h5", type=Path, default=None, help="Optional targets HDF5 (for evaluation).")
    p.add_argument("--dof", type=str, default="Heave", help="Target DoF (e.g., HEAVE, PITCH, fairlead3).")
    p.add_argument("--multi-dir", action="store_true", help="Use DIRECTIONALWAVES/WaveDir1..N inputs.")
    p.add_argument("--dir-num", type=int, default=3, help="Number of directions if --multi-dir is set.")
    p.add_argument("--norm-json", type=Path, required=True, help="Normalization JSON with min/max for inputs/outputs.")
    p.add_argument("--checkpoint", type=Path, required=True, help="Path to trained model weights (.pt or state_dict).")
    p.add_argument("--out-h5", type=Path, default=Path("predictions.h5"), help="Where to store HDF5 predictions.")
    p.add_argument("--out-csv-dir", type=Path, default=None, help="Optional folder to dump per-case CSV predictions.")
    p.add_argument("--seq-len", type=int, default=80, help="Sequence length for sliding windows.")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size for inference.")
    p.add_argument("--hidden-size", type=int, default=128, help="LSTM hidden size.")
    p.add_argument("--num-layers", type=int, default=5, help="LSTM layers.")
    p.add_argument("--dropout", type=float, default=0.3, help="LSTM dropout.")
    p.add_argument("--cuda", action="store_true", help="Use CUDA if available.")
    p.add_argument("--seq2seq", action="store_true",
                   help="If set, predict sequence-per-window (default: seq-to-one).")
    p.add_argument("--stride", type=int, default=1, help="Optional decimation stride for inputs/targets.")
    p.add_argument("--nn-pyc", type=Path, default=DEFAULT_PYC,
                   help=f"Path to compiled model .pyc (default: {DEFAULT_PYC}).")
    return p

def main():
    args = build_argparser().parse_args()

    run_inference(
        inputs_h5=args.inputs_h5,
        outputs_h5=args.outputs_h5,
        dof_name=args.dof,
        normalization_json=args.norm_json,
        checkpoint=args.checkpoint,
        out_h5=args.out_h5,
        out_csv_dir=args.out_csv_dir,
        sequence_length=args.seq_len,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_cuda=args.cuda,
        seq2seq=args.seq2seq,
        stride=args.stride,
        multi_dir=args.multi_dir,
        dir_num=args.dir_num,
        nn_pyc=args.nn_pyc,
    )

if __name__ == "__main__":
    main()
