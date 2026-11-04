"""
pstam/ann.py
============
Forward-pass inference for the three pSTAM ANN surrogate models.

Architecture: single hidden layer, fully connected.
    hidden = act1( X @ W1 + b1 )
    output = act2( hidden @ W2 + b2 )

Normalisation uses min-max scaling (input → [0,1], output → [0,1])
with bounds read from data/normalisation.xlsx.

─────────────────────────────────────────────────────────────────────
Model   Description                     Inputs (order)
─────────────────────────────────────────────────────────────────────
M1      Static Contraction              L, D, T, F, P
M2      Inflation Time  (dynamic)       L, D, T, P, F, stroke%
M3      Deflation Time  (dynamic)       L, D, T, P, F, stroke%
─────────────────────────────────────────────────────────────────────

Variable key
  L        Muscle length            mm       [150, 350]
  D        Outer diameter           mm       [14,   26]
  T        Wall thickness           mm       [3,     6]
  F        Axial load               N        [10,  150]
  P        Actuation pressure       kPa      [20,  200]
  stroke%  % of total stroke        %        [20,   90]

Output ranges (from normalisation file)
  M1  Contraction      mm   [0.119,      89.641     ]
  M2  Inflation time   ms   [25.16129,   1652.473118]
  M3  Deflation time   ms   [8.602151,   2852.903226]
"""

import os
import numpy as np
import pandas as pd


# ─── file locations ────────────────────────────────────────────────
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

_WEIGHT_FILES = {
    "M1": os.path.join(_DATA_DIR, "M1_weights.csv"),
    "M2": os.path.join(_DATA_DIR, "M2_weights.csv"),
    "M3": os.path.join(_DATA_DIR, "M3_weights.csv"),
}

# ─── hard-coded normalisation bounds (from normalisation.xlsx) ─────
# Input bounds  →  (min, max)
INPUT_BOUNDS = {
    "L":       (150.0, 350.0),
    "D":       ( 14.0,  26.0),
    "T":       (  3.0,   6.0),
    "F":       ( 10.0, 150.0),
    "P":       ( 20.0, 200.0),
    "stroke":  ( 20.0,  90.0),
}

# Output bounds  →  (min, max)
OUTPUT_BOUNDS = {
    "M1": (  0.119,     89.641    ),   # contraction  mm
    "M2": ( 25.16129, 1652.473118 ),   # inflation    ms
    "M3": (  8.602151, 2852.903226),   # deflation    ms
}

# ─── network metadata ──────────────────────────────────────────────
_MODEL_META = {
    #          n_inputs  n_hidden  act1       act2
    "M1": dict(n_in=5,   n_hid=4,  act1="sigmoid", act2="sigmoid"),
    "M2": dict(n_in=6,   n_hid=10, act1="sigmoid", act2="sigmoid"),
    "M3": dict(n_in=6,   n_hid=6,  act1="relu",    act2="sigmoid"),
}


# ─── helpers ───────────────────────────────────────────────────────

def _norm(value: float, key: str) -> float:
    lo, hi = INPUT_BOUNDS[key]
    return (value - lo) / (hi - lo)


def _denorm(value: float, model: str) -> float:
    lo, hi = OUTPUT_BOUNDS[model]
    return value * (hi - lo) + lo


def _activation(z: np.ndarray, name: str) -> np.ndarray:
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    elif name == "relu":
        return np.maximum(0.0, z)
    elif name == "tanh":
        return np.tanh(z)
    return z  # linear / identity


def _forward(x: np.ndarray,
             params: np.ndarray,
             n_in: int,
             n_hid: int,
             act1: str,
             act2: str) -> float:
    """Single forward pass. x must already be normalised."""
    x = x.reshape(1, -1).astype(float)
    ptr = 0
    W1 = params[ptr: ptr + n_in * n_hid].reshape(n_in, n_hid); ptr += n_in * n_hid
    b1 = params[ptr: ptr + n_hid].reshape(1, n_hid);            ptr += n_hid
    W2 = params[ptr: ptr + n_hid].reshape(n_hid, 1);            ptr += n_hid
    b2 = params[ptr: ptr + 1].reshape(1, 1);                    ptr += 1
    h  = _activation(x @ W1 + b1, act1)
    o  = _activation(h @ W2 + b2, act2)
    return float(o[0, 0])


def _load_weights(model: str) -> np.ndarray:
    """Extract flattened weight vector from a model CSV file."""
    df = pd.read_csv(_WEIGHT_FILES[model])
    w  = pd.to_numeric(df["model weights"], errors="coerce").dropna().values
    return w.astype(float)


# ─── main model class ──────────────────────────────────────────────

class PAMModels:
    """
    Loads all three pSTAM ANN models and exposes predict_M1/M2/M3.

    All weight CSVs are read once at construction time.

    Parameters
    ----------
    data_dir : str, optional
        Override path to the folder containing M1/M2/M3_weights.csv.
        Defaults to the project's data/ directory.
    """

    def __init__(self, data_dir: str | None = None):
        if data_dir is not None:
            # Patch global paths if caller supplies custom directory
            global _WEIGHT_FILES
            _WEIGHT_FILES = {k: os.path.join(data_dir, f"{k}_weights.csv")
                             for k in ("M1", "M2", "M3")}
        self._weights = {m: _load_weights(m) for m in ("M1", "M2", "M3")}

    # ── M1 ─────────────────────────────────────────────────────────

    def predict_M1(self,
                   L: float, D: float, T: float,
                   F: float, P: float) -> float:
        """
        M1 — Static Contraction.

        Parameters
        ----------
        L : muscle length          (mm)  [150 – 350]
        D : outer diameter         (mm)  [14  –  26]
        T : wall thickness         (mm)  [3   –   6]
        F : axial load             (N)   [10  – 150]
        P : actuation pressure     (kPa) [20  – 200]

        Returns
        -------
        Static contraction  (mm)
        """
        x   = np.array([_norm(L,"L"), _norm(D,"D"), _norm(T,"T"),
                         _norm(F,"F"), _norm(P,"P")])
        meta = _MODEL_META["M1"]
        raw  = _forward(x, self._weights["M1"], **meta)
        return _denorm(raw, "M1")

    # ── M2 ─────────────────────────────────────────────────────────

    def predict_M2(self,
                   L: float, D: float, T: float,
                   P: float, F: float,
                   stroke_pct: float) -> float:
        """
        M2 — Inflation Time for dynamic contraction under axial load.

        Parameters
        ----------
        L          : muscle length          (mm)  [150 – 350]
        D          : outer diameter         (mm)  [14  –  26]
        T          : wall thickness         (mm)  [3   –   6]
        P          : actuation pressure     (kPa) [20  – 200]
        F          : axial load             (N)   [10  – 150]
        stroke_pct : % of total stroke      (%)   [20  –  90]

        Returns
        -------
        Inflation time  (ms)
        """
        x    = np.array([_norm(L,"L"), _norm(D,"D"), _norm(T,"T"),
                          _norm(P,"P"), _norm(F,"F"), _norm(stroke_pct,"stroke")])
        meta = _MODEL_META["M2"]
        raw  = _forward(x, self._weights["M2"], **meta)
        return _denorm(raw, "M2")

    # ── M3 ─────────────────────────────────────────────────────────

    def predict_M3(self,
                   L: float, D: float, T: float,
                   P: float, F: float,
                   stroke_pct: float) -> float:
        """
        M3 — Deflation Time for dynamic contraction under axial load.

        Parameters
        ----------
        L          : muscle length          (mm)  [150 – 350]
        D          : outer diameter         (mm)  [14  –  26]
        T          : wall thickness         (mm)  [3   –   6]
        P          : actuation pressure     (kPa) [20  – 200]
        F          : axial load             (N)   [10  – 150]
        stroke_pct : % of total stroke      (%)   [20  –  90]

        Returns
        -------
        Deflation time  (ms)
        """
        x    = np.array([_norm(L,"L"), _norm(D,"D"), _norm(T,"T"),
                          _norm(P,"P"), _norm(F,"F"), _norm(stroke_pct,"stroke")])
        meta = _MODEL_META["M3"]
        raw  = _forward(x, self._weights["M3"], **meta)
        return _denorm(raw, "M3")

    # ── convenience: predict all three at once ──────────────────────

    def predict_all(self,
                    L: float, D: float, T: float,
                    P: float, F: float,
                    stroke_pct: float) -> dict:
        """
        Run all three models for the same (L, D, T, P, F, stroke_pct)
        combination and return a labelled dict.

        Note: M1 uses F and P;  M2/M3 additionally use stroke_pct.
        """
        return {
            "M1_contraction_mm":   self.predict_M1(L, D, T, F, P),
            "M2_inflation_time_ms": self.predict_M2(L, D, T, P, F, stroke_pct),
            "M3_deflation_time_ms": self.predict_M3(L, D, T, P, F, stroke_pct),
        }
