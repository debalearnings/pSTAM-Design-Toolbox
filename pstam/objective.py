"""
pstam/objective.py
==================
Cost function for pSTAM PSO-based design optimisation.

Design variables searched by PSO
---------------------------------
  x[0]  L   — muscle length          mm
  x[1]  D   — outer diameter         mm
  x[2]  T   — wall thickness         mm
  x[3]  P   — actuation pressure     kPa

Objective
---------
A weighted sum of one or more "terms", each of which evaluates a single
model (M1, M2, or M3) at fixed non-design inputs, optionally sweeping one
extra input over a range and aggregating the results.

    objective = Σ  weight_i × aggregate_i(model_i predictions)

The user sets mode = "maximize" or "minimize" globally.

Constraint blocks
-----------------
Up to 4 constraint blocks, each following the same term schema as the
objective but comparing the aggregated output to a threshold:

    penalty = weight × max(0, violation)

where violation = output − threshold  (if direction "≤")
             or = threshold − output  (if direction "≥")

Term / block schema (from config.py)
--------------------------------------
{
    "model":       "M1" | "M2" | "M3",
    "fixed": {                           # non-design inputs, fixed values
        "F":       float,                # axial load N         (M1, M2, M3)
        "stroke":  float,                # % of total stroke    (M2, M3 only)
    },
    "sweep": {                           # optional – sweep one input
        "variable": "F" | "stroke",
        "from":  float,
        "to":    float,                  # inclusive upper bound
        "step":  float,
    } | None,
    "aggregate": "single" | "mean" | "min" | "max",
    "weight":    float,                  # for objective terms
    # constraint-only keys:
    "enabled":        bool,
    "threshold":      float,
    "direction":      "≤" | "≥",
    "penalty_weight": float,
}
"""

from __future__ import annotations
import numpy as np
from pstam.ann import PAMModels


class CostFunction:
    """
    Evaluates the PSO cost for a candidate design vector.

    Parameters
    ----------
    models : PAMModels
    cfg    : full config dict (see config.py)
    """

    def __init__(self, models: PAMModels, cfg: dict):
        self._m   = models
        self._cfg = cfg

    # ─── single model inference ───────────────────────────────────────

    def _infer(self, model: str,
               L: float, D: float, T: float, P: float,
               F: float, stroke: float) -> float:
        """Call the correct model method with the right arguments."""
        if model == "M1":
            return self._m.predict_M1(L, D, T, F, P)
        elif model == "M2":
            return self._m.predict_M2(L, D, T, P, F, stroke)
        elif model == "M3":
            return self._m.predict_M3(L, D, T, P, F, stroke)
        else:
            raise ValueError(f"Unknown model: {model}")

    # ─── evaluate a single term / constraint block ────────────────────

    def _evaluate_block(self, block: dict,
                        L: float, D: float,
                        T: float, P: float) -> float:
        """
        Evaluate one objective term or constraint block.
        Returns the aggregated scalar prediction.
        """
        model     = block["model"]
        fixed     = block.get("fixed", {})
        sweep_cfg = block.get("sweep", None)
        agg       = block.get("aggregate", "single")

        # Base fixed inputs (with safe defaults)
        F_base      = float(fixed.get("F",      75.0))
        stroke_base = float(fixed.get("stroke", 55.0))

        # Build list of (F, stroke) combinations to evaluate
        if sweep_cfg is None or agg == "single":
            combos = [(F_base, stroke_base)]
        else:
            var    = sweep_cfg["variable"]          # "F" or "stroke"
            vals   = np.arange(
                float(sweep_cfg["from"]),
                float(sweep_cfg["to"]) + 1e-9,
                float(sweep_cfg["step"])
            )
            if var == "F":
                combos = [(v, stroke_base) for v in vals]
            elif var == "stroke":
                combos = [(F_base, v) for v in vals]
            else:
                combos = [(F_base, stroke_base)]

        predictions = [self._infer(model, L, D, T, P, F_val, s_val)
                       for F_val, s_val in combos]

        if   agg == "mean":   return float(np.mean(predictions))
        elif agg == "min":    return float(np.min(predictions))
        elif agg == "max":    return float(np.max(predictions))
        else:                 return float(predictions[0])   # "single"

    # ─── penalty ──────────────────────────────────────────────────────

    @staticmethod
    def _penalty(value: float, threshold: float,
                 direction: str, weight: float) -> float:
        if direction == "≤":
            violation = value - threshold
        else:
            violation = threshold - value
        return weight * max(0.0, violation)

    # ─── single-particle evaluation ───────────────────────────────────

    def evaluate(self, x: np.ndarray) -> float:
        """
        Compute cost for one candidate design vector x = [L, D, T, P].
        PSO minimises this value.
        """
        L, D, T, P = float(x[0]), float(x[1]), float(x[2]), float(x[3])
        cfg  = self._cfg
        mode = cfg["objective"]["mode"]   # "maximize" | "minimize"

        # ── objective: weighted sum of terms ──────────────────────────
        obj_value = 0.0
        for term in cfg["objective"]["terms"]:
            pred       = self._evaluate_block(term, L, D, T, P)
            obj_value += float(term.get("weight", 1.0)) * pred

        cost = -obj_value if mode == "maximize" else obj_value

        # ── constraints ───────────────────────────────────────────────
        for block in cfg["constraints"]:
            if not block.get("enabled", True):
                continue
            pred  = self._evaluate_block(block, L, D, T, P)
            cost += self._penalty(pred,
                                  block["threshold"],
                                  block["direction"],
                                  block["penalty_weight"])

        return cost

    # ─── batch evaluation (PySwarms interface) ────────────────────────

    def batch_evaluate(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.evaluate(X[i]) for i in range(X.shape[0])])

    # ─── detailed result at best-found design ─────────────────────────

    def detailed_report(self, x: np.ndarray) -> dict:
        """
        Return a dict of all term and constraint predictions
        at the best-found design vector x = [L, D, T, P].
        """
        L, D, T, P = float(x[0]), float(x[1]), float(x[2]), float(x[3])
        cfg = self._cfg

        obj_terms = []
        for i, term in enumerate(cfg["objective"]["terms"]):
            pred = self._evaluate_block(term, L, D, T, P)
            obj_terms.append({
                "term_index":   i + 1,
                "model":        term["model"],
                "aggregate":    term.get("aggregate", "single"),
                "weight":       term.get("weight", 1.0),
                "prediction":   pred,
                "weighted":     pred * float(term.get("weight", 1.0)),
            })

        constraints = []
        for i, block in enumerate(cfg["constraints"]):
            if not block.get("enabled", True):
                continue
            pred     = self._evaluate_block(block, L, D, T, P)
            violated = (pred > block["threshold"] if block["direction"] == "≤"
                        else pred < block["threshold"])
            constraints.append({
                "constraint_index": i + 1,
                "model":            block["model"],
                "aggregate":        block.get("aggregate", "single"),
                "prediction":       pred,
                "threshold":        block["threshold"],
                "direction":        block["direction"],
                "violated":         violated,
            })

        total_weighted = sum(t["weighted"] for t in obj_terms)

        return {
            "design":           {"L_mm": L, "D_mm": D, "T_mm": T, "P_kPa": P},
            "objective_terms":  obj_terms,
            "total_objective":  total_weighted,
            "constraints":      constraints,
            "mode":             cfg["objective"]["mode"],
        }
