"""
pstam/optimizer.py
==================
Wraps PySwarms GlobalBestPSO and drives the optimisation.
Results are written to a timestamped CSV inside results/.

Progress tracking is done by counting cost-function batch calls
(one per PSO iteration) — no monkey-patching of PySwarms internals.
"""

import os
import time
import numpy as np
import pandas as pd

from pstam.ann import PAMModels
from pstam.objective import CostFunction


class PAMOptimizer:
    """
    PSO-based design optimisation for pSTAMs.

    Parameters
    ----------
    cfg : dict
        Full configuration dict (see config.py).
    data_dir : str, optional
        Folder containing M1/M2/M3_weights.csv.
    results_dir : str, optional
        Folder where result CSVs are written.
    progress_callback : callable(current_iter, total_iters), optional
        Called once per completed PSO iteration.
    """

    def __init__(self,
                 cfg: dict,
                 data_dir: str | None = None,
                 results_dir: str = "results",
                 progress_callback=None):
        self.cfg               = cfg
        self.results_dir       = results_dir
        self.progress_callback = progress_callback

        models       = PAMModels(data_dir)
        self.cost_fn = CostFunction(models, cfg)
        os.makedirs(results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Cost wrapper — counts batch calls to infer iteration number
    # ------------------------------------------------------------------

    def _make_cost_fn(self, n_iterations: int):
        """
        PySwarms calls the cost function exactly once per iteration
        with a matrix of shape (n_particles, 4).  Counting those calls
        gives us the current iteration without patching any internals.
        """
        call_count = [0]

        def _batch(X: np.ndarray) -> np.ndarray:
            costs = self.cost_fn.batch_evaluate(X)
            call_count[0] += 1
            if self.progress_callback is not None:
                self.progress_callback(
                    min(call_count[0], n_iterations), n_iterations)
            return costs

        return _batch

    # ------------------------------------------------------------------

    def run(self) -> dict:
        """
        Execute PSO optimisation.

        Returns
        -------
        dict with keys:
            best_cost, best_position, cost_history, report, csv_path
        """
        import pyswarms as ps

        pso = self.cfg["pso"]
        ds  = self.cfg["design_space"]

        n_particles  = int(pso["n_particles"])
        n_iterations = int(pso["n_iterations"])

        options = {
            "c1": float(pso["c1"]),
            "c2": float(pso["c2"]),
            "w":  float(pso["w"]),
        }
        bounds = (
            np.array([ds["L_min"], ds["D_min"],
                      ds["T_min"], ds["P_min"]], dtype=float),
            np.array([ds["L_max"], ds["D_max"],
                      ds["T_max"], ds["P_max"]], dtype=float),
        )

        optimizer = ps.single.GlobalBestPSO(
            n_particles = n_particles,
            dimensions  = 4,
            options     = options,
            bounds      = bounds,
        )

        best_cost, best_pos = optimizer.optimize(
            self._make_cost_fn(n_iterations),
            iters   = n_iterations,
            verbose = False,
        )

        report   = self.cost_fn.detailed_report(best_pos)
        csv_path = self._save(best_pos, optimizer.cost_history, report)

        return {
            "best_cost":     best_cost,
            "best_position": best_pos,
            "cost_history":  optimizer.cost_history,
            "report":        report,
            "csv_path":      csv_path,
        }

    # ------------------------------------------------------------------

    def _save(self, best_pos, cost_history, report) -> str:
        ts   = round(time.time())
        path = os.path.join(self.results_dir, f"pstam_result_{ts}.csv")

        rows_hist = pd.DataFrame(cost_history,
                                  columns=["best_cost_per_iter"])
        rows_pos  = pd.DataFrame(
            [[best_pos[0], best_pos[1], best_pos[2], best_pos[3]]],
            columns=["L_mm", "D_mm", "T_mm", "P_kPa"],
        )

        term_data = {}
        for t in report["objective_terms"]:
            i = t["term_index"]
            term_data[f"term{i}_model"]      = t["model"]
            term_data[f"term{i}_prediction"] = t["prediction"]
            term_data[f"term{i}_weight"]     = t["weight"]
        term_data["total_objective"] = report["total_objective"]

        c_data = {}
        for c in report["constraints"]:
            i = c["constraint_index"]
            c_data[f"c{i}_model"]     = c["model"]
            c_data[f"c{i}_pred"]      = c["prediction"]
            c_data[f"c{i}_threshold"] = c["threshold"]
            c_data[f"c{i}_violated"]  = c["violated"]

        combined = pd.concat([
            rows_hist,
            rows_pos,
            pd.DataFrame([term_data]),
            pd.DataFrame([c_data]),
        ], axis=1)
        combined.to_csv(path, index=False)
        return path
