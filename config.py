"""
config.py
=========
Central configuration for pSTAM design optimisation.

All user-tunable parameters live here.  The GUI reads and writes
this dictionary; you can also edit it directly and run the CLI.

─────────────────────────────────────────────────────────────────
Models available
─────────────────────────────────────────────────────────────────
  M1  Static Contraction          output: mm
  M2  Inflation Time (dynamic)    output: ms
  M3  Deflation Time (dynamic)    output: ms

Non-design inputs (set per-term / per-constraint)
  F        Axial load           N    [10  – 150]
  stroke   % of total stroke    %    [20  –  90]

Design variables (searched by PSO)
  L   Muscle length      mm   [150 – 350]
  D   Outer diameter     mm   [14  –  26]
  T   Wall thickness     mm   [3   –   6]
  P   Actuation pressure kPa  [20  – 200]

─────────────────────────────────────────────────────────────────
Objective term schema
─────────────────────────────────────────────────────────────────
{
    "model":     "M1" | "M2" | "M3",
    "fixed": {
        "F":      <float>,       # axial load (N) – used by all 3 models
        "stroke": <float>,       # % stroke       – used by M2 and M3 only
    },
    "sweep": {                   # optional – omit or set to None for single
        "variable": "F" | "stroke",
        "from":  <float>,
        "to":    <float>,        # inclusive
        "step":  <float>,
    },
    "aggregate": "single" | "mean" | "min" | "max",
    "weight":    <float>,        # contribution weight in the combined objective
}

─────────────────────────────────────────────────────────────────
Constraint block schema  (adds "enabled", "threshold",
                          "direction", "penalty_weight")
─────────────────────────────────────────────────────────────────
{
    "enabled":        True | False,
    "model":          "M1" | "M2" | "M3",
    "fixed":          { "F": ..., "stroke": ... },
    "sweep":          { ... } | None,
    "aggregate":      "single" | "mean" | "min" | "max",
    "threshold":      <float>,
    "direction":      "≤" | "≥",
    "penalty_weight": <float>,
}
"""


def get_default_config() -> dict:
    """Return a fresh copy of the default configuration."""
    return {

        # ── Design space ────────────────────────────────────────────
        "design_space": {
            "L_min": 150.0,   # mm
            "L_max": 350.0,
            "D_min":  14.0,   # mm
            "D_max":  26.0,
            "T_min":   3.0,   # mm
            "T_max":   6.0,
            "P_min":  20.0,   # kPa
            "P_max": 200.0,
        },

        # ── Objective ───────────────────────────────────────────────
        # mode: "maximize" or "minimize"
        # terms: list of term dicts (at least one).
        #   The final objective = Σ weight_i × term_i
        "objective": {
            "mode": "maximize",
            "terms": [
                {
                    "model":     "M1",
                    "fixed":     {"F": 75.0, "stroke": 55.0},
                    "sweep":     None,            # single inference
                    "aggregate": "single",
                    "weight":    1.0,
                },
                # --- Uncomment to add a second term ---
                # {
                #     "model":     "M2",
                #     "fixed":     {"F": 75.0, "stroke": 55.0},
                #     "sweep":     {"variable": "stroke",
                #                   "from": 20, "to": 90, "step": 10},
                #     "aggregate": "mean",
                #     "weight":    0.0,   # set > 0 to activate
                # },
            ],
        },

        # ── Constraints ─────────────────────────────────────────────
        # Each item follows the constraint block schema above.
        # Add or remove items freely; set enabled=False to skip.
        "constraints": [
            {
                "enabled":        True,
                "model":          "M1",
                "fixed":          {"F": 10.0, "stroke": 55.0},
                "sweep":          None,
                "aggregate":      "single",
                "threshold":      85.0,        # contraction must be ≤ 85 mm
                "direction":      "≤",
                "penalty_weight": 100_000,
            },
            {
                "enabled":        True,
                "model":          "M2",
                "fixed":          {"F": 75.0, "stroke": 55.0},
                "sweep":          {"variable": "stroke",
                                   "from": 20, "to": 90, "step": 10},
                "aggregate":      "max",        # worst-case inflation time
                "threshold":      800.0,        # must be ≤ 800 ms
                "direction":      "≤",
                "penalty_weight": 100_000,
            },
            {
                "enabled":        True,
                "model":          "M3",
                "fixed":          {"F": 75.0, "stroke": 55.0},
                "sweep":          {"variable": "stroke",
                                   "from": 20, "to": 90, "step": 10},
                "aggregate":      "max",        # worst-case deflation time
                "threshold":      1200.0,       # must be ≤ 1200 ms
                "direction":      "≤",
                "penalty_weight": 100_000,
            },
            {
                "enabled":        False,        # spare slot
                "model":          "M2",
                "fixed":          {"F": 150.0, "stroke": 90.0},
                "sweep":          None,
                "aggregate":      "single",
                "threshold":      500.0,
                "direction":      "≤",
                "penalty_weight": 100_000,
            },
        ],

        # ── PSO hyperparameters ─────────────────────────────────────
        "pso": {
            "n_particles":  20,
            "n_iterations": 50,
            "c1":           0.5,   # cognitive acceleration
            "c2":           0.3,   # social acceleration
            "w":            0.9,   # inertia weight
        },
    }
