"""
run_optimization.py
===================
CLI entry point for the pSTAM Design Optimiser.
Edit config.py, then:  python run_optimization.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_default_config
from pstam.optimizer import PAMOptimizer

def main():
    print("="*60)
    print("  pSTAM Design Optimiser — PSO + ANN surrogate")
    print("="*60)
    cfg = get_default_config()
    ds, pso = cfg["design_space"], cfg["pso"]
    print(f"\n  Design space:  L [{ds['L_min']}–{ds['L_max']} mm]  "
          f"D [{ds['D_min']}–{ds['D_max']} mm]  "
          f"T [{ds['T_min']}–{ds['T_max']} mm]  "
          f"P [{ds['P_min']}–{ds['P_max']} kPa]")
    print(f"  Objective   :  {cfg['objective']['mode']}  "
          f"({len(cfg['objective']['terms'])} term(s))")
    active_c = sum(1 for c in cfg['constraints'] if c.get('enabled'))
    print(f"  Constraints :  {active_c} active")
    print(f"  PSO         :  {pso['n_particles']} particles × {pso['n_iterations']} iters\n")

    def progress(it, total):
        bar = "█"*int(30*it/total) + "░"*(30-int(30*it/total))
        print(f"\r  [{bar}] {it}/{total}", end="", flush=True)

    base = os.path.dirname(os.path.abspath(__file__))
    opt  = PAMOptimizer(cfg=cfg,
                        data_dir=os.path.join(base,"data"),
                        results_dir=os.path.join(base,"results"),
                        progress_callback=progress)
    result = opt.run()
    print()

    pos = result["best_position"]
    rep = result["report"]
    print("\n"+"="*60+"\n  RESULT\n"+"="*60)
    print(f"  L={pos[0]:.2f}mm  D={pos[1]:.2f}mm  T={pos[2]:.2f}mm  P={pos[3]:.2f}kPa")
    print(f"\n  Objective terms:")
    for t in rep["objective_terms"]:
        unit={"M1":"mm","M2":"ms","M3":"ms"}[t["model"]]
        print(f"    Term {t['term_index']} [{t['model']}] {t['aggregate']:>6}: "
              f"{t['prediction']:.3f} {unit}  × {t['weight']} = {t['weighted']:.3f}")
    print(f"  Total: {rep['total_objective']:.4f}")
    print(f"\n  Constraints:")
    for c in rep["constraints"]:
        unit={"M1":"mm","M2":"ms","M3":"ms"}[c["model"]]
        ok="VIOLATED" if c["violated"] else "OK"
        print(f"    C{c['constraint_index']} [{c['model']}] {c['aggregate']:>6}: "
              f"{c['prediction']:.3f} {unit}  ({c['direction']} {c['threshold']})  [{ok}]")
    print(f"\n  PSO cost: {result['best_cost']:.6f}")
    print(f"  Saved  : {result['csv_path']}\n")

if __name__ == "__main__":
    main()
