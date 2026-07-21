"""
optimize_ctrl_freak.py

STANDALONE speed-comparison script: runs the exact same reservoir problem
(same simulate(), same objectives.evaluate(), same decision vector, same
bounds) through ctrl-freak's nsga2() instead of pymoo's, to assess relative
engine throughput. Does NOT modify optimize.py, objectives.py, simulator.py,
policy.py, or data_loader.py in any way -- imports them unchanged.

IMPORTANT CAVEAT -- not solving an identical problem, on purpose:
  The dam-safety hard constraint (flood_control_level, pymoo's n_ieq_constr=1
  in optimize.py) is NOT enforced here. ctrl-freak has no native constrained-
  domination mechanism (confirmed by reading its docs) -- its documented
  approach is repair-in-operators or penalty-as-objective, either of which
  would change the problem formulation, not just the engine. For a PURE
  SPEED comparison this constraint is simply dropped rather than faked with
  a hand-rolled workaround. This means:
    - Any front produced here may include designs that violate dam safety.
      DO NOT use this script's output for actual design decisions -- it
      exists only to benchmark raw optimizer throughput.
    - The comparison is still meaningful for that purpose: the dominant
      cost in both engines is calling the numba-accelerated simulate()
      thousands of times, which is identical either way; constrained- vs
      unconstrained-domination bookkeeping is a rounding error next to that.

Usage
-----
    uv run python Module/optimize_ctrl_freak.py --data-dir Data/Mandrare_248 --preset quick
    uv run python Module/optimize_ctrl_freak.py --data-dir Data/Mandrare_248 --preset full --compare
        (--compare also runs pymoo's existing run_optimization() in the same
        process, back to back, for a direct side-by-side timing comparison)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from ctrl_freak import nsga2, polynomial_mutation, sbx_crossover

from data_loader import case_name_from_data_dir, load_reservoir_data, prompt_for_data_folder
from objectives import evaluate as objectives_evaluate
from policy import N_POLICY_PARAMS, compute_forecast_categories

# Same presets as optimize.py, for a direct comparison at matching effort
PRESETS = {
    "quick": {"pop_size": 30, "n_gen": 30},
    "full": {"pop_size": 250, "n_gen": 300},
}


def run_ctrl_freak(data, categories, pop_size: int, n_gen: int, seed: int = 1):
    """
    Runs NSGA-II via ctrl-freak on the exact same 111-variable reservoir
    problem objectives.evaluate() already defines. Returns (X, F, elapsed_seconds).
    """
    xl = np.concatenate([
        [data.scalars["design_discharge_hydro_min"]],
        np.zeros(N_POLICY_PARAMS),
    ])
    xu = np.concatenate([
        [data.scalars["design_discharge_hydro_max"]],
        np.ones(N_POLICY_PARAMS),
    ])
    n_var = xl.shape[0]

    def init(rng):
        return rng.uniform(xl, xu)

    def evaluate(x):
        # objectives.evaluate() already returns F in ctrl-freak's minimize
        # convention (negative energy, negative reliability, positive
        # spillage) -- no sign changes needed. G (dam safety) is deliberately
        # NOT used here -- see module docstring.
        return objectives_evaluate(data, categories, x).F

    # eta=15 (SBX) / eta=20 (polynomial mutation) match pymoo's own defaults,
    # so both engines are using the same genetic operators, not just the
    # same problem -- isolates the comparison to the engine itself.
    crossover = sbx_crossover(eta=15.0, bounds=(xl, xu), seed=seed)
    mutate = polynomial_mutation(eta=20.0, prob=1.0 / n_var, bounds=(xl, xu), seed=seed)

    t0 = time.perf_counter()
    result = nsga2(
        init=init, evaluate=evaluate, crossover=crossover, mutate=mutate,
        pop_size=pop_size, n_generations=n_gen, seed=seed,
    )
    elapsed = time.perf_counter() - t0

    front = result.pareto_front
    return front.x, front.objectives, elapsed, result.evaluations


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Speed-compare ctrl-freak's NSGA-II against pymoo's on the same problem.")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--preset", type=str, default="quick", choices=list(PRESETS))
    parser.add_argument("--pop-size", type=int, default=None)
    parser.add_argument("--n-gen", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--compare", action="store_true",
                         help="Also run pymoo's existing run_optimization() in this same process for a direct timing comparison.")
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    pop_size = args.pop_size if args.pop_size is not None else preset["pop_size"]
    n_gen = args.n_gen if args.n_gen is not None else preset["n_gen"]

    if args.data_dir is None:
        default_data_dir = Path(__file__).resolve().parent.parent / "Data"
        data_dir = prompt_for_data_folder(default_data_dir)
    else:
        data_dir = args.data_dir

    print(f"Loading data from {data_dir} ...")
    data = load_reservoir_data(data_dir)
    categories = compute_forecast_categories(data.months, data.inflow_m3s)
    case_name = case_name_from_data_dir(data_dir)

    print(f"\n{'=' * 60}\nctrl-freak NSGA-II  (pop={pop_size}, n_gen={n_gen})\n{'=' * 60}")
    X_cf, F_cf, elapsed_cf, n_evals_cf = run_ctrl_freak(data, categories, pop_size, n_gen, seed=args.seed)
    print(f"Elapsed: {elapsed_cf:.2f}s  ({n_evals_cf:,} evaluations, "
          f"{elapsed_cf / n_evals_cf * 1000:.3f} ms/eval)")
    print(f"Pareto front size: {len(X_cf)}")
    print(f"Energy range: {-F_cf[:, 0].max():.2f} to {-F_cf[:, 0].min():.2f} GWh/yr")
    print(f"Reliability range: {-F_cf[:, 1].max():.1%} to {-F_cf[:, 1].min():.1%}")
    print(f"Spillage range: {F_cf[:, 2].min():.1f} to {F_cf[:, 2].max():.1f} Mm3/yr")

    output_dir = Path(__file__).resolve().parent.parent / "Output" / f"{case_name}_ctrlfreak"
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_cols = ",".join(f"policy_{i}" for i in range(N_POLICY_PARAMS))
    np.savetxt(output_dir / "pareto_X.csv", X_cf, delimiter=",",
               header=f"design_discharge_hydro,{policy_cols}", comments="")
    np.savetxt(output_dir / "pareto_F.csv", F_cf, delimiter=",",
               header="neg_energy_GWh_per_year,neg_irrig_reliability,spillage_Mm3_per_year", comments="")
    print(f"Saved {output_dir}/pareto_X.csv, pareto_F.csv "
          f"(NOTE: dam-safety constraint NOT enforced -- see module docstring, do not use for design decisions)")

    if args.compare:
        from optimize import run_optimization

        print(f"\n{'=' * 60}\npymoo NSGA-II  (pop={pop_size}, n_gen={n_gen})\n{'=' * 60}")
        t0 = time.perf_counter()
        run_optimization(
            data_dir=data_dir,
            output_dir=str(Path(__file__).resolve().parent.parent / "Output" / f"{case_name}_pymoo_compare"),
            pop_size=pop_size, n_gen=n_gen, seed=args.seed, verbose=True,
        )
        elapsed_pymoo = time.perf_counter() - t0
        n_evals_pymoo = pop_size * n_gen

        print(f"\n{'=' * 60}\nSIDE-BY-SIDE COMPARISON\n{'=' * 60}")
        print(f"{'Engine':<15} {'Elapsed (s)':>12} {'Evaluations':>14} {'ms/eval':>10}")
        print(f"{'ctrl-freak':<15} {elapsed_cf:>12.2f} {n_evals_cf:>14,} {elapsed_cf/n_evals_cf*1000:>10.3f}")
        print(f"{'pymoo':<15} {elapsed_pymoo:>12.2f} {n_evals_pymoo:>14,} {elapsed_pymoo/n_evals_pymoo*1000:>10.3f}")
        print(f"\nSpeed ratio (pymoo / ctrl-freak): {elapsed_pymoo / elapsed_cf:.2f}x")
        print("(Remember: pymoo's run enforces the dam-safety constraint, ctrl-freak's doesn't -- "
              "this is a speed comparison, not a like-for-like feasibility comparison.)")