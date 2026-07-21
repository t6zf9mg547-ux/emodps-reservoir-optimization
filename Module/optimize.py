"""
optimize.py

pymoo NSGA-II driver for the joint EMODPS-style design + operating policy
optimization. Decision vector (110 variables):
    x[0]   = design_discharge_hydro   (m3/s, real bounds from config_scalars.csv)
    x[1:]  = 109 policy genes (unit hypercube -- see policy.py)

Irrigation design discharge is NOT a decision variable -- it's fixed at
design_discharge_irrig_m3s in config_scalars.csv (per-conversation
decision: unlike hydropower, more irrigation capacity beyond peak
monthly demand buys nothing, since delivery is capped at
min(mult * demand, capacity) and reliability is capped at meeting a
FIXED demand -- there's no open-ended benefit to searching over it the
way there is for hydropower capacity).

Objectives (see objectives.py): maximize hydropower energy, maximize
irrigation reliability, minimize spillage. Flood/dam safety is a hard
CONSTRAINT, not an objective (monthly timestep can't represent
flood-peak dynamics -- per-conversation decision, handled separately).

Algorithm: NSGA-II -- the standard choice for <=3 objectives. NSGA-III
is built for many-objective (4+) problems and isn't needed here.

Convergence: rather than an arbitrary fixed generation count, hypervolume
of the current non-dominated front is tracked every generation via a
callback, printed live, and saved to Output/hypervolume_history.csv.
Run long enough that the curve visibly flattens; if it's still climbing
at n_gen, increase it and re-run (pymoo supports resuming, but this
driver re-runs from scratch for simplicity -- ask if you want
warm-start/checkpointing added).

Population size vs. generations -- these control different things:
  - GENERATIONS determine how close the search gets to the true efficient
    frontier (convergence). Watch the hypervolume curve: once it flattens,
    more generations won't meaningfully improve things.
  - POPULATION SIZE determines how many distinct points get retained
    ALONG that frontier (resolution/density). A converged front with a
    small population can still have gaps between points -- this is
    resolution, not a convergence problem, and needs more population,
    not more generations, to fill in.

Two presets are provided (--preset quick / --preset full), or set
--pop-size/--n-gen directly for full manual control (explicit flags
always override a preset):
  - "quick"  (pop=30,  n_gen=30 ) -- fast sanity check that the pipeline
    runs end-to-end against your data, NOT meant for interpreting results.
  - "full"   (pop=250, n_gen=300) -- recommended default for actual
    analysis: higher resolution than earlier runs (was pop=150) at a
    modest extra time cost, per-conversation decision.

Performance: simulate() is numba-jitted (see simulator.py) -- measured at
~0.55 ms/evaluation on a 30-year record, ~28x faster than the pre-numba
pure-Python baseline (~15.5 ms/eval). A full 250x300=75,000-evaluation
run on that 30-year record was measured directly at ~44 seconds. Runtime
scales roughly linearly with inflow record length, so on a ~70-year
record (like Mandrare) expect roughly 2x that -- an estimate, not a
direct measurement on every machine/dataset, so treat it as a ballpark:
  - quick (30x30=900 evals):            a few seconds, most record lengths
  - full, 30-yr record (75,000 evals):   ~44 seconds (measured)
  - full, ~70-yr record (75,000 evals):  ~1.5-2 minutes (estimated)
One-time cost: the FIRST simulate() call on a fresh machine/environment
pays a one-off numba JIT-compilation cost (~4s uncached, ~0.3s if numba's
on-disk compilation cache from a previous run is still present) -- this
happens once per machine, not once per run, and is negligible against
the runtimes above.
Evaluations are currently serial (one pymoo worker); pymoo supports
parallelizing via elementwise_runner (multiprocessing/joblib/MPI) if this
becomes a bottleneck -- not added by default since it isn't needed given
the numba speedup already achieved.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.problem import ElementwiseProblem
from pymoo.indicators.hv import HV
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize

from data_loader import case_name_from_data_dir, load_reservoir_data, prompt_for_data_folder
from objectives import evaluate
from policy import N_POLICY_PARAMS, compute_forecast_categories

POP_SIZE = 250       # "full" preset default -- see module docstring for pop_size vs n_gen guidance
N_GENERATIONS = 300
RANDOM_SEED = 1

PRESETS = {
    "quick": {"pop_size": 30, "n_gen": 30},     # fast pipeline sanity check, NOT for interpreting results
    "full": {"pop_size": POP_SIZE, "n_gen": N_GENERATIONS},  # recommended default for actual analysis
}


class ReservoirProblem(ElementwiseProblem):
    """
    pymoo problem wrapping objectives.evaluate(). One row of X = one full
    110-gene decision vector (hydro design discharge in physical units,
    policy genes in [0, 1] -- pymoo handles the per-variable bounds via
    xl/xu). Irrigation design discharge is fixed (design_discharge_irrig_m3s
    in config_scalars.csv), not a decision variable -- see objectives.py.
    """

    def __init__(self, data, categories, hydro_min: float | None = None, hydro_max: float | None = None):
        """
        hydro_min/hydro_max override config_scalars.csv's design_discharge_hydro_min/max
        if given -- useful for a targeted re-optimization within a narrowed capacity
        band (e.g. to test whether a gap in a previous Pareto front is a real
        discontinuity or just under-sampling), without editing the CSV by hand.
        """
        xl = np.concatenate([
            [hydro_min if hydro_min is not None else data.scalars["design_discharge_hydro_min"]],
            np.zeros(N_POLICY_PARAMS),
        ])
        xu = np.concatenate([
            [hydro_max if hydro_max is not None else data.scalars["design_discharge_hydro_max"]],
            np.ones(N_POLICY_PARAMS),
        ])
        super().__init__(n_var=xl.shape[0], n_obj=3, n_ieq_constr=1, xl=xl, xu=xu)
        self.data = data
        self.categories = categories

    def _evaluate(self, x, out, *args, **kwargs):
        result = evaluate(self.data, self.categories, x)
        out["F"] = result.F
        out["G"] = result.G


class HypervolumeCallback(Callback):
    """Tracks + prints hypervolume of the current non-dominated front each generation."""

    def __init__(self, ref_point: np.ndarray, print_every: int = 5):
        super().__init__()
        self.indicator = HV(ref_point=ref_point)
        self.print_every = print_every
        self.data["hv"] = []
        self.data["n_gen"] = []

    def notify(self, algorithm):
        F = algorithm.opt.get("F")
        hv = self.indicator(F) if F is not None and len(F) > 0 else 0.0
        self.data["hv"].append(hv)
        self.data["n_gen"].append(algorithm.n_gen)
        if algorithm.n_gen % self.print_every == 0 or algorithm.n_gen == 1:
            print(f"  gen {algorithm.n_gen:4d}  |  pareto size {len(F):3d}  |  hypervolume {hv:.4f}")


def run_optimization(
    data_dir: str | None = None,
    output_dir: str | None = None,
    pop_size: int = POP_SIZE,
    n_gen: int = N_GENERATIONS,
    seed: int = RANDOM_SEED,
    hydro_min: float | None = None,
    hydro_max: float | None = None,
    verbose: bool = True,
):
    """
    data_dir=None (the default) opens an interactive folder-picker dialog
    (see data_loader.prompt_for_data_folder), falling back to the project's
    default Data/ folder if tkinter isn't available or the dialog is
    cancelled. Pass an explicit path to skip the dialog entirely -- useful
    for scripted/batch runs.

    output_dir=None (the default) saves results under Output/<case_name>/,
    where <case_name> is derived from data_dir's own folder name (e.g.
    Data/Mandrare -> Output/Mandrare/) -- so results from different
    datasets/scenarios never overwrite each other. Pass an explicit path
    to bypass this and use it verbatim instead.

    hydro_min/hydro_max override config_scalars.csv's
    design_discharge_hydro_min/max if given -- for a targeted
    re-optimization within a narrowed capacity band (e.g. to test whether
    a gap in a previous Pareto front is a real discontinuity or just
    under-sampling), without editing the CSV by hand. When used, saving
    to a DIFFERENT output_dir than your main run is strongly recommended
    (e.g. append a suffix to the case name) so this doesn't overwrite your
    full-range results.
    """
    if data_dir is None:
        default_data_dir = Path(__file__).resolve().parent.parent / "Data"
        data_dir = prompt_for_data_folder(default_data_dir)
        print(f"Loading data from: {data_dir}")

    case_name = case_name_from_data_dir(data_dir)
    if output_dir is None:
        project_root = Path(__file__).resolve().parent.parent
        output_dir = str(project_root / "Output" / case_name)
    if verbose:
        print(f"Case: {case_name}  ->  results will be saved to {output_dir}/")
        if hydro_min is not None or hydro_max is not None:
            print(f"  NOTE: hydro design discharge bounds overridden to [{hydro_min}, {hydro_max}]")

    data = load_reservoir_data(data_dir)
    categories = compute_forecast_categories(data.months, data.inflow_m3s)
    problem = ReservoirProblem(data, categories, hydro_min=hydro_min, hydro_max=hydro_max)

    # Hypervolume reference point: worst plausible value in each (minimized) dimension.
    # -energy and -reliability are both bounded above by 0 (zero energy / zero reliability).
    # spillage's worst case is bounded by total mean annual inflow (spilling everything).
    n_years = data.n_steps / 12.0
    mean_annual_inflow_Mm3 = data.m3s_to_Mm3(data.inflow_m3s, data.days_in_month).sum() / n_years
    ref_point = np.array([0.0, 0.0, mean_annual_inflow_Mm3])

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )

    if verbose:
        print(f"Running NSGA-II: pop_size={pop_size}, n_gen={n_gen}, n_var={problem.n_var}")

    callback = HypervolumeCallback(ref_point)
    res = minimize(
        problem, algorithm,
        termination=("n_gen", n_gen),
        seed=seed,
        callback=callback,
        verbose=False,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    policy_cols = ",".join(f"policy_{i}" for i in range(N_POLICY_PARAMS))
    np.savetxt(
        out_dir / "pareto_X.csv", res.X, delimiter=",",
        header=f"design_discharge_hydro,{policy_cols}", comments="",
    )
    np.savetxt(
        out_dir / "pareto_F.csv", res.F, delimiter=",",
        header="neg_energy_GWh_per_year,neg_irrig_reliability,spillage_Mm3_per_year", comments="",
    )
    hv_hist = np.column_stack([callback.data["n_gen"], callback.data["hv"]])
    np.savetxt(out_dir / "hypervolume_history.csv", hv_hist, delimiter=",", header="n_gen,hypervolume", comments="")

    if verbose:
        print(f"\nDone. {res.F.shape[0]} Pareto-optimal solutions found.")
        print(f"Saved: {out_dir/'pareto_X.csv'}, {out_dir/'pareto_F.csv'}, {out_dir/'hypervolume_history.csv'}")
    return res


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the reservoir design + operating policy optimization.")
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to the Data folder. If omitted, opens an interactive folder-picker dialog "
             "(falls back to the project's own Data/ folder if cancelled or tkinter unavailable).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Where to save results. Default: Output/<case_name>/, where <case_name> is derived "
             "from --data-dir's folder name (e.g. Data/Mandrare -> Output/Mandrare/).",
    )
    parser.add_argument(
        "--preset", type=str, default=None, choices=list(PRESETS.keys()),
        help="Named pop_size/n_gen combination. 'quick' (pop=30, n_gen=30): fast pipeline "
             "sanity check, NOT for interpreting results. 'full' (pop=250, n_gen=300): "
             "recommended default for actual analysis. --pop-size/--n-gen below override "
             "the preset if both are given.",
    )
    parser.add_argument(
        "--pop-size", type=int, default=None,
        help=f"NSGA-II population size. Overrides --preset if given. Default if neither is "
             f"given: {POP_SIZE} (the 'full' preset's value).",
    )
    parser.add_argument(
        "--n-gen", type=int, default=None,
        help=f"Number of generations. Overrides --preset if given. Default if neither is "
             f"given: {N_GENERATIONS} (the 'full' preset's value).",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help=f"Random seed (default: {RANDOM_SEED}).")
    parser.add_argument(
        "--hydro-min", type=float, default=None,
        help="Override design_discharge_hydro_min for this run only (doesn't touch config_scalars.csv). "
             "Useful for a targeted re-optimization within a narrowed capacity band, e.g. to test "
             "whether a gap in a previous Pareto front is real or just under-sampling.",
    )
    parser.add_argument(
        "--hydro-max", type=float, default=None,
        help="Override design_discharge_hydro_max for this run only. See --hydro-min.",
    )
    args = parser.parse_args()

    preset = PRESETS.get(args.preset, {})
    pop_size = args.pop_size if args.pop_size is not None else preset.get("pop_size", POP_SIZE)
    n_gen = args.n_gen if args.n_gen is not None else preset.get("n_gen", N_GENERATIONS)

    run_optimization(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        pop_size=pop_size,
        n_gen=n_gen,
        seed=args.seed,
        hydro_min=args.hydro_min,
        hydro_max=args.hydro_max,
    )