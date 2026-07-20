"""
diagnostics.py

Given any decision vector (a row from the Pareto front, or a candidate
you're testing by hand), reruns the simulator and reports failure counts
and stats beyond the single aggregated objective values -- e.g.
"irrigation shortfall occurred in 34 of 360 months", not just an overall
reliability fraction.

Uses the SAME warm-up-excluded evaluation window as objectives.py, so
these numbers are directly comparable to (and consistent with) the
Pareto front's F values -- irrig_reliability here should exactly match
-F[1] for the same x, for example.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from objectives import IRRIGATION_SHORTFALL_TOLERANCE_M3S, _warmup_mask
from policy import decode_policy
from simulator import simulate


@dataclass
class DiagnosticsReport:
    n_months_evaluated: int

    # Irrigation
    irrig_months_with_shortfall: int
    irrig_reliability: float                 # fraction of months with NO shortfall
    irrig_max_shortfall_m3s: float
    irrig_mean_shortfall_when_failing_m3s: float
    irrig_total_shortfall_volume_Mm3: float   # cumulative unmet demand, evaluated period

    # Water supply (modeled the same way as irrigation)
    ws_months_with_shortfall: int
    ws_reliability: float
    ws_max_shortfall_m3s: float
    ws_mean_shortfall_when_failing_m3s: float
    ws_total_shortfall_volume_Mm3: float
    ws_total_delivered_volume_Mm3: float

    # Hydropower
    hydro_months_offline: int                 # hydro_release ~ 0 (maintenance, infeasible head, or Restricted zone)
    hydro_months_online: int
    hydro_mean_energy_when_online_MWh: float
    hydro_total_energy_GWh: float
    hydro_total_volume_Mm3: float              # total water volume that passed through the turbine, evaluated period
    hydro_design_discharge_m3s: float           # the installed capacity itself, for reference alongside the utilization metrics
    hydro_capacity_utilization: float           # mean release (ALL evaluated months, including 0) / design discharge -- 0-1
    hydro_months_near_full_capacity: int        # months where release >= 90% of design discharge
    irrig_total_delivered_volume_Mm3: float    # total water volume actually delivered to irrigation, evaluated period

    # Spillway / flood management (informational -- NOT the dam-safety constraint)
    spill_months_active: int                  # spillway_release > 0
    spill_max_m3s: float
    spill_total_volume_Mm3: float

    # Dam safety -- the hard pymoo constraint
    dam_safety_violation_months: int
    dam_safety_max_violation_Mm3: float

    def print_report(self):
        n = self.n_months_evaluated
        print(f"Diagnostics over {n} evaluated months ({n / 12:.1f} years, post-warmup)")
        print()
        print("Irrigation")
        print(f"  Reliability: {self.irrig_reliability:.1%} "
              f"({n - self.irrig_months_with_shortfall}/{n} months fully met)")
        print(f"  Months with shortfall: {self.irrig_months_with_shortfall}")
        if self.irrig_months_with_shortfall > 0:
            print(f"  Max shortfall: {self.irrig_max_shortfall_m3s:.2f} m3/s")
            print(f"  Mean shortfall (failing months only): {self.irrig_mean_shortfall_when_failing_m3s:.2f} m3/s")
        print(f"  Total unmet demand volume: {self.irrig_total_shortfall_volume_Mm3:.1f} Mm3")
        print()
        print("Water Supply (modeled the same way as irrigation)")
        print(f"  Reliability: {self.ws_reliability:.1%} "
              f"({n - self.ws_months_with_shortfall}/{n} months fully met)")
        print(f"  Months with shortfall: {self.ws_months_with_shortfall}")
        if self.ws_months_with_shortfall > 0:
            print(f"  Max shortfall: {self.ws_max_shortfall_m3s:.2f} m3/s")
            print(f"  Mean shortfall (failing months only): {self.ws_mean_shortfall_when_failing_m3s:.2f} m3/s")
        print(f"  Total unmet demand volume: {self.ws_total_shortfall_volume_Mm3:.1f} Mm3")
        print(f"  Total delivered volume: {self.ws_total_delivered_volume_Mm3:.1f} Mm3")
        print()
        print("Hydropower")
        print(f"  Months online: {self.hydro_months_online}/{n}")
        print(f"  Months offline: {self.hydro_months_offline}")
        print(f"  Total energy: {self.hydro_total_energy_GWh:.1f} GWh")
        print(f"  Total water turbined: {self.hydro_total_volume_Mm3:.1f} Mm3")
        print(f"  Installed (design) discharge: {self.hydro_design_discharge_m3s:.2f} m3/s")
        print(f"  Capacity utilization: {self.hydro_capacity_utilization:.1%} "
              f"(mean release / design discharge, across ALL evaluated months)")
        print(f"  Months near full capacity (>=90%): {self.hydro_months_near_full_capacity}/{n}")
        print(f"  Mean energy per online month: {self.hydro_mean_energy_when_online_MWh:.1f} MWh")
        print()
        print(f"Irrigation total delivered volume: {self.irrig_total_delivered_volume_Mm3:.1f} Mm3")
        print()
        print("Spillway (informational)")
        print(f"  Months with spill: {self.spill_months_active}/{n}")
        print(f"  Max spill rate: {self.spill_max_m3s:.1f} m3/s")
        print(f"  Total spilled volume: {self.spill_total_volume_Mm3:.1f} Mm3")
        print()
        print("Dam safety (hard constraint)")
        print(f"  Violation months: {self.dam_safety_violation_months}")
        status = "  <-- INFEASIBLE DESIGN" if self.dam_safety_violation_months > 0 else "  (feasible)"
        print(f"  Max violation: {self.dam_safety_max_violation_Mm3:.2f} Mm3{status}")

    def to_csv(self, path: str | Path, extra_metadata: dict | None = None):
        """
        Saves this report as a simple metric,value CSV -- easy to open in
        Excel or append to a record of solutions you've inspected.
        extra_metadata (e.g. solution index, design discharge, seed) is
        written as additional leading rows if given.
        """
        rows = list((extra_metadata or {}).items()) + list(asdict(self).items())
        with open(path, "w") as f:
            f.write("metric,value\n")
            for k, v in rows:
                f.write(f"{k},{v}\n")


def diagnose(data, categories: np.ndarray, x: np.ndarray) -> DiagnosticsReport:
    design_discharge_hydro = x[0]
    design_discharge_irrig = data.scalars["design_discharge_irrig_m3s"]
    policy_genes = x[1:]

    mol_vol = data.volume_from_elevation(data.scalars["min_operating_level"])
    fsl_vol = data.volume_from_elevation(data.scalars["max_operating_level"])
    policy = decode_policy(policy_genes, mol_vol, fsl_vol)

    result = simulate(data, policy, categories, design_discharge_hydro, design_discharge_irrig)

    mask = _warmup_mask(data)
    days = data.days_in_month[mask]
    n_eval = int(mask.sum())

    shortfall = result.irrig_shortfall_m3s[mask]
    failing = shortfall > IRRIGATION_SHORTFALL_TOLERANCE_M3S
    n_fail = int(failing.sum())

    ws_shortfall = result.ws_shortfall_m3s[mask]
    ws_failing = ws_shortfall > IRRIGATION_SHORTFALL_TOLERANCE_M3S
    ws_n_fail = int(ws_failing.sum())
    ws_delivered = result.ws_release_m3s[mask]
    ws_delivered_volume_Mm3 = data.m3s_to_Mm3(ws_delivered, days).sum()

    hydro = result.hydro_release_m3s[mask]
    online = hydro > 1e-9
    energy = result.energy_MWh[mask]
    hydro_volume_Mm3 = data.m3s_to_Mm3(hydro, days).sum()

    irrig_delivered = result.irrig_release_m3s[mask]
    irrig_delivered_volume_Mm3 = data.m3s_to_Mm3(irrig_delivered, days).sum()

    spill = result.spillway_release_m3s[mask]
    spill_active = spill > 1e-9

    dam_violation = result.dam_safety_violation_Mm3[mask]

    return DiagnosticsReport(
        n_months_evaluated=n_eval,
        irrig_months_with_shortfall=n_fail,
        irrig_reliability=float(1.0 - n_fail / n_eval) if n_eval else 0.0,
        irrig_max_shortfall_m3s=float(shortfall.max()) if n_eval else 0.0,
        irrig_mean_shortfall_when_failing_m3s=float(shortfall[failing].mean()) if n_fail else 0.0,
        irrig_total_shortfall_volume_Mm3=float(data.m3s_to_Mm3(shortfall, days).sum()),
        ws_months_with_shortfall=ws_n_fail,
        ws_reliability=float(1.0 - ws_n_fail / n_eval) if n_eval else 0.0,
        ws_max_shortfall_m3s=float(ws_shortfall.max()) if n_eval else 0.0,
        ws_mean_shortfall_when_failing_m3s=float(ws_shortfall[ws_failing].mean()) if ws_n_fail else 0.0,
        ws_total_shortfall_volume_Mm3=float(data.m3s_to_Mm3(ws_shortfall, days).sum()),
        ws_total_delivered_volume_Mm3=float(ws_delivered_volume_Mm3),
        hydro_months_offline=int((~online).sum()),
        hydro_months_online=int(online.sum()),
        hydro_mean_energy_when_online_MWh=float(energy[online].mean()) if online.any() else 0.0,
        hydro_total_energy_GWh=float(energy.sum() / 1000.0),
        hydro_total_volume_Mm3=float(hydro_volume_Mm3),
        hydro_design_discharge_m3s=float(design_discharge_hydro),
        hydro_capacity_utilization=float(hydro.mean() / design_discharge_hydro) if design_discharge_hydro > 0 else 0.0,
        hydro_months_near_full_capacity=int((hydro >= 0.9 * design_discharge_hydro).sum()),
        irrig_total_delivered_volume_Mm3=float(irrig_delivered_volume_Mm3),
        spill_months_active=int(spill_active.sum()),
        spill_max_m3s=float(spill.max()) if n_eval else 0.0,
        spill_total_volume_Mm3=float(data.m3s_to_Mm3(spill, days).sum()),
        dam_safety_violation_months=int((dam_violation > 0).sum()),
        dam_safety_max_violation_Mm3=float(dam_violation.max()) if n_eval else 0.0,
    )


if __name__ == "__main__":
    import argparse

    from data_loader import case_name_from_data_dir, load_reservoir_data, prompt_for_data_folder
    from policy import compute_forecast_categories

    parser = argparse.ArgumentParser(
        description="Diagnose a solution's detailed failure counts/stats beyond the aggregated objectives."
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to the Data folder (e.g. Data/Mandrare). If omitted, opens an interactive "
             "folder-picker dialog (falls back to the project's own Data/ folder if cancelled "
             "or tkinter unavailable).",
    )
    parser.add_argument(
        "--pareto-x", type=str, default=None,
        help="Path to a saved pareto_X.csv. Default: Output/<case_name>/pareto_X.csv, where "
             "<case_name> is derived from --data-dir's folder name (e.g. Data/Mandrare -> "
             "Output/Mandrare/pareto_X.csv).",
    )
    parser.add_argument(
        "--solution-index", type=int, default=None,
        help="Which row (0-indexed) of pareto_X.csv to diagnose. If neither this nor "
             "--min-reliability is given, defaults to --min-reliability 0.9.",
    )
    parser.add_argument(
        "--min-reliability", type=float, default=None,
        help="Auto-select the solution with the HIGHEST energy among those meeting AT LEAST "
             "this irrigation reliability (0-1). Overrides --solution-index if both given. "
             "Default when neither is given: 0.9.",
    )
    parser.add_argument(
        "--output-csv", type=str, default=None,
        help="Where to save this report as a CSV. Default: "
             "Output/<case_name>/diagnostics_solution_<index>.csv (same folder optimize.py "
             "saves to). Pass --no-csv to skip saving entirely.",
    )
    parser.add_argument("--no-csv", action="store_true", help="Don't save a CSV, print to terminal only.")
    args = parser.parse_args()

    if args.data_dir is None:
        default_data_dir = Path(__file__).resolve().parent.parent / "Data"
        data_dir = prompt_for_data_folder(default_data_dir)
    else:
        data_dir = args.data_dir

    case_name = case_name_from_data_dir(data_dir)
    data = load_reservoir_data(data_dir)
    categories = compute_forecast_categories(data.months, data.inflow_m3s)

    project_root = Path(__file__).resolve().parent.parent
    pareto_x_path = Path(args.pareto_x) if args.pareto_x else project_root / "Output" / case_name / "pareto_X.csv"
    if pareto_x_path.exists():
        X = np.loadtxt(pareto_x_path, delimiter=",", skiprows=1)
        F_path = pareto_x_path.parent / "pareto_F.csv"
        n_solutions = X.shape[0] if X.ndim > 1 else 1

        if args.solution_index is not None:
            solution_index = args.solution_index
        elif F_path.exists():
            import pandas as pd
            from plot_results import select_by_min_reliability
            F = pd.read_csv(F_path)
            solution_index = select_by_min_reliability(F, args.min_reliability if args.min_reliability is not None else 0.9)
        else:
            print(f"No {F_path} found -- can't auto-select by reliability; defaulting to solution 0.")
            solution_index = 0

        if solution_index >= n_solutions:
            raise ValueError(f"solution index {solution_index} out of range (only {n_solutions} solutions in {pareto_x_path}).")
        x = X[solution_index] if X.ndim > 1 else X
        print(f"Diagnosing solution {solution_index} of {n_solutions} from {pareto_x_path}")
    else:
        print(f"No {pareto_x_path} found -- diagnosing a candidate at mid-range hydro capacity instead.")
        from policy import N_POLICY_PARAMS
        rng = np.random.default_rng(0)
        x = np.concatenate([
            [0.5 * (data.scalars["design_discharge_hydro_min"] + data.scalars["design_discharge_hydro_max"])],
            rng.random(N_POLICY_PARAMS),
        ])
        solution_index = "adhoc"

    report = diagnose(data, categories, x)
    print()
    report.print_report()

    if not args.no_csv:
        if args.output_csv is not None:
            csv_path = Path(args.output_csv)
        else:
            csv_path = project_root / "Output" / case_name / f"diagnostics_solution_{solution_index}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(csv_path, extra_metadata={
            "solution_index": solution_index,
            "design_discharge_hydro_m3s": x[0],
            "design_discharge_irrig_m3s": data.scalars["design_discharge_irrig_m3s"],
        })
        print(f"\nSaved {csv_path}")