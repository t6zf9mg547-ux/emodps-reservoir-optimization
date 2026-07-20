"""
data_loader.py

Reads all raw CSV inputs from Data/ and assembles them into a single
validated, numpy-based container (ReservoirData). This is the ONLY place
that touches pandas / disk I/O. Everything downstream (simulator, MOEA
objective function) works on plain numpy arrays for speed, since the
simulation will be called O(10^4-10^6) times inside the optimizer.

Design notes (why it's built this way):
- np.interp for the EVAC curve instead of scipy.interpolate objects:
  np.interp is a compiled C loop with near-zero per-call Python overhead,
  which matters when it's called every month, every simulation, every
  generation of the MOEA. scipy.interpolate.interp1d/CubicSpline give
  smoother curves but each call carries object-method overhead that adds
  up over millions of calls. If you later decide the piecewise-linear
  EVAC assumption is too coarse, we can switch to a monotonic cubic
  (PCHIP) fit ONCE at load time and still export a dense lookup table
  consumed via np.interp -- keeping the hot loop unchanged.
- Monthly climatology arrays (evaporation, demand, flood threshold) are
  stored as fixed length-12 arrays indexed by (month - 1), not merged
  into the main time series -- avoids re-reading/re-indexing every step.
- All comment lines in the CSV templates start with '#' and are skipped
  via `comment='#'` in read_csv.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

MONTHS_IN_YEAR = 12
_DAYS_IN_MONTH = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])


@dataclass
class ReservoirData:
    # --- EVAC curve (ascending elevation) ---
    evac_elevation: np.ndarray      # (n,) m
    evac_volume: np.ndarray         # (n,) Mm3
    evac_area: np.ndarray           # (n,) km2

    # --- Inflow time series ---
    years: np.ndarray               # (T,) int
    months: np.ndarray              # (T,) int, 1-12
    inflow_m3s: np.ndarray          # (T,) m3/s

    # --- Monthly climatology (index 0 = January) ---
    evaporation_mm: np.ndarray      # (12,) mm/month
    seepage_Mm3: np.ndarray         # (12,) Mm3/month
    irrig_demand_m3s: np.ndarray    # (12,) m3/s
    irrig_priority: np.ndarray      # (12,) dimensionless weight
    water_supply_demand_m3s: np.ndarray  # (12,) m3/s -- modeled the same way as irrigation, see simulator.py
    hydro_availability: np.ndarray  # (12,) fraction [0,1], derates design discharge for maintenance
    env_flow_m3s: np.ndarray        # (12,) mandatory monthly environmental flow

    # --- Spillway rating curve (elevation between FSL and flood_control_level) ---
    spillway_elevation: np.ndarray  # (k,) m, ascending
    spillway_discharge_m3s: np.ndarray  # (k,) m3/s

    # --- Tailwater rating curve (discharge = hydro release + spillway release only) ---
    tailwater_discharge_m3s: np.ndarray  # (j,) m3/s, ascending
    tailwater_elevation_m: np.ndarray    # (j,) m

    # --- Scalars ---
    scalars: dict = field(default_factory=dict)

    # ---------- derived / convenience ----------
    @property
    def n_steps(self) -> int:
        return self.inflow_m3s.shape[0]

    @property
    def days_in_month(self) -> np.ndarray:
        """(T,) calendar days per month for this record, leap-year aware."""
        is_leap = (self.years % 4 == 0) & ((self.years % 100 != 0) | (self.years % 400 == 0))
        d = _DAYS_IN_MONTH[self.months - 1].copy()
        d = np.where((self.months == 2) & is_leap, 29, d)
        return d

    # ---------- EVAC lookups (vectorized, fast) ----------
    def elevation_from_volume(self, volume_Mm3):
        return np.interp(volume_Mm3, self.evac_volume, self.evac_elevation)

    def volume_from_elevation(self, elevation_m):
        return np.interp(elevation_m, self.evac_elevation, self.evac_volume)

    def area_from_elevation(self, elevation_m):
        return np.interp(elevation_m, self.evac_elevation, self.evac_area)

    def area_from_volume(self, volume_Mm3):
        elev = self.elevation_from_volume(volume_Mm3)
        return self.area_from_elevation(elev)

    def spillway_discharge_from_elevation(self, elevation_m):
        """
        Physical spillway discharge (m3/s) at a given pool elevation.
        Flat-extrapolates beyond the rating curve's given range (see
        spillway_rating_curve.csv header comment for the assumption this implies).
        """
        return np.interp(elevation_m, self.spillway_elevation, self.spillway_discharge_m3s)

    def tailwater_elevation_from_discharge(self, discharge_m3s):
        """
        Tailrace elevation for a given downstream discharge (hydro release +
        spillway release ONLY -- see tailwater_rating_curve.csv header
        comment). Flat-extrapolates beyond the curve's given range.
        """
        return np.interp(discharge_m3s, self.tailwater_discharge_m3s, self.tailwater_elevation_m)

    # ---------- unit conversions ----------
    @staticmethod
    def m3s_to_Mm3(flow_m3s, days):
        """Mean monthly flow (m3/s) -> monthly volume (Mm3)."""
        return flow_m3s * days * 86400.0 / 1.0e6

    @staticmethod
    def Mm3_to_m3s(volume_Mm3, days):
        """Monthly volume (Mm3) -> mean monthly flow (m3/s)."""
        return volume_Mm3 * 1.0e6 / (days * 86400.0)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#")


def load_reservoir_data(data_dir: str | Path) -> ReservoirData:
    data_dir = Path(data_dir)

    evac = _read_csv(data_dir / "reservoir_evac.csv")
    inflow = _read_csv(data_dir / "inflows_monthly.csv")
    evap = _read_csv(data_dir / "evaporation_monthly.csv")
    irrig = _read_csv(data_dir / "irrigation_demand_monthly.csv")
    water_supply = _read_csv(data_dir / "water_supply_demand_monthly.csv")
    hydro_avail = _read_csv(data_dir / "hydropower_availability_monthly.csv")
    env_flow = _read_csv(data_dir / "environmental_flow_monthly.csv")
    spillway = _read_csv(data_dir / "spillway_rating_curve.csv")
    tailwater = _read_csv(data_dir / "tailwater_rating_curve.csv")
    scal = _read_csv(data_dir / "config_scalars.csv")

    # --- scalars: build a {name: float | bool | str} dict ---
    scalars = {}
    for _, row in scal.iterrows():
        raw = str(row["value"]).strip()
        if raw.lower() in ("true", "false"):
            val = raw.lower() == "true"
        else:
            try:
                val = float(raw)
            except ValueError:
                val = raw
        scalars[row["parameter"]] = val

    data = ReservoirData(
        evac_elevation=evac["elevation_m"].to_numpy(float),
        evac_volume=evac["volume_Mm3"].to_numpy(float),
        evac_area=evac["surface_area_km2"].to_numpy(float),
        years=inflow["year"].to_numpy(int),
        months=inflow["month"].to_numpy(int),
        inflow_m3s=inflow["inflow_m3s"].to_numpy(float),
        evaporation_mm=evap.sort_values("month")["evaporation_mm"].to_numpy(float),
        seepage_Mm3=evap.sort_values("month")["seepage_Mm3"].to_numpy(float),
        irrig_demand_m3s=irrig.sort_values("month")["demand_m3s"].to_numpy(float),
        irrig_priority=irrig.sort_values("month")["priority_weight"].to_numpy(float),
        water_supply_demand_m3s=water_supply.sort_values("month")["demand_m3s"].to_numpy(float),
        hydro_availability=hydro_avail.sort_values("month")["availability_fraction"].to_numpy(float),
        env_flow_m3s=env_flow.sort_values("month")["flow_m3s"].to_numpy(float),
        spillway_elevation=spillway["elevation_m"].to_numpy(float),
        spillway_discharge_m3s=spillway["discharge_m3s"].to_numpy(float),
        tailwater_discharge_m3s=tailwater["discharge_m3s"].to_numpy(float),
        tailwater_elevation_m=tailwater["tailwater_elevation_m"].to_numpy(float),
        scalars=scalars,
    )

    _validate(data)
    return data


def _validate(d: ReservoirData) -> None:
    errors = []

    if not np.all(np.diff(d.evac_elevation) > 0):
        errors.append("reservoir_evac.csv: elevation must be strictly increasing.")
    if not np.all(np.diff(d.evac_volume) >= 0):
        errors.append("reservoir_evac.csv: volume must be non-decreasing with elevation.")

    for name, arr in [
        ("evaporation_monthly.csv", d.evaporation_mm),
        ("evaporation_monthly.csv (seepage)", d.seepage_Mm3),
        ("irrigation_demand_monthly.csv", d.irrig_demand_m3s),
        ("water_supply_demand_monthly.csv", d.water_supply_demand_m3s),
        ("hydropower_availability_monthly.csv", d.hydro_availability),
        ("environmental_flow_monthly.csv", d.env_flow_m3s),
    ]:
        if arr.shape[0] != MONTHS_IN_YEAR:
            errors.append(f"{name}: expected 12 rows (one per month), got {arr.shape[0]}.")

    if np.any((d.hydro_availability < 0) | (d.hydro_availability > 1)):
        errors.append("hydropower_availability_monthly.csv: availability_fraction must be in [0, 1].")

    if not np.all(np.diff(d.spillway_elevation) > 0):
        errors.append("spillway_rating_curve.csv: elevation must be strictly increasing.")
    if not np.all(np.diff(d.spillway_discharge_m3s) >= 0):
        errors.append("spillway_rating_curve.csv: discharge must be non-decreasing with elevation.")
    if "max_operating_level" in d.scalars and d.spillway_elevation.size > 0:
        if abs(d.spillway_elevation.min() - d.scalars["max_operating_level"]) > 1e-6:
            errors.append(
                "spillway_rating_curve.csv: lowest elevation should equal max_operating_level "
                "(the spillway crest / FSL), where discharge should be 0."
            )
    if "flood_control_level" in d.scalars and d.spillway_elevation.size > 0:
        if d.spillway_elevation.max() < d.scalars["flood_control_level"]:
            errors.append(
                "spillway_rating_curve.csv: highest elevation should reach at least "
                "flood_control_level, otherwise the rating curve is flat-extrapolated "
                "below the true dam-safety ceiling, which may understate spill capacity."
            )

    if not np.all(np.diff(d.tailwater_discharge_m3s) >= 0):
        errors.append("tailwater_rating_curve.csv: discharge must be strictly increasing.")
    if not np.all(np.diff(d.tailwater_elevation_m) >= 0):
        errors.append("tailwater_rating_curve.csv: tailwater_elevation_m must be non-decreasing with discharge.")

    if np.any(np.isnan(d.inflow_m3s)):
        errors.append("inflows_monthly.csv: contains NaN inflow values.")
    if d.n_steps > 0:
        # Records don't need to start in January or span a whole number of
        # years (e.g. Nov 1950 - Sep 2022 is perfectly fine) -- just check
        # that (year, month) advances by exactly one calendar month, every
        # row, from whatever the record's actual first entry is.
        start_year, start_month = int(d.years[0]), int(d.months[0])
        expected_years = np.empty(d.n_steps, dtype=int)
        expected_months = np.empty(d.n_steps, dtype=int)
        y, mth = start_year, start_month
        for i in range(d.n_steps):
            expected_years[i] = y
            expected_months[i] = mth
            mth += 1
            if mth > 12:
                mth = 1
                y += 1
        if not (np.array_equal(d.months, expected_months) and np.array_equal(d.years, expected_years)):
            errors.append(
                "inflows_monthly.csv: (year, month) must advance by exactly one calendar "
                "month every row, with no gaps or duplicates (the record can start in any "
                "month and need not span a whole number of years)."
            )

    required_scalars = [
        "min_operating_level", "max_operating_level", "flood_control_level",
        "initial_level", "max_release_capacity_m3s",
        "min_operating_level_hydro", "min_operating_level_irrig", "min_operating_level_water_supply",
        "environmental_flow_turbined", "bypass_outlet_capacity_m3s",
        "turbine_efficiency", "min_hydropower_head", "alpha_headloss_coeff",
        "design_discharge_hydro_min", "design_discharge_hydro_max",
        "design_discharge_irrig_m3s", "design_discharge_water_supply_m3s",
    ]
    missing = [p for p in required_scalars if p not in d.scalars]
    if missing:
        errors.append(f"config_scalars.csv: missing required parameters: {missing}")

    if "bypass_outlet_capacity_m3s" in d.scalars and d.env_flow_m3s.size > 0:
        if d.scalars["bypass_outlet_capacity_m3s"] < d.env_flow_m3s.max():
            errors.append(
                "config_scalars.csv: bypass_outlet_capacity_m3s must be >= the maximum monthly "
                "value in environmental_flow_monthly.csv, otherwise the environmental flow cannot "
                "always be guaranteed when turbines are off."
            )

    if "environmental_flow_turbined" in d.scalars and not isinstance(d.scalars["environmental_flow_turbined"], bool):
        errors.append(
            "config_scalars.csv: environmental_flow_turbined must be 'True' or 'False', "
            f"got {d.scalars['environmental_flow_turbined']!r}."
        )

    if "min_operating_level" in d.scalars and "max_operating_level" in d.scalars:
        if d.scalars["min_operating_level"] >= d.scalars["max_operating_level"]:
            errors.append("config_scalars.csv: min_operating_level must be < max_operating_level.")
        lo, hi = d.evac_elevation.min(), d.evac_elevation.max()
        if not (lo <= d.scalars["min_operating_level"] <= hi):
            errors.append("config_scalars.csv: min_operating_level is outside the EVAC elevation range.")
        if not (lo <= d.scalars["max_operating_level"] <= hi):
            errors.append("config_scalars.csv: max_operating_level is outside the EVAC elevation range.")

        for name in ("min_operating_level_hydro", "min_operating_level_irrig", "min_operating_level_water_supply"):
            if name in d.scalars:
                if not (d.scalars["min_operating_level"] <= d.scalars[name] <= d.scalars["max_operating_level"]):
                    errors.append(
                        f"config_scalars.csv: {name} ({d.scalars[name]}) must be between "
                        f"min_operating_level ({d.scalars['min_operating_level']}) and "
                        f"max_operating_level ({d.scalars['max_operating_level']})."
                    )

    if "design_discharge_irrig_m3s" in d.scalars and d.irrig_demand_m3s.size > 0:
        peak_demand = d.irrig_demand_m3s.max()
        if d.scalars["design_discharge_irrig_m3s"] < peak_demand:
            errors.append(
                f"config_scalars.csv: design_discharge_irrig_m3s ({d.scalars['design_discharge_irrig_m3s']}) "
                f"is below peak monthly demand ({peak_demand}) in irrigation_demand_monthly.csv -- this would "
                "silently undersize the canal below what's needed to ever fully meet demand. If this is "
                "intentional, this check needs a deliberate override; if not, raise the capacity to at "
                "least the peak demand."
            )

    if "design_discharge_water_supply_m3s" in d.scalars and d.water_supply_demand_m3s.size > 0:
        peak_ws_demand = d.water_supply_demand_m3s.max()
        if d.scalars["design_discharge_water_supply_m3s"] < peak_ws_demand:
            errors.append(
                f"config_scalars.csv: design_discharge_water_supply_m3s "
                f"({d.scalars['design_discharge_water_supply_m3s']}) is below peak monthly demand "
                f"({peak_ws_demand}) in water_supply_demand_monthly.csv -- this would silently undersize "
                "the intake below what's needed to ever fully meet demand. If this is intentional, this "
                "check needs a deliberate override; if not, raise the capacity to at least the peak demand."
            )

    if errors:
        raise ValueError("Input data validation failed:\n- " + "\n- ".join(errors))


def prompt_for_data_folder(default_dir: Path) -> Path:
    """
    Open a folder-selection dialog (tkinter) so the user can pick the Data/
    folder interactively, matching this project template's convention.
    Falls back to `default_dir` if tkinter isn't available (e.g. a headless
    environment) or the dialog is cancelled -- so scripts that call this
    still work non-interactively without modification.

    Kept separate from load_reservoir_data() on purpose: that function stays
    a pure "path in, data out" call with no UI side effects, so it's easy to
    call repeatedly/programmatically (e.g. from optimize.py) without ever
    triggering a dialog by accident.
    """
    print(
        "No data folder specified -- opening a folder-picker window.\n"
        "  -> Select the folder containing your input CSVs (e.g. Data/Mandrare),\n"
        "     NOT a specific CSV file.\n"
        "  -> Look for the dialog window; it may open BEHIND your terminal/editor.\n"
        "  -> To skip this next time, pass --data-dir <path> on the command line\n"
        "     (or data_dir=\"<path>\" if calling run_optimization()/load_reservoir_data() directly)."
    )
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        print(f"tkinter not available -- using default Data folder: {default_dir}")
        return default_dir

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askdirectory(
        title="Select your DATA FOLDER (e.g. Data/Mandrare) -- not a CSV file",
        initialdir=str(default_dir) if default_dir.exists() else str(Path.cwd()),
    )
    root.destroy()

    if not selected:
        print(f"No folder selected -- using default Data folder: {default_dir}")
        return default_dir
    print(f"Using data folder: {selected}")
    return Path(selected)


def case_name_from_data_dir(data_dir: str | Path) -> str:
    """
    Derive a short case/scenario label from a Data folder path, so
    Output/ and Plot/ can be kept organized per dataset instead of every
    run overwriting the last one -- e.g. .../Data/Mandrare -> "Mandrare".
    Falls back to whatever the folder's own name is (e.g. "Data" if you
    point directly at the top-level Data/ folder rather than a named
    subfolder like Data/Mandrare -- using a named subfolder per case/river
    is recommended precisely so this label is meaningful).
    """
    return Path(data_dir).resolve().name


if __name__ == "__main__":
    default_data_dir = Path(__file__).resolve().parent.parent / "Data"
    data_dir = prompt_for_data_folder(default_data_dir)
    print(f"Loading data from: {data_dir}")

    d = load_reservoir_data(data_dir)
    print(f"Loaded {d.n_steps} months ({d.n_steps / 12:.1f} years) of inflow data.")
    print(f"EVAC curve: {len(d.evac_elevation)} points, "
          f"elevation range [{d.evac_elevation.min()}, {d.evac_elevation.max()}] m")
    print(f"Storage at max operating level: "
          f"{d.volume_from_elevation(d.scalars['max_operating_level']):.1f} Mm3")
    print("Validation passed.")