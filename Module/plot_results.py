"""
plot_results.py

Visualizes optimization results:
  1. Hypervolume convergence (Output/hypervolume_history.csv)
  2. Pareto front, as a pairwise 2D scatter matrix (3 objectives -> 3 pairs,
     each colored by the third objective) -- easier to read than a single
     3D plot, which is hard to interpret as a static image.
  3. For one selected solution: the operating RULE CURVE as a zone diagram.
  4. For one selected solution: a RESERVOIR LEVEL HEAT MAP -- year (rows)
     x month (columns), colored by end-of-month elevation. Makes drought/
     wet-period clusters and long-term trends visible at a glance across
     the whole record.
  5. For one selected solution: the average monthly WATER BALANCE -- a
     stacked bar of where water went each month (hydropower, irrigation,
     environmental flow, spillage, evaporation) against a net inflow line.
  6. For one selected solution: a monthly CLIMATOLOGY view (average
     storage/level/releases/energy by calendar month, across the whole
     evaluated record).
  7. For the same solution: a windowed RAW time series (a handful of
     consecutive years) so you can see actual inter-annual variability,
     not just the smoothed average.

Run directly (see --help), or import the individual plot_* functions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend: this script only ever saves
                        # figures to PNG (fig.savefig), never displays a window
                        # (plt.show is never called). Forcing this here avoids
                        # editors/debuggers (e.g. VS Code) auto-selecting an
                        # interactive GUI backend, which can pop up blank
                        # windows or hang instead of just writing the file.
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from objectives import IRRIGATION_SHORTFALL_TOLERANCE_M3S, _warmup_mask
from policy import CATEGORY_NAMES, decode_policy
from simulator import simulate

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# When the CLI is run with neither --solution-index nor --min-reliability, this is
# the default selection criterion: max energy subject to >= this irrigation
# reliability, rather than an arbitrary Pareto front row index.
DEFAULT_MIN_RELIABILITY = 0.9


# ---------------------------------------------------------------------------
# Loading saved optimization outputs
# ---------------------------------------------------------------------------

def load_pareto(output_dir: str | Path):
    output_dir = Path(output_dir)
    F = pd.read_csv(output_dir / "pareto_F.csv")
    X = pd.read_csv(output_dir / "pareto_X.csv")
    hv = pd.read_csv(output_dir / "hypervolume_history.csv")
    return F, X, hv


# ---------------------------------------------------------------------------
# 1. Hypervolume convergence
# ---------------------------------------------------------------------------

def plot_hypervolume(hv: pd.DataFrame, out_path: str | Path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(hv["n_gen"], hv["hypervolume"], color="#2b6cb0", linewidth=1.8)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Hypervolume")
    ax.set_title("NSGA-II convergence")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Pareto front -- pairwise 2D scatter matrix
# ---------------------------------------------------------------------------

def plot_pareto_matrix(F: pd.DataFrame, X: pd.DataFrame, out_path: str | Path, highlight_index: int | None = None,
                        data=None, categories: np.ndarray | None = None):
    """
    data/categories are optional -- if given, a 4th panel is added comparing
    irrigation reliability directly against water supply reliability
    (colored by energy), since water supply isn't one of the 3 objectives
    NSGA-II actually optimizes (see objectives.py/optimize.py) and so has no
    natural home in pareto_F.csv. Computing this requires re-simulating
    every solution in X (ws_reliability isn't stored anywhere) -- cheap
    with the numba-accelerated simulator, but skipped entirely if data/
    categories aren't provided (keeps this function usable from contexts
    that only have F/X, e.g. quick ad-hoc front inspection).
    """
    energy = -F["neg_energy_GWh_per_year"]
    reliability = -F["neg_irrig_reliability"]
    spillage = F["spillage_Mm3_per_year"]
    hydro_Q = X["design_discharge_hydro"]

    # Linear fit energy <-> hydro design discharge, for the secondary axis below.
    # This is an APPROXIMATION, not an exact 1:1 mapping -- flagged in the
    # footnote, since actual energy also depends on the operating policy,
    # maintenance derating, and head, not capacity alone.
    slope, intercept = np.polyfit(energy, hydro_Q, 1)
    energy_to_hydro_Q = lambda e: slope * e + intercept
    hydro_Q_to_energy = lambda q: (q - intercept) / slope
    r = np.corrcoef(energy, hydro_Q)[0, 1]

    pairs = [
        (energy, reliability, spillage, "Energy (GWh/yr)", "Irrigation reliability", "Spillage (Mm3/yr)", True),
        (energy, spillage, reliability, "Energy (GWh/yr)", "Spillage (Mm3/yr)", "Irrigation reliability", True),
        (reliability, spillage, energy, "Irrigation reliability", "Spillage (Mm3/yr)", "Energy (GWh/yr)", False),
    ]

    ws_reliability = None
    if data is not None and categories is not None:
        mask = _warmup_mask(data)
        n_eval = mask.sum()
        ws_rel_list = []
        for i in range(len(X)):
            result = _simulate_solution(data, categories, X.iloc[i].to_numpy())
            failing = result.ws_shortfall_m3s[mask] > IRRIGATION_SHORTFALL_TOLERANCE_M3S
            ws_rel_list.append(1.0 - failing.sum() / n_eval)
        ws_reliability = pd.Series(ws_rel_list, index=X.index)
        pairs.append((reliability, ws_reliability, energy,
                      "Irrigation reliability", "Water supply reliability", "Energy (GWh/yr)", False))

    n_panels = len(pairs)
    fig, axes = plt.subplots(1, n_panels, figsize=(16 * n_panels / 3, 5.5))
    for ax, (x, y, c, xlabel, ylabel, clabel, energy_is_x) in zip(axes, pairs):
        sc = ax.scatter(x, y, c=c, cmap="viridis", s=45, edgecolor="white", linewidth=0.4)
        if highlight_index is not None:
            ax.scatter(x.iloc[highlight_index], y.iloc[highlight_index],
                       s=220, facecolor="none", edgecolor="red", linewidth=2.2, zorder=5,
                       label=f"solution {highlight_index}")
            ax.legend(loc="best", fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label(clabel, fontsize=9)
        if "reliability" in xlabel.lower() or "reliability" in ylabel.lower():
            if "reliability" in xlabel.lower():
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
            if "reliability" in ylabel.lower():
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

        if energy_is_x:
            secax = ax.secondary_xaxis("top", functions=(energy_to_hydro_Q, hydro_Q_to_energy))
            secax.set_xlabel("Hydro design discharge (m3/s, approx.)", fontsize=8.5)

    fig.suptitle(f"Pareto front ({len(F)} solutions)")
    footnote = (
        f"Top axes: linear approximation of hydro design discharge from energy "
        f"(correlation r={r:.2f} on this front) -- not an exact 1:1 mapping, since "
        f"actual energy also depends on the operating policy, maintenance derating, "
        f"and head. Irrigation reliability is always measured against the fixed "
        f"monthly demand in irrigation_demand_monthly.csv -- 100% means every "
        f"evaluated month fully met that demand, not some larger potential delivery."
    )
    if ws_reliability is not None:
        footnote += (
            " Water supply reliability is NOT one of the 3 objectives NSGA-II "
            "optimizes -- it's computed here just for this comparison, using the "
            "same shared zone multiplier as irrigation (see simulator.py)."
        )
    fig.text(0.5, 0.005, footnote, ha="center", va="bottom", fontsize=7.5, color="#555555", wrap=True)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Shared: run the simulator for one decision vector, keep full result
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Shared: decode the policy portion of a decision vector
# ---------------------------------------------------------------------------

def _decode_solution_policy(data, x):
    design_discharge_hydro = x[0]
    design_discharge_irrig = data.scalars["design_discharge_irrig_m3s"]
    policy_genes = x[1:]
    mol_vol = data.volume_from_elevation(data.scalars["min_operating_level"])
    fsl_vol = data.volume_from_elevation(data.scalars["max_operating_level"])
    policy = decode_policy(policy_genes, mol_vol, fsl_vol)
    return policy, design_discharge_hydro, design_discharge_irrig


def _simulate_solution(data, categories, x):
    policy, design_discharge_hydro, design_discharge_irrig = _decode_solution_policy(data, x)
    return simulate(data, policy, categories, design_discharge_hydro, design_discharge_irrig)


# ---------------------------------------------------------------------------
# 3. Rule curve chart -- the actual operating policy, as a zone diagram
# ---------------------------------------------------------------------------

ZONE_COLORS = {
    "Flood": "#fed7d7",
    "Conservation": "#c6f6d5",
    "Buffer": "#fefcbf",
    "Restricted": "#feb2b2",
}


def plot_rule_curve(data, x, out_path: str | Path, solution_index: int | None = None):
    """
    The classic reservoir zone-operation diagram: month on the x-axis,
    elevation on the y-axis, with the Flood/Conservation/Buffer/Restricted
    zone boundaries shown -- one subplot per forecast category (dry/normal/wet).
    This is the actual operating policy an operator would be handed.
    """
    policy, design_discharge_hydro, design_discharge_irrig = _decode_solution_policy(data, x)

    mol = data.scalars["min_operating_level"]
    fsl = data.scalars["max_operating_level"]
    flood_control = data.scalars["flood_control_level"]
    months = np.arange(1, 13)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
    title_suffix = f" -- solution {solution_index}" if solution_index is not None else ""
    fig.suptitle(f"Operating rule curve{title_suffix}  "
                 f"(hydro design Q={design_discharge_hydro:.1f} m3/s, "
                 f"irrig design Q={design_discharge_irrig:.1f} m3/s)")

    for ax, cat_idx, cat_name in zip(axes, range(len(CATEGORY_NAMES)), CATEGORY_NAMES):
        b1 = np.array([data.elevation_from_volume(v) for v in policy.b1[cat_idx]])
        b2 = np.array([data.elevation_from_volume(v) for v in policy.b2[cat_idx]])
        b3 = np.array([data.elevation_from_volume(v) for v in policy.b3[cat_idx]])

        # close the loop visually (Dec -> Jan) by repeating the first month at the end
        m = np.append(months, 13)
        b1c, b2c, b3c = (np.append(a, a[0]) for a in (b1, b2, b3))
        mol_line = np.full_like(m, mol, dtype=float)
        fsl_line = np.full_like(m, fsl, dtype=float)
        flood_line = np.full_like(m, flood_control, dtype=float)

        ax.fill_between(m, b1c, flood_line, color=ZONE_COLORS["Flood"], label="Flood")
        ax.fill_between(m, b2c, b1c, color=ZONE_COLORS["Conservation"], label="Conservation")
        ax.fill_between(m, b3c, b2c, color=ZONE_COLORS["Buffer"], label="Buffer")
        ax.fill_between(m, mol_line, b3c, color=ZONE_COLORS["Restricted"], label="Restricted")

        ax.plot(m, b1c, color="black", linewidth=1)
        ax.plot(m, b2c, color="black", linewidth=1)
        ax.plot(m, b3c, color="black", linewidth=1)
        ax.axhline(flood_control, color="red", linestyle="--", linewidth=1)

        ax.set_title(f"{cat_name.capitalize()} year")
        ax.set_xticks(months)
        ax.set_xticklabels(MONTH_LABELS)
        ax.set_xlim(1, 13)
        ax.set_ylim(mol - 0.03 * (flood_control - mol), flood_control + 0.03 * (flood_control - mol))
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Elevation (m)")
    axes[0].legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.text(
        0.5, 0.005,
        "Buffer zone curtails HYDROPOWER ONLY (irrigation/water supply stay at 100%). "
        "Restricted zone: hydropower's discretionary target is 0%; irrigation/water "
        "supply taper linearly to 0% at their own MOL (see policy.py).",
        ha="center", va="bottom", fontsize=7.5, color="#555555", wrap=True,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def export_rule_curve_csv(data, x, out_path: str | Path):
    """
    Numeric export of the same rule curve as plot_rule_curve, for handing
    to an operator or opening in Excel -- one row per (category, month),
    boundary elevations in meters.
    """
    policy, design_discharge_hydro, design_discharge_irrig = _decode_solution_policy(data, x)

    rows = []
    for cat_idx, cat_name in enumerate(CATEGORY_NAMES):
        for m in range(12):
            rows.append({
                "category": cat_name,
                "month": m + 1,
                "b3_restricted_top_elevation_m": data.elevation_from_volume(policy.b3[cat_idx, m]),
                "b2_buffer_top_elevation_m": data.elevation_from_volume(policy.b2[cat_idx, m]),
                "b1_conservation_top_elevation_m": data.elevation_from_volume(policy.b1[cat_idx, m]),
            })
    df = pd.DataFrame(rows)

    meta_lines = [
        f"# min_operating_level_m,{data.scalars['min_operating_level']}",
        f"# max_operating_level_m,{data.scalars['max_operating_level']}",
        f"# flood_control_level_m,{data.scalars['flood_control_level']}",
        f"# design_discharge_hydro_m3s,{design_discharge_hydro}",
        f"# design_discharge_irrig_m3s,{design_discharge_irrig}",
        f"# hydro_buffer_floor_fraction,{policy.hydro_buffer_floor}",
        f"# note,irrigation/water_supply are flat at 100% through Buffer; hydropower alone tapers there",
        f"# note,irrigation/water_supply taper linearly through Restricted down to their own MOL (see policy.py)",
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(meta_lines) + "\n")
        df.to_csv(f, index=False)


# ---------------------------------------------------------------------------
# 4. Reservoir level heat map -- year (rows) x month (columns)
# ---------------------------------------------------------------------------

def plot_reservoir_level_heatmap(data, categories, x, out_path: str | Path, solution_index: int | None = None):
    """
    Year x month heat map of end-of-month reservoir elevation. Makes
    drought/wet-period clusters and any long-term trend visible at a
    glance across the whole record, including the warm-up years (shown
    for context, not excluded here the way objectives are).
    """
    result = _simulate_solution(data, categories, x)

    years_unique = np.arange(int(data.years.min()), int(data.years.max()) + 1)
    matrix = np.full((len(years_unique), 12), np.nan)
    for t in range(data.n_steps):
        y_idx = data.years[t] - years_unique[0]
        m_idx = data.months[t] - 1
        matrix[y_idx, m_idx] = result.level_m[t]

    n_years = len(years_unique)
    fig, ax = plt.subplots(figsize=(11, max(5, n_years * 0.16)))
    cmap = plt.get_cmap("RdYlBu").copy()
    cmap.set_bad("white")
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, aspect="auto", cmap=cmap, interpolation="nearest")

    ax.set_xticks(range(12))
    ax.set_xticklabels(MONTH_LABELS)
    ax.set_yticks(range(n_years))
    ax.set_yticklabels(years_unique, fontsize=max(5, min(9, 300 / n_years)))
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    title_suffix = f" -- solution {solution_index}" if solution_index is not None else ""
    ax.set_title(f"Reservoir Level Heat Map [m a.s.l.]{title_suffix}")

    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Reservoir level [m a.s.l.]")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Average monthly water balance -- stacked bar + net inflow line
# ---------------------------------------------------------------------------

def plot_water_balance(data, categories, x, out_path: str | Path, solution_index: int | None = None):
    """
    Average monthly water balance: a stacked bar of where water went each
    month (hydropower, irrigation, environmental flow, spillage,
    evaporation), against a net (naturalized) inflow line -- same
    evaluation window as the objectives (warm-up excluded). Bars won't
    necessarily sum to the inflow line in any given month: the difference
    is storage building up or drawing down that month, which is real and
    expected, not an imbalance.
    """
    result = _simulate_solution(data, categories, x)
    mask = _warmup_mask(data)
    months = data.months[mask]
    days = data.days_in_month[mask]

    def clim_vol_from_m3s(arr_m3s):
        vol = data.m3s_to_Mm3(arr_m3s[mask], days)
        return np.array([vol[months == m].mean() for m in range(1, 13)])

    def clim_vol_direct(arr_vol):
        arr_vol = arr_vol[mask]
        return np.array([arr_vol[months == m].mean() for m in range(1, 13)])

    hydro_vol = clim_vol_from_m3s(result.hydro_release_m3s)
    irrig_vol = clim_vol_from_m3s(result.irrig_release_m3s)
    ws_vol = clim_vol_from_m3s(result.ws_release_m3s)
    # Use the BYPASS portion only, not total env_release_m3s -- when environmental
    # flow is turbined, it's already inside hydro_release_m3s (a floor, not
    # additive); stacking the full total again here would double-count that
    # water in the chart. This also matches how the Excel legacy model presents
    # it ("Hydropower incl. ecological flow", not shown as a separate bar).
    env_vol = clim_vol_from_m3s(result.env_bypass_release_m3s)
    spill_vol = clim_vol_from_m3s(result.spillway_release_m3s)
    evap_vol = clim_vol_direct(result.evaporation_Mm3)
    inflow_vol = clim_vol_from_m3s(data.inflow_m3s)

    fig, ax = plt.subplots(figsize=(13, 6))
    x_pos = np.arange(12)
    bottom = np.zeros(12)
    components = [
        ("Hydropower", hydro_vol, "#2196F3"),
        ("Irrigation", irrig_vol, "#4CAF50"),
        ("Water supply", ws_vol, "#00BCD4"),
        ("Environmental flow (bypass only)", env_vol, "#9C27B0"),
        ("Spillage", spill_vol, "#E53935"),
        ("Evaporation", evap_vol, "#FF9800"),
    ]
    for label, vol, color in components:
        ax.bar(x_pos, vol, bottom=bottom, label=label, color=color, edgecolor="white", linewidth=0.5)
        bottom += vol

    ax.plot(x_pos, inflow_vol, color="#1565C0", marker="o", linewidth=2, label="Net Inflow", zorder=5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(MONTH_LABELS)
    ax.set_ylabel("Volume [Mm3]")
    ax.set_xlabel("Month")
    title_suffix = f" -- solution {solution_index}" if solution_index is not None else ""
    n_years = mask.sum() / 12.0
    ax.set_title(f"Average Monthly Water Balance{title_suffix}\n"
                 f"(mean of each calendar month across {n_years:.0f} evaluated years, "
                 f"warm-up excluded)", fontsize=12)
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Monthly climatology for one solution
# ---------------------------------------------------------------------------

def plot_solution_climatology(data, categories, x, out_path: str | Path, solution_index: int | None = None):
    result = _simulate_solution(data, categories, x)
    mask = _warmup_mask(data)
    months = data.months[mask]

    def clim(arr):
        arr = arr[mask]
        return np.array([arr[months == m].mean() for m in range(1, 13)])

    storage_clim = clim(result.storage_Mm3[1:])
    level_clim = clim(result.level_m)
    hydro_clim = clim(result.hydro_release_m3s)
    irrig_clim = clim(result.irrig_release_m3s)
    irrig_demand_clim = data.irrig_demand_m3s  # already a 12-length climatology
    ws_clim = clim(result.ws_release_m3s)
    ws_demand_clim = data.water_supply_demand_m3s
    env_clim = clim(result.env_release_m3s)
    spill_clim = clim(result.spillway_release_m3s)
    energy_clim = clim(result.energy_MWh)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    title_suffix = f" -- solution {solution_index}" if solution_index is not None else ""
    n_years = mask.sum() / 12.0
    fig.suptitle(f"Monthly climatology{title_suffix}\n"
                 f"(mean of each calendar month across {n_years:.0f} evaluated years, "
                 f"warm-up excluded)", fontsize=12)

    ax = axes[0, 0]
    ax.plot(MONTH_LABELS, level_clim, color="#2b6cb0", marker="o")
    ax.set_ylabel("Reservoir level (m)")
    ax.set_title("Level")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(MONTH_LABELS, hydro_clim, label="Hydropower release", color="#c05621", marker="o")
    ax.plot(MONTH_LABELS, env_clim, label="Environmental release", color="#38a169", marker="o")
    ax.plot(MONTH_LABELS, spill_clim, label="Spillway release", color="#805ad5", marker="o")
    ax.set_ylabel("m3/s")
    ax.set_title("Hydropower / environmental / spillway releases")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 2]
    ax.bar(MONTH_LABELS, energy_clim / 1000.0, color="#2b6cb0")
    ax.set_ylabel("GWh")
    ax.set_title("Hydropower energy")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 0]
    ax.plot(MONTH_LABELS, irrig_demand_clim, label="Demand", color="gray", linestyle="--", marker="o")
    ax.plot(MONTH_LABELS, irrig_clim, label="Delivered", color="#c05621", marker="o")
    ax.fill_between(MONTH_LABELS, irrig_clim, irrig_demand_clim, color="red", alpha=0.15, label="Shortfall")
    ax.set_ylabel("m3/s")
    ax.set_title("Irrigation: demand vs. delivered")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(MONTH_LABELS, ws_demand_clim, label="Demand", color="gray", linestyle="--", marker="o")
    ax.plot(MONTH_LABELS, ws_clim, label="Delivered", color="#00838F", marker="o")
    ax.fill_between(MONTH_LABELS, ws_clim, ws_demand_clim, color="red", alpha=0.15, label="Shortfall")
    ax.set_ylabel("m3/s")
    ax.set_title("Water supply: demand vs. delivered")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.delaxes(axes[1, 2])  # unused slot

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Windowed raw time series for one solution
# ---------------------------------------------------------------------------

def plot_solution_timeseries(data, categories, x, out_path: str | Path, n_years: int | str = 10,
                              start_offset_years: int = 0, solution_index: int | None = None):
    """
    n_years="full" (or any n_years large enough to exceed the actual
    evaluated record) plots the WHOLE evaluated period in one chart.
    Figure width scales with the actual number of years plotted (clipped
    to a sane range) so a full 70-year record stays legible instead of
    being crammed into a fixed-width image sized for a 10-year window.
    """
    result = _simulate_solution(data, categories, x)
    mask = _warmup_mask(data)
    start = np.where(mask)[0][0] + start_offset_years * 12

    n_years_requested = data.n_steps if n_years == "full" else int(n_years)
    end = min(start + n_years_requested * 12, data.n_steps)
    actual_years = (end - start) / 12.0

    dates = pd.to_datetime({"year": data.years[start:end], "month": data.months[start:end], "day": 1})

    fig_width = float(np.clip(actual_years * 0.45, 13, 55))
    fig, axes = plt.subplots(3, 1, figsize=(fig_width, 9), sharex=True)
    title_suffix = f" -- solution {solution_index}" if solution_index is not None else ""
    fig.suptitle(f"Simulation trace, {actual_years:.1f} years starting "
                 f"{data.years[start]}-{data.months[start]:02d}{title_suffix}")

    ax = axes[0]
    ax.plot(dates, result.level_m[start:end], color="#2b6cb0")
    ax.axhline(data.scalars["min_operating_level"], color="gray", linestyle=":", linewidth=1, label="MOL")
    ax.axhline(data.scalars["max_operating_level"], color="gray", linestyle="--", linewidth=1, label="FSL")
    ax.axhline(data.scalars["flood_control_level"], color="red", linestyle="--", linewidth=1, label="flood_control_level")
    ax.set_ylabel("Level (m)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(dates, result.hydro_release_m3s[start:end], label="Hydropower", color="#c05621")
    ax.plot(dates, result.irrig_release_m3s[start:end], label="Irrigation", color="#38a169")
    ax.plot(dates, result.ws_release_m3s[start:end], label="Water supply", color="#00838F")
    ax.plot(dates, result.spillway_release_m3s[start:end], label="Spillway", color="#805ad5")
    ax.set_ylabel("Release (m3/s)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(dates, data.inflow_m3s[start:end], label="Inflow", color="gray")
    ax.fill_between(dates, 0, result.irrig_shortfall_m3s[start:end], color="red", alpha=0.4, label="Irrigation shortfall")
    ax.set_ylabel("m3/s")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_xlabel("Date")

    # Yearly tick marks always; labeled interval scales with the plotted span
    # so a short --years window still gets readable labels (e.g. every year)
    # while a long/full window gets labels every 10 years as requested,
    # rather than one fixed spacing that's wrong at either extreme.
    if actual_years <= 12:
        label_interval = 1
    elif actual_years <= 30:
        label_interval = 5
    else:
        label_interval = 10
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator(label_interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(mdates.YearLocator(1))
        ax.grid(which="minor", axis="x", alpha=0.1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def select_by_min_reliability(F: pd.DataFrame, min_reliability: float) -> int:
    """
    Selects the solution with the HIGHEST energy among those meeting AT
    LEAST min_reliability (0-1) irrigation reliability. If no solution on
    the front meets the threshold, falls back to the highest-reliability
    solution available and prints a warning -- the front simply doesn't
    contain anything better in that case.
    """
    energy = -F["neg_energy_GWh_per_year"]
    reliability = -F["neg_irrig_reliability"]
    candidates = np.where(reliability >= min_reliability)[0]
    if len(candidates) == 0:
        idx = int(reliability.idxmax())
        print(
            f"WARNING: no solution on this front reaches {min_reliability:.1%} irrigation "
            f"reliability. Falling back to the highest available ({reliability[idx]:.1%}, "
            f"solution {idx})."
        )
        return idx
    idx = int(candidates[np.argmax(energy.iloc[candidates])])
    print(
        f"Selected solution {idx}: energy={energy[idx]:.2f} GWh/yr, "
        f"reliability={reliability[idx]:.1%}, spillage={F['spillage_Mm3_per_year'][idx]:.1f} Mm3/yr "
        f"(highest energy among {len(candidates)} solutions with reliability >= {min_reliability:.1%})"
    )
    return idx


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    from data_loader import case_name_from_data_dir, load_reservoir_data, prompt_for_data_folder
    from policy import compute_forecast_categories

    parser = argparse.ArgumentParser(description="Plot Pareto front + a selected solution's simulation trace.")
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to the Data folder (e.g. Data/Mandrare). If omitted, opens an interactive "
             "folder-picker dialog (falls back to the project's own Data/ folder if cancelled "
             "or tkinter unavailable). ALWAYS required (even for just the Pareto front/"
             "hypervolume plots) so results/figures can be organized per dataset -- see "
             "--output-dir/--plot-dir below.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Where pareto_X/F.csv + hypervolume_history.csv live. Default: Output/<case_name>/, "
             "where <case_name> is derived from --data-dir's folder name (e.g. Data/Mandrare -> "
             "Output/Mandrare/) -- matching where optimize.py saved them.",
    )
    parser.add_argument(
        "--plot-dir", type=str, default=None,
        help="Where to save generated figures. Default: Plot/<case_name>/, matching --output-dir's "
             "convention, so figures from different datasets never overwrite each other.",
    )
    parser.add_argument("--solution-index", type=int, default=None,
                         help="Which row of pareto_X.csv to detail (climatology + timeseries). "
                              f"If neither this nor --min-reliability is given, defaults to "
                              f"--min-reliability {DEFAULT_MIN_RELIABILITY} (see below).")
    parser.add_argument(
        "--min-reliability", type=float, default=None,
        help="Auto-select the solution with the HIGHEST energy among those meeting AT LEAST "
             "this irrigation reliability (0-1, e.g. 0.9 for 90%%). Overrides --solution-index "
             f"if both are given. DEFAULT when neither this nor --solution-index is given: "
             f"{DEFAULT_MIN_RELIABILITY} -- pass --front-only to skip solution selection entirely "
             "and only make the Pareto front + hypervolume plots.",
    )
    parser.add_argument(
        "--front-only", action="store_true",
        help="Skip solution selection/detail plots entirely -- only make the Pareto front "
             "matrix + hypervolume convergence plots, with no highlighted solution.",
    )
    parser.add_argument("--years", type=str, default="10",
                         help="Years shown in the windowed time series plot (default: 10). "
                              "Pass 'full' to plot the entire evaluated period instead of a window "
                              "(figure width scales automatically to stay legible).")
    parser.add_argument("--start-offset-years", type=int, default=0,
                         help="Years into the (post-warmup) record where the time series window starts (default: 0).")
    args = parser.parse_args()

    # data_dir is ALWAYS resolved first -- this is what determines the
    # case-named Output/Plot subfolders, regardless of whether a specific
    # solution is being detailed.
    if args.data_dir is None:
        default_data_dir = Path(__file__).resolve().parent.parent / "Data"
        data_dir = prompt_for_data_folder(default_data_dir)
    else:
        data_dir = args.data_dir

    case_name = case_name_from_data_dir(data_dir)
    project_root = Path(__file__).resolve().parent.parent
    output_dir = args.output_dir if args.output_dir is not None else str(project_root / "Output" / case_name)
    plot_dir = Path(args.plot_dir) if args.plot_dir is not None else project_root / "Plot" / case_name
    print(f"Case: {case_name}  ->  reading results from {output_dir}/, saving figures to {plot_dir}/")

    plot_dir.mkdir(parents=True, exist_ok=True)

    F, X, hv = load_pareto(output_dir)
    data = load_reservoir_data(data_dir)
    categories = compute_forecast_categories(data.months, data.inflow_m3s)

    if args.front_only:
        solution_index = None
    elif args.solution_index is not None:
        solution_index = args.solution_index
    elif args.min_reliability is not None:
        solution_index = select_by_min_reliability(F, args.min_reliability)
    else:
        # DEFAULT when nothing is specified: max energy subject to >= DEFAULT_MIN_RELIABILITY
        # irrigation reliability, rather than an arbitrary row index.
        solution_index = select_by_min_reliability(F, DEFAULT_MIN_RELIABILITY)

    plot_hypervolume(hv, plot_dir / "hypervolume_convergence.png")
    print(f"Saved {plot_dir / 'hypervolume_convergence.png'}")

    plot_pareto_matrix(F, X, plot_dir / "pareto_front_matrix.png", highlight_index=solution_index,
                       data=data, categories=categories)
    print(f"Saved {plot_dir / 'pareto_front_matrix.png'}")

    if solution_index is not None:
        x = X.iloc[solution_index].to_numpy()

        rule_curve_path = plot_dir / f"solution_{solution_index}_rule_curve.png"
        plot_rule_curve(data, x, rule_curve_path, solution_index=solution_index)
        print(f"Saved {rule_curve_path}")

        rule_curve_csv_path = Path(output_dir) / f"solution_{solution_index}_rule_curve.csv"
        export_rule_curve_csv(data, x, rule_curve_csv_path)
        print(f"Saved {rule_curve_csv_path}")

        heatmap_path = plot_dir / f"solution_{solution_index}_level_heatmap.png"
        plot_reservoir_level_heatmap(data, categories, x, heatmap_path, solution_index=solution_index)
        print(f"Saved {heatmap_path}")

        water_balance_path = plot_dir / f"solution_{solution_index}_water_balance.png"
        plot_water_balance(data, categories, x, water_balance_path, solution_index=solution_index)
        print(f"Saved {water_balance_path}")

        clim_path = plot_dir / f"solution_{solution_index}_climatology.png"
        plot_solution_climatology(data, categories, x, clim_path, solution_index=solution_index)
        print(f"Saved {clim_path}")

        ts_path = plot_dir / f"solution_{solution_index}_timeseries.png"
        years_arg = "full" if args.years.lower() == "full" else int(args.years)
        plot_solution_timeseries(data, categories, x, ts_path, n_years=years_arg,
                                  start_offset_years=args.start_offset_years, solution_index=solution_index)
        print(f"Saved {ts_path}")