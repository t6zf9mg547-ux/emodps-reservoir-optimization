"""
simulator.py

Monthly mass-balance simulator. Combines the zone-based rule curve policy
(policy.py) with fixed rules for environmental flow routing, hydropower
maintenance derating, and flood/dam-safety management.

Flood / dam-safety design
--------------------------
Two storage ceilings are distinguished:
  - FSL (max_operating_level) : top of the policy-controlled live storage,
    i.e. the top of the Flood zone as defined in policy.py; also the
    spillway crest elevation (zero discharge at/below FSL).
  - flood_control_level        : the ABSOLUTE dam-safety ceiling. Storage
    must NEVER exceed this. This is exposed to the optimizer as a hard
    pymoo CONSTRAINT (see optimize.py), not a soft objective -- matching
    "it is not possible to go above it."

No downstream flood-risk objective is modeled here -- monthly-mean flow
cannot represent flood-peak routing, so per-conversation decision, that
is handled separately (a daily/sub-daily study), out of scope for this
long-term design/policy optimization.

Spill is NOT a policy decision. Once storage rises above FSL, the
spillway discharges physically according to spillway_rating_curve.csv
(elevation -> discharge), same as a real uncontrolled/gated spillway --
there is no "choice" involved, unlike the hydropower/irrigation targets.
Because spillway discharge depends on pool elevation, and elevation
depends on how much has already spilled THIS month, the routing is
solved with a short fixed-point iteration (spill -> new elevation ->
updated spill estimate), analogous to the pre/post-release level
averaging already used for the hydropower head calculation.

Release priority each month, in order:
  1. Environmental flow (always prioritized, via turbine or bypass per
     environmental_flow_turbined).
  2. 100% hydropower + 100% irrigation target (already the Flood-zone
     policy rule -- see policy.delivery_multipliers).
  3. If storage is still above FSL after (1)+(2): the spillway
     physically discharges according to the rating curve at the
     (iteratively solved) representative pool elevation.
  4. If even the rating curve's discharge at flood_control_level is not
     enough to keep ending storage <= flood_control_level: this is a
     genuine design failure (spillway undersized for that inflow), and
     is flagged (dam_safety_violation_Mm3 > 0) rather than silently
     clipped -- clipping would break the mass balance and hide the
     failure from the optimizer's hard constraint.

Service-specific minimum operating levels
-------------------------------------------
min_operating_level (MOL) is the absolute reservoir floor -- the bottom
of the live-storage range the zone policy operates within. Separately,
min_operating_level_hydro and min_operating_level_irrig represent each
service's own PHYSICAL intake elevation: below that level the intake is
dry and the service is forced to zero, regardless of what the zone
policy says. These sit at or above MOL (a physical intake is normally
higher than the absolute dead-storage bottom) and can differ from each
other, since the two intakes are rarely at the same elevation.

Restricted-zone hydropower rule (refined)
--------------------------------------------
"Physically available" (can the turbine run AT ALL this month --
maintenance derating, minimum net head, min_operating_level_hydro) is
kept SEPARATE from "is there a discretionary generation target"
(hydro_mult > 0, i.e. NOT in the Restricted zone). In the Restricted
zone, hydro_mult is 0, but the turbine can still be physically capable
of running:
  - environmental_flow_turbined = True  -> turbine runs at EXACTLY the
    environmental flow rate (a floor with a target of 0, so it's not
    additive) -- free energy from water that has to be released anyway
    for compliance, not a discretionary generation decision.
  - environmental_flow_turbined = False -> turbine stays at 0, the
    environmental flow goes through the bypass instead, exactly as in
    every other zone when not turbined.
This also means environmental_flow_turbined now correctly applies in
the Restricted zone -- previously it was ignored there by mistake, and
environmental flow always went through the bypass regardless of the
setting.

Monthly-timestep limitation (explicit)
----------------------------------------
A monthly mass balance CANNOT represent within-month flood-peak routing
-- a multi-day flood wave is smoothed into one monthly mean inflow. This
model captures "seasonal/chronic" flood risk (can the reservoir safely
absorb an anomalously wet month/run of months without exceeding the
dam-safety ceiling) but is NOT a substitute for a sub-daily flood-routing
study against a design flood (PMF, 100-yr hydrograph) for spillway
sizing -- that remains a separate hydraulic engineering exercise.

Other modeling assumptions (flagged explicitly, easy to revisit):
  - Evaporation uses the surface area at START-of-month storage (not
    iterated to convergence with end-of-month storage).
  - Hydropower net head = gross head (reservoir level - tailwater level)
    minus a waterway headloss DH = alpha * hydro_discharge^2 (alpha from
    config_scalars.csv). Tailwater level comes from tailwater_rating_curve.csv
    as a function of downstream discharge = hydro release + spillway
    release + environmental-flow BYPASS portion only (irrigation is
    excluded; if environmental flow was routed through the turbine that
    month instead, it's already counted within the hydropower discharge,
    not added again -- see env_bypass_release_m3s in SimulationResult).
  - Minimum-head turbine feasibility is checked using a PRELIMINARY
    estimate: pre-release (start-of-month, post-inflow) reservoir level,
    and tailwater/headloss evaluated at the ZONE TARGET hydro discharge
    alone (spillway release isn't known yet at this point in the monthly
    routine -- it's resolved afterward, see spillway routing above). The
    FINAL energy calculation below re-evaluates tailwater and headloss
    using the actual, final hydro + spillway releases once both are
    known, so energy itself is accurate; only the feasibility gate uses
    the cheaper approximation. This mirrors the same
    "approximate-then-refine" pattern already used for spillway routing.
  - Hydropower energy uses the AVERAGE of pre- and post-release levels
    as the representative monthly reservoir level (before subtracting
    tailwater + headloss).
  - Irrigation delivery target = irrig_mult * monthly demand, capped by
    the design (intake) discharge decision variable. Hydropower target =
    hydro_mult * design discharge (capacity), since hydropower has no
    exogenous "demand" the way irrigation does.
  - Environmental flow requirement is now MONTHLY (environmental_flow_monthly.csv),
    not a single fixed scalar -- it's looked up per timestep by calendar month.

Performance note
-----------------
The monthly loop is inherently sequential (storage[t+1] depends on
storage[t]), so it cannot be vectorized across time. It CAN be written to
avoid pandas and minimize Python overhead per step, which is what's done
here. If profiling later shows this loop dominates optimizer runtime,
numba's @njit is the natural next step (the function is already written
in a numba-friendly style: plain scalars and arrays, no object attribute
access inside the hot loop beyond simple lookups) -- flag if you want
that added now instead of after profiling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from policy import PolicyParams, delivery_multipliers

KW_PER_M3S_M = 9.81  # rho * g / 1000, standard hydropower power formula: P[kW] = 9.81 * Q * H * eta


@njit(cache=True)
def _simulate_core(
    months, cats, inflow_m3s, days,
    evac_elevation, evac_volume, evac_area,
    spillway_elevation, spillway_discharge_m3s,
    tailwater_discharge_m3s, tailwater_elevation_m,
    evaporation_mm, seepage_Mm3, irrig_demand_m3s, water_supply_demand_m3s,
    hydro_availability, env_flow_m3s,
    policy_b2, policy_b3, hydro_buffer_floor, irrig_buffer_floor,
    fsl_vol, safety_vol, mol_hydro, mol_irrig, mol_water_supply, min_head, turbine_eff, alpha,
    env_turbined, bypass_cap_m3s, max_release_cap_m3s,
    design_discharge_hydro, design_discharge_irrig, design_discharge_water_supply, initial_storage,
):
    """
    Numba-jitted core of the monthly loop. Line-for-line equivalent of the
    pure-Python version below (see simulate() and the module docstring for
    the full explanation of every rule implemented here) -- this function
    exists ONLY for speed, not to change any behavior. Takes plain arrays
    and scalars (no ReservoirData/PolicyParams objects, since numba can't
    jit arbitrary Python class methods) -- delivery_multipliers' logic is
    inlined directly using policy_b2/policy_b3 + the two floor fractions,
    and every data.<method>() lookup becomes a direct np.interp() call
    against the corresponding raw array.

    Water supply is modeled the same way as irrigation (predetermined
    demand, own physical intake MOL, own fixed design capacity) but shares
    irrig_mult -- both are protected/curtailed together, matching the
    legacy Excel model this project was compared against.
    """
    T = months.shape[0]
    storage = np.empty(T + 1)
    storage[0] = initial_storage
    level = np.empty(T)
    hydro_release = np.zeros(T)
    irrig_release = np.zeros(T)
    ws_release = np.zeros(T)
    env_release = np.zeros(T)
    env_bypass_release = np.zeros(T)
    spillway_release = np.zeros(T)
    total_release = np.zeros(T)
    energy = np.zeros(T)
    evaporation = np.zeros(T)
    irrig_shortfall = np.zeros(T)
    ws_shortfall = np.zeros(T)
    dam_safety_violation = np.zeros(T)

    for t in range(T):
        m = months[t] - 1
        cat = cats[t]
        d_days = days[t]

        inflow_vol = inflow_m3s[t] * d_days * 86400.0 / 1.0e6

        area = np.interp(storage[t], evac_volume, evac_area)
        evap_vol = evaporation_mm[m] * 1e-3 * area
        seepage_vol = seepage_Mm3[m]

        s_avail = max(storage[t] + inflow_vol - evap_vol - seepage_vol, 0.0)
        evaporation[t] = evap_vol

        # --- zone-based delivery targets (delivery_multipliers, inlined) ---
        b2v = policy_b2[cat, m]
        b3v = policy_b3[cat, m]
        if s_avail >= b2v:
            hydro_mult = 1.0
            irrig_mult = 1.0
        elif s_avail >= b3v:
            frac = (s_avail - b3v) / max(b2v - b3v, 1e-9)
            hydro_mult = hydro_buffer_floor + frac * (1.0 - hydro_buffer_floor)
            irrig_mult = irrig_buffer_floor + frac * (1.0 - irrig_buffer_floor)
        else:
            hydro_mult = 0.0
            irrig_mult = 1.0

        env_flow_t = env_flow_m3s[m]
        env_target_vol = env_flow_t * d_days * 86400.0 / 1.0e6
        env_actual_vol = min(env_target_vol, s_avail)

        avail_frac = hydro_availability[m]
        pre_release_level = np.interp(s_avail, evac_volume, evac_elevation)
        if hydro_mult > 0:
            trial_hydro_m3s = hydro_mult * design_discharge_hydro
        else:
            trial_hydro_m3s = env_flow_t
        prelim_tailwater = np.interp(trial_hydro_m3s, tailwater_discharge_m3s, tailwater_elevation_m)
        prelim_gross_head = pre_release_level - prelim_tailwater
        prelim_net_head = prelim_gross_head - alpha * trial_hydro_m3s ** 2
        turbine_physically_available = (
            (avail_frac > 0) and (prelim_net_head >= min_head) and (pre_release_level >= mol_hydro)
        )

        turbine_capacity_m3s = avail_frac * design_discharge_hydro
        hydro_target_m3s = (hydro_mult * design_discharge_hydro) if turbine_physically_available else 0.0

        if turbine_physically_available and env_turbined:
            hydro_release_m3s = min(max(hydro_target_m3s, env_flow_t), turbine_capacity_m3s)
            hydro_release_vol_trial = hydro_release_m3s * d_days * 86400.0 / 1.0e6
            env_via_turbine_vol = min(env_actual_vol, hydro_release_vol_trial)
            bypass_needed_vol = max(0.0, env_actual_vol - env_via_turbine_vol)
        else:
            hydro_release_m3s = hydro_target_m3s
            env_via_turbine_vol = 0.0
            bypass_needed_vol = env_actual_vol

        bypass_actual_vol = min(bypass_needed_vol, bypass_cap_m3s * d_days * 86400.0 / 1.0e6)
        env_delivered_vol = env_via_turbine_vol + bypass_actual_vol
        hydro_release_vol = hydro_release_m3s * d_days * 86400.0 / 1.0e6

        irrigation_physically_available = pre_release_level >= mol_irrig
        irrig_target_m3s = (
            min(irrig_mult * irrig_demand_m3s[m], design_discharge_irrig)
            if irrigation_physically_available else 0.0
        )
        irrig_release_vol = irrig_target_m3s * d_days * 86400.0 / 1.0e6

        # Water supply: modeled the same way as irrigation -- predetermined demand,
        # own physical intake MOL, own fixed design capacity, but shares irrig_mult
        # (both are protected/curtailed together -- matches the legacy Excel model,
        # where irrigation and water supply failed in the exact same months).
        ws_physically_available = pre_release_level >= mol_water_supply
        ws_target_m3s = (
            min(irrig_mult * water_supply_demand_m3s[m], design_discharge_water_supply)
            if ws_physically_available else 0.0
        )
        ws_release_vol = ws_target_m3s * d_days * 86400.0 / 1.0e6

        baseline_vol = hydro_release_vol + bypass_actual_vol + irrig_release_vol + ws_release_vol
        max_baseline_vol = min(s_avail, max_release_cap_m3s * d_days * 86400.0 / 1.0e6)
        if baseline_vol > max_baseline_vol:
            scale = (max_baseline_vol / baseline_vol) if baseline_vol > 0 else 0.0
            env_delivered_vol *= scale
            hydro_release_vol *= scale
            bypass_actual_vol *= scale
            irrig_release_vol *= scale
            ws_release_vol *= scale
            baseline_vol = max_baseline_vol

        s_after_baseline = s_avail - baseline_vol

        spill_vol = 0.0
        if s_after_baseline > fsl_vol:
            s_end_estimate = s_after_baseline
            for _ in range(4):
                rep_storage = 0.5 * (s_after_baseline + s_end_estimate)
                rep_level = np.interp(rep_storage, evac_volume, evac_elevation)
                spill_rate_m3s = np.interp(rep_level, spillway_elevation, spillway_discharge_m3s)
                spill_vol_estimate = spill_rate_m3s * d_days * 86400.0 / 1.0e6
                s_end_estimate = max(s_after_baseline - spill_vol_estimate, fsl_vol)
            spill_vol = s_after_baseline - s_end_estimate
            if s_end_estimate > safety_vol:
                dam_safety_violation[t] = s_end_estimate - safety_vol

        total_release_vol = baseline_vol + spill_vol
        storage[t + 1] = s_avail - total_release_vol

        level[t] = np.interp(storage[t + 1], evac_volume, evac_elevation)
        hydro_release[t] = hydro_release_vol * 1.0e6 / (d_days * 86400.0)
        irrig_release[t] = irrig_release_vol * 1.0e6 / (d_days * 86400.0)
        ws_release[t] = ws_release_vol * 1.0e6 / (d_days * 86400.0)
        env_release[t] = env_delivered_vol * 1.0e6 / (d_days * 86400.0)
        env_bypass_release[t] = bypass_actual_vol * 1.0e6 / (d_days * 86400.0)
        spillway_release[t] = spill_vol * 1.0e6 / (d_days * 86400.0)
        total_release[t] = total_release_vol * 1.0e6 / (d_days * 86400.0)

        irrig_shortfall[t] = max(0.0, irrig_demand_m3s[m] - irrig_release[t])
        ws_shortfall[t] = max(0.0, water_supply_demand_m3s[m] - ws_release[t])

        downstream_discharge_m3s = hydro_release[t] + spillway_release[t] + env_bypass_release[t]
        tailwater_final = np.interp(downstream_discharge_m3s, tailwater_discharge_m3s, tailwater_elevation_m)
        avg_level = 0.5 * (pre_release_level + level[t])
        gross_head = max(avg_level - tailwater_final, 0.0)
        headloss = alpha * hydro_release[t] ** 2
        net_head = max(gross_head - headloss, 0.0)
        power_kW = KW_PER_M3S_M * hydro_release[t] * net_head * turbine_eff
        energy[t] = power_kW * d_days * 24.0 / 1000.0

    return (storage, level, hydro_release, irrig_release, ws_release, env_release, env_bypass_release,
            spillway_release, total_release, energy, evaporation, irrig_shortfall, ws_shortfall,
            dam_safety_violation)


@dataclass
class SimulationResult:
    storage_Mm3: np.ndarray                    # (T+1,) end-of-month storage (index 0 = initial)
    level_m: np.ndarray                         # (T,) end-of-month elevation
    hydro_release_m3s: np.ndarray                # (T,)
    irrig_release_m3s: np.ndarray                # (T,)
    ws_release_m3s: np.ndarray                    # (T,) water supply, modeled same as irrigation
    env_release_m3s: np.ndarray                  # (T,) actually delivered (turbine + bypass)
    env_bypass_release_m3s: np.ndarray            # (T,) portion of env_release_m3s NOT through the turbine
    spillway_release_m3s: np.ndarray              # (T,) physical spillway discharge (elevation-driven)
    total_release_m3s: np.ndarray                # (T,)
    energy_MWh: np.ndarray                       # (T,)
    evaporation_Mm3: np.ndarray                   # (T,) monthly evaporation loss volume
    irrig_shortfall_m3s: np.ndarray               # (T,) demand - delivered, >= 0
    ws_shortfall_m3s: np.ndarray                  # (T,) demand - delivered, >= 0
    dam_safety_violation_Mm3: np.ndarray          # (T,) > 0 only if spillway capacity was insufficient


def simulate(
    data,
    policy: PolicyParams,
    categories: np.ndarray,
    design_discharge_hydro: float,
    design_discharge_irrig: float,
    design_discharge_water_supply: float | None = None,
    initial_level_override: float | None = None,
) -> SimulationResult:
    """
    design_discharge_water_supply defaults to data.scalars["design_discharge_water_supply_m3s"]
    if not given (matching how design_discharge_irrig is typically pulled from
    data.scalars["design_discharge_irrig_m3s"] by callers) -- it's fixed, not a
    decision variable, same reasoning as irrigation.

    initial_level_override, if given, replaces data.scalars["initial_level"]
    for this call only (data itself is never mutated) -- used by
    closure.find_initial_level_closure() to search for a self-consistent
    starting level without needing a separate copy of the whole ReservoirData
    object for every trial.
    """
    if design_discharge_water_supply is None:
        design_discharge_water_supply = data.scalars["design_discharge_water_supply_m3s"]

    mol_vol = data.volume_from_elevation(data.scalars["min_operating_level"])
    fsl_vol = data.volume_from_elevation(data.scalars["max_operating_level"])
    safety_vol = data.volume_from_elevation(data.scalars["flood_control_level"])
    initial_level = initial_level_override if initial_level_override is not None else data.scalars["initial_level"]
    initial_storage = data.volume_from_elevation(initial_level)

    (storage, level, hydro_release, irrig_release, ws_release, env_release, env_bypass_release,
     spillway_release, total_release, energy, evaporation, irrig_shortfall, ws_shortfall,
     dam_safety_violation) = _simulate_core(
        data.months, categories, data.inflow_m3s, data.days_in_month,
        data.evac_elevation, data.evac_volume, data.evac_area,
        data.spillway_elevation, data.spillway_discharge_m3s,
        data.tailwater_discharge_m3s, data.tailwater_elevation_m,
        data.evaporation_mm, data.seepage_Mm3, data.irrig_demand_m3s, data.water_supply_demand_m3s,
        data.hydro_availability, data.env_flow_m3s,
        policy.b2, policy.b3, policy.hydro_buffer_floor, policy.irrig_buffer_floor,
        fsl_vol, safety_vol,
        data.scalars["min_operating_level_hydro"], data.scalars["min_operating_level_irrig"],
        data.scalars["min_operating_level_water_supply"],
        data.scalars["min_hydropower_head"], data.scalars["turbine_efficiency"],
        data.scalars["alpha_headloss_coeff"], bool(data.scalars["environmental_flow_turbined"]),
        data.scalars["bypass_outlet_capacity_m3s"], data.scalars["max_release_capacity_m3s"],
        design_discharge_hydro, design_discharge_irrig, design_discharge_water_supply, initial_storage,
    )

    return SimulationResult(
        storage_Mm3=storage,
        level_m=level,
        hydro_release_m3s=hydro_release,
        irrig_release_m3s=irrig_release,
        ws_release_m3s=ws_release,
        env_release_m3s=env_release,
        env_bypass_release_m3s=env_bypass_release,
        spillway_release_m3s=spillway_release,
        total_release_m3s=total_release,
        energy_MWh=energy,
        evaporation_Mm3=evaporation,
        irrig_shortfall_m3s=irrig_shortfall,
        ws_shortfall_m3s=ws_shortfall,
        dam_safety_violation_Mm3=dam_safety_violation,
    )