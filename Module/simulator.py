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

from policy import PolicyParams, delivery_multipliers

KW_PER_M3S_M = 9.81  # rho * g / 1000, standard hydropower power formula: P[kW] = 9.81 * Q * H * eta


@dataclass
class SimulationResult:
    storage_Mm3: np.ndarray                    # (T+1,) end-of-month storage (index 0 = initial)
    level_m: np.ndarray                         # (T,) end-of-month elevation
    hydro_release_m3s: np.ndarray                # (T,)
    irrig_release_m3s: np.ndarray                # (T,)
    env_release_m3s: np.ndarray                  # (T,) actually delivered (turbine + bypass)
    env_bypass_release_m3s: np.ndarray            # (T,) portion of env_release_m3s NOT through the turbine
    spillway_release_m3s: np.ndarray              # (T,) physical spillway discharge (elevation-driven)
    total_release_m3s: np.ndarray                # (T,)
    energy_MWh: np.ndarray                       # (T,)
    evaporation_Mm3: np.ndarray                   # (T,) monthly evaporation loss volume
    irrig_shortfall_m3s: np.ndarray               # (T,) demand - delivered, >= 0
    dam_safety_violation_Mm3: np.ndarray          # (T,) > 0 only if spillway capacity was insufficient


def simulate(
    data,
    policy: PolicyParams,
    categories: np.ndarray,
    design_discharge_hydro: float,
    design_discharge_irrig: float,
) -> SimulationResult:
    T = data.n_steps
    days = data.days_in_month

    mol_vol = data.volume_from_elevation(data.scalars["min_operating_level"])
    fsl_vol = data.volume_from_elevation(data.scalars["max_operating_level"])
    safety_vol = data.volume_from_elevation(data.scalars["flood_control_level"])
    mol_hydro = data.scalars["min_operating_level_hydro"]
    mol_irrig = data.scalars["min_operating_level_irrig"]
    min_head = data.scalars["min_hydropower_head"]
    turbine_eff = data.scalars["turbine_efficiency"]
    alpha = data.scalars["alpha_headloss_coeff"]
    env_turbined = bool(data.scalars["environmental_flow_turbined"])
    bypass_cap_m3s = data.scalars["bypass_outlet_capacity_m3s"]
    max_release_cap_m3s = data.scalars["max_release_capacity_m3s"]  # non-spillway outlets combined

    storage = np.empty(T + 1)
    storage[0] = data.volume_from_elevation(data.scalars["initial_level"])

    level = np.empty(T)
    hydro_release = np.zeros(T)
    irrig_release = np.zeros(T)
    env_release = np.zeros(T)
    env_bypass_release = np.zeros(T)
    spillway_release = np.zeros(T)
    total_release = np.zeros(T)
    energy = np.zeros(T)
    evaporation = np.zeros(T)
    irrig_shortfall = np.zeros(T)
    dam_safety_violation = np.zeros(T)

    for t in range(T):
        m = data.months[t] - 1  # 0-indexed month
        cat = categories[t]
        d_days = days[t]

        inflow_vol = data.m3s_to_Mm3(data.inflow_m3s[t], d_days)

        # --- losses (evaporation uses start-of-month area; see module docstring) ---
        area = data.area_from_volume(storage[t])
        evap_vol = data.evaporation_mm[m] * 1e-3 * area  # mm * km2 -> Mm3
        seepage_vol = data.seepage_Mm3[m]

        s_avail = max(storage[t] + inflow_vol - evap_vol - seepage_vol, 0.0)
        evaporation[t] = evap_vol

        # --- zone-based delivery targets ---
        hydro_mult, irrig_mult = delivery_multipliers(s_avail, m, cat, policy)

        # --- environmental flow: always prioritized (monthly requirement) ---
        env_flow_t = data.env_flow_m3s[m]
        env_target_vol = data.m3s_to_Mm3(env_flow_t, d_days)
        env_actual_vol = min(env_target_vol, s_avail)

        # --- turbine feasibility: maintenance derating + minimum NET head + intake MOL ---
        # Preliminary estimate only -- spillway release isn't known yet (resolved
        # below), so tailwater/headloss here use a trial discharge. The final
        # energy calculation re-evaluates both precisely once hydro release AND
        # spillway release are final (see module docstring).
        #
        # IMPORTANT: "physically available" (can the turbine run AT ALL this
        # month) is now separate from "is there a discretionary generation
        # target" (hydro_mult > 0, i.e. NOT in the Restricted zone). This
        # matters because in the Restricted zone, hydro_mult is 0 but the
        # turbine can still be physically capable of running -- and if
        # environmental_flow_turbined is True, it SHOULD run, at exactly the
        # environmental flow rate (free energy from water that has to pass
        # through anyway), not be forced to 0 just because there's no
        # discretionary target. If environmental_flow_turbined is False, the
        # discretionary target being 0 in the Restricted zone naturally
        # results in hydro_release_m3s = 0 further below, as intended.
        avail_frac = data.hydro_availability[m]
        pre_release_level = data.elevation_from_volume(s_avail)
        trial_hydro_m3s = (hydro_mult * design_discharge_hydro) if hydro_mult > 0 else env_flow_t
        prelim_tailwater = data.tailwater_elevation_from_discharge(trial_hydro_m3s)
        prelim_gross_head = pre_release_level - prelim_tailwater
        prelim_net_head = prelim_gross_head - alpha * trial_hydro_m3s ** 2
        turbine_physically_available = (
            (avail_frac > 0)
            and (prelim_net_head >= min_head)
            and (pre_release_level >= mol_hydro)
        )
        turbine_capacity_m3s = avail_frac * design_discharge_hydro
        hydro_target_m3s = (hydro_mult * design_discharge_hydro) if turbine_physically_available else 0.0

        if turbine_physically_available and env_turbined:
            # environmental flow is a floor under hydro release, not additive; capped by turbine capacity.
            # In the Restricted zone hydro_target_m3s is 0, so this correctly passes EXACTLY the
            # environmental flow through the turbine (not more) -- see note above.
            hydro_release_m3s = min(max(hydro_target_m3s, env_flow_t), turbine_capacity_m3s)
            env_via_turbine_vol = min(env_actual_vol, data.m3s_to_Mm3(hydro_release_m3s, d_days))
            bypass_needed_vol = max(0.0, env_actual_vol - env_via_turbine_vol)
        else:
            # either the turbine physically can't run this month, or environmental flow
            # isn't routed through it -- always bypass in both cases
            hydro_release_m3s = hydro_target_m3s
            env_via_turbine_vol = 0.0
            bypass_needed_vol = env_actual_vol

        bypass_actual_vol = min(bypass_needed_vol, data.m3s_to_Mm3(bypass_cap_m3s, d_days))
        env_delivered_vol = env_via_turbine_vol + bypass_actual_vol  # REPORTING metric only -- see note below
        hydro_release_vol = data.m3s_to_Mm3(hydro_release_m3s, d_days)

        # --- irrigation: target = mult * demand, capped by design (intake) capacity,
        # forced to zero below the irrigation intake's own minimum operating level ---
        irrigation_physically_available = pre_release_level >= mol_irrig
        irrig_target_m3s = (
            min(irrig_mult * data.irrig_demand_m3s[m], design_discharge_irrig)
            if irrigation_physically_available else 0.0
        )
        irrig_release_vol = data.m3s_to_Mm3(irrig_target_m3s, d_days)

        # Mass-balance release: hydro_release_vol ALREADY includes any turbined
        # environmental-flow water (a floor under hydro release, not additive --
        # see above), so it must NOT be added again here. Only hydro_release_vol
        # (turbine, whatever its composition) + bypass_actual_vol (a physically
        # separate outlet) + irrig_release_vol are physically distinct
        # withdrawals from storage. env_delivered_vol is kept purely as a
        # REPORTING metric (env_release_m3s below) -- summing it into the mass
        # balance here would double-count the turbined portion of the
        # environmental flow, over-depleting storage every month the turbine
        # runs with environmental_flow_turbined=True.
        baseline_vol = hydro_release_vol + bypass_actual_vol + irrig_release_vol
        # cap at BOTH water availability AND the physical capacity of the combined
        # power/irrigation outlets (max_release_capacity_m3s -- distinct from the
        # spillway, which is handled separately below)
        max_baseline_vol = min(s_avail, data.m3s_to_Mm3(max_release_cap_m3s, d_days))
        if baseline_vol > max_baseline_vol:
            scale = (max_baseline_vol / baseline_vol) if baseline_vol > 0 else 0.0
            env_delivered_vol *= scale
            hydro_release_vol *= scale
            bypass_actual_vol *= scale
            irrig_release_vol *= scale
            baseline_vol = max_baseline_vol

        s_after_baseline = s_avail - baseline_vol

        # --- spillway routing (elevation-driven, not a policy decision) ---
        spill_vol = 0.0
        if s_after_baseline > fsl_vol:
            # Fixed-point iteration: spill depends on elevation, elevation depends on
            # spill. Start from the pre-spill (post-baseline-release) level as the
            # first estimate, then refine using the average of pre/post-spill levels
            # (same averaging approach used for the hydropower head below).
            s_end_estimate = s_after_baseline
            for _ in range(4):
                rep_storage = 0.5 * (s_after_baseline + s_end_estimate)
                rep_level = data.elevation_from_volume(rep_storage)
                spill_rate_m3s = data.spillway_discharge_from_elevation(rep_level)
                spill_vol_estimate = data.m3s_to_Mm3(spill_rate_m3s, d_days)
                # spillway physically stops discharging once the pool recedes to FSL
                s_end_estimate = max(s_after_baseline - spill_vol_estimate, fsl_vol)
            spill_vol = s_after_baseline - s_end_estimate

            if s_end_estimate > safety_vol:
                # Even the rating curve's discharge at this elevation can't hold the
                # ceiling -- genuine spillway-undersized failure. Do NOT clip storage
                # here: clipping would silently break the mass balance and hide the
                # failure from the hard constraint in optimize.py.
                dam_safety_violation[t] = s_end_estimate - safety_vol

        total_release_vol = baseline_vol + spill_vol
        storage[t + 1] = s_avail - total_release_vol

        # --- record ---
        level[t] = data.elevation_from_volume(storage[t + 1])
        hydro_release[t] = data.Mm3_to_m3s(hydro_release_vol, d_days)
        irrig_release[t] = data.Mm3_to_m3s(irrig_release_vol, d_days)
        env_release[t] = data.Mm3_to_m3s(env_delivered_vol, d_days)
        env_bypass_release[t] = data.Mm3_to_m3s(bypass_actual_vol, d_days)
        spillway_release[t] = data.Mm3_to_m3s(spill_vol, d_days)
        total_release[t] = data.Mm3_to_m3s(total_release_vol, d_days)

        irrig_shortfall[t] = max(0.0, data.irrig_demand_m3s[m] - irrig_release[t])

        # --- final energy calculation: precise tailwater + headloss, using the
        # ACTUAL final hydro + spillway + env-bypass releases (all now known).
        # NOTE: if env flow was turbined this month, it's already inside
        # hydro_release[t] -- only the bypass portion is added separately,
        # to avoid double-counting it in the downstream discharge.
        downstream_discharge_m3s = hydro_release[t] + spillway_release[t] + env_bypass_release[t]
        tailwater_final = data.tailwater_elevation_from_discharge(downstream_discharge_m3s)
        avg_level = 0.5 * (pre_release_level + level[t])
        gross_head = max(avg_level - tailwater_final, 0.0)
        headloss = alpha * hydro_release[t] ** 2
        net_head = max(gross_head - headloss, 0.0)
        power_kW = KW_PER_M3S_M * hydro_release[t] * net_head * turbine_eff
        energy[t] = power_kW * d_days * 24.0 / 1000.0  # MWh

    return SimulationResult(
        storage_Mm3=storage,
        level_m=level,
        hydro_release_m3s=hydro_release,
        irrig_release_m3s=irrig_release,
        env_release_m3s=env_release,
        env_bypass_release_m3s=env_bypass_release,
        spillway_release_m3s=spillway_release,
        total_release_m3s=total_release,
        energy_MWh=energy,
        evaporation_Mm3=evaporation,
        irrig_shortfall_m3s=irrig_shortfall,
        dam_safety_violation_Mm3=dam_safety_violation,
    )