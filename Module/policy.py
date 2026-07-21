"""
policy.py

Zone-based rule curve operating policy, chosen over an RBF/EMODPS policy
because it is directly interpretable and auditable by an operator: at any
time, "which zone am I in, what does the rule say to deliver".

Structure
---------
Live storage (between MOL and FSL) is split into 4 zones, top to bottom:
    Flood        : above the Conservation top boundary (b1), up to FSL/surcharge
                   -> 100% target supply to all three services (fixed rule),
                      PLUS flood/spill release handled by simulator.py
    Conservation : between b2 and b1   -> 100% target supply to all three services
    Buffer       : between b3 and b2   -> HYDROPOWER ONLY tapers linearly
                   (decision variable: hydro_buffer_floor). Irrigation and
                   water supply stay at 100% target throughout this zone --
                   hydropower absorbs the entire Buffer-zone curtailment
                   alone, protecting the other two services longer.
    Restricted   : between each service's own physical MOL and b3 ->
                   hydropower's discretionary target is 0% (fixed rule --
                   still subject to the environmental-flow floor in
                   simulator.py if environmental_flow_turbined=True).
                   Irrigation and water supply now ALSO taper -- linearly
                   from 100% at b3 down to 0% at their own physical MOL
                   (min_operating_level_irrig / min_operating_level_water_supply)
                   -- a graceful decline into the hard physical cutoff,
                   rather than the discontinuous "100% then cliff to 0%"
                   shape an earlier version of this policy had.

Design history / rejected alternatives (kept for the record)
--------------------------------------------------------------
An earlier version of this policy applied the SAME Buffer-zone taper to
irrigation as to hydropower, then had irrigation/water-supply snap back to
100% in the Restricted zone. That shape is NON-MONOTONIC (delivery briefly
recovers as storage keeps falling) and was rejected as indefensible
operating practice once diagrammed and reviewed.

The current design (irrigation/water-supply flat through Buffer, tapering
only in Restricted down to their own MOL) was chosen over two alternatives:
  1. Flat 100% all the way to a hard MOL cliff (no Restricted-zone taper
     at all) -- monotonic and simple, but gives irrigation/water-supply NO
     graduated warning before an abrupt 100% -> 0% cutoff.
  2. The version implemented here: reuses the EXISTING MOL boundary as the
     taper's zero-point, at NO cost to the decision vector -- no new
     optimizable parameters.

POSSIBLE FUTURE IMPLEMENTATION (not yet done, flagging for the record):
  A genuinely tunable Restricted-zone floor per service (analogous to
  hydro_buffer_floor) -- e.g. irrig_restricted_floor_fraction -- would let
  the optimizer control the STEEPNESS of irrigation/water-supply's decline
  through the Restricted zone (gentle vs. sharp), rather than the taper's
  shape being entirely fixed by wherever MOL happens to sit. This would
  add new decision variables (growing N_POLICY_PARAMS) and hasn't been
  implemented -- revisit if the current fixed-slope taper looks too
  abrupt or too gentle in practice on real results.

There are 3 independent boundary sets (b1, b2, b3), one per inflow-forecast
category (dry / normal / wet), and each category has its own set of 12
monthly boundary values. The forecast category is determined once, up
front, from PERFECT FORESIGHT of the actual 3-month-ahead cumulative
inflow relative to that calendar month's historical terciles -- this is
a standard simplification for design-stage studies (see e.g. Loucks &
van Beek, 2005, ch. 4 on rule curve derivation) but is an optimistic
upper bound on what a real forecast-driven policy could achieve. Flagging
this explicitly since you may want to swap in a real forecast model later
-- the interface (an array of per-timestep category labels) is designed
so that swap doesn't touch anything else.

Performance note
-----------------
compute_forecast_categories() depends only on the (fixed) historical
inflow record, NOT on the decision vector, so it is computed ONCE outside
the optimization loop and cached. decode_policy() is cheap (just array
reshuffling) and can be called once per MOEA evaluation. The per-timestep
delivery-multiplier lookup used inside the simulator's monthly loop is
pure numpy indexing, no interpolation, no branching in Python (and is, in
fact, inlined directly into simulator.py's numba-jitted core for speed --
delivery_multipliers() below is the readable reference version, kept in
sync with that inlined logic, but not itself on the hot path).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_CATEGORIES = 3          # 0=dry, 1=normal, 2=wet
CATEGORY_NAMES = ("dry", "normal", "wet")
N_MONTHS = 12
N_BOUNDARY_FRACTIONS = 3  # increments defining b3 < b2 < b1 within live storage
N_FLOOR_PARAMS = 1        # hydro_buffer_floor only -- irrigation/water supply no longer
                           # taper in the Buffer zone (see module docstring), so they have
                           # no Buffer-zone floor parameter of their own. Their Restricted-
                           # zone taper reuses the existing MOL boundary, at no cost to the
                           # decision vector -- see "POSSIBLE FUTURE IMPLEMENTATION" above.

N_POLICY_PARAMS = N_CATEGORIES * N_MONTHS * N_BOUNDARY_FRACTIONS + N_FLOOR_PARAMS
# = 3 * 12 * 3 + 1 = 109.  (+1 design-discharge variable lives outside this
# module, appended to the full MOEA decision vector by optimize.py)


@dataclass
class PolicyParams:
    """Decoded, ready-to-use policy. All boundary arrays are in Mm3."""
    b1: np.ndarray  # (N_CATEGORIES, N_MONTHS) top of Conservation zone
    b2: np.ndarray  # (N_CATEGORIES, N_MONTHS) top of Buffer zone
    b3: np.ndarray  # (N_CATEGORIES, N_MONTHS) top of Restricted zone (bottom of Buffer)
    hydro_buffer_floor: float


def decode_policy(x: np.ndarray, mol_volume_Mm3: float, fsl_volume_Mm3: float) -> PolicyParams:
    """
    Decode the policy portion of a flat decision vector (length
    N_POLICY_PARAMS, all genes in [0, 1] -- set these as the pymoo variable
    bounds) into physically valid, ordered zone boundaries.

    Ordering (MOL < b3 < b2 < b1 < FSL) is guaranteed by construction: the
    3 boundary genes per (category, month) are treated as successive
    fractions of the REMAINING live storage above the previous boundary,
    not as absolute levels. This means every point in the unit hypercube
    decodes to a feasible, ordered rule curve -- no infeasible individuals
    from bad boundary ordering, which would otherwise waste a lot of MOEA
    evaluations.
    """
    x = np.asarray(x, dtype=float)
    assert x.shape[0] == N_POLICY_PARAMS, f"expected {N_POLICY_PARAMS} policy genes, got {x.shape[0]}"

    n_boundary_genes = N_CATEGORIES * N_MONTHS * N_BOUNDARY_FRACTIONS
    boundary_genes = x[:n_boundary_genes].reshape(N_CATEGORIES, N_MONTHS, N_BOUNDARY_FRACTIONS)
    floor_genes = x[n_boundary_genes:]

    live_range = fsl_volume_Mm3 - mol_volume_Mm3

    f3 = boundary_genes[..., 0]                       # fraction of live_range for b3 (>= MOL)
    f2 = boundary_genes[..., 1]                        # fraction of REMAINING range for b2 (>= b3)
    f1 = boundary_genes[..., 2]                        # fraction of REMAINING range for b1 (>= b2)

    b3 = mol_volume_Mm3 + f3 * live_range
    b2 = b3 + f2 * (mol_volume_Mm3 + live_range - b3)
    b1 = b2 + f1 * (mol_volume_Mm3 + live_range - b2)

    return PolicyParams(
        b1=b1, b2=b2, b3=b3,
        hydro_buffer_floor=float(floor_genes[0]),
    )


def compute_forecast_categories(months: np.ndarray, inflow_m3s: np.ndarray) -> np.ndarray:
    """
    Perfect-foresight dry/normal/wet classification, computed ONCE from the
    historical inflow record (see module docstring for rationale).

    For each calendar month m, looks at the historical distribution (across
    all years in the record) of the 3-month-ahead cumulative inflow
    (t+1, t+2, t+3), and classifies each timestep into a tercile:
        0 = dry    (<= 33rd percentile for that calendar month)
        1 = normal
        2 = wet    (>= 67th percentile for that calendar month)

    Edge handling: the final 1-2 timesteps of the record don't have a full
    3-month lookahead. ASSUMPTION: these are classified using whatever
    lookahead months are available (1 or 2 instead of 3) rather than
    dropped, since dropping would shorten the usable simulation record.
    Flagging this -- if you'd rather truncate the record by 3 months
    instead, tell me and I'll change this.
    """
    T = inflow_m3s.shape[0]
    lookahead_sum = np.full(T, np.nan)
    for t in range(T):
        window = inflow_m3s[t + 1: min(t + 4, T)]
        if window.size > 0:
            lookahead_sum[t] = window.mean() * window.size  # scale-consistent even if truncated

    categories = np.full(T, 1, dtype=int)  # default 'normal' where undefined
    for m in range(1, N_MONTHS + 1):
        mask = (months == m) & ~np.isnan(lookahead_sum)
        if mask.sum() < 3:
            continue  # not enough history to form terciles; leave as 'normal'
        vals = lookahead_sum[mask]
        q33, q67 = np.percentile(vals, [33.33, 66.67])
        idx = np.where(mask)[0]
        categories[idx[lookahead_sum[idx] <= q33]] = 0
        categories[idx[(lookahead_sum[idx] > q33) & (lookahead_sum[idx] < q67)]] = 1
        categories[idx[lookahead_sum[idx] >= q67]] = 2
    return categories


def delivery_multipliers(
    storage_Mm3: float, month_idx0: int, category: int, policy: PolicyParams,
    mol_irrig_volume_Mm3: float, mol_water_supply_volume_Mm3: float,
):
    """
    Given current storage (Mm3, BEFORE release), the calendar month index
    (0=Jan..11=Dec), the forecast category, and each service's own MOL
    (converted to a volume, since storage/b2/b3 are all in volume terms)
    -- return (hydro_mult, irrig_mult, ws_mult) in [0, 1], the fraction of
    full demand/design-capacity each service should target this month.
    Flood-zone spill and mass balance / constraint enforcement are handled
    in simulator.py, which calls this function once per timestep.

    Reference version only -- see module docstring's Performance note:
    the actual hot-path logic is inlined directly into simulator.py's
    numba-jitted core, kept in sync with this function by hand.
    """
    b2 = policy.b2[category, month_idx0]
    b3 = policy.b3[category, month_idx0]

    if storage_Mm3 >= b2:  # Conservation zone or Flood zone: 100% target for all three services
        return 1.0, 1.0, 1.0

    if storage_Mm3 >= b3:  # Buffer zone: HYDROPOWER ONLY tapers, from 1.0 (at b2) to its floor (at b3).
        # Irrigation and water supply stay flat at 100% here -- hydropower
        # absorbs the entire Buffer-zone curtailment alone (see module
        # docstring for the design rationale and rejected alternatives).
        frac = (storage_Mm3 - b3) / max(b2 - b3, 1e-9)
        hydro = policy.hydro_buffer_floor + frac * (1.0 - policy.hydro_buffer_floor)
        return hydro, 1.0, 1.0

    # Restricted zone (storage < b3): hydropower's discretionary target is 0
    # (fixed rule -- still subject to the environmental-flow floor elsewhere
    # in simulator.py). Irrigation and water supply now taper linearly from
    # 1.0 at b3 down to 0.0 at their OWN physical MOL -- a graceful decline
    # into the hard physical cutoff already enforced by simulator.py,
    # rather than staying at 100% and then cliff-dropping to 0% at MOL.
    def _restricted_taper(mol_volume_Mm3: float) -> float:
        if storage_Mm3 <= mol_volume_Mm3:
            return 0.0
        return min((storage_Mm3 - mol_volume_Mm3) / max(b3 - mol_volume_Mm3, 1e-9), 1.0)

    irrig = _restricted_taper(mol_irrig_volume_Mm3)
    ws = _restricted_taper(mol_water_supply_volume_Mm3)
    return 0.0, irrig, ws