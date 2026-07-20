"""
policy.py

Zone-based rule curve operating policy, chosen over an RBF/EMODPS policy
because it is directly interpretable and auditable by an operator: at any
time, "which zone am I in, what does the rule say to deliver".

Structure
---------
Live storage (between MOL and FSL) is split into 4 zones, top to bottom:
    Flood        : above the Conservation top boundary (b1), up to FSL/surcharge
                   -> 100% target supply to hydropower & irrigation (fixed rule),
                      PLUS flood/spill release handled by simulator.py
    Conservation : between b2 and b1   -> 100% target supply to hydropower & irrigation
    Buffer       : between b3 and b2   -> deliveries taper linearly (decision variables)
    Restricted   : between MOL and b3  -> hydropower OFF (fixed 0%), irrigation
                   targets 100% of demand (fixed rule) -- i.e. all available
                   water goes to irrigation; actual delivery is whatever mass
                   balance allows, so shortfalls here are real and are what
                   the irrigation reliability objective measures.

Only the Buffer zone taper is decision-variable-controlled (hydro_buffer_floor,
irrig_buffer_floor). Flood, Conservation, and Restricted zone rules are fixed
by design choice, not tuned by the optimizer.

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
pure numpy indexing, no interpolation, no branching in Python.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_CATEGORIES = 3          # 0=dry, 1=normal, 2=wet
CATEGORY_NAMES = ("dry", "normal", "wet")
N_MONTHS = 12
N_BOUNDARY_FRACTIONS = 3  # increments defining b3 < b2 < b1 within live storage
N_FLOOR_PARAMS = 2        # hydro_buffer_floor, irrig_buffer_floor
                           # (restricted-zone rule is fixed: hydro=0, irrig=1 -- not decided by MOEA)

N_POLICY_PARAMS = N_CATEGORIES * N_MONTHS * N_BOUNDARY_FRACTIONS + N_FLOOR_PARAMS
# = 3 * 12 * 3 + 2 = 110.  (+2 design-discharge variables live outside this
# module, appended to the full MOEA decision vector by optimize.py)


@dataclass
class PolicyParams:
    """Decoded, ready-to-use policy. All boundary arrays are in Mm3."""
    b1: np.ndarray  # (N_CATEGORIES, N_MONTHS) top of Conservation zone
    b2: np.ndarray  # (N_CATEGORIES, N_MONTHS) top of Buffer zone
    b3: np.ndarray  # (N_CATEGORIES, N_MONTHS) top of Restricted zone (bottom of Buffer)
    hydro_buffer_floor: float
    irrig_buffer_floor: float


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
        irrig_buffer_floor=float(floor_genes[1]),
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


def delivery_multipliers(storage_Mm3: float, month_idx0: int, category: int, policy: PolicyParams):
    """
    Given current storage (Mm3, BEFORE release), the calendar month index
    (0=Jan..11=Dec), and the forecast category, return (hydro_mult,
    irrig_mult) in [0, 1] -- the fraction of full demand/design-capacity
    each service should target this month. Flood-zone spill and mass
    balance / constraint enforcement are handled in simulator.py, which
    calls this function once per timestep.
    """
    b2 = policy.b2[category, month_idx0]
    b3 = policy.b3[category, month_idx0]

    if storage_Mm3 >= b2:  # Conservation zone or Flood zone: 100% target for both services
        return 1.0, 1.0

    if storage_Mm3 >= b3:  # Buffer zone: taper from 1.0 (at b2) down to buffer floor (at b3)
        frac = (storage_Mm3 - b3) / max(b2 - b3, 1e-9)
        hydro = policy.hydro_buffer_floor + frac * (1.0 - policy.hydro_buffer_floor)
        irrig = policy.irrig_buffer_floor + frac * (1.0 - policy.irrig_buffer_floor)
        return hydro, irrig

    # Restricted zone (storage < b3, down to MOL): hydropower off, all water to irrigation.
    # Fixed rule, not decision-variable-controlled. Irrigation TARGETS full demand here;
    # whether it actually gets it depends on real water availability in the mass balance --
    # that shortfall is exactly what the irrigation reliability objective should capture.
    return 0.0, 1.0
