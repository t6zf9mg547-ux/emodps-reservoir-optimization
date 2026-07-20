"""
objectives.py

Wraps simulator.py to produce the 3 objectives and 1 hard constraint that
optimize.py's pymoo problem will use. Per-conversation decision: no
flood-risk objective (monthly timestep can't represent flood-peak
dynamics; handled separately by the user). Flood/dam safety is instead a
hard CONSTRAINT (flood_control_level must never be exceeded), consistent
with "it is not possible to go above it."

Objectives (all returned in MINIMIZE form, since pymoo minimizes by
convention -- maximize objectives are negated):
  1. -mean_annual_energy_GWh        (maximize hydropower energy)
  2. -irrigation_reliability        (maximize; time-based, see below)
  3. mean_annual_spillage_Mm3        (minimize -- water lost over the
                                       spillway, distinct from the hard
                                       dam-safety constraint: spill can be
                                       nonzero with zero constraint
                                       violation, e.g. mild FSL exceedance
                                       events)

Irrigation reliability: TIME-BASED (Hashimoto et al., 1982) -- the
fraction of evaluated months where demand is fully met (shortfall below
a small numerical tolerance), not a volumetric delivered/demanded ratio.
ASSUMPTION, flagged: if you'd rather use a volumetric ratio instead (or
in addition -- e.g. as a 4th objective or a resilience/vulnerability
metric), that's a small, isolated change here.

Warm-up handling: the first `warmup_years` (config_scalars.csv) of the
simulation are EXCLUDED from all objectives AND from the constraint
evaluation, since they carry transient bias from the arbitrary initial
storage level. Both use the same evaluation window for consistency.

Constraint (pymoo convention: g(x) <= 0 is feasible):
  g1 = max(dam_safety_violation_Mm3 over evaluated months)
     (exactly 0 when flood_control_level was never exceeded; > 0 -- and
     therefore infeasible -- the instant it is, by any amount)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from policy import PolicyParams, decode_policy
from simulator import simulate

IRRIGATION_SHORTFALL_TOLERANCE_M3S = 1e-6


@dataclass
class ObjectiveResult:
    F: np.ndarray  # (3,) [-energy_GWh_per_year, -irrigation_reliability, spillage_Mm3_per_year]
    G: np.ndarray  # (1,) [max_dam_safety_violation_Mm3]  (<=0 feasible)


def _warmup_mask(data) -> np.ndarray:
    warmup_months = int(round(data.scalars["warmup_years"] * 12))
    mask = np.ones(data.n_steps, dtype=bool)
    mask[:warmup_months] = False
    if not np.any(mask):
        raise ValueError(
            f"warmup_years ({data.scalars['warmup_years']}) leaves no months to evaluate "
            f"objectives on (record is {data.n_steps / 12:.1f} years) -- shorten warmup_years "
            "or lengthen the inflow record."
        )
    return mask


def evaluate(
    data,
    categories: np.ndarray,
    x: np.ndarray,
) -> ObjectiveResult:
    """
    x: full 111-gene MOEA decision vector = [design_discharge_hydro] +
    [110 policy genes], each gene in [0, 1] for the policy portion -- see
    optimize.py for how variable bounds map design_discharge_hydro's gene
    to its real [min, max] range before this function is called (this
    function expects it already scaled to physical units, policy genes
    still in [0, 1]).

    Irrigation design discharge is NOT a decision variable -- it's fixed
    at design_discharge_irrig_m3s in config_scalars.csv (per-conversation
    decision: unlike hydropower, more irrigation capacity beyond peak
    monthly demand buys nothing, since irrig_target is capped at
    min(mult * demand, capacity) and reliability is capped at meeting a
    FIXED demand).
    """
    design_discharge_hydro = x[0]
    design_discharge_irrig = data.scalars["design_discharge_irrig_m3s"]
    policy_genes = x[1:]

    mol_vol = data.volume_from_elevation(data.scalars["min_operating_level"])
    fsl_vol = data.volume_from_elevation(data.scalars["max_operating_level"])
    policy = decode_policy(policy_genes, mol_vol, fsl_vol)

    result = simulate(data, policy, categories, design_discharge_hydro, design_discharge_irrig)

    mask = _warmup_mask(data)
    n_years = mask.sum() / 12.0

    energy_GWh_per_year = result.energy_MWh[mask].sum() / 1000.0 / n_years

    shortfall = result.irrig_shortfall_m3s[mask]
    reliability = float(np.mean(shortfall < IRRIGATION_SHORTFALL_TOLERANCE_M3S))

    days = data.days_in_month[mask]
    spillage_Mm3_per_year = data.m3s_to_Mm3(result.spillway_release_m3s[mask], days).sum() / n_years

    max_violation = float(result.dam_safety_violation_Mm3[mask].max()) if mask.any() else 0.0

    F = np.array([-energy_GWh_per_year, -reliability, spillage_Mm3_per_year])
    G = np.array([max_violation])
    return ObjectiveResult(F=F, G=G)