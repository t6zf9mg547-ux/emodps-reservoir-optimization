# EMODPS_Reservoir_Optimization

A Python toolkit for jointly optimizing reservoir infrastructure sizing
(hydropower turbine capacity) and monthly operating rule curves. Balances
hydropower energy, irrigation reliability, and spillage under a hard
dam-safety constraint, using NSGA-II (via [pymoo](https://pymoo.org/))
following the EMODPS -- Evolutionary Multi-Objective Direct Policy Search
-- framework (Giuliani, Castelletti, Reed et al.). Driven by a physically
detailed monthly mass-balance simulator: environmental flow routing
(turbine vs. bypass), spillway and tailwater rating curves, waterway
headloss, service-specific minimum operating levels, and dry/normal/wet
forecast-adaptive rule curves. Built with [uv](https://docs.astral.sh/uv/)
for dependency management.

## What's included
EMODPS_Reservoir_Optimization/
├── Data/
│   └── template/    # example input CSVs, tracked in git (schema reference)
│                     # real project data goes directly in Data/, untracked
├── Module/          # project source code
│   ├── data_loader.py   # reads + validates all Data/ CSVs into one object
│   ├── policy.py         # zone-based rule curve operating policy
│   ├── simulator.py      # monthly mass-balance simulation
│   ├── objectives.py     # wraps the simulator into MOEA objectives + constraint
│   ├── optimize.py       # pymoo NSGA-II driver
│   ├── diagnostics.py    # per-solution failure counts/stats beyond the objectives
│   └── plot_results.py   # Pareto front, rule curve, level heat map, water balance, climatology, simulation trace figures
├── Output/          # Pareto front results per case, e.g. Output/Mandrare/ (not tracked in git)
├── Plot/            # generated figures per case, e.g. Plot/Mandrare/ (not tracked in git)
├── pyproject.toml   # project metadata + dependencies (uv-managed)
└── .gitignore       # excludes venv, cache files, Output/Plot, real Data/, OS junk, etc.

`Output/<case_name>/` and `Plot/<case_name>/` always resolve to the SAME
location at the project root, regardless of which folder you happen to
run a script from (terminal, VS Code's Run button, or the debugger) --
each script anchors to its own file location internally, not the current
working directory, specifically to avoid results silently landing in
different places depending on how you launched it. Pass
`--output-dir`/`--plot-dir` explicitly to override this.

## Setup

Install dependencies and create the virtual environment:
```bash
uv sync
```

## Running it

On a fresh clone, `Data/` only has the tracked `template/` subfolder. Copy
those into `Data/` directly to run against the example dataset, then
replace them with your real site data (both stay untracked in git):
```bash
cp Data/template/*.csv Data/
```

All commands below are run from inside `Module/`:
```bash
cd Module
```

Sanity-check your `Data/` files after editing them:
```bash
uv run python data_loader.py
```
This opens an interactive folder-picker dialog (`tkinter`) so you can
point it at any `Data/` folder, defaulting to the project's own `Data/`
if you cancel. Falls back to that default automatically in headless
environments where `tkinter` isn't available.

Run the optimization:
```bash
uv run python optimize.py --data-dir /path/to/your/Data --preset quick
uv run python optimize.py --data-dir /path/to/your/Data --preset full
```
Two named presets, or set `--pop-size`/`--n-gen` directly for full manual
control (explicit flags always override a preset):
- **`quick`** (pop=30, n_gen=30) -- fast pipeline sanity check (~1 min),
  confirms everything runs end-to-end against your data. NOT meant for
  interpreting results.
- **`full`** (pop=250, n_gen=300) -- recommended default for actual
  analysis. On a 30-year record this is roughly 16 min; on a longer
  record (e.g. 70+ years) closer to 40 min, since runtime scales with
  inflow record length.

**Population size vs. generations control different things.** Generations
determine how close the search gets to the true efficient frontier
(convergence) -- watch the printed hypervolume progression; once it
flattens, more generations won't help. Population size determines how
many distinct points get retained along that frontier (resolution) -- a
converged front with a small population can still have visible gaps
between points, which needs a bigger population, not more generations, to
fill in. If you suspect a specific gap might be a real discontinuity in
the achievable trade-off space (e.g. from a discrete policy switch) rather
than just under-sampling, a cheaper diagnostic than scaling up the whole
search is a separate run with the relevant decision variable's bounds
narrowed to bracket just that region.

Results land in `Output/<case_name>/pareto_X.csv`,
`Output/<case_name>/pareto_F.csv`, `Output/<case_name>/hypervolume_history.csv`
(e.g. `Output/Mandrare/...`), always at the project root regardless of
which folder you happen to run the script from.

Inspect any solution's detailed failure counts (irrigation shortfall
months, hydropower online/offline months, spill events, dam-safety
violations) beyond the single aggregated objective values:
```bash
uv run python diagnostics.py --data-dir /path/to/your/Data --solution-index 0
```
Diagnoses the first row of `pareto_X.csv` if it exists (falls back
to a random decision vector otherwise, for a quick sanity check before
running a full optimization). To diagnose a specific solution
programmatically: `diagnose(data, categories, x)` returns a
`DiagnosticsReport` -- see `diagnostics.py` for the full field list.

Generate figures from a saved optimization run:
```bash
uv run python plot_results.py --data-dir /path/to/your/Data
```
Always saves the **Pareto front matrix** (3 pairwise scatter plots, colored
by the third objective, with a secondary top axis showing the
approximate corresponding hydro design discharge) and the **hypervolume
convergence curve**.

By default (no extra flags needed) it ALSO picks a specific solution to
detail -- the one with the HIGHEST energy among those meeting at least
`DEFAULT_MIN_RELIABILITY` (90%) irrigation reliability (see the constant
near the top of `plot_results.py`) -- and generates:
- **operating rule curve** (PNG zone diagram + CSV export)
- **reservoir level heat map** (year x month grid, whole record including warm-up)
- **average monthly water balance** (stacked bar: hydropower/irrigation/
  environmental flow/spillage/evaporation, against a net inflow line)
- **monthly climatology** (level, releases, irrigation demand vs. delivered, energy)
- **windowed raw simulation trace** (level vs. MOL/FSL/flood_control_level,
  all release streams, inflow, irrigation shortfall)

Three ways to change which solution gets detailed:
```bash
uv run python plot_results.py --data-dir /path/to/your/Data --min-reliability 0.8   # same auto-selection, different reliability floor
uv run python plot_results.py --data-dir /path/to/your/Data --solution-index 42     # a specific row of pareto_X.csv by hand
uv run python plot_results.py --data-dir /path/to/your/Data --front-only            # skip solution selection entirely, only the front + hypervolume plots
```

Controlling the raw simulation trace window:
```bash
uv run python plot_results.py --data-dir /path/to/your/Data --years full                      # whole evaluated record instead of a 10-year window (figure width scales automatically to stay legible)
uv run python plot_results.py --data-dir /path/to/your/Data --years 15 --start-offset-years 20  # a specific 15-year window, starting 20 years into the evaluated period
```
The x-axis always shows yearly tick marks, with labels spaced to stay
readable (every year for short windows, every 5-10 years for longer ones
including `--years full`).

Flags combine freely -- e.g. `--min-reliability 0.9 --years full` details
the 90%-reliability candidate across the entire record in one command.

### VS Code: use "Run and Debug", not the plain Run button

Every script above (`optimize.py`, `diagnostics.py`, `plot_results.py`)
needs command-line arguments (`--data-dir`, `--solution-index`, etc.) to
do anything useful. VS Code has two different ways to run a Python file,
and only one of them can pass arguments:

- **The plain ▷ "Run Python File" button** (editor toolbar, or right-click
  -> Run Python File) ALWAYS runs with zero arguments, no exceptions, and
  has no knowledge of `launch.json`. For `plot_results.py` specifically,
  this silently skips the rule curve / climatology / timeseries figures
  and only produces the Pareto front matrix + hypervolume plot, since
  `--solution-index` never gets passed.
- **The Run and Debug panel** (sidebar icon, or `F5`) DOES read
  `.vscode/launch.json` and passes whatever `"args"` are configured
  there -- despite the word "Debug", this isn't about breakpoints or
  stepping through code, it's just "run with configured arguments."

**Always use the Run and Debug panel for this project.** Pick a
configuration from the dropdown at the top (e.g. "Plot results
(Mandrare, solution 0 detail)") and click the green ▷ next to it. See
`.vscode/launch.json` for the available presets, or add your own
following the same pattern.

## Dependencies

- numpy
- pandas
- pymoo

Add or remove packages as the project needs:
```bash
uv add <package>
uv remove <package>
```

## Data/template/ file inventory

| File | Status | Purpose |
|---|---|---|
| `reservoir_evac.csv` | template — REPLACE | Elevation-Volume-Area curve (any number of points) |
| `inflows_monthly.csv` | template — REPLACE | Monthly inflow record, **30+ years** (year, month, m3/s) |
| `evaporation_monthly.csv` | template — REPLACE | Monthly evaporation (mm) + seepage (Mm3), climatology |
| `irrigation_demand_monthly.csv` | template — REPLACE | Monthly irrigation demand (m3/s) + priority weight |
| `water_supply_demand_monthly.csv` | template — REPLACE | Monthly water supply (domestic/municipal) demand (m3/s), modeled the same way as irrigation |
| `hydropower_availability_monthly.csv` | template — REPLACE | Monthly derating factor for scheduled turbine maintenance |
| `environmental_flow_monthly.csv` | template — REPLACE | Mandatory monthly environmental flow (m3/s) |
| `spillway_rating_curve.csv` | template — REPLACE | Elevation-discharge spillway rating curve (FSL to flood_control_level) |
| `tailwater_rating_curve.csv` | template — REPLACE | Discharge-tailwater elevation rating curve (hydro + spillway + env-flow bypass, excludes irrigation) |
| `config_scalars.csv` | template — REPLACE (values are placeholder assumptions) | Operating levels, outlet/turbine params, bypass outlet capacity, headloss coefficient, decision variable bounds |

All template CSVs (tracked in git under `Data/template/`) contain
illustrative dummy values (including a 30-year synthetic seasonal inflow
record) so the code runs end-to-end before your real data arrives. **Every
numeric value in `config_scalars.csv` and the climatology files is
currently a placeholder assumption, not real site data** -- do not use
these for any actual design decision until replaced.
Lines starting with `#` in any CSV are treated as comments and skipped by
the loader.

## Model design notes

- **Operating policy**: zone-based rule curve (Flood / Conservation /
  Buffer / Restricted), with separate boundary sets for dry / normal /
  wet inflow-forecast categories (perfect-foresight 3-month-ahead
  classification). See `policy.py` docstring for the full zone logic.
- **Decision vector** (111 variables, fed to NSGA-II): design discharge
  for hydropower + 110 policy genes (unit hypercube, decoded into a
  guaranteed-feasible, ordered rule curve -- see `policy.decode_policy`).
  Irrigation design discharge is NOT a decision variable -- it's fixed at
  `design_discharge_irrig_m3s` in `config_scalars.csv` (recommended:
  peak monthly demand + a small margin). Unlike hydropower, capacity
  beyond peak demand buys nothing for irrigation, since delivery is
  capped at `min(mult * demand, capacity)` and reliability is capped at
  meeting a fixed demand -- there's no open-ended benefit to searching
  over it the way there is for hydropower capacity.
- **Water supply**: modeled the same way as irrigation -- a predetermined
  monthly demand (`water_supply_demand_monthly.csv`), a fixed design
  capacity (`design_discharge_water_supply_m3s`, not a decision variable,
  same reasoning as irrigation), and its own physical intake elevation
  (`min_operating_level_water_supply`). Shares irrigation's zone
  multiplier -- both are protected/curtailed together, matching the
  legacy Excel model this project was compared against (irrigation and
  water supply failed in the exact same months there). Not currently a
  4th Pareto objective -- tracked as its own reliability/shortfall metric
  in `diagnostics.py`, but its water use already affects all 3 existing
  objectives (energy, irrigation reliability, spillage) through the mass
  balance, correctly.
- **Service-specific minimum operating levels**: `min_operating_level`
  (MOL) is the absolute reservoir floor the zone policy operates within.
  `min_operating_level_hydro` and `min_operating_level_irrig` are each
  service's own physical intake elevation -- below that level the intake
  is dry and the service is forced to zero, regardless of what the zone
  policy says. Both sit at or above MOL and can differ from each other.
- **Objectives**: maximize hydropower energy, maximize irrigation
  reliability (time-based), minimize spillage. No flood-risk objective --
  a monthly timestep can't represent flood-peak routing, so flood
  protection is managed separately by the user with a sub-daily study.
- **Dam safety**: `flood_control_level` (the absolute ceiling) is a hard
  pymoo constraint, not an objective -- storage must never exceed it.
  Spill itself is not a policy decision; it's computed physically from
  `spillway_rating_curve.csv` at the current pool elevation.
- **Environmental flow**: defined per calendar month (`environmental_flow_monthly.csv`,
  not a single fixed value), always prioritized ahead of hydropower/
  irrigation allocation; routed through the turbine or a dedicated
  bypass outlet depending on `environmental_flow_turbined` in
  `config_scalars.csv`, with maintenance-derated turbine availability
  from `hydropower_availability_monthly.csv`. In the Restricted zone
  specifically, the turbine has no discretionary generation target, but
  if `environmental_flow_turbined=True` it still runs at EXACTLY the
  environmental flow rate (not more) whenever physically able to --
  free energy from water that has to be released anyway for compliance.
  If `environmental_flow_turbined=False`, the turbine stays off in the
  Restricted zone and the environmental flow goes through the bypass.
- **Hydropower head**: net head = gross head (reservoir level − tailwater
  level) − waterway headloss, with headloss = `alpha_headloss_coeff` ×
  discharge² (`config_scalars.csv`). Tailwater level comes from
  `tailwater_rating_curve.csv` as a function of downstream discharge =
  hydropower release + spillway release + the environmental-flow
  **bypass** portion only (irrigation excluded; if environmental flow is
  routed through the turbine that month it's already inside the
  hydropower discharge, not added again -- see `env_bypass_release_m3s`
  in `simulator.SimulationResult`).
- **Diagnostics**: `diagnostics.py` reruns the simulator for any decision
  vector and reports failure counts/stats (irrigation shortfall months,
  hydropower online/offline months, spill events, dam-safety violation
  months) beyond the single aggregated objective values -- useful once
  you're inspecting individual Pareto solutions.

## Notes

- `[tool.uv] package = false` in `pyproject.toml` marks this as a scripts
  project rather than an installable package -- required so `uv sync`/
  `uv add` don't try (and fail) to build a wheel.
- `Output/` and `Plot/` are excluded from git, since results are
  regenerated. Real data placed directly in `Data/` is also excluded --
  only `Data/template/` (the example/schema-reference CSVs) is tracked.
  Adjust `.gitignore` if you want this handled differently.
- Performance: ~13 ms/evaluation on a 30-year record, ~31 ms/evaluation
  on a 71.9-year real record (runtime scales with inflow record length).
  See "Running it" above for preset timings. Evaluations are currently
  serial; pymoo supports parallelizing via `elementwise_runner` if this
  becomes a bottleneck.