# NoLoftFix

Reference-profile midcourse lofting for Nuclear Option ARH missiles, after
CEAS-GNC-2026-016 (Tahk, Kim, Jeong and Lee, KAIST, EuroGNC 2026).

The plugin disables the game's `loftAmount` loft bias and replaces it with an
offline-optimised reference trajectory. The missile follows that trajectory by aimpoint
placement, without any change to the terminal guidance phase.

This document describes the offline simulator in `sim/`, the table format that the
plugin loads, the command line of `sim/grid_dymos.py` and `sim/plot_trajectories.py`, and
the runtime plugin in `mod/`. `README.md` is the short public description.

- [1. Overview](#1-overview)
- [2. Requirements](#2-requirements)
- [3. Build and installation](#3-build-and-installation)
- [4. Runtime plugin](#4-runtime-plugin)
  - [4.1 Hooks](#41-hooks)
  - [4.2 Configuration](#42-configuration)
  - [4.3 Target motion](#43-target-motion)
  - [4.4 Manoeuvre authority](#44-manoeuvre-authority)
- [5. Simulator](#5-simulator)
- [6. Data model](#6-data-model)
- [7. Quick start](#7-quick-start)
- [8. `grid_dymos.py` command reference](#8-grid_dymospy-command-reference)
  - [8.1 Solving cells](#81-solving-cells)
  - [8.2 Constraint settings](#82-constraint-settings)
  - [8.3 Reading, checking and merging](#83-reading-checking-and-merging)
  - [8.4 Re-optimisation options](#84-re-optimisation-options)
  - [8.5 Cell filters](#85-cell-filters)
- [9. Cell file fields](#9-cell-file-fields)
- [10. Constraints and floors](#10-constraints-and-floors)
- [11. Warm starts](#11-warm-starts)
- [12. Re-optimisation](#12-re-optimisation)
  - [12.1 Relationship to `--check`](#121-relationship-to---check)
  - [12.2 Behaviour](#122-behaviour)
  - [12.3 Commands](#123-commands)
  - [12.4 Cells that do not improve](#124-cells-that-do-not-improve)
- [13. Validation with `--check`](#13-validation-with---check)
  - [13.1 Reachable range](#131-reachable-range)
  - [13.2 Limitations of validation](#132-limitations-of-validation)
- [14. Run monitoring](#14-run-monitoring)
  - [14.1 Stopping a run](#141-stopping-a-run)
- [15. Performance and accuracy](#15-performance-and-accuracy)
- [16. Plotting with `plot_trajectories.py`](#16-plotting-with-plot_trajectoriespy)
- [17. File layout](#17-file-layout)
- [18. Limitations](#18-limitations)

## 1. Overview

`ARHSeeker.Seek()` adds a loft bias to the lead vector:

```csharp
float num = Mathf.Min(timeToTarget * timeToTarget * 4.905f * loftAmount,
                      targetDist * loftAmount);
leadVectorWithAccel += num * Vector3.up;
```

The AAM-29 Scythe prefab sets `loftAmount` to 0.7; the code default is 0.2. The bias is a
function of time-to-go alone, and `timeToTarget` is refreshed at 1 Hz in `SlowChecks()`
and decremented on every fixed tick. Since the bias scales with `t_go²`, its effect varies
strongly with range: below approximately 45 nmi the aimpoint is below the launch altitude
and the missile dives; in the 50 to 65 nmi band the resulting profile is usable; at longer
ranges the missile over-lofts and arrives with little energy. A Python replica of the game
physics using the parsed prefab coefficients reproduces the reported apex altitudes to
within a few percent, including the reported 141 000 ft peak at 105 nmi against 137 900 ft
in simulation.

The aimpoint is also recomputed from a predicted intercept point that moves on each
datalink update, so the missile follows a bias rather than a persistent path.

The plugin instead solves the trajectory offline and flies its shape:

```
xi    = x / X_f                     x = downrange travelled, X_f = launch to PIP distance
h_ref = sum_k poly_a[k] * xi^k      polynomial of order 7, per the paper
x_ap  = x + lookahead * speed
ap    = launch + dir * x_ap
ap.y  = poly(x_ap / X_f) + trackGain * (h_ref - h_actual)
```

The APG acceleration law `a_com = N·V·(λ_ref − λ)` from the paper is not implemented
directly. The game does not expose an acceleration command, and pursuit toward a
lookahead point on a reference path is equivalent to trajectory following for this
purpose.

## 2. Requirements

* **Plugin**: .NET SDK, Windows or Linux, and the game's managed assemblies (see
  section 3).
* **Simulator**: Python 3 with the packages listed in `sim/requirements.txt` (numpy,
  scipy, matplotlib, dymos, openmdao), plus the parsed coefficient file
  `sim/coeffs_*.json` for the missile being solved.

## 3. Build and installation

```bash
cd mod
dotnet build                                            # netstandard2.1
dotnet build -p:ManagedDir="/path/to/NuclearOption_Data/Managed"
```

The build compiles against the game's own managed assemblies: `Assembly-CSharp`,
`Newtonsoft.Json` and `Mirage`. `mod/Directory.Build.props` locates these in the standard
Steam installation directory, or in the server variant of it, and stops with the
`-p:ManagedDir=` message above if `Assembly-CSharp.dll` cannot be found. Publicising is
applied to `Assembly-CSharp` and `Mirage`, which is what makes the private
`loftAmount`, `knownPos`, `targetDist` and `terminalRange` fields visible to the plugin.
The first build requires network access for the NuGet restore (BepInEx.Core and the
publiciser are fetched from nuget.org and nuget.bepinex.dev).

Deployment is separate from compilation. Copy `mod/Local.props.example.txt` to
`mod/Local.props` and set `NuclearOptionDir` to the game installation. Without that file
`dotnet build` compiles the plugin but does not copy it into the game, and the game
continues to load whichever DLL was last installed by hand. `Local.props` is not intended
to be committed.

The installed layout is:

```
<game>/BepInEx/plugins/NOLoftFix/
    NOLoftFix.dll
    Tables/
        loft_table_scythe.json
        loft_table_scimitar.json
```

Every `*.json` file below `Tables/` is loaded recursively, one file per missile. Each
table names the missile it applies to; at runtime `TableRouter` matches that name against
the missile component (case-insensitive substring match, with `default` as a fallback,
and a single table used for all missiles). Parsing uses Newtonsoft rather than Unity's
`JsonUtility`, which silently ignores fields whose types it does not support and reports
no error when it does so.

## 4. Runtime plugin

### 4.1 Hooks

The plugin installs two Harmony postfixes. There are no transpilers.

| Hook                           | Effect                                                                                                                                                             |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `ARHSeeker.Initialize` postfix | Sets `loftAmount` to 0, which disables the bias in both `Seek()` and `SlowChecks()`. Captures the launch geometry from `seeker.knownPos` and selects a table cell. |
| `ARHSeeker.Seek` postfix       | Runs after the game's `SetAimpoint()` and overwrites the aimpoint with the profile lookahead point.                                                                |

**Side effect of `loftAmount = 0`.** `ARHSeeker.SlowChecks` computes `targetDist` inside
the `if (loftAmount > 0f)` block, so setting the field to zero prevents `targetDist` from
being updated and it remains at `0f`. The game reads that field in two places unrelated
to lofting: the close-in-under-jamming branch (`targetDist < 5000f`, which becomes true at
any range) and jink evasion (`targetDist > terminalRange`, which becomes never true and
disables `ApplyJink`; the AAM-29 sets `jinkEvasion.amount = 0.2`). `LoftGuidance.MaintainTargetDist`
recomputes `targetDist` from `knownPos` on every tick, which restores the value
`SlowChecks` would have written. Guidance itself uses a live slant range and does not
read the field.

Both hooks are wrapped in try/catch handlers. After five failures the plugin disarms
itself and the game's original behaviour resumes.

### 4.2 Configuration

The configuration file is `BepInEx/config/NOLoftFix.cfg`. Editing it does not require a
restart: the plugin watches the file and calls `ConfigFile.Reload()`, and each setting is
read through its `ConfigEntry` rather than copied at load time. The three settings marked
*restart* are exceptions.

| Key                               | Default | Meaning                                                                                                                                                                       |
| --------------------------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Following/LookaheadSeconds`      | 3.0     | Lookahead distance in seconds. The primary tracking parameter: values that are too small make the missile cut corners, and values that are too large make it lag the profile. |
| `Following/MaxTrackAngleDeg`      | 12      | Maximum deviation of the aimpoint from the profile slope. Bounds the recovery rate so that a decelerating missile cannot be commanded into an increasingly steep dive.        |
| `Following/TrackGain`             | 0.5     | Proportional gain on altitude tracking error. A value of 0 gives pure lookahead.                                                                                              |
| `Following/TargetAltLagRateMPerS` | 250     | Maximum rate at which the reference endpoint may move. Must exceed the target's own vertical rate.                                                                            |
| `Following/FloorAltM`             | 60      | Lower bound on the aimpoint altitude. The plugin has no terrain query.                                                                                                        |
| `Lookup/MaxNormalisedDistance`    | 0.5     | Distance from any solved cell beyond which the plugin does not engage and the game's original lofting applies.                                                                |
| `General/MinCellTerminalSpeedMs`  | 800     | Cells predicting a slower arrival are discarded at load time. *restart*                                                                                                       |
| `General/RejectFlatCells`         | true    | Discards cells that never lofted (apex equal to launch altitude). *restart*                                                                                                   |
| `General/Enabled`                 | true    | Master switch, checked before patching. *restart*                                                                                                                             |
| `General/VerboseLogging`          | true    | Enables the `[flight]`, `[terminal]` and `[law]` log lines.                                                                                                                   |
| `Following/FlightLogIntervalS`    | 1       | Interval in seconds between those lines.                                                                                                                                      |
| `Following/DebugDraw`             | true    | Draws the remaining reference profile with a `LineRenderer`.                                                                                                                  |

Twelve keys remain of the original twenty-four. The removed entries (`Neighbours`,
`IdwPower`, `MinLookaheadM`, `MaxLookaheadM`, `XfLagRateMPerS`, `BumpFadeSeconds`,
`RebaseWindowSeconds`, `PipChangeThresholdM`) were internal values that were never
changed in practice and are now constants in `Core/LoftSettings.cs`. Two settings were
removed outright: `MaxApexM`, a runtime apex ceiling, and the `Relookup*` pair together
with the cell re-selection code they controlled. Re-selecting a cell in flight steps the
profile, because `BumpCoeffs` pins the endpoints and not the shape.

### 4.3 Target motion

A solved profile is an absolute polynomial pinned to the launch and target altitudes it
was generated for, so it cannot be used unchanged when the target moves. The polynomial
is decomposed into a bump term, defined as the polynomial minus the straight line joining
its own endpoints, which is zero at both ends by construction. The runtime reference is
then rebuilt as

```
h(s) = h_launch + (h_pip_now - h_launch) * s + B(s)
```

so the offline shape is preserved while both endpoints track the live predicted intercept
point. The intercept point is obtained from `seeker.knownPos` through
`TargetCalc.GetLeadVectorWithAccel(...)`, the same call used by `ARHSeeker.Seek`, so the
lead is preserved.

Three mechanisms keep the reference continuous:

1. `s` is normalised by `xfUsed`, a first-order lag (`XfLagRateMPerS`) on the true
   launch-to-PIP distance. Without the lag, a target turning away increases `X_f`,
   decreases `s`, and the missile is commanded as if it were earlier in the profile and
   must climb.
2. Progress is computed as `xf0 − remaining range` rather than as a projection onto the
   current axis. A projection collapses toward zero if the target turns 90 degrees, which
   would command the launch altitude in mid-flight.
3. Cell re-selection cross-fades over `BumpFadeSeconds` rather than switching directly,
   so selecting a different table cell cannot produce a step in the reference.

The behaviour was verified against the shipped `LoftTable.cs` (`BumpCoeffs`, `Poly.Eval`):
`B(0)` and `B(1)` are zero; the polynomial rebuilt at 0.25, 0.5 and 0.75 equals the
original to 1e-6; a retarget to a moved intercept point leaves `h(0)` and `h(1)` exact;
and a 10 m target climb changes the reference by at most 10.0 m. The counters
`pipRefreshes` and `xfDrift` are written to the verbose flight log.

### 4.4 Manoeuvre authority

The offline objective maximises terminal speed subject to `nload <= gLimit`. That is an
upper bound, so a missile with little available load factor satisfies it trivially and the
optimiser will produce trajectories in air too thin to turn in. With the AAM-29 curves
(`Cl_max` 0.999, `finArea` 0.5, 100 kg dry mass):

| Altitude                               | Game density | Density vs ISA | Load available at 1000 m/s |
| -------------------------------------- | ------------ | -------------- | -------------------------- |
| 25 km (82 000 ft)                      | 0.0277       | 0.59x          | 7.1 g                      |
| 30 km (98 400 ft)                      | 0.0049       | 0.20x          | 1.3 g                      |
| 32 km (105 000 ft)                     | 0.0018       | 0.09x          | 0.45 g                     |
| 35 km and above (115 000 ft and above) | 0.000526     | 0.04x          | 0.13 g                     |

`airDensityAltitude` flattens at 0.000526 above its last key (35 km), so above that
altitude the missile has negligible available load factor at any speed. This is the
failure mode described in CEAS-GNC-2026-016: "if the target maneuvers to produce any PIP
change, the trajectory correction of the missile can be a challenging task to handle at
high altitudes."

The `min_authority_g` constraint (section 10) addresses this by requiring a minimum
available load factor along the path. On a 110 km shot it reduced the apex from 34 439 m
to 25 496 m and the terminal speed from 797 m/s to 710 m/s.

Tables generated before the constraint existed should be regenerated. A runtime `MaxApexM`
ceiling was previously available as a stopgap; it was removed because clamping a profile
after the fact distorts a trajectory that the optimiser shaped deliberately, and no
measured failure required it.

## 5. Simulator

`sim/grid_dymos.py` solves individual trajectories offline and writes them to disk. It
supports several modes, all selected by command-line arguments: solving a grid, solving a
reachable-range envelope, reporting status, validating results, re-optimising cells, and
merging cells into a table.

A worker solves a *ray*: all cells that share a launch and target condition, ordered by
range. Each cell after the first is warm-started from its neighbour's solution.

## 6. Data model

Three terms are used throughout the simulator and this document.

**Grid spec.** A JSON file in `sim/grids/` containing a list of axes (`range_km`,
`launch_alt_m`, `launch_speed_ms`, `target_alt_m`, `target_speed_ms`) and the settings
used for every solve. One spec defines one grid, whose cells are the cross product of its
axes. `sim/grids/scythe-v2.json` defines 480 cells.

**Cell.** One solved trajectory, identified by its launch and target geometry and stored
in `runs/<name>/cells/<key>.json`. The key lists the geometry values in `AXIS_ORDER`,
for example `75_4000_320_200_0` for 75 km range, 4 000 m launch altitude, 320 m/s launch
speed, 200 m target altitude and 0 m/s target speed. The target altitude field is in
metres.

**Table.** The file that the plugin ships, `mod/Tables/loft_table_*.json`, containing the
polynomial coefficients of every usable cell. `--merge` builds it from a run directory.
The merge never copies trajectories, so the table size does not depend on whether the run
used `--store-solutions`. If an envelope run is present, each cell also carries
`exp_max_range_km` and `reachable`.

All quantities are metric: kilometres, metres and metres per second. Tables written before
the metric change still load; the loader converts the legacy `range_nmi`, `launch_alt_ft`
and `launch_kt` keys.

All states, controls and ODE inputs declare units (`m`, `m/s`, `rad`, `kg`, `s`), which
allows OpenMDAO to check the connections. All values in the pipeline are already SI, so
no conversion takes place: of the 43 connections that carry units on both ends, every
conversion factor is 1.0.

Runs are resumable. A cell file is written to a temporary name and renamed, so a cell file
is either complete or absent. Re-running the same command skips cells that are already
present.

## 7. Quick start

```bash
cd sim
pip install -r requirements.txt

# 1. pipeline check (takes seconds)
python3 grid_dymos.py --grid grids/scythe.json --limit 3 --nodes 12 --jobs 3

# 2. inspect the result
python3 plot_trajectories.py --runs runs/scythe

# 3. full run (480 cells; resumable)
python3 grid_dymos.py --grid grids/scythe-v2.json --jobs 8 --store-solutions

# 4. reports (these solve nothing)
python3 grid_dymos.py --grid grids/scythe-v2.json --status
python3 grid_dymos.py --grid grids/scythe-v2.json --check

# 5. repair cells (see section 12)
python3 grid_dymos.py --grid grids/scythe-v2.json --reoptimise --max-weaves 2 \
    --best-of flyable --reopt-runs 3

# 6. write the table
python3 grid_dymos.py --grid grids/scythe-v2.json --merge --out ../mod/Tables/loft_table_scythe.json
```

Step 1 with `--limit` and `--nodes` verifies the pipeline, the coefficient file and the
constraint floors. Step 3 is the long run; see section 15 for expected cost and tuning.
Steps 4 and 5 are optional. Step 5 can be repeated, and each pass can only improve or keep
the existing cells.

## 8. `grid_dymos.py` command reference

### 8.1 Solving cells

| Option                                       | Meaning                                                                                                                                                                                     |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--grid FILE`                                | Grid spec. Required.                                                                                                                                                                        |
| `--jobs N`                                   | Number of worker processes. Default: number of CPUs.                                                                                                                                        |
| `--nodes N`                                  | Overrides the collocation node count from the spec.                                                                                                                                         |
| `--limit N`                                  | Solves at most N cells. Intended for smoke tests.                                                                                                                                           |
| `--timeout S`                                | Per-cell wall-clock limit. Default 900.                                                                                                                                                     |
| `--max-tasks N`                              | Restarts each worker after N cells. Default 8. OpenMDAO leaks memory between solves.                                                                                                        |
| `--no-warm`                                  | Solves every cell from a cold start, one cell at a time rather than as part of a ray. Results are typically worse; useful to confirm that an observation is not an effect of warm starting. |
| `--store-solutions` / `--no-store-solutions` | Stores the solved trajectory (`_sol`) in each cell file. Overrides the spec setting. `_sol` is required for the re-integration check and for plotting trajectory detail.                    |
| `--envelope`                                 | Solves the maximum-range problem for each launch condition and writes `envelope.json`, instead of solving the grid.                                                                         |
| `--runs DIR`                                 | Run directory. Default `runs/` inside the `sim/` directory.                                                                                                                                 |
| `--retry-failed`                             | Removes the records of failed cells so that the next run attempts them again.                                                                                                               |

### 8.2 Constraint settings

These options override the corresponding grid spec keys (section 10).

| Option                  | Spec key            | Meaning                                                                                                                       |
| ----------------------- | ------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `--min-authority G`     | `min_authority_g`   | Minimum available load factor in g along the path. 0 disables.                                                                |
| `--min-speed MS`        | `min_speed_ms`      | Speed floor along the path in m/s, or `"self_destruct"` for the game's own value. Unset means no path floor.                  |
| `--max-q KPA`           | none                | Dynamic pressure ceiling in kPa. Disabled by default; cells for this game reach 600 to 2 600 kPa in the terminal phase.       |
| `--alpha-rate DEGS`     | none                | Angle-of-attack rate limit in degrees per second. Promotes alpha to a state and its rate to the control. Disabled by default. |
| `--max-dive DEG`        | `max_dive_deg`      | Steepest permitted descent angle. 0 disables.                                                                                 |
| `--maxiter N`           | `maxiter`           | SLSQP iteration budget. Default 250.                                                                                          |
| `--refine-iterations N` | `refine_iterations` | hp-adaptive mesh refinement passes per solve. 0 disables.                                                                     |
| `--refine-tol E`        | `refine_tol`        | Refinement error tolerance. Default 1e-4.                                                                                     |

### 8.3 Reading, checking and merging

| Option               | Meaning                                                                                                                                                             |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--status`           | Prints progress, estimated time remaining, and the number of cells whose flight time ends on a duration bound.                                                      |
| `--check`            | Prints a verdict per cell: `ok`, `woven` or `INFEASIBLE` (section 13). Solves nothing and exits 1 if any cell is infeasible.                                        |
| `--table PATH`       | With `--check`, validates a merged table instead of a run directory.                                                                                                |
| `--max-weaves N`     | Number of turning points above which a profile is reported as woven. Default 2. Also used as a filter and as a `--best-of flyable` threshold.                       |
| `--speed-tol X`      | Tolerance for the re-integration check: a cell may claim at most this much more terminal speed than re-integrating its stored alpha history produces. Default 0.25. |
| `--merge --out FILE` | Writes the table.                                                                                                                                                   |
| `--purge`            | Moves unusable cell files to `runs/<name>/rejected/`.                                                                                                               |

### 8.4 Re-optimisation options

| Option                             | Meaning                                                                                                                                        |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `--reoptimise`, `--reoptimize`     | Re-solves the selected cells and keeps the better result of each attempt.                                                                      |
| `--optimise-all`, `--optimize-all` | With `--reoptimise`, re-solves every cell. A bare `--reoptimise` already does this.                                                            |
| `--reopt-runs N`                   | Attempts per cell per invocation. Default 1. The first attempt is warm-started from the current solution; subsequent attempts are cold starts. |
| `--iterations N`                   | Re-solves a cell until it converges or until it has had N attempts, whichever comes first. Overrides `--reopt-runs`.                           |
| `--best-of RULE`                   | Selects the winning attempt: `best` (default), `converged`, `fastest`, `first` or `flyable`. See section 12.                                   |
| `--best-of-only`                   | Re-picks the winner among the attempts already on disk and solves nothing.                                                                     |
| `--keep-attempts`                  | Keeps every attempt file. By default, attempts that lose are deleted when the winner is written.                                               |
| `--prune-attempts`                 | Deletes the attempts that are not the winner, for every cell in a run. Solves nothing.                                                         |
| `--compact-state`                  | Rewrites `state.jsonl` with one record per cell. Solves nothing.                                                                               |
| `--dry-run`                        | With `--reoptimise`, lists the selected cells and the reason for each selection, and exits without solving.                                    |

### 8.5 Cell filters

Filters select cells for `--reoptimise`. They are optional and combine as a union. With no
filter, every cell is selected; nothing inside a cell marks it as a bad result, so the
filters are the only way to target a subset.

| Filter                    | Selects                                                                                                                                                                                                     |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--min-terminal-speed MS` | Cells whose `terminal_speed_ms` is below this value. Also used by `--status`.                                                                                                                               |
| `--max-terminal-speed MS` | Cells whose `terminal_speed_ms` is above this value. Used to find solver outliers.                                                                                                                          |
| `--max-tof S`             | Cells whose time of flight exceeds this value.                                                                                                                                                              |
| `--max-rmse M`            | Cells whose polynomial fit error exceeds this value, but only when `--rmse-filter` is also given. Otherwise it is used only by the winner rule, which prefers an attempt whose fit is within the threshold. |
| `--rmse-filter`           | Makes `--max-rmse` a selection filter.                                                                                                                                                                      |
| `--max-weaves N`          | Cells with more turning points than the threshold.                                                                                                                                                          |
| `--infeasible`            | Cells whose stored trajectory fails the re-integration check. Requires `--store-solutions` on the run that produced them.                                                                                   |

## 9. Cell file fields

| Field                                                                            | Meaning                                                                                                                                                                                                          |
| -------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `range_km`, `launch_alt_m`, `launch_speed_ms`, `target_alt_m`, `target_speed_ms` | Launch and target geometry.                                                                                                                                                                                      |
| `terminal_speed_ms`                                                              | Speed at the target. This is the objective; the winner rule reads this field and nothing else.                                                                                                                   |
| `poly_a`                                                                         | Coefficients of the polynomial that the plugin flies (order 7).                                                                                                                                                  |
| `tof_s`                                                                          | Time of flight.                                                                                                                                                                                                  |
| `apex_alt_m`, `apex_downrange_m`                                                 | Maximum altitude and the downrange distance at which it occurs.                                                                                                                                                  |
| `fit_rmse_m`                                                                     | Root-mean-square error of the polynomial against the solved trajectory. This is not a pass criterion. It is used as a preference between attempts (section 12) and, with `--rmse-filter`, as a selection filter. |
| `converged`                                                                      | Whether SLSQP reached convergence, as opposed to stopping on the iteration budget. Convergence is uncommon; cells that stop on the budget usually contain a usable trajectory.                                   |
| `maxiter`                                                                        | Iteration budget used for this cell.                                                                                                                                                                             |
| `refine_error`                                                                   | Refinement error estimate, if refinement was enabled.                                                                                                                                                            |
| `feasible`, `violation`                                                          | Result of the constraint audit: whether the trajectory satisfies the floors it was solved with, and the first violation found.                                                                                   |
| `v_min_ms`, `nav_min_g`                                                          | Minimum speed and minimum available load factor along the trajectory. These are recorded whether or not a floor was set.                                                                                         |
| `authority_floor_g`, `speed_floor_ms`                                            | Floors used for this solve. 0 means none.                                                                                                                                                                        |
| `t_bound`, `duration_bounds_s`                                                   | Whether the flight time ended on its own duration bound: `"lower"`, `"upper"` or null. A cell ending on a bound is box-limited rather than converged.                                                            |
| `health`                                                                         | Verdict recomputed by `--check`: `ok`, `woven`, `infeasible` or `unverified`.                                                                                                                                    |
| `defect_ratio`                                                                   | Ratio of the recorded arrival speed to the speed the stored alpha history produces, used when the trajectory is no longer available. 1.0 means the two agree.                                                    |
| `_elapsed_s`, `_settings`                                                        | Solve duration and the settings used.                                                                                                                                                                            |
| `_sol`                                                                           | Solved trajectory, written only with `--store-solutions`. Trajectories roughly triple the size of a run directory.                                                                                               |

## 10. Constraints and floors

Two path constraints are available, both specified in the grid spec and both optional.

**`min_authority_g`** requires a minimum available load factor along the whole path. The
rationale is in section 4.4. The value that a spec omitting the key receives is 3.0 g;
every spec shipped in `sim/grids/` sets 0.1 g.

**`min_speed_ms`** applies the game's own speed rule as a path constraint rather than only
at the terminal point. The missile is lost if its speed drops below the game's
self-destruct speed at any time after t = 2 s, so a spec can request that condition with
`"min_speed_ms": "self_destruct"`, or with a number in m/s.

Grid spec settings and their defaults:

| Key                 | Default | Notes                                |
| ------------------- | ------- | ------------------------------------ |
| `nodes`             | 12      | Collocation nodes per segment.       |
| `order`             | 7       | Order of the reference polynomial.   |
| `min_authority_g`   | 3.0     | Code default. Shipped specs use 0.1. |
| `min_speed_ms`      | none    | No path floor.                       |
| `maxiter`           | 250     | SLSQP iteration budget.              |
| `refine_iterations` | 0       | hp refinement disabled.              |
| `refine_tol`        | 1e-4    |                                      |
| `store_solutions`   | false   |                                      |
| `max_dive_deg`      | 0       | Descent unconstrained.               |

A floor must be attainable at the start of the cell. A speed floor above the launch speed
makes the problem infeasible at t = 0, so the floor is capped at the launch value. For
this reason `scythe-v2.json` raises the lowest launch speed to 210 m/s, above the game's
200 m/s self-destruct speed, rather than lowering the floor. A solution that sits exactly
on a floor is not a violation; the audit allows 1e-3 of slack.

`--check` reports the trajectory's own minima (`v_min_ms`, `nav_min_g`) whether or not a
floor was set. These recorded values are the appropriate basis for choosing a floor.

## 11. Warm starts

A worker solves all cells that share a launch and target condition in order of range, and
each cell is warm-started from the solution of its neighbour. The neighbour's flight time
belongs to a different range, so it is scaled with the path and clamped into the current
cell's duration bounds, and the state histories are clamped into their bounds.

The declared bounds are stored in one place and are used both for the phase declarations
and for validating start values, so the values given to the solver cannot drift from the
values a guess is checked against. A start value that lies outside its bounds is reported:

```
start values: 4 value(s) were outside their bounds (by up to 0.15 on traj.mid.controls:alpha); clamped
```

A value that lies exactly on a bound can return from interpolation one unit in the last
place outside that bound. Values that are clamped onto a bound after interpolation are
therefore written back inside their bounds, which keeps a run free of `DriverWarning`
messages. Values that are clamped by an amount within rounding error are not reported
unless verbose output is enabled.

A cell whose flight time ends on a duration bound records `t_bound`. The objective drives
flight time down, so this is a normal outcome, but it means the value is limited by the
bound rather than by the optimiser. `--status` reports the count.

## 12. Re-optimisation

### 12.1 Relationship to `--check`

`--check` and `--reoptimise` are independent. `--check` reads and reports; it solves
nothing, changes nothing, and exits 1 if any cell is infeasible, so it can be used as a
gate. `--reoptimise` selects cells using the filters described in section 8.5 and
re-solves them. It does not consult `--check` and does not require it to have been run.

`--check` is the appropriate way to determine which filters to use. A typical sequence is
to run the grid, run `--check`, and then use `--dry-run` with the reported categories to
list the cells that a re-optimisation would select.

### 12.2 Behaviour

```
select cells -> for each cell: solve up to --reopt-runs attempts -> keep the best by --best-of
```

* The first attempt is warm-started from the cell's stored trajectory, if one exists, and
  can only lose. Later attempts are cold starts, which is where the variation between
  attempts originates.
* An attempt replaces the incumbent only if it ranks better. The losing attempt file is
  deleted when the winner is written, unless `--keep-attempts` is given.
* `--iterations N` re-solves a cell until it converges or until it has had N attempts. It
  is appropriate when convergence rather than variation is the problem.

The winner rule compares attempts in tiers, because the table ships the polynomial rather
than the trajectory:

1. A cell that fails validation (no polynomial, or no positive finite terminal speed)
   ranks below all others.
2. Otherwise, an attempt whose fit error is within `--max-rmse` ranks above one whose fit
   error is not.
3. Within a tier, the rule named by `--best-of` applies:
   * `best` (default): highest terminal speed.
   * `converged`: as `best`, with a bonus for a converged attempt.
   * `fastest`: as `best`, without the fit preference.
   * `first`: keeps the existing cell.
   * `flyable`: as `best`, with two additional tiers: an attempt whose re-integration
     disagrees with its own arrival speed ranks below all others, and a woven profile
     ranks below every straight one.

### 12.3 Commands

```bash
# list the cells that would be re-solved, and the reason for each (no solving)
python3 grid_dymos.py --grid grids/scythe-v2.json --reoptimise --max-weaves 2 --infeasible --dry-run

# re-solve slow, badly fitted or woven cells with three attempts each
python3 grid_dymos.py --grid grids/scythe-v2.json --reoptimise --reopt-runs 3 \
    --min-terminal-speed 900 --max-rmse 2000 --max-weaves 2 --best-of flyable

# re-solve cells whose stored trajectory fails re-integration (requires _sol)
python3 grid_dymos.py --grid grids/scythe-v2.json --reoptimise --infeasible

# re-solve every cell with five attempts each
python3 grid_dymos.py --grid grids/scythe-v2.json --reoptimise --optimise-all \
    --reopt-runs 5 --best-of flyable

# re-apply the winner rule to attempts already on disk (no solving)
python3 grid_dymos.py --grid grids/scythe-v2.json --best-of-only --best-of flyable

# maintenance
python3 grid_dymos.py --grid grids/scythe-v2.json --status
python3 grid_dymos.py --grid grids/scythe-v2.json --prune-attempts
python3 grid_dymos.py --grid grids/scythe-v2.json --compact-state
```

### 12.4 Cells that do not improve

Terminal speed claims are not always reproducible. On the shipped table,
`300_4000_600_200_0` records 3 912 m/s, which does not survive re-integration;
`--best-of best` retains that value indefinitely because the rule reads
`terminal_speed_ms` alone. `--best-of flyable` selects a 1 067 m/s trajectory that is
reproducible instead. Validation with `--check --table` reports 67 of the 480 cells in
that table as woven.

## 13. Validation with `--check`

```bash
python3 grid_dymos.py --grid grids/scythe-v2.json --check
python3 grid_dymos.py --grid grids/scythe-v2.json --check --table mod/Tables/loft_table_scythe.json
```

`--check` prints a verdict for each cell.

* `ok`: the trajectory is consistent.
* `woven`: the altitude profile has more turning points than `--max-weaves`. The cell is
  flyable, but the profile is not a loft; it is typically a stall that the optimiser
  recovered from.
* `INFEASIBLE`: one of two conditions, identified in the output.

```
INFEASIBLE 50_9000_600_200_0  claims 6365 m/s, re-integrating the same alpha
                              history arrives at 4720 m/s (1.3x), 1 turning
                              points; paths end 90 km apart
INFEASIBLE 30_9000_600_200_0  did not satisfy its own constraint audit:
                              min_speed: 180.0 m/s < 200.0 m/s
```

The first form is a **re-integration failure**: the stored trajectory was flown again
through the same ODE from the same initial state using the same alpha history, and the
result does not reproduce the recorded arrival speed. The second form is a **constraint
audit failure**: the solve did not satisfy one of the floors it was given (section 10),
and the output names the constraint. Both conditions also report the number of turning
points and the distance between the two path endpoints.

`--check` exits 1 if any cell is infeasible, so it can be used directly as a gate.

### 13.1 Reachable range

`--envelope` solves the free-final-range problem once per launch condition and writes
`runs/<name>/envelope.json`. `--merge` then records `exp_max_range_km` and `reachable` on
each cell. For the 180 m/s launch family in the shipped table, cells are recorded at
300 km and 400 km where the best envelope start reached 98 km; those cells are outside the
reachable set as measured. The envelope value should be read as a bound on what the
optimiser reached, not as a proven limit: it is one non-convex NLP from one start per
launch condition, and separate starts have differed by nearly a factor of three.

### 13.2 Limitations of validation

* With `--table`, cells can only be checked for turning points, because a table contains
  no trajectory. Cells that are not woven are reported as `unverified` and the command
  states this.
* Without `--store-solutions`, the trajectory is not available, so the verdict is derived
  from the `defect_ratio` recorded at solve time.
* Re-integration is a reconstruction, not a certificate. The control nodes are placed
  where dymos places them and the ODE is the same, but a cell that stopped on the
  iteration budget can re-integrate to between 0.71 and 1.10 of its own claim while
  remaining acceptable. The verdict is intended to catch gross disagreement, defined as
  more than `--speed-tol` (0.25 by default) of optimism.

## 14. Run monitoring

Each completed cell produces one line, printed by the worker that finished it:

```
[137/480] 130_9000_600_200_0   1084.3 m/s    41.2s  ITER-LIMIT
```

`ITER-LIMIT` indicates that the solve stopped on the iteration budget rather than on
convergence, which is the common outcome. `FAILED` indicates that the cell produced no
usable result.

Output from the solver itself (the sparsity coloring report, the SLSQP exit summary and
the failure banner that OpenMDAO prints unconditionally) is captured rather than printed,
because it obscures the per-cell lines on a large run. The text remains available in the
result's `msg`, and if a solve raises an exception the captured output is attached to it.
Running `grid_dymos.py` with `verbose=True` restores the previous level of detail.

`--status` prints progress, estimated time remaining and duration-bound counts without
solving. `--dry-run` with `--reoptimise` lists the cells that would be selected.

### 14.1 Stopping a run

Ctrl-C stops a grid run, as do `kill` (SIGTERM) and a dropped terminal (SIGHUP). Workers
are terminated, the run reports how many cells stopped, and the process exits with 130,
143 or 129 respectively. If the parent is killed with `-9`, the workers detect the loss of
their parent and exit within about one second.

A stopped run retains every cell that the parent has already stored, which means every
cell belonging to a ray that had finished. Cells inside a ray that was still running are
lost. Re-running the same command solves them again; no other recovery step is needed.

## 15. Performance and accuracy

**Nodes.** Increasing the node count improves transcription accuracy rather than the
objective. On a 30 km cell (9 000 m launch, 320 m/s, 200 m target altitude):

| Nodes | Recorded terminal speed | Speed produced by re-integrating its own alpha history | Time   |
| ----- | ----------------------- | ------------------------------------------------------ | ------ |
| 12    | 965.7 m/s               | 0.65x the recorded value                               | 12.0 s |
| 40    | 956.5 m/s               | 1.11x the recorded value                               | 75.7 s |

The finer mesh records a slightly lower terminal speed because the coarser mesh's value
does not survive re-integration. The `defect_ratio` and `health` fields record this
condition per cell.

**Iterations.** `--maxiter` controls the SLSQP budget. Raising it from the default of 250
is inexpensive, but a stalled solve can remain at a feasible flat trajectory indefinitely,
so additional iterations do not guarantee improvement. Beyond a few hundred iterations,
results change little.

**Refinement.** hp-adaptive refinement (`--refine-iterations`, `--refine-tol`) addresses
discretisation error and is expensive. Measured on a short cell, two passes increased one
solve from 8 s to 470 s and left the error estimate at 1.9e-2. Increasing `nodes` is
generally more effective: a 40-node mesh costs about six times a 12-node mesh and requires
no refinement. `scythe-v2.json` uses `nodes 40` with one refinement pass because it is
intended for an overnight run.

**Cost.** Cost varies strongly with range and with the constraint floors. The original
252-cell grid measured approximately 150 s per cell at 12 nodes, about 80 minutes on eight
cores; a 30 km cell at the same node count takes about 12 s. Short-range cells are faster
and tend to converge cleanly. Long-range cells often stop on the iteration budget and
still produce usable trajectories.

**Other parameters.** `--jobs` sets the number of workers, `--timeout` sets the per-cell
limit (default 900 s), and `--no-warm` disables warm starting. Per-cell post-processing
outside the solve is small: the re-integration behind `health` takes about 0.07 s per
cell.

## 16. Plotting with `plot_trajectories.py`

`plot_trajectories.py` renders the contents of a merged table or a run directory. It
requires numpy and matplotlib only; dymos is not imported, so a table can be inspected on
a machine that has never solved anything.

The plotter reads everything a cell records, not only the polynomial: `--check`'s verdict
(`health`), whether the solve satisfied its own constraint audit (`feasible`, `violation`),
the lowest speed and lowest available load factor anywhere on the path (`v_min_ms`,
`nav_min_g`), the recorded/re-integrated speed ratio (`defect_ratio`), whether the flight
time ended on its duration bound (`t_bound`), and the envelope marks (`exp_max_range_km`,
`reachable`). Line style carries the verdict, and a cell with no verdict recorded is drawn
as such rather than as a pass.

```bash
python3 plot_trajectories.py --runs runs/scythe                    # every solved cell
python3 plot_trajectories.py --runs runs/scythe --overview         # apex, terminal speed, tof, rmse
python3 plot_trajectories.py --runs runs/scythe --cell 75_4000_320_200_0
python3 plot_trajectories.py --table ../mod/Tables/loft_table_scythe.json \
    --fix launch_alt_m=9000 --hide-flat
python3 plot_trajectories.py --flight ../../BepInEx/LogOutput.log --save flight.png
```

| Option                                                                                          | Meaning                                                                                                                                                                                                                     |
| ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--table FILE`                                                                                  | Merged table to plot. May be repeated; `--compare` labels each table separately.                                                                                                                                            |
| `--runs DIR`                                                                                    | Run directory to plot, using `runs/<name>/cells/*.json`.                                                                                                                                                                    |
| `--list`                                                                                        | Prints every cell key with apex, gain, terminal speed, flight time, fit error and flags.                                                                                                                                    |
| `--cell KEY`                                                                                    | Plots one cell. The key is the same in a table and in a run directory.                                                                                                                                                      |
| `--overview`                                                                                    | Grid health panels: apex, terminal speed, flight time and fit error against range, plus a panel each for the path minima and the defect ratio when the cells carry them (a table merged before those fields existed falls back to the four original panels). Cells beyond the solved envelope are drawn as hollow red squares regardless of verdict.|
| `--fix AXIS=VALUE`                                                                              | Restricts the plot to one value of one axis.                                                                                                                                                                                |
| `--rows`, `--cols`, `--sweep`                                                                   | Select which free axis becomes the panel grid and which becomes the curves.                                                                                                                                                 |
| `--colour AXIS`                                                                                 | Colour of the curves.                                                                                                                                                                                                       |
| `--normalised`                                                                                  | Uses percent of `Xf` on the x axis.                                                                                                                                                                                         |
| `--palette NAME`                                                                                | Colour scheme: `auto` (default; tab10 up to ten lines, viridis beyond), `okabe-ito` (colour-blind safe for the lines), or a sequential map such as `cividis` for `--overview`'s colour scale.                               |
| `--min-terminal-speed`, `--max-terminal-speed`, `--max-rmse`, `--hide-flat`, `--only-converged` | Filters on the recorded numbers: terminal speed, fit error, flat cells, and cells the solver did not converge.                                                                                                              |
| `--health STATE`                                                                                | Keep only cells with this `--check` verdict (`ok`, `woven`, `infeasible`, `unverified`). Repeatable. Cells with no recorded verdict are excluded and counted, because "not checked" is not "passed".                        |
| `--reachable` / `--unreachable`                                                                 | Keep only cells inside the solved envelope, or only those beyond it. Both need a table merged with an `envelope.json`.                                                                                                      |
| `--min-v-along MS`                                                                              | Keep only cells whose lowest speed along the path is at least this.                                                                                                                                                         |
| `--max-defect RATIO`                                                                            | Keep only cells whose recorded/re-integrated ratio is at most this (1.0 means the claim flies).                                                                                                                             |
| `--include-attempts`                                                                            | Also plots `<key>.attempt<N>.json` files. Without this option, attempts are ignored because they repeat the same cell.                                                                                                      |
| `--reoptimise`                                                                                  | Colors attempts as one cell's attempts.                                                                                                                                                                                     |
| `--annotate`, `--no-annotate`, `--legend`, `--no-legend`, `--max-legend N`                      | Label controls. Curves are labelled at the point where they are furthest from the other curves in the panel, because all curves in a panel end at the same aimpoint. Above 12 curves the legend is replaced by annotations. |
| `--max-panels N`                                                                                | Maximum number of panels, default 36. Panels are thinned by even spacing that preserves both ends, so the longest range is never dropped. The values that were omitted are reported on stderr.                              |
| `--detail-extra`                                                                                | With `--cell`, plots speed, flight path angle, angle of attack and the polynomial residual at the solution nodes. Requires `--store-solutions`. The speed panel marks the path minimum the audit measured against the floor it was solved with.|
| `--html FILE`                                                                                   | Writes one self-contained page: verdict tiles, the figures as inline SVG, and a sortable, filterable table of every cell. No network and no external assets, so it renders offline.                                         |
| `--html-interactive`                                                                            | Draws the page with plotly instead of as a static SVG, so curves can be hovered (cell key, verdict, terminal speed), grouped by verdict and zoomed. The plotly runtime is inlined, so the page stays self-contained; without plotly installed it falls back to the static figure and says so on the page. On its own it writes `trajectories.html` and reports the name it chose; with `--html FILE` that file is the one made interactive. Requires the optional `plotly` from `requirements.txt`.|
| `--flight LOG`                                                                                  | Parses the `[flight]` lines of a BepInEx log and plots, per round: altitude against the reference it was commanded to follow, the tracking error, the slope loop's command against what the airframe achieved, and speed with the game's self-destruct and load-time-filter speeds marked. The log carries a state section and a slope section, and the two reuse key names (`ref` is an altitude in one and an angle in the other); the parser keeps them apart.|
| `--save FILE`, `--html`, `--show`, `--dpi`                                                      | Output controls. `--html` writes a self-contained page.                                                                                                                                                                     |

## 17. File layout

```
sim/
    grid_dymos.py          solver and command-line interface
    dymos_opt.py           transcription, solve and audit of one cell
    missile_sim.py         replica of the game's missile physics
    coeffs_*.json          parsed prefab coefficients (density, thrust, mass, aero)
    grids/*.json           grid specs
    plot_trajectories.py   plotting tool
    runs/<name>/
        cells/<key>.json   one solved cell, plus <key>.attempt<N>.json if kept
        state.jsonl        append-only record of cell results
        rejected/          cell files moved by --purge
        envelope.json      written by --envelope
mod/
    Plugin.cs              BepInEx entry point
    Core/                  patches, guidance, table loading, settings
    Tables/                loft_table_*.json, what the plugin loads
    Directory.Build.props  locates the game's managed assemblies
    Local.props.example.txt  copy to Local.props and set the game directory
```

## 18. Limitations

* **No terrain awareness.** The aimpoint is clamped to `FloorAltM`, not to the ground.
  Low-level trajectories require tables generated with terrain in the loop.
* **The terminal phase is unaffected.** Inside `terminalRange` (12 000 m) the plugin
  returns without overriding, so the game's own radar-locked lead calculation applies.
  The plugin covers the midcourse only.
* **Blending cells with different endpoints.** The polynomials are pinned to their own
  launch and target altitudes, so an inverse-distance blend of two cells matches neither
  endpoint exactly. Measured: querying 1 000 m between cells at 305 m and 3 048 m gives
  `h(1) = 679 m`. The effect is limited because the plugin stops overriding inside
  `terminalRange`, and it is the reason a denser `target_alt_m` axis is more valuable
  than a denser `range_km` axis.
* **`target_speed_ms` is carried but not solved.** Reference profiles are generated
  against a fixed predicted intercept point, as in the paper. Target motion is handled at
  runtime; see section 4.3.
* **The game is server-authoritative.** Missiles are simulated only where `LocalSim` is
  true. On a dedicated server the plugin must be installed there as well.
* **`--fail-rmse X` is not implemented.** The equivalent workflow is
  `--reoptimise --max-rmse X --rmse-filter --dry-run` followed by a re-optimisation with
  that filter. No persistent failed flag is recorded in a cell.
* **The optimiser is not reproducible.** Two runs of the same command produce different
  results. One fixed 15 km solve returned flight times between 13.08 s and 15.43 s across
  eight runs. A change should therefore not be attributed to a patch on the basis of a
  single run, and comparisons between trees should be made on counts rather than hashes.
* **Plugin behaviour that requires in-game verification.** `TrackGain` has no rate
  damping, so the tracking law can chase its own lag. `Debug.DrawLine` in
  `LoftGuidance.Draw` is obsolete now that a `LineRenderer` is used. The load-time filters
  (`MinCellTerminalSpeedMs`, `RejectFlatCells`) discard cells rather than reporting them,
  which is why 45 flat cells can remain in a shipped table without notice.
* **Validation is a reconstruction.** The re-integration check reproduces the stored
  trajectory through the same ODE, but it is not an independent verification of the
  dynamics; see section 13.2.
