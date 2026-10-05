#!/usr/bin/env python3
"""
plot_trajectories.py -- look at the planned loft trajectories without flying them.

    cd sim

    # every solved cell, as a grid of altitude-vs-downrange curves
    python3 plot_trajectories.py --runs runs/scythe

    # the table the mod actually loads
    python3 plot_trajectories.py --table ../mod/Tables/loft_table_scythe.json

    # slice it: fix two axes, sweep the rest
    python3 plot_trajectories.py --table ../mod/Tables/loft_table_scythe.json \
        --fix launch_alt_m=9000 --fix launch_speed_ms=320 --fix target_alt_m=200

    # the health of the whole grid at once: apex, terminal speed, fit error, TOF
    python3 plot_trajectories.py --runs runs/scythe --overview --colour launch_alt_m

    # one cell in full, including V / gamma / alpha when the solution was stored
    python3 plot_trajectories.py --runs runs/scythe --cell 75_4000_320_200_0

    # what the missiles actually did, from a game log
    python3 plot_trajectories.py --flight ../../uploads/log5normal.txt

    # something you can leave open in a browser tab
    python3 plot_trajectories.py --runs runs/scythe --html plan.html --save plan.png

WHY THIS EXISTS

The optimiser only ever hands back `poly_a` plus six diagnostics, so the only way
to see what a cell is going to make the missile do was to fly it. A profile that
is wrong is usually wrong in a way that is obvious on a plot -- a flat cell, a
poly that dips below the launch altitude, ringing near the endpoints, an apex
later than the target, two neighbouring cells with incompatible shapes -- and all
of that is invisible in the table JSON and in `--status`. This renders it.

WHAT IT PLOTS

"Family" mode (the default) is altitude against downrange, one line per cell:
one panel per (rows x cols) pair of grid axes, one line per value of a third
axis. Which axes those are is inferred from what you left free, and printed in
the title, so there is no hidden state. Fix axes with `--fix axis=value` until
what is left is what you want to look at; override the choice with `--rows`,
`--cols` and `--sweep`.

`--overview` is the health check: apex gain, terminal speed, fit RMSE and TOF,
each against range, all cells, coloured by whichever axis you nominate. A flat
cell (apex == launch altitude) is drawn as a red cross everywhere because that
is the failure that looks most like success in the JSON: it fits its own flat
answer exactly, so it reports the *lowest* fit RMSE in the table.

`--cell KEY` is one cell, plus the stored Dymos solution when there is one
(`V`, `gamma`, `alpha` and the polynomial residual). Cells solved before
`grid_dymos.py --store-solutions` existed only have the polynomial.

`--flight LOG` is a different axis of the same problem: it parses the mod's own
`[flight]` lines out of a BepInEx log and plots what the missile did against
what it was told to do (`ref`), the tracking error, and speed, per round. The
`[flight]` line carries exactly these numbers so the comparison is possible; see
the flight-telemetry notes in the README.

INPUTS

  --runs DIR     a grid run directory (`runs/<name>`). Attempt files left by
                 grid_dymos.py --reoptimise are ignored unless --include-attempts.
  --table FILE   a merged table (grid_dymos.py --merge). Repeatable; with
                 --compare, same-cell lines from two tables are drawn against
                 each other, which is how a re-optimise run is judged.
  --runs DIR     a grid run directory (`runs/<name>`), i.e. the per-cell files
                 before merging. These are the richer source: they can carry the
                 solved trajectories.

Both are tolerant: a cell with no `poly_a`, a legacy imperial cell from before
the metric switch, or a JSON file that is not a cell at all is skipped with a
counted reason rather than a traceback.

Requires numpy and matplotlib only -- it does NOT import dymos, so it can be run
on a machine that has never solved anything, against a table copied off the
machine that did.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

AXES = ["range_km", "launch_alt_m", "launch_speed_ms", "target_alt_m", "target_speed_ms"]

# Axis order == "which is worth seeing vary first". Used to choose rows/cols/sweep
# when the user has not said, and to lay out the overview.
ORDER = list(AXES)

LABEL = {
    "range_km": "range (km)",
    "launch_alt_m": "launch alt (m)",
    "launch_speed_ms": "launch speed (m/s)",
    "target_alt_m": "target alt (m)",
    "target_speed_ms": "target speed (m/s)",
}
SHORT = {
    "range_km": "R", "launch_alt_m": "h0", "launch_speed_ms": "V0",
    "target_alt_m": "ht", "target_speed_ms": "Vt",
}

# Accepted spellings for --fix / --rows / --cols / --sweep / --colour.
ALIAS = {}
for _a in AXES:
    ALIAS[_a] = _a
    ALIAS[_a.replace("_m", "")] = _a
    ALIAS[_a.replace("_ms", "")] = _a
ALIAS.update({
    "range": "range_km", "r": "range_km",
    "launch_alt": "launch_alt_m", "h0": "launch_alt_m", "launchalt": "launch_alt_m",
    "launch_speed": "launch_speed_ms", "v0": "launch_speed_ms",
    "target_alt": "target_alt_m", "ht": "target_alt_m",
    "target_speed": "target_speed_ms", "vt": "target_speed_ms",
})

# Legacy imperial -> metric, same factors as grid_dymos.LEGACY_AXES.
LEGACY = {
    "range_nmi": ("range_km", 1.852),
    "launch_alt_ft": ("launch_alt_m", 0.3048),
    "launch_kt": ("launch_speed_ms", 0.514444),
    "target_alt_ft": ("target_alt_m", 0.3048),
    "target_kt": ("target_speed_ms", 0.514444),
}

# The mod's own thresholds, drawn as reference lines in --overview.
SELF_DESTRUCT_SPEED = 200.0     # ARHSeeker.selfDestructAtSpeed
DEFAULT_MIN_TERMINAL = 800.0    # General/MinCellTerminalSpeedMs in the README


# --------------------------------------------------------------------------
# cells
# --------------------------------------------------------------------------
@dataclass
class Cell:
    """One solved grid cell, whatever it was read from."""

    coords: dict
    poly: list
    meta: dict = field(default_factory=dict)
    sol: dict | None = None
    source: str = ""
    key: str = ""

    # --- derived ---------------------------------------------------------
    @property
    def xf(self) -> float:
        return float(self.coords.get("range_km", 0.0)) * 1000.0

    @property
    def h0(self) -> float:
        return float(self.coords.get("launch_alt_m", 0.0))

    @property
    def ht(self) -> float:
        return float(self.coords.get("target_alt_m", 0.0))

    @property
    def apex(self) -> float:
        v = self.meta.get("apex_alt_m")
        return float(v) if v is not None else float("nan")

    @property
    def apex_gain(self) -> float:
        return self.apex - self.h0

    @property
    def v_term(self) -> float:
        v = self.meta.get("terminal_speed_ms")
        return float(v) if v is not None else float("nan")

    @property
    def rmse(self) -> float:
        v = self.meta.get("fit_rmse_m")
        return float(v) if v is not None else float("nan")

    @property
    def tof(self) -> float:
        v = self.meta.get("tof_s")
        return float(v) if v is not None else float("nan")

    @property
    def converged(self) -> bool:
        return bool(self.meta.get("converged", False))

    @property
    def flat(self) -> bool:
        """Never lofted: apex is the launch altitude. The table's silent failure."""
        return abs(self.apex_gain) < 1.0

    def alt(self, xi):
        """h(xi), Horner. Matches LoftCell.AltitudeAt so a plot cannot disagree
        with what the mod computes."""
        c = self.poly
        acc = c[-1]
        for k in range(len(c) - 2, -1, -1):
            acc = acc * xi + c[k]
        return acc

    def alt_curve(self, n: int = 200):
        import numpy as np
        xi = np.linspace(0.0, 1.0, n)
        return xi, np.array([self.alt(x) for x in xi])

    def coord(self, axis: str):
        return self.coords.get(axis)

    def short(self) -> str:
        """One line for a legend or a log: the axes that matter, with units."""
        return (f"{self.coords['range_km']:.0f}km "
                f"h0={self.coords['launch_alt_m']:.0f} "
                f"V0={self.coords['launch_speed_ms']:.0f} "
                f"ht={self.coords['target_alt_m']:.0f}")

    def flags(self) -> str:
        f = []
        if self.flat:
            f.append("FLAT")
        if self.rmse == self.rmse and self.rmse > 1000.0:
            f.append(f"rmse={self.rmse:.0f}")
        if not self.converged:
            f.append("no-conv")
        if self.v_term == self.v_term and self.v_term < SELF_DESTRUCT_SPEED * 4:
            f.append(f"v={self.v_term:.0f}")
        return " ".join(f)


def _get(d: dict, *names, default=None):
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return default


def cell_from_json(obj: dict, source: str, key: str = "") -> Cell | None:
    """Build a Cell, or None with a reason if the record is not usable.

    Deliberately mirrors the mod's own tolerance: legacy imperial keys are
    converted (LoftCell.Normalise), and a cell with no polynomial is skipped --
    that is the only field the runtime cannot do without.
    """
    if not isinstance(obj, dict):
        return None
    coords: dict = {}
    for axis in AXES:
        v = obj.get(axis)
        if v is None:
            for old, (new, k) in LEGACY.items():
                if new == axis and old in obj and obj[old] is not None:
                    v = float(obj[old]) * k
                    break
        if v is None:
            v = 0.0
        try:
            coords[axis] = float(v)
        except (TypeError, ValueError):
            return None

    poly = obj.get("poly_a")
    if not isinstance(poly, list) or len(poly) < 2:
        return None
    try:
        poly = [float(v) for v in poly]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in poly):
        return None

    meta = {k: obj[k] for k in
            ("apex_alt_m", "apex_downrange_m", "terminal_speed_ms", "tof_s",
             "fit_rmse_m", "converged", "maxiter", "_elapsed_s", "_attempts",
             "_settings", "generated")
            if k in obj}
    sol = obj.get("_sol") if isinstance(obj.get("_sol"), dict) else None
    if sol is not None:
        need = ("x", "h")
        if not all(isinstance(sol.get(k), list) and len(sol[k]) > 1 for k in need):
            sol = None

    if not key or key.startswith("#"):
        # Prefer the cell's own _key, else derive it from the axes exactly the way
        # grid_dymos.py does (AXIS_ORDER joined with "_", %g formatting). A merged
        # table does not carry _key -- --merge drops it -- so without this a table
        # could only be addressed by position (#12), which changes the moment the
        # file is regenerated, and which disagreed with the key the same cell has
        # in runs/<name>/cells/. Now `--cell 75_4000_320_200_0` means the same
        # thing for a table and for a run directory.
        derived = "_".join(f"{coords[a]:g}" for a in AXES)
        key = obj.get("_key") or derived
    meta.setdefault("_index", key if key.startswith("#") else None)
    return Cell(coords=coords, poly=poly, meta=meta, sol=sol, source=source, key=key)


def load_table(path: Path, label: str | None = None) -> tuple[list[Cell], list[str]]:
    """Read a merged table. Returns (cells, notes) -- notes are for the user."""
    notes: list[str] = []
    try:
        d = json.loads(path.read_text())
    except Exception as e:
        return [], [f"{path}: cannot read ({type(e).__name__}: {e})"]
    if not isinstance(d, dict) or "cells" not in d:
        return [], [f"{path}: no 'cells' key -- not a merged table?"]
    src = label or path.name
    cells = []
    for i, obj in enumerate(d["cells"]):
        c = cell_from_json(obj, src, key=f"#{i}")
        if c is None:
            notes.append(f"{src}: cell {i} skipped (no usable poly_a)")
            continue
        cells.append(c)
    if d.get("generated"):
        notes.append(f"{src}: {len(cells)} cells, generated {d['generated']}"
                     + (f", missile {d['missile']}" if d.get("missile") else ""))
    else:
        notes.append(f"{src}: {len(cells)} cells")
    return cells, notes


ATTEMPT_RE = re.compile(r"\.attempt\d+$")


def load_runs(root: Path, include_attempts: bool = False) -> tuple[list[Cell], list[str]]:
    """Read runs/<name>/cells/*.json, the pre-merge grid output.

    `--reoptimise` leaves every attempt it made on disk, as <key>.attempt<N>.json,
    next to the winner <key>.json. They are the same cell, so counting them as
    more cells would make every layer of a family plot repeat itself; they are
    skipped unless --include-attempts asks for them, which is how you look at
    the spread the optimiser landed in.
    """
    notes: list[str] = []
    cells_dir = root / "cells" if (root / "cells").is_dir() else root
    if not cells_dir.is_dir():
        return [], [f"{root}: no cells/ directory"]
    skipped = 0
    n_att = 0
    cells = []
    for p in sorted(cells_dir.glob("*.json")):
        if ATTEMPT_RE.search(p.stem):
            if not include_attempts:
                n_att += 1
                continue
            cells_att = True
        try:
            obj = json.loads(p.read_text())
        except Exception:
            skipped += 1
            continue
        c = cell_from_json(obj, root.name, key=p.stem)
        if c is None:
            skipped += 1
            continue
        if ATTEMPT_RE.search(p.stem):
            c.meta.setdefault("_note", "attempt " + ATTEMPT_RE.search(p.stem).group()[8:])
        elif c.meta.get("_attempts", 1) > 1:
            c.meta.setdefault("_note", f"best of {c.meta['_attempts']}")
        cells.append(c)
    notes.append(f"{root.name}: {len(cells)} cells from {cells_dir}"
                 + (f" ({skipped} unusable files skipped)" if skipped else "")
                 + (f" ({n_att} attempt files ignored, --include-attempts to see them)"
                    if n_att else ""))
    return cells, notes


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------
def resolve_axis(name: str) -> str:
    a = ALIAS.get(name.strip().lower())
    if a is None:
        raise SystemExit(f"unknown axis '{name}'. Known: {', '.join(AXES)} "
                         f"(aliases: {', '.join(sorted(k for k in ALIAS if k not in AXES))})")
    return a


def parse_fix(pairs: list[str]) -> dict:
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--fix needs axis=value, got '{p}'")
        k, v = p.split("=", 1)
        axis = resolve_axis(k)
        try:
            out[axis] = float(v)
        except ValueError:
            raise SystemExit(f"--fix {p}: '{v}' is not a number")
    return out


def matches(value: float, want: float, tol: float) -> bool:
    return abs(value - want) <= tol


def select(cells: list[Cell], args) -> list[Cell]:
    fix = parse_fix(args.fix)
    out = []
    for c in cells:
        if any(not matches(c.coord(a), v, args.fix_tol) for a, v in fix.items()):
            continue
        if args.min_terminal_speed is not None and not (c.v_term >= args.min_terminal_speed):
            continue
        if args.max_rmse is not None and not (c.rmse <= args.max_rmse):
            continue
        if args.max_terminal_speed is not None and c.v_term > args.max_terminal_speed:
            continue
        if args.only_converged and not c.converged:
            continue
        if args.hide_flat and c.flat:
            continue
        out.append(c)
    return out


def values_of(cells: list[Cell], axis: str) -> list[float]:
    return sorted({round(c.coord(axis), 6) for c in cells})


def free_axes(cells: list[Cell], cut: int = 1) -> list[str]:
    """Axes with more than `cut` distinct value in the current selection, in
    ORDER, i.e. most interesting first."""
    return [a for a in ORDER if len(values_of(cells, a)) > cut]


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------
def import_mpl(show: bool):
    try:
        import matplotlib
    except Exception as e:  # pragma: no cover - environment problem
        raise SystemExit(f"matplotlib is required for plotting ({e}). "
                         f"pip install -r requirements.txt")
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 110,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })
    return plt


def curve_label(c: Cell, sweep_axis: str | None) -> str:
    """Legend text for one curve.

    Deliberately just the sweep VALUE: the legend is deduplicated across panels,
    so anything cell-specific in here (terminal speed, FLAT, convergence) would
    make every panel a new entry and produce a 480-line legend. Those go on the
    curve as an annotation instead.
    """
    if sweep_axis is None:
        return c.short()
    return f"{SHORT[sweep_axis]}={c.coord(sweep_axis):g}"


def curve_annotation(c: Cell) -> str:
    """Short per-curve note drawn at the right-hand end of the line."""
    extra = []
    if c.v_term == c.v_term:
        extra.append(f"{c.v_term:.0f}")
    if c.flat:
        extra.append("FLAT")
    elif not c.converged:
        extra.append("nc")
    return "  ".join(extra)


def label_curves(ax, curves, min_gap_frac: float = 0.05):
    """Annotate each curve in a panel, without the labels landing on each other.

    Every curve in a panel ends at the same aimpoint -- that is what a cell is --
    so labelling them at their ends stacks them on one spot by construction. The
    earlier version did exactly that and printed three labels on top of each
    other, which reads as one garbled string ("3489F nc").

    So each label goes where its own curve is FURTHEST from every other curve in
    the panel, weighted toward the end of the trajectory, which is where a
    reader looks for it anyway. Curves that are about to converge are avoided,
    which is the whole point: a label is only useful where you can tell which
    line it belongs to. A faint white box keeps it legible over the grid, and
    labels that still land close together are pushed apart vertically.

    `curves` is [(xs, ys, text, colour)] in plot order; the ys arrays must share
    a length (they are all sampled on the same xi grid), which is what makes
    "the point where this curve is most separated" comparable between curves.
    """
    import numpy as np
    curves = [c for c in curves if c[2]]
    if not curves:
        return

    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap_frac
    n = min(len(c[1]) for c in curves)
    if n < 8 or len(curves) < 2:
        spots = [(c[0][-1], c[1][-1]) for c in curves]
    else:
        stack = np.array([c[1][:n] for c in curves], dtype=float)
        # late points are worth more than early ones: a label two thirds along
        # the trajectory is where you would put it by hand
        weight = 0.4 + np.linspace(0.0, 1.0, n)
        spots = []
        for i, (_xs, ys, _t, _c) in enumerate(curves):
            others = np.delete(stack, i, axis=0)
            separation = np.min(np.abs(others - stack[i]), axis=0)
            j = int(np.argmax(separation * weight))
            spots.append((_xs[j], ys[j]))

    # push apart anything that is still close, in plot order
    placed = []
    for (x, y), (_xs, _ys, text, colour) in zip(spots, curves):
        pushed = list(placed)
        for px, py in pushed:
            if abs(x - px) < 0.18 * (ax.get_xlim()[1] - ax.get_xlim()[0]) and abs(y - py) < gap:
                y = py + gap if y >= py else py - gap
        placed.append((x, y))
        ax.annotate(text, xy=(x, y), xytext=(0, 4), textcoords="offset points",
                    fontsize=6.5, color=colour, ha="center", va="bottom",
                    bbox=dict(fc="white", ec="none", alpha=0.72, pad=0.8))


def colour_for(i: int, n: int, plt):
    import numpy as np
    if n <= 10:
        return plt.get_cmap("tab10")(i % 10)
    return plt.get_cmap("viridis")(i / max(n - 1, 1)) if n > 10 else "C0"


def plot_family(cells: list[Cell], args, plt, desc: str):
    """Altitude vs downrange, one panel per (rows x cols), one line per sweep."""
    if not cells:
        raise SystemExit("nothing to plot: no cells matched. Loosen --fix / --min-* .")

    free = free_axes(cells)
    fixed = {a: values_of(cells, a)[0] for a in ORDER if a not in free}

    def pick(name, default_axis=None):
        if name:
            return resolve_axis(name)
        return default_axis

    rows_axis = pick(args.rows, free[0] if len(free) > 1 else None)
    cols_axis = pick(args.cols, free[1] if len(free) > 2 else None)
    used = {a for a in (rows_axis, cols_axis) if a}
    rest = [a for a in free if a not in used]
    sweep_axis = pick(args.sweep, rest[0] if rest else None)
    if sweep_axis in used:
        raise SystemExit(f"--sweep {sweep_axis} is also used for rows/cols")
    if sweep_axis:
        rest = [a for a in rest if a != sweep_axis]
    if rest:
        print(f"NOTE: {', '.join(rest)} also varies but is not shown. "
              f"Fix it with --fix {rest[0]}=<value> to see it.", file=sys.stderr)

    row_vals = values_of(cells, rows_axis) if rows_axis else [None]
    col_vals = values_of(cells, cols_axis) if cols_axis else [None]

    # Panel cap. Sub-sample the *outer* axis rather than refusing to draw: a 6x6
    # wall of panels is already unreadable, and the user asked for a glance.
    #
    # Sub-sample by even spacing INCLUDING both ends, and say exactly which
    # values survived. A plain stride (row_vals[::step]) drops the last value of
    # the axis, which is usually the one you were looking at -- on the Scythe
    # grid the 400 km row never appeared in a default plot, and the note said
    # "showing every 2th row", which does not tell you it is missing either.
    def thin(vals, keep):
        n = len(vals)
        if keep < 1 or n <= keep:
            return vals
        if keep == 1:
            return [vals[0]]
        idx = sorted({round(i * (n - 1) / (keep - 1)) for i in range(keep)})
        return [vals[i] for i in idx]

    n_rows, n_cols = len(row_vals), len(col_vals)
    if n_rows * n_cols > args.max_panels:
        keep = max(1, args.max_panels // max(n_cols, 1))
        kept = thin(row_vals, keep)
        dropped = [v for v in row_vals if v not in kept]
        print(f"NOTE: {n_rows}x{n_cols} = {n_rows * n_cols} panels > --max-panels "
              f"{args.max_panels}; showing {len(kept)} rows "
              f"({', '.join(f'{v:g}' for v in kept)}) and dropping {len(dropped)} "
              f"({', '.join(f'{v:g}' for v in dropped)}). "
              f"Raise --max-panels or --fix {rows_axis or 'an axis'} to see them.",
              file=sys.stderr)
        row_vals = kept

    n_rows, n_cols = len(row_vals), len(col_vals)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.6 * n_cols + 0.8, 2.7 * n_rows + 1.4),
                             squeeze=False, sharex=False, sharey=False)
    where = []
    if rows_axis:
        where.append(f"rows: {LABEL[rows_axis]}")
    if cols_axis:
        where.append(f"cols: {LABEL[cols_axis]}")
    if sweep_axis:
        where.append(f"lines: {LABEL[sweep_axis]}")
    fig.suptitle(f"planned loft trajectories -- {desc}\n" + " | ".join(where),
                 fontsize=11)

    # Curves are labelled by sweep value only, and the label set is shared across
    # every panel, so the legend belongs to the figure and not to any one axes.
    legend_handles = {}
    for i, rv in enumerate(row_vals):
        for j, cv in enumerate(col_vals):
            ax = axes[i][j]
            sub = [c for c in cells
                   if (rows_axis is None or matches(c.coord(rows_axis), rv, args.fix_tol))
                   and (cols_axis is None or matches(c.coord(cols_axis), cv, args.fix_tol))]
            if not sub:
                ax.set_visible(False)
                continue

            sweep_vals = values_of(sub, sweep_axis) if sweep_axis else [None]
            labels = []            # [(xs, ys, text, colour)] -- placed after the axes settle
            for si, sv in enumerate(sweep_vals):
                sel = [c for c in sub if sv is None or matches(c.coord(sweep_axis), sv, args.fix_tol)]
                if not sel:
                    continue
                c = sel[0]
                lw = 1.8 if sweep_axis else 2.0
                ls = "--" if c.flat else "-"
                col = colour_for(si, len(sweep_vals), plt) if sweep_axis else "C0"
                xi, h = c.alt_curve()
                if args.normalised:
                    xs, ys = xi * 100.0, h / 1000.0
                else:
                    xs, ys = xi * c.xf / 1000.0, h / 1000.0
                (ln,) = ax.plot(xs, ys, color=col, lw=lw, ls=ls,
                                label=curve_label(c, sweep_axis))
                legend_handles.setdefault(ln.get_label(), ln)
                if args.annotate and len(sweep_vals) <= args.max_legend:
                    labels.append((xs, ys, curve_annotation(c), col))

            # Endpoint reference lines, only when they are constant in this panel.
            # (before the labels: they set the y limits the labels are placed in)
            if len({round(c.h0) for c in sub}) == 1:
                ax.axhline(sub[0].h0 / 1000.0, color="0.55", lw=0.8, ls=":")
            if len({round(c.ht) for c in sub}) == 1 and not args.normalised:
                ax.axhline(sub[0].ht / 1000.0, color="0.8", lw=0.8, ls=":")

            if not args.normalised:
                for c in sub:
                    ax.plot([c.xf / 1000.0], [c.ht / 1000.0], marker="o", ms=2.5,
                            color="0.5")

            if labels and not args.normalised:
                label_curves(ax, labels)

            panel = []
            if rows_axis:
                panel.append(f"{SHORT[rows_axis]}={rv:g}")
            if cols_axis:
                panel.append(f"{SHORT[cols_axis]}={cv:g}")
            if panel:
                ax.text(0.03, 0.94, "  ".join(panel), transform=ax.transAxes,
                        fontsize=8, va="top",
                        bbox=dict(fc="white", ec="0.8", alpha=0.8, pad=1.5))
            ax.margins(x=0.04)
            ax.set_xlabel("range along launch axis (%)" if args.normalised else "downrange (km)")
            ax.set_ylabel("altitude (km)")
    if legend_handles:
        if len(legend_handles) <= args.max_legend:
            fig.legend(list(legend_handles.values()), list(legend_handles.keys()),
                       loc="lower center", ncol=min(len(legend_handles),
                                                    max(2, 6)),
                       framealpha=0.9, fontsize=8)
            fig.tight_layout(rect=(0, 0.05 + 0.02 * min(len(legend_handles), 6) // 2, 1, 0.94))
        else:
            print(f"NOTE: {len(legend_handles)} values on the line axis -- legend "
                  f"suppressed, each curve is annotated with its terminal speed and "
                  f"the axis is in the panel labels.", file=sys.stderr)
            fig.tight_layout(rect=(0, 0, 1, 0.94))
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def plot_overview(cells: list[Cell], args, plt, desc: str):
    """The health of the grid: apex, terminal speed, fit RMSE, TOF vs range."""
    if not cells:
        raise SystemExit("nothing to plot: no cells matched.")
    import numpy as np

    colour_axis = resolve_axis(args.colour) if args.colour else (
        "launch_alt_m" if len(values_of(cells, "launch_alt_m")) > 1 else "launch_speed_ms")
    cv = np.array([c.coord(colour_axis) for c in cells])
    xr = np.array([c.coord("range_km") for c in cells])
    flat = np.array([c.flat for c in cells])
    conv = np.array([c.converged for c in cells])

    panels = [
        ("apex gain (m)", np.array([c.apex_gain for c in cells]), False),
        ("terminal speed (m/s)", np.array([c.v_term for c in cells]), False),
        ("fit RMSE (m)", np.array([c.rmse for c in cells]), True),
        ("time of flight (s)", np.array([c.tof for c in cells]), False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.0), squeeze=False)
    for (name, y, logy), ax in zip(panels, axes.ravel()):
        good = ~flat
        sc = ax.scatter(xr[good], y[good], c=cv[good], cmap="viridis", s=18,
                        edgecolors="none")
        if np.any(flat):
            ax.scatter(xr[flat], y[flat], marker="x", s=26, color="crimson",
                       label=f"never lofted ({int(flat.sum())})")
        if np.any(~conv):
            ax.scatter(xr[~conv & ~flat], y[~conv & ~flat], s=42, facecolors="none",
                       edgecolors="grey", linewidths=0.7, label=f"not converged ({int((~conv & ~flat).sum())})")
        ax.set_xlabel(LABEL["range_km"])
        ax.set_ylabel(name)
        if logy:
            ax.set_yscale("log")
        if args.legend:
            ax.legend(loc="best", framealpha=0.9)
        fig.colorbar(sc, ax=ax, label=LABEL[colour_axis], pad=0.02)
    # Reference lines the mod cares about.
    ax = axes[0][1]
    ax.axhline(DEFAULT_MIN_TERMINAL, color="0.4", lw=0.9, ls="--")
    ax.axhline(SELF_DESTRUCT_SPEED, color="crimson", lw=0.9, ls=":")
    # bottom-right: the top-left corner is where the legends land
    ax.text(0.99, 0.02,
            f"dashed {DEFAULT_MIN_TERMINAL:.0f} m/s = the mod's old load-time filter\n"
            f"dotted {SELF_DESTRUCT_SPEED:.0f} m/s = ARHSeeker self-destruct",
            transform=ax.transAxes, fontsize=7, color="0.35", ha="right", va="bottom",
            bbox=dict(fc="white", ec="0.85", alpha=0.85, pad=1.5))
    axes[1][0].axhline(1000.0, color="0.4", lw=0.9, ls="--")
    if args.max_rmse is not None:
        axes[1][0].axhline(args.max_rmse, color="orange", lw=0.9, ls="-.")
    fig.suptitle(f"grid health -- {desc}   "
                 f"({len(cells)} cells, {int(flat.sum())} flat, "
                 f"{int(conv.sum())} converged)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_detail(cells: list[Cell], args, plt, desc: str):
    """One cell: the polynomial, and the stored solution if there is one."""
    import numpy as np
    if not cells:
        raise SystemExit("no cell matched --cell")
    c = cells[0]
    has_sol = c.sol is not None
    if has_sol:
        n = 3 if args.detail_extra else 1
        ncols, nrows = (2, 2)
        fig, axes = plt.subplots(nrows, ncols, figsize=(11, 6.4), squeeze=False)
        axlist = list(axes.ravel())
    else:
        fig, axes = plt.subplots(1, 1, figsize=(8, 4.6), squeeze=False)
        axlist = [axes[0][0]]

    xi, h = c.alt_curve()
    ax = axlist[0]
    ax.plot(xi * c.xf / 1000.0, h / 1000.0, color="C0", lw=2, label="planned (poly_a)")
    if has_sol:
        xs = np.asarray(c.sol["x"], dtype=float)
        hs = np.asarray(c.sol["h"], dtype=float)
        ax.plot(xs / 1000.0, hs / 1000.0, "o", ms=3, color="k",
                label="Dymos solution nodes")
    ax.axhline(c.h0 / 1000.0, color="0.5", ls=":", lw=0.9, label="launch alt")
    ax.axhline(c.ht / 1000.0, color="0.8", ls=":", lw=0.9, label="target alt")
    ax.plot([0, c.xf / 1000.0], [c.h0 / 1000.0, c.ht / 1000.0], color="0.6",
            lw=0.9, ls="--", label="endpoint chord")
    ax.plot([c.meta.get("apex_downrange_m", float("nan")) / 1000.0], [c.apex / 1000.0],
            marker="*", ms=9, color="orange", label="apex (recorded)")
    ax.set_xlabel("downrange (km)")
    ax.set_ylabel("altitude (km)")
    ax.set_title(f"{c.source} {c.key}  {c.short()}   "
                 f"V_term={c.v_term:.0f} m/s  tof={c.tof:.0f}s  rmse={c.rmse:.1f} m"
                 f"{'  FLAT' if c.flat else ''}")
    ax.legend(loc="best", framealpha=0.9)

    if has_sol:
        # poly - solution, at the solution nodes. `h` is the dense 200-point
        # curve and must not be used here: the arrays are different lengths and
        # the residual is only defined where the solver actually evaluated.
        poly_at_nodes = np.array([c.alt(x / c.xf) if c.xf > 0 else 0.0 for x in xs])
        res = poly_at_nodes - hs
        ax = axlist[1]
        ax.plot(xs / 1000.0, res, color="crimson", lw=1.4)
        ax.set_xlabel("downrange (km)")
        ax.set_ylabel("poly - solution (m)")
        ax.set_title(f"fit residual (rmse {np.sqrt(np.mean(res ** 2)):.1f} m)")
        if not args.detail_extra:
            axlist[2].axis("off")
            axlist[3].axis("off")
        else:
            ax = axlist[2]
            ax.plot(xs / 1000.0, np.asarray(c.sol["V"], dtype=float), color="C2", lw=1.6)
            ax.axhline(SELF_DESTRUCT_SPEED, color="crimson", ls=":", lw=0.9)
            ax.set_xlabel("downrange (km)")
            ax.set_ylabel("speed (m/s)")
            ax.set_title("speed")
            ax = axlist[3]
            for nm, col in (("gamma", "C3"), ("alpha", "C4")):
                if nm in c.sol:
                    ax.plot(xs / 1000.0, np.degrees(np.asarray(c.sol[nm], dtype=float)),
                            color=col, lw=1.5, label=nm)
            ax.axhline(0.0, color="0.6", lw=0.8)
            ax.set_xlabel("downrange (km)")
            ax.set_ylabel("deg")
            ax.set_title("flight path angle / AoA")
            ax.legend(loc="best", framealpha=0.9)
    fig.suptitle(f"cell detail -- {desc}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


# --------------------------------------------------------------------------
# flight logs
# --------------------------------------------------------------------------
FLIGHT_RE = re.compile(r"(M\d+)\s+\[flight\]\s+(.*)$")
KV_RE = re.compile(r"([A-Za-z_][A-Za-z_0-9]*)=(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")


def parse_flight_log(path: Path) -> dict:
    """{round id: {field: [values]}} from the mod's own [flight] lines.

    The line is:
        M1 [flight] t=1.0s s=0.006 alt=5969 ref=6294 B=344 hT=5271/5271
           xf=63.8/66.3km v=478 d=1435 err=326 pip=1 cell=67km | slope=44.1 ...
    Fields that carry a pair (hT/live/lagged, xf/live/lagged) keep the first
    number, which is the live one; the lagged one is only interesting when
    diagnosing the lag itself, and the pair would otherwise be ambiguous here.
    """
    rounds: dict[str, dict] = {}
    with path.open(errors="replace") as fh:
        for line in fh:
            m = FLIGHT_RE.search(line)
            if not m:
                continue
            rid, rest = m.group(1), m.group(2)
            d = rounds.setdefault(rid, {})
            for k, v in KV_RE.findall(rest):
                d.setdefault(k, []).append(float(v))
    return rounds


def plot_flight(logs: list[Path], args, plt):
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 9.0), sharex=False)
    shown = 0
    for path in logs:
        rounds = parse_flight_log(path)
        if not rounds:
            print(f"{path}: no [flight] lines (is this a BepInEx log?)", file=sys.stderr)
            continue
        for rid, d in sorted(rounds.items()):
            if len(d.get("t", [])) < 3:
                continue
            shown += 1
            t = d["t"]
            s = d.get("s", [float("nan")] * len(t))
            xf = d.get("xf", [float("nan")] * len(t))
            # xf is logged in km, s is the fraction of it: s*xf is km downrange
            x = [si * xi for si, xi in zip(s, xf)]
            lab = f"{path.stem} {rid}"
            axes[0].plot(x, [v / 1000.0 for v in d.get("alt", [])], lw=1.3, label=lab)
            if "ref" in d:
                axes[0].plot(x, [v / 1000.0 for v in d["ref"]], lw=1.0, ls="--",
                             color=axes[0].lines[-1].get_color())
            if "err" in d:
                axes[1].plot(s, d["err"], lw=1.3, label=lab)
            if "v" in d:
                axes[2].plot(t, d["v"], lw=1.3, label=lab)
    if not shown:
        raise SystemExit(f"no usable [flight] data in {', '.join(str(p) for p in logs)}")
    axes[0].set_xlabel("downrange (km)")
    axes[0].set_ylabel("altitude (km)")
    axes[0].set_title("solid = missile, dashed = reference it was commanded to follow")
    axes[0].legend(loc="best", fontsize=7)
    axes[1].axhline(0.0, color="0.6", lw=0.8)
    axes[1].set_xlabel("s (fraction of Xf)")
    axes[1].set_ylabel("tracking error (m)")
    axes[1].legend(loc="best", fontsize=7)
    axes[2].axhline(SELF_DESTRUCT_SPEED, color="crimson", ls=":", lw=0.9)
    axes[2].axhline(DEFAULT_MIN_TERMINAL, color="0.4", ls="--", lw=0.9)
    axes[2].set_xlabel("t since launch (s)")
    axes[2].set_ylabel("speed (m/s)")
    axes[2].legend(loc="best", fontsize=7)
    fig.suptitle("flight logs -- " + ", ".join(p.name for p in logs), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


# --------------------------------------------------------------------------
# --list
# --------------------------------------------------------------------------
def cmd_list(cells: list[Cell], args):
    rows = sorted(cells, key=lambda c: (c.coord("range_km"), c.coord("launch_alt_m"),
                                        c.coord("launch_speed_ms"), c.coord("target_alt_m")))
    head = (f"{'key':<22} {'range':>6} {'h0':>6} {'V0':>4} {'ht':>6} {'apex':>7} "
            f"{'gain':>7} {'V_term':>7} {'tof':>6} {'rmse':>8} {'conv':>4}  flags")
    print(head)
    print("-" * len(head))
    for c in rows:
        print(f"{c.key:<22} {c.coord('range_km'):6.0f} {c.coord('launch_alt_m'):6.0f} "
              f"{c.coord('launch_speed_ms'):4.0f} {c.coord('target_alt_m'):6.0f} "
              f"{c.apex:7.0f} {c.apex_gain:7.0f} {c.v_term:7.0f} {c.tof:6.1f} "
              f"{c.rmse:8.1f} {'yes' if c.converged else 'no':>4}  {c.flags()}")
    n_flat = sum(1 for c in rows if c.flat)
    slow = sum(1 for c in rows if c.v_term == c.v_term and c.v_term < DEFAULT_MIN_TERMINAL)
    print(f"\n{len(rows)} cells | {n_flat} never lofted | {slow} below "
          f"{DEFAULT_MIN_TERMINAL:.0f} m/s | {sum(1 for c in rows if c.converged)} converged")


# --------------------------------------------------------------------------
# --html
# --------------------------------------------------------------------------
def fig_to_svg(fig, plt) -> str:
    from io import StringIO
    buf = StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    svg = buf.getvalue()
    i = svg.find("<svg")
    if i > 0:
        svg = svg[i:]
    return svg.replace("<svg ", '<svg style="max-width:100%;height:auto" ', 1)


def write_html(path: Path, figs: list[tuple[str, object]], cells: list[Cell], desc: str, plt):
    """Self-contained page: inline SVG, no scripts, no network. It renders the
    same offline as it does on the machine that made it."""
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>NOLoftFix trajectories - {desc}</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:24px;background:#fafafa;"
        "color:#222;max-width:1500px}h1{font-size:20px}h2{font-size:15px;margin-top:28px}"
        "table{border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums}"
        "th,td{border-bottom:1px solid #ddd;padding:2px 8px;text-align:right}"
        "th{background:#eee;position:sticky;top:0}tr.flat{background:#fde8e8}"
        ".note{color:#666;font-size:12px}</style></head><body>",
        f"<h1>NOLoftFix planned trajectories</h1><p class='note'>{desc}</p>",
    ]
    for title, fig in figs:
        parts.append(f"<h2>{title}</h2>")
        parts.append(fig_to_svg(fig, plt))
    if cells:
        rows = sorted(cells, key=lambda c: (c.coord("range_km"), c.coord("launch_alt_m"),
                                            c.coord("launch_speed_ms"), c.coord("target_alt_m")))
        parts.append("<h2>cells shown</h2><table><tr>"
                     "<th>range km</th><th>launch alt m</th><th>launch m/s</th>"
                     "<th>target alt m</th><th>apex m</th><th>apex gain m</th>"
                     "<th>V_term m/s</th><th>tof s</th><th>rmse m</th><th>conv</th>"
                     "<th>source</th></tr>")
        for c in rows:
            cls = " class='flat'" if c.flat else ""
            parts.append(
                f"<tr{cls}><td>{c.coord('range_km'):.0f}</td>"
                f"<td>{c.coord('launch_alt_m'):.0f}</td>"
                f"<td>{c.coord('launch_speed_ms'):.0f}</td>"
                f"<td>{c.coord('target_alt_m'):.0f}</td><td>{c.apex:.0f}</td>"
                f"<td>{c.apex_gain:.0f}</td><td>{c.v_term:.0f}</td>"
                f"<td>{c.tof:.1f}</td><td>{c.rmse:.1f}</td>"
                f"<td>{'yes' if c.converged else 'no'}</td><td>{c.source}</td></tr>")
        parts.append("</table>")
    parts.append("</body></html>")
    path.write_text("\n".join(parts))
    print(f"wrote {path}")


# --------------------------------------------------------------------------
def describe(cells: list[Cell], args) -> str:
    bits = []
    fix = parse_fix(args.fix)
    for a in ORDER:
        if a in fix:
            bits.append(f"{SHORT[a]}={fix[a]:g}")
    for k, v in (("minV", args.min_terminal_speed), ("maxV", args.max_terminal_speed),
                 ("maxRmse", args.max_rmse)):
        if v is not None:
            bits.append(f"{k}={v:g}")
    if args.hide_flat:
        bits.append("flat hidden")
    if args.only_converged:
        bits.append("converged only")
    bits.append(f"{len(cells)} cells")
    return " ".join(bits)


def main():
    ap = argparse.ArgumentParser(
        description="Plot planned loft trajectories from a merged table or a grid run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("WHY THIS EXISTS")[0].strip())
    src = ap.add_argument_group("input (at least one)")
    src.add_argument("--table", action="append", default=[], metavar="FILE",
                     help="merged table JSON (grid_dymos.py --merge). Repeatable.")
    src.add_argument("--include-attempts", action="store_true",
                     help="with --runs, also load the <key>.attempt<N>.json files "
                          "--reoptimise leaves behind, as separate trajectories")
    src.add_argument("--runs", metavar="DIR",
                     help="grid run directory, e.g. runs/scythe (reads cells/*.json)")
    src.add_argument("--flight", action="append", default=[], metavar="LOG",
                     help="BepInEx log with [flight] lines; plots what actually flew")
    sel = ap.add_argument_group("selection")
    sel.add_argument("--fix", action="append", default=[], metavar="AXIS=VALUE",
                     help="keep only cells on this axis value. Repeatable.")
    sel.add_argument("--fix-tol", type=float, default=1e-3,
                     help="tolerance for --fix matching (default 1e-3)")
    sel.add_argument("--min-terminal-speed", type=float, default=None,
                     help=f"hide cells below this V_term (the mod's filter is "
                          f"{DEFAULT_MIN_TERMINAL:.0f})")
    sel.add_argument("--max-terminal-speed", type=float, default=None,
                     help="hide cells above this V_term (catches solver outliers)")
    sel.add_argument("--max-rmse", type=float, default=None,
                     help="hide cells whose poly fit is worse than this")
    sel.add_argument("--hide-flat", action="store_true",
                     help="hide cells that never lofted (apex == launch alt)")
    sel.add_argument("--only-converged", action="store_true",
                     help="hide cells the solver flagged as not converged")
    lay = ap.add_argument_group("layout")
    lay.add_argument("--rows", metavar="AXIS", help="axis to put down the grid")
    lay.add_argument("--cols", metavar="AXIS", help="axis to put across the grid")
    lay.add_argument("--sweep", metavar="AXIS", help="axis that becomes the lines")
    lay.add_argument("--colour", metavar="AXIS",
                     help="axis to colour by in --overview (default launch_alt_m)")
    lay.add_argument("--normalised", action="store_true",
                     help="x axis as %% of Xf instead of km")
    lay.add_argument("--max-panels", type=int, default=36)
    lay.add_argument("--legend", action="store_true", default=True)
    lay.add_argument("--no-legend", dest="legend", action="store_false")
    lay.add_argument("--annotate", action="store_true", default=True,
                     help="label each curve with its terminal speed (default on)")
    lay.add_argument("--no-annotate", dest="annotate", action="store_false")
    lay.add_argument("--max-legend", type=int, default=12,
                     help="suppress the legend above this many lines (default 12)")
    out = ap.add_argument_group("output")
    out.add_argument("--overview", action="store_true", help="grid health panels instead")
    out.add_argument("--cell", metavar="KEY", help="detail view of one cell (key or index)")
    out.add_argument("--detail-extra", action="store_true",
                     help="in --cell, also plot speed and gamma/alpha when stored")
    out.add_argument("--list", action="store_true", help="print the cells as a table")
    out.add_argument("--compare", action="store_true",
                     help="with two or more --table, label each table's cells separately")
    out.add_argument("--save", metavar="FILE",
                     help="write a PNG here (default: trajectories.png when nothing else is asked)")
    out.add_argument("--html", metavar="FILE", help="write a self-contained HTML page")
    out.add_argument("--show", action="store_true", help="open a window (needs a display)")
    out.add_argument("--dpi", type=int, default=140)
    args = ap.parse_args()

    # ---- load ----------------------------------------------------------
    cells: list[Cell] = []
    notes: list[str] = []
    if args.table:
        multi = len(args.table) > 1
        for p in args.table:
            cs, nt = load_table(Path(p), label=(Path(p).stem if multi else None))
            cells += cs
            notes += nt
    if args.runs:
        cs, nt = load_runs(Path(args.runs), args.include_attempts)
        cells += cs
        notes += nt
    if not cells and not args.flight:
        for n in notes:
            print(n, file=sys.stderr)
        raise SystemExit("no cells loaded: pass --table FILE or --runs DIR (--help for usage)")
    for n in notes:
        print(n, file=sys.stderr)

    if args.flight:
        plt = import_mpl(args.show)
        figs = [("flight logs", plot_flight([Path(p) for p in args.flight], args, plt))]
        save_figs(figs, args, plt, cells=[])
        return

    cells = select(cells, args)
    if not cells:
        raise SystemExit("selection is empty -- check --fix values and --min/--max filters")

    if args.list:
        cmd_list(cells, args)
        if not (args.save or args.html or args.show):
            return

    plt = import_mpl(args.show)
    desc = describe(cells, args)

    # ---- figures -------------------------------------------------------
    figs = []
    if args.cell is not None:
        want = args.cell
        pick = [c for c in cells if c.key == want]
        if not pick and want.lstrip("#").isdigit():
            i = int(want.lstrip("#"))
            pick = [c for c in cells if c.meta.get("_index") == f"#{i}"] or (
                [cells[i % len(cells)]] if cells else [])
        if not pick:
            keys = ", ".join(c.key for c in cells[:4])
            raise SystemExit(
                f"no cell with key '{want}' in the selection ({len(cells)} cells). "
                f"Keys look like {keys} -- the axis values in order "
                f"(range_km_launch_alt_m_launch_speed_ms_target_alt_m_target_speed_ms), "
                f"or a position like #12. --list prints every one.")
        figs.append((f"cell {pick[0].key}", plot_detail(pick, args, plt, desc)))
    elif args.overview:
        figs.append(("grid health", plot_overview(cells, args, plt, desc)))
    else:
        multi = args.compare and len(args.table) > 1
        if multi:
            # one figure per source, plus the combined grid with the sweep axis
            # forced onto the source so the two families are directly comparable.
            for src in sorted({c.source for c in cells}):
                sub = [c for c in cells if c.source == src]
                figs.append((f"{src} ({len(sub)} cells)",
                             plot_family(sub, args, plt, f"{desc} | {src}")))
            args_sweep = argparse.Namespace(**vars(args))
            args_sweep.sweep = args.sweep
            args_sweep.rows = args.rows
            args_sweep.cols = args.cols
            args_sweep.legend = True
            figs.append(("all sources", plot_family(cells, args_sweep, plt, desc)))
        else:
            figs.append(("trajectories", plot_family(cells, args, plt, desc)))

    save_figs(figs, args, plt, cells)


def save_figs(figs, args, plt, cells):
    if args.html:
        write_html(Path(args.html), figs, cells, describe(cells, args) if cells else "", plt)
    if args.show:
        plt.show()
    if args.save or not args.html:
        dest = Path(args.save) if args.save else Path("trajectories.png")
        figs[0][1].savefig(dest, dpi=args.dpi, bbox_inches="tight")
        print(f"wrote {dest}")
        for extra in figs[1:]:
            alt = dest.with_name(f"{dest.stem}-{_slug(extra[0])}{dest.suffix}")
            extra[1].savefig(alt, dpi=args.dpi, bbox_inches="tight")
            print(f"wrote {alt}")


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40]


if __name__ == "__main__":
    main()
