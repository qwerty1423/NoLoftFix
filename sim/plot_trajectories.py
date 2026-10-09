#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

AXES = ["range_km", "launch_alt_m", "launch_speed_ms",
        "target_alt_m", "target_speed_ms"]

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

LEGACY = {
    "range_nmi": ("range_km", 1.852),
    "launch_alt_ft": ("launch_alt_m", 0.3048),
    "launch_kt": ("launch_speed_ms", 0.514444),
    "target_alt_ft": ("target_alt_m", 0.3048),
    "target_kt": ("target_speed_ms", 0.514444),
}

PANEL_ROWS = 6

SELF_DESTRUCT_SPEED = 200.0
DEFAULT_MIN_TERMINAL = 800.0
DEFAULT_SPEED_TOL = 0.25

HEALTH_STATES = ("ok", "woven", "infeasible", "unverified")
STATUS_ORDER = HEALTH_STATES + ("unknown",)
STATUS_LINE = {
    "ok": "-",
    "woven": (0, (7, 2)),
    "infeasible": (0, (2, 1.6)),
    "unverified": (0, (1, 1.6)),
    "unknown": "-",
}
STATUS_MARKER = {"ok": "o", "woven": "^", "infeasible": "x",
                 "unverified": "o", "unknown": "o"}
STATUS_NOTE = {"ok": "", "woven": "WOVEN", "infeasible": "INFEASIBLE",
               "unverified": "unverified", "unknown": ""}


@dataclass
class Cell:
    coords: dict
    poly: list
    meta: dict = field(default_factory=dict)
    sol: dict | None = None
    source: str = ""
    key: str = ""

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
    def health(self) -> str | None:
        h = self.meta.get("health")
        return h if h in HEALTH_STATES else None

    @property
    def status(self) -> str:
        return self.health or "unknown"

    @property
    def feasible(self) -> bool | None:
        f = self.meta.get("feasible")
        return None if f is None else bool(f)

    @property
    def infeasible(self) -> bool:
        return self.status == "infeasible"

    @property
    def woven(self) -> bool:
        return self.status == "woven"

    @property
    def violation(self) -> str | None:
        v = self.meta.get("violation")
        if isinstance(v, dict):
            v = _get(v, "violation", "text", "reason")
        return str(v) if v else None

    @property
    def defect(self) -> float:
        return _fnum(self.meta.get("defect_ratio"))

    @property
    def v_min(self) -> float:
        return _fnum(self.meta.get("v_min_ms"))

    @property
    def nav_min(self) -> float:
        return _fnum(self.meta.get("nav_min_g"))

    @property
    def authority_floor(self) -> float:
        return _fnum(self.meta.get("authority_floor_g"))

    @property
    def speed_floor(self) -> float:
        return _fnum(self.meta.get("speed_floor_ms"))

    @property
    def t_bound(self) -> str | None:
        b = self.meta.get("t_bound")
        return str(b) if b else None

    @property
    def exp_max_range(self) -> float:
        return _fnum(self.meta.get("exp_max_range_km"))

    @property
    def reachable(self) -> bool | None:
        r = self.meta.get("reachable")
        return None if r is None else bool(r)

    @property
    def beyond_envelope(self) -> bool:
        return self.reachable is False

    @property
    def attempts(self) -> int:
        try:
            return int(self.meta.get("_attempts", 1) or 1)
        except (TypeError, ValueError):
            return 1

    @property
    def elapsed(self) -> float:
        return _fnum(self.meta.get("_elapsed_s"))

    @property
    def flat(self) -> bool:
        return False

    def alt(self, xi):
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
        return (f"{self.coords['range_km']:.0f}km "
                f"h0={self.coords['launch_alt_m']:.0f} "
                f"V0={self.coords['launch_speed_ms']:.0f} "
                f"ht={self.coords['target_alt_m']:.0f}")

    def flags(self) -> str:
        f = []
        if self.status == "infeasible":
            f.append("INFEASIBLE")
        elif self.status == "woven":
            f.append("woven")
        if self.beyond_envelope:
            f.append("beyond-envelope")
        if self.flat:
            f.append("FLAT")
        if self.rmse == self.rmse and self.rmse > 1000.0:
            f.append(f"rmse={self.rmse:.0f}")
        if self.defect == self.defect and self.defect > 1.0 + DEFAULT_SPEED_TOL:
            f.append(f"defect={self.defect:.2f}")
        if self.t_bound:
            f.append(f"t={self.t_bound}")
        if self.v_min == self.v_min and self.v_min < SELF_DESTRUCT_SPEED:
            f.append(f"v_min={self.v_min:.0f}")
        if not self.converged:
            f.append("no-conv")
        if self.status == "unknown":
            f.append("unchecked")
        return " ".join(f)


def _fnum(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _get(d: dict, *names, default=None):
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return default


def cell_from_json(obj: dict, source: str, key: str = "") -> Cell | None:
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
             "_settings", "generated",
             "health", "feasible", "violation", "defect_ratio",
             "v_min_ms", "nav_min_g", "authority_floor_g", "speed_floor_ms",
             "t_bound", "duration_bounds_s", "refine_error", "refine_segments",
             "exp_max_range_km", "reachable")
            if k in obj}
    sol = obj.get("_sol") if isinstance(obj.get("_sol"), dict) else None
    if sol is not None:
        need = ("x", "h")
        if not all(isinstance(sol.get(k), list) and len(sol[k]) > 1 for k in need):
            sol = None

    if not key or key.startswith("#"):
        derived = "_".join(f"{coords[a]:g}" for a in AXES)
        key = obj.get("_key") or derived
    meta.setdefault("_index", key if key.startswith("#") else None)
    return Cell(coords=coords, poly=poly, meta=meta, sol=sol, source=source, key=key)


def load_table(path: Path, label: str | None = None) -> tuple[list[Cell], list[str]]:
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
            c.meta.setdefault("_note", "attempt " +
                              ATTEMPT_RE.search(p.stem).group()[8:])
        elif c.meta.get("_attempts", 1) > 1:
            c.meta.setdefault("_note", f"best of {c.meta['_attempts']}")
        cells.append(c)
    notes.append(f"{root.name}: {len(cells)} cells from {cells_dir}"
                 + (f" ({skipped} unusable files skipped)" if skipped else "")
                 + (f" ({n_att} attempt files ignored, --include-attempts to see them)"
                    if n_att else ""))
    return cells, notes


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
    want_health = set(args.health or [])
    dropped: dict[str, int] = {}

    def drop(why: str):
        dropped[why] = dropped.get(why, 0) + 1

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
        if want_health and c.status not in want_health:
            drop(f"status not in {','.join(sorted(want_health))}")
            continue
        if args.reachable_only and c.reachable is not True:
            drop("no envelope verdict" if c.reachable is None else "beyond envelope")
            continue
        if args.unreachable and not c.beyond_envelope:
            drop("no envelope verdict" if c.reachable is None else "within envelope")
            continue
        if args.min_v_along is not None and not (c.v_min >= args.min_v_along):
            drop("no v_min recorded" if c.v_min !=
                 c.v_min else "v_min below the floor")
            continue
        if args.max_defect is not None and not (c.defect <= args.max_defect):
            drop("no defect ratio" if c.defect !=
                 c.defect else "defect above the limit")
            continue
        out.append(c)
    for why, n in sorted(dropped.items()):
        print(f"NOTE: {n} cell(s) dropped by a filter: {why}.",
              file=sys.stderr)
    return out


def values_of(cells: list[Cell], axis: str) -> list[float]:
    return sorted({round(c.coord(axis), 6) for c in cells})


def free_axes(cells: list[Cell], cut: int = 1) -> list[str]:
    return [a for a in ORDER if len(values_of(cells, a)) > cut]


def import_mpl(show: bool):
    try:
        import matplotlib
    except Exception as e:
        raise SystemExit(f"matplotlib is required for plotting ({e}). "
                         f"pip install -r requirements.txt")
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 120,
        "figure.facecolor": "white",
        "axes.grid": True,
        "grid.color": "#d8dde3",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.9,
        "axes.axisbelow": True,
        "axes.edgecolor": "#9aa3ad",
        "axes.linewidth": 0.8,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "axes.titleweight": "semibold",
        "legend.fontsize": 8,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.color": "#4b5563",
        "ytick.color": "#4b5563",
        "font.size": 9,
    })
    return plt


def curve_label(c: Cell, sweep_axis: str | None) -> str:
    if sweep_axis is None:
        return c.short()
    return f"{SHORT[sweep_axis]}={c.coord(sweep_axis):g}"


def curve_annotation(c: Cell) -> str:
    extra = []
    if c.v_term == c.v_term:
        extra.append(f"{c.v_term:.0f}")
    note = STATUS_NOTE[c.status]
    if note:
        extra.append(note)
    elif not c.converged:
        extra.append("nc")
    if c.flat:
        extra.append("FLAT")
    if c.beyond_envelope:
        extra.append("beyond")
    return "  ".join(extra)


def label_curves(ax, curves, min_gap_frac: float = 0.05):
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
        weight = 0.4 + np.linspace(0.0, 1.0, n)
        spots = []
        for i, (_xs, ys, _t, _c) in enumerate(curves):
            others = np.delete(stack, i, axis=0)
            separation = np.min(np.abs(others - stack[i]), axis=0)
            j = int(np.argmax(separation * weight))
            spots.append((_xs[j], ys[j]))

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


PALETTES = {
    "auto": None,
    "tab10": "tab10",
    "okabe-ito": ["#0072B2", "#E69F00", "#009E73", "#D55E00",
                  "#CC79A7", "#56B4E9", "#F0E442", "#333333"],
    "viridis": "viridis",
    "cividis": "cividis",
}
SEQUENTIAL = ("viridis", "cividis", "plasma", "magma", "Greys")


def colour_for(i: int, n: int, plt, palette: str = "auto"):
    import numpy as np
    pal = PALETTES.get(palette)
    if isinstance(pal, list):
        return pal[i % len(pal)]
    if pal is None:
        pal = "tab10" if n <= 10 else "viridis"
    if isinstance(pal, str) and pal == "tab10":
        return plt.get_cmap("tab10")(i % 10)
    return plt.get_cmap(pal)(i / max(n - 1, 1))


def plot_family(cells: list[Cell], args, plt, desc: str):
    if not cells:
        raise SystemExit(
            "nothing to plot: no cells matched. Loosen --fix / --min-* .")

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

    pairs = [(rv, cv) for rv in row_vals for cv in col_vals]
    if len(pairs) > PANEL_ROWS * max(len(col_vals), 1):
        n_cols = math.ceil(len(pairs) / PANEL_ROWS)
        n_rows = math.ceil(len(pairs) / n_cols)
    else:
        n_rows, n_cols = max(len(row_vals), 1), max(len(col_vals), 1)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.5 * n_cols + 0.7, 2.5 * n_rows + 1.3),
                             squeeze=False, sharex=False, sharey=False,
                             layout="constrained")
    for k in range(len(pairs), n_rows * n_cols):
        axes[k // n_cols][k % n_cols].set_visible(False)
    where = []
    if rows_axis:
        where.append(f"rows: {LABEL[rows_axis]}")
    if cols_axis:
        where.append(f"cols: {LABEL[cols_axis]}")
    if sweep_axis:
        where.append(f"lines: {LABEL[sweep_axis]}")
    fig.suptitle(f"planned loft trajectories -- {desc}\n" + " | ".join(where),
                 fontsize=11)

    legend_handles = {}
    for idx, (rv, cv) in enumerate(pairs):
        ax = axes[idx // n_cols][idx % n_cols]
        if True:
            sub = [c for c in cells
                   if (rows_axis is None or matches(c.coord(rows_axis), rv, args.fix_tol))
                   and (cols_axis is None or matches(c.coord(cols_axis), cv, args.fix_tol))]
            if not sub:
                ax.set_visible(False)
                continue

            sweep_vals = values_of(sub, sweep_axis) if sweep_axis else [None]

            labels = []
            for si, sv in enumerate(sweep_vals):
                sel = [c for c in sub if sv is None or matches(
                    c.coord(sweep_axis), sv, args.fix_tol)]
                if not sel:
                    continue
                c = sel[0]
                lw = 1.8 if sweep_axis else 2.0

                ls = STATUS_LINE[c.status]
                if c.flat:
                    ls = (0, (4, 1.6, 1, 1.6))
                col = colour_for(si, len(sweep_vals), plt,
                                 args.palette) if sweep_axis else "C0"
                xi, h = c.alt_curve()
                if args.normalised:
                    xs, ys = xi * 100.0, h / 1000.0
                else:
                    xs, ys = xi * c.xf / 1000.0, h / 1000.0
                (ln,) = ax.plot(xs, ys, color=col, lw=lw, ls=ls, alpha=0.92,
                                label=curve_label(c, sweep_axis))
                legend_handles.setdefault(ln.get_label(), ln)

                mfc = "none" if (c.beyond_envelope or c.status in (
                    "unverified", "unknown")) else col
                ax.plot([xs[-1]], [ys[-1]], marker=STATUS_MARKER[c.status], ms=3.6,
                        color=col, markerfacecolor=mfc, markeredgewidth=0.9,
                        linestyle="none")
                if args.annotate and len(sweep_vals) <= args.max_legend:
                    labels.append((xs, ys, curve_annotation(c), col))

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
            ax.set_xlabel(
                "range along launch axis (%)" if args.normalised else "downrange (km)")
            ax.set_ylabel("altitude (km)")

    from matplotlib.lines import Line2D
    present = {c.status for c in cells}
    for s in STATUS_ORDER:
        if s not in present or s == "ok":
            continue
        legend_handles.setdefault(
            {"woven": "woven -- up-down-up, not a loft",
             "infeasible": "infeasible -- does not fly its own claim",
             "unverified": "no verdict recorded",
             "unknown": "no verdict recorded"}[s],
            Line2D([], [], color="0.35", ls=STATUS_LINE[s], lw=1.4))
    if any(c.beyond_envelope for c in cells):
        legend_handles.setdefault("beyond the solved envelope (hollow marker)",
                                  Line2D([], [], color="crimson", ls="none", marker="s",
                                         ms=4, mfc="none", markeredgewidth=0.9))
    if legend_handles:
        if len(legend_handles) <= args.max_legend:
            fig.legend(list(legend_handles.values()), list(legend_handles.keys()),
                       loc="outside lower center",
                       ncol=min(len(legend_handles), 6),
                       framealpha=0.92, fontsize=8)
        else:
            print(f"NOTE: {len(legend_handles)} values on the line axis -- legend "
                  f"suppressed, each curve is annotated with its terminal speed and "
                  f"the axis is in the panel labels.", file=sys.stderr)
    return fig


def overview_panels(cells: list[Cell]) -> list[tuple[str, list, bool]]:
    cands = [
        ("apex gain (m)", [c.apex_gain for c in cells], False, True),
        ("terminal speed (m/s)", [c.v_term for c in cells], False, True),
        ("lowest speed along the path (m/s)",
         [c.v_min for c in cells], False, False),
        ("lowest available load factor (g)", [
         c.nav_min for c in cells], True, False),
        ("fit RMSE (m)", [c.rmse for c in cells], True, True),
        ("time of flight (s)", [c.tof for c in cells], False, True),
        ("recorded / re-integrated speed",
         [c.defect for c in cells], False, False),
    ]
    import numpy as np
    out = []
    for name, vals, logy, always in cands:
        v = np.asarray(vals, dtype=float)
        if always or np.isfinite(v).any():
            out.append((name, vals, logy))
    return out


def plot_overview(cells: list[Cell], args, plt, desc: str):
    if not cells:
        raise SystemExit("nothing to plot: no cells matched.")
    import numpy as np

    colour_axis = resolve_axis(args.colour) if args.colour else (
        "launch_alt_m" if len(values_of(cells, "launch_alt_m")) > 1 else "launch_speed_ms")
    cv = np.array([c.coord(colour_axis) for c in cells])
    xr = np.array([c.coord("range_km") for c in cells])
    panels = overview_panels(cells)

    n = len(panels)
    ncols = min(3, n)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.7 * ncols + 0.7, 2.9 * nrows + 1.0),
                             squeeze=False, layout="constrained")
    axlist = [ax for row in axes for ax in row]
    for ax in axlist[n:]:
        ax.set_visible(False)

    unreach = np.array([c.beyond_envelope for c in cells])
    sc = None
    for (name, yv, logy), ax in zip(panels, axlist):
        y = np.asarray(yv, dtype=float)
        okrange = np.isfinite(y)
        for status, marker, s in (("infeasible", "x", 34), ("woven", "^", 26),
                                  ("unverified", "o", 16), ("unknown", "o", 16)):
            m = np.array([c.status == status for c in cells]
                         ) & okrange & ~unreach
            if not np.any(m):
                continue
            if status in ("infeasible", "woven"):
                ax.scatter(xr[m], y[m], marker=marker, s=s, color="crimson" if status == "infeasible" else "darkorange",
                           edgecolors="none", label=status, zorder=4)
            else:
                ax.scatter(xr[m], y[m], marker=marker, s=s, facecolors="none",
                           edgecolors="0.6", linewidths=0.7, label="no verdict recorded", zorder=3)
        m = np.array([c.status == "ok" for c in cells]) & okrange & ~unreach
        if np.any(m):
            cmap = args.palette if args.palette in SEQUENTIAL else "viridis"
            sc = ax.scatter(xr[m], y[m], c=cv[m], cmap=cmap, s=18, edgecolors="none",
                            zorder=2)
        m = unreach & okrange
        if np.any(m):
            ax.scatter(xr[m], y[m], marker="s", s=44, facecolors="none", edgecolors="crimson",
                       linewidths=1.0, zorder=5, label="beyond envelope")
        ax.set_xlabel(LABEL["range_km"])
        ax.set_ylabel(name)
        if logy:
            ax.set_yscale("log")
    if sc is not None:
        fig.colorbar(
            sc, ax=axlist, label=LABEL[colour_axis], pad=0.015, fraction=0.02)

    by_name = {name: ax for (name, _v, _l), ax in zip(panels, axlist)}
    ax = by_name.get("terminal speed (m/s)")
    if ax is not None:
        ax.axhline(DEFAULT_MIN_TERMINAL, color="0.4", lw=0.9, ls="--")
        ax.axhline(SELF_DESTRUCT_SPEED, color="crimson", lw=0.9, ls=":")
        ax.text(0.99, 0.02,
                f"dashed {DEFAULT_MIN_TERMINAL:.0f} m/s = the mod's load-time filter\n"
                f"dotted {SELF_DESTRUCT_SPEED:.0f} m/s = self-destruct",
                transform=ax.transAxes, fontsize=7, color="0.35", ha="right", va="bottom",
                bbox=dict(fc="white", ec="0.85", alpha=0.85, pad=1.5))
    ax = by_name.get("lowest speed along the path (m/s)")
    if ax is not None:
        ax.axhline(SELF_DESTRUCT_SPEED, color="crimson", lw=0.9, ls=":")
        floors = [c.speed_floor for c in cells if c.speed_floor ==
                  c.speed_floor and c.speed_floor > 0]
        if floors:
            ax.axhline(max(floors), color="0.4", lw=0.9, ls="--")
            ax.text(0.99, 0.02, f"dashed {max(floors):.0f} m/s = the floor these cells were "
                    f"solved with", transform=ax.transAxes, fontsize=7, color="0.35",
                    ha="right", va="bottom", bbox=dict(fc="white", ec="0.85", alpha=0.85, pad=1.5))
    ax = by_name.get("lowest available load factor (g)")
    if ax is not None:
        auth = [c.authority_floor for c in cells
                if c.authority_floor == c.authority_floor and c.authority_floor > 0]
        if auth:
            ax.axhline(max(auth), color="0.4", lw=0.9, ls="--")
            ax.text(0.99, 0.95, f"dashed {max(auth):g} g = the authority floor",
                    transform=ax.transAxes, fontsize=7, color="0.35", ha="right", va="top",
                    bbox=dict(fc="white", ec="0.85", alpha=0.85, pad=1.5))
    ax = by_name.get("recorded / re-integrated speed")
    if ax is not None:
        ax.axhline(1.0, color="0.3", lw=0.9)
        ax.axhline(1.0 + DEFAULT_SPEED_TOL, color="crimson", lw=0.9, ls="--")
        ax.text(0.99, 0.95, f"1.0 = the claim is what flying it gives\n"
                f"dashed {1.0 + DEFAULT_SPEED_TOL:g} = --speed-tol",
                transform=ax.transAxes, fontsize=7, color="0.35", ha="right", va="top",
                bbox=dict(fc="white", ec="0.85", alpha=0.85, pad=1.5))
    ax = by_name.get("fit RMSE (m)")
    if ax is not None:
        ax.axhline(1000.0, color="0.4", lw=0.9, ls="--")
        if args.max_rmse is not None:
            ax.axhline(args.max_rmse, color="orange", lw=0.9, ls="-.")

    fig.suptitle(f"grid health -- {desc}", fontsize=11)
    handles = []
    for ax in axlist:
        if ax.get_visible() and ax.get_legend_handles_labels()[0]:
            handles = ax.get_legend_handles_labels()
            break
    if handles[0]:
        axlist[0].legend(*handles, loc="upper left",
                         fontsize=7, framealpha=0.9)
    return fig


def plot_detail(cells: list[Cell], args, plt, desc: str):
    import numpy as np
    if not cells:
        raise SystemExit("no cell matched --cell")
    c = cells[0]
    has_sol = c.sol is not None
    if has_sol:
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.6), squeeze=False,
                                 layout="constrained")
        axlist = list(axes.ravel())
    else:
        fig, axes = plt.subplots(1, 1, figsize=(8.5, 4.8), squeeze=False,
                                 layout="constrained")
        axlist = [axes[0][0]]

    xi, h = c.alt_curve()
    ax = axlist[0]
    ax.plot(xi * c.xf / 1000.0, h / 1000.0,
            color="C0", lw=2, label="planned (poly_a)")
    if has_sol:
        xs = np.asarray(sol_series(c.sol, "x") or [], dtype=float)
        hs = np.asarray(sol_series(c.sol, "h") or [], dtype=float)
        if xs.size and xs.size == hs.size:
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

    ax.set_title(f"{c.source} {c.key}   {c.short()}\n"
                 f"V_term {c.v_term:.0f} m/s   tof {c.tof:.1f} s   "
                 f"apex {c.apex:.0f} m   rmse {c.rmse:.1f} m", fontsize=8.5)
    ax.legend(loc="best", framealpha=0.92, fontsize=7.5)

    facts = []
    verdict = {"infeasible": "INFEASIBLE", "woven": "woven profile",
               "unknown": "no verdict recorded"}.get(c.status, "")
    if verdict:
        facts.append(f"--check: {verdict}" +
                     (f" -- {c.violation}" if c.violation else ""))
    if c.flat:
        facts.append("apex == launch altitude")
    if c.beyond_envelope:
        facts.append(f"beyond the solved envelope ({c.exp_max_range:.0f} km)")
    if c.t_bound:
        facts.append(f"flight time ends on its {c.t_bound} duration bound")
    if c.feasible is not None:
        facts.append("constraint audit: " + ("satisfied" if c.feasible
                                             else f"VIOLATED -- {c.violation or 'see the cell'}"))
    if c.defect == c.defect:
        facts.append(f"recorded / re-integrated speed: {c.defect:.2f}"
                     + ("  (over the 1.25 tolerance)" if c.defect > 1.0 + DEFAULT_SPEED_TOL else ""))
    if c.v_min == c.v_min:
        facts.append(f"lowest speed on the path: {c.v_min:.0f} m/s"
                     + (f"  (floor {c.speed_floor:.0f})" if c.speed_floor > 0 else ""))
    if c.nav_min == c.nav_min:
        facts.append(f"lowest available load factor: {c.nav_min:.2f} g"
                     + (f"  (floor {c.authority_floor:g})" if c.authority_floor > 0 else ""))
    if c.attempts > 1:
        facts.append(f"best of {c.attempts} attempts")
    if c.elapsed == c.elapsed:
        facts.append(f"solved in {c.elapsed:.0f} s")
    if facts:
        ax.text(0.02, 0.02, "\n".join(facts), transform=ax.transAxes, fontsize=7,
                color="0.25", ha="left", va="bottom",
                bbox=dict(fc="white", ec="0.85", alpha=0.9, pad=2.2))

    if has_sol and xs.size and xs.size == hs.size:
        poly_at_nodes = np.array(
            [c.alt(x / c.xf) if c.xf > 0 else 0.0 for x in xs])
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
            vv = np.asarray(sol_series(c.sol, "V") or [], dtype=float)
            if vv.size != xs.size:
                vv = np.array([])
            ax.plot(xs[:vv.size] / 1000.0, vv, color="C2", lw=1.6)
            ax.axhline(SELF_DESTRUCT_SPEED, color="crimson", ls=":", lw=0.9,
                       label=f"self-destruct {SELF_DESTRUCT_SPEED:.0f}")

            if c.v_min == c.v_min and vv.size:
                j = int(np.nanargmin(vv))
                ax.plot([xs[j] / 1000.0], [c.v_min], marker="v", ms=6, color="crimson",
                        label=f"path minimum {c.v_min:.0f}")
            ax.legend(loc="best", fontsize=7, framealpha=0.9)
            ax.set_xlabel("downrange (km)")
            ax.set_ylabel("speed (m/s)")
            ax.set_title("speed")
            ax = axlist[3]
            for nm, col in (("gamma", "C3"), ("alpha", "C4")):
                y = np.asarray(sol_series(c.sol, nm) or [], dtype=float)
                if not y.size:
                    continue

                if len(y) == len(xs):
                    xv = xs
                elif len(y) == len(xs) - 1:
                    xv = 0.5 * (xs[:-1] + xs[1:])
                else:
                    continue
                ax.plot(xv / 1000.0, np.degrees(y),
                        color=col, lw=1.5, label=nm)
            ax.axhline(0.0, color="0.6", lw=0.8)
            ax.set_xlabel("downrange (km)")
            ax.set_ylabel("deg")
            ax.set_title("flight path angle / AoA")
            ax.legend(loc="best", framealpha=0.9)
    return fig


FLIGHT_RE = re.compile(r"(M\d+)\s+\[flight\]\s+(.*)$")
KV_RE = re.compile(
    r"([A-Za-z_][A-Za-z_0-9]*)=(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")


def parse_flight_log(path: Path) -> dict:
    rounds: dict[str, dict] = {}
    with path.open(errors="replace") as fh:
        for line in fh:
            m = FLIGHT_RE.search(line)
            if not m:
                continue
            rid, rest = m.group(1), m.group(2)
            state, _, tail = rest.partition("|")
            d = rounds.setdefault(rid, {})
            for k, v in KV_RE.findall(state):
                d.setdefault(k, []).append(float(v))
            for k, v in KV_RE.findall(tail):
                d.setdefault("slope_" + k, []).append(float(v))
    return rounds


def _paired(d: dict, xkey: str, ykey: str, scale: float = 1.0):
    y = d.get(ykey) or []
    x = d.get(xkey) or []
    n = min(len(x), len(y))
    return x[:n], [v * scale for v in y[:n]]


def plot_flight(logs: list[Path], args, plt):
    fig, axes = plt.subplots(4, 1, figsize=(
        10, 11.5), sharex=False, layout="constrained")
    shown = 0
    for path in logs:
        rounds = parse_flight_log(path)
        if not rounds:
            print(
                f"{path}: no [flight] lines (is this a BepInEx log?)", file=sys.stderr)
            continue
        for rid, d in sorted(rounds.items()):
            if len(d.get("t", [])) < 3:
                continue
            shown += 1
            lab = f"{path.stem} {rid}"

            x = [si * xi for si, xi in zip(d.get("s", []), d.get("xf", []))]
            ax = axes[0]
            ax.plot(x, [v / 1000.0 for v in d.get("alt", [])],
                    lw=1.4, label=lab)
            if "ref" in d:
                ax.plot(x, [v / 1000.0 for v in d["ref"]], lw=1.1, ls="--",
                        color=ax.lines[-1].get_color(), alpha=0.9)
            axes[1].plot(*_paired(d, "s", "err"), lw=1.3, label=lab)
            if "slope_cmd" in d:
                xs, ys = _paired(d, "t", "slope_cmd")
                axes[2].plot(xs, ys, lw=1.4, label=f"{lab} cmd")
                if "slope_act" in d:
                    axes[2].plot(*_paired(d, "t", "slope_act"), lw=1.2, ls="--",
                                 color=axes[2].lines[-1].get_color(), alpha=0.9)
            axes[3].plot(*_paired(d, "t", "v"), lw=1.3, label=lab)
    if not shown:
        raise SystemExit(
            f"no usable [flight] data in {', '.join(str(p) for p in logs)}")
    axes[0].set_ylabel("altitude (km)")
    axes[0].set_title(
        "solid = missile, dashed = the reference it was commanded to follow")
    axes[1].axhline(0.0, color="0.6", lw=0.8)
    axes[1].set_ylabel("tracking error (m)")
    axes[1].set_xlabel("s (fraction of Xf)")
    axes[2].axhline(0.0, color="0.6", lw=0.8)
    axes[2].set_ylabel("slope (deg)")
    axes[2].set_title("slope loop: solid = command, dashed = achieved")
    axes[3].axhline(SELF_DESTRUCT_SPEED, color="crimson", ls=":", lw=0.9)
    axes[3].axhline(DEFAULT_MIN_TERMINAL, color="0.4", ls="--", lw=0.9)
    axes[3].set_ylabel("speed (m/s)")
    axes[3].set_xlabel("t since launch (s)")
    for ax in axes:
        ax.legend(loc="best", fontsize=7)
        ax.margins(x=0.02)
    axes[0].set_xlabel("downrange (km)")
    fig.suptitle("flight logs -- " +
                 ", ".join(p.name for p in logs), fontsize=11)
    return fig


SHORT_STATUS = {"ok": "ok", "woven": "wov", "infeasible": "inf",
                "unverified": "?", "unknown": "-"}


def cmd_list(cells: list[Cell], args):
    rows = sorted(cells, key=lambda c: (c.coord("range_km"), c.coord("launch_alt_m"),
                                        c.coord("launch_speed_ms"), c.coord("target_alt_m")))

    def num(v, fmt, dash="-"):
        return dash if v != v else format(v, fmt)

    head = (f"{'key':<22} {'range':>6} {'h0':>6} {'V0':>4} {'ht':>6} {'apex':>7} "
            f"{'gain':>7} {'V_term':>7} {'tof':>6} {'rmse':>8} {'health':>5} "
            f"{'V_min':>6} {'nav_g':>6} {'defect':>6} {'reach':>5} {'t':>5}  flags")
    print(head)
    print("-" * len(head))
    for c in rows:
        reach = {None: "-", True: "yes", False: "no"}[c.reachable]
        print(f"{c.key:<22} {c.coord('range_km'):6.0f} {c.coord('launch_alt_m'):6.0f} "
              f"{c.coord('launch_speed_ms'):4.0f} {c.coord('target_alt_m'):6.0f} "
              f"{c.apex:7.0f} {c.apex_gain:7.0f} {c.v_term:7.0f} {c.tof:6.1f} "
              f"{c.rmse:8.1f} {SHORT_STATUS[c.status]:>5} "
              f"{num(c.v_min, '6.0f'):>6} {num(c.nav_min, '6.2f'):>6} "
              f"{num(c.defect, '6.2f'):>6} {reach:>5} {c.t_bound or '-':>5}  {c.flags()}")
    print(f"\n{summary(cells)}")
    slow = sum(1 for c in rows if c.v_term ==
               c.v_term and c.v_term < DEFAULT_MIN_TERMINAL)
    if slow:
        print(f"{slow} cell(s) arrive below the mod's {DEFAULT_MIN_TERMINAL:.0f} m/s "
              f"load-time filter")


def fig_to_svg(fig, plt) -> str:
    from io import StringIO
    buf = StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    svg = buf.getvalue()
    i = svg.find("<svg")
    if i > 0:
        svg = svg[i:]
    return svg.replace("<svg ", '<svg style="max-width:100%;height:auto" ', 1)


def _json_safe(v):
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    if isinstance(v, (int, str, bool)) or v is None:
        return v
    try:
        import numpy as np
        if isinstance(v, np.ndarray):
            return [_json_safe(x) for x in v.tolist()]
        if isinstance(v, np.generic):
            return _json_safe(v.item())
    except Exception:
        pass
    return str(v)


def sol_series(sol: dict | None, key: str):
    if not isinstance(sol, dict) or key not in sol:
        return None
    v = sol[key]
    if isinstance(v, (int, float)):
        return None
    if isinstance(v, (list, tuple)):
        try:
            return [float(x) for x in v]
        except (TypeError, ValueError):
            return None
    try:
        import numpy as np
        arr = np.asarray(v, dtype=float).ravel()
    except Exception:
        return None
    return list(arr) if arr.size > 1 else None


def sol_scalar(sol: dict | None, key: str) -> float:
    if not isinstance(sol, dict) or key not in sol:
        return float("nan")
    return _fnum(sol[key])


def cells_json(cells: list[Cell], with_curves: bool = False) -> str:
    import numpy as np
    rows = []
    keys = [("key", lambda c: c.key), ("source", lambda c: c.source),
            ("short", lambda c: c.short()), ("status", lambda c: c.status),
            ("health", lambda c: c.health), ("feasible", lambda c: c.feasible),
            ("violation", lambda c: c.violation), ("defect",
                                                   lambda c: _json_safe(c.defect)),
            ("v_min", lambda c: _json_safe(c.v_min)
             ), ("nav_min", lambda c: _json_safe(c.nav_min)),
            ("authority_floor", lambda c: _json_safe(c.authority_floor)),
            ("speed_floor", lambda c: _json_safe(c.speed_floor)
             ), ("t_bound", lambda c: c.t_bound),
            ("apex", lambda c: _json_safe(c.apex)
             ), ("apex_gain", lambda c: _json_safe(c.apex_gain)),
            ("apex_downrange_km",
             lambda c: _json_safe(c.meta.get("apex_downrange_m", float("nan")) / 1000.0)),
            ("v_term", lambda c: _json_safe(c.v_term)
             ), ("tof", lambda c: _json_safe(c.tof)),
            ("rmse", lambda c: _json_safe(c.rmse)
             ), ("converged", lambda c: c.converged),
            ("flat", lambda c: c.flat), ("elapsed",
                                         lambda c: _json_safe(c.elapsed)),
            ("attempts", lambda c: c.attempts),
            ("exp_max_range_km", lambda c: _json_safe(c.exp_max_range)),
            ("reachable", lambda c: c.reachable), ("flags", lambda c: c.flags())]
    for c in sorted(cells, key=lambda c: (c.coord("range_km"), c.coord("launch_alt_m"),
                                          c.coord("launch_speed_ms"), c.coord("target_alt_m"))):
        rec = {name: fn(c) for name, fn in keys}
        rec["coords"] = {a: c.coord(a) for a in AXES}
        rec["poly"] = list(c.poly)
        if with_curves:
            xi = np.linspace(0.0, 1.0, 120)
            rec["x_km"] = [float(v) * c.xf / 1000.0 for v in xi]
            rec["h_km"] = [float(c.alt(v)) / 1000.0 for v in xi]
        if c.sol is not None:
            sol: dict = {}
            for k in ("x", "h", "V", "gamma", "alpha", "mass"):
                ser = sol_series(c.sol, k)
                if ser is not None:
                    sol[k] = [_json_safe(v) for v in ser]
            tof = sol_scalar(c.sol, "tof")
            if tof == tof:
                sol["tof"] = _json_safe(tof)
            if sol:
                rec["sol"] = sol
        rows.append(rec)
    return json.dumps(rows, allow_nan=False)


HTML_CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#1d2129;--muted:#6b7280;--line:#e3e6ea;
      --ok:#1a7f37;--woven:#b45309;--bad:#b91c1c;--unknown:#6b7280}
@media (prefers-color-scheme: dark){
 :root{--bg:#15181c;--card:#1d2126;--ink:#e6e8eb;--muted:#9aa3ad;--line:#2c3238;
       --ok:#4ade80;--woven:#fbbf24;--bad:#f87171;--unknown:#9aa3ad}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
     font:14px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1680px;margin:0 auto;padding:22px 20px 60px}
h1{font-size:20px;margin:0 0 4px}
h2{font-size:15px;margin:26px 0 10px;font-weight:600}
.sub{color:var(--muted);margin:0 0 16px}
.tiles{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0 6px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;
      padding:9px 13px;min-width:104px}
.tile .n{font-size:19px;font-weight:650;font-variant-numeric:tabular-nums}
.tile .k{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em}
.tile.ok .n{color:var(--ok)}.tile.woven .n{color:var(--woven)}
.tile.bad .n{color:var(--bad)}.tile.unknown .n{color:var(--unknown)}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;
      padding:12px;margin:12px 0;overflow:hidden}
.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:10px 0}
.controls input,.controls select{background:var(--card);color:var(--ink);
      border:1px solid var(--line);border-radius:8px;padding:6px 9px;font:inherit}
.controls input[type=text]{min-width:230px}
.tblwrap{overflow:auto;max-height:70vh;border:1px solid var(--line);border-radius:10px}
table{border-collapse:separate;border-spacing:0;width:100%;font-size:12.5px;
      font-variant-numeric:tabular-nums}
th,td{padding:6px 9px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line)}
th{position:sticky;top:0;background:var(--card);cursor:pointer;font-weight:600;user-select:none}
th:first-child,td:first-child{text-align:left}
th.sorted:after{content:" \\25be";color:var(--muted)}
th.sorted.asc:after{content:" \\25b4"}
tbody tr:hover{background:rgba(127,127,127,.08)}
td.st-ok{color:var(--ok)}td.st-woven{color:var(--woven)}td.st-infeasible{color:var(--bad)}
td.st-unknown,td.st-unverified{color:var(--unknown)}
tr.bad td{background:rgba(220,38,38,.08)}
tr.woven td{background:rgba(217,119,6,.08)}
.note{color:var(--muted);font-size:12px}
footer{color:var(--muted);font-size:12px;margin-top:26px}
"""


def _tile(label: str, value, cls: str = "") -> str:
    return f"<div class='tile {cls}'><div class='n'>{value}</div><div class='k'>{label}</div></div>"


def html_summary_tiles(cells: list[Cell]) -> str:
    counts = status_counts(cells)
    n_env_known = sum(1 for c in cells if c.reachable is not None)
    n_unr = sum(1 for c in cells if c.beyond_envelope)
    worst = max((c.defect for c in cells if c.defect ==
                c.defect), default=float("nan"))
    tiles = [
        _tile("cells", len(cells)),
        _tile("ok", counts.get("ok", 0), "ok"),
        _tile("woven", counts.get("woven", 0), "woven"),
        _tile("infeasible", counts.get("infeasible", 0), "bad"),
        _tile("no verdict", counts.get("unverified", 0) +
              counts.get("unknown", 0), "unknown"),
        _tile("never lofted", sum(1 for c in cells if c.flat), "woven"),
        _tile("worst defect", f"{worst:.2f}" if worst == worst else "-",
              "bad" if worst == worst and worst > 1.0 + DEFAULT_SPEED_TOL else ""),
    ]
    if n_env_known:
        tiles.append(
            _tile("beyond envelope", f"{n_unr} / {n_env_known}", "bad" if n_unr else ""))
    return "<div class='tiles'>" + "".join(tiles) + "</div>"


def html_table(cells: list[Cell], table_id: str = "cells") -> str:
    cols = [("key", "key", "s"), ("range_km", "range", "n"), ("launch_alt_m", "h0", "n"),
            ("launch_speed_ms", "V0", "n"), ("target_alt_m", "ht", "n"),
            ("apex", "apex", "n"), ("apex_gain",
                                    "gain", "n"), ("v_term", "V_term", "n"),
            ("tof", "tof", "n"), ("rmse", "rmse", "n"), ("status", "health", "s"),
            ("v_min", "V_min", "n"), ("nav_min",
                                      "nav_g", "n"), ("defect", "defect", "n"),
            ("reachable", "reach", "s"), ("t_bound", "t", "s"), ("flags", "flags", "s")]
    head = "".join(f"<th data-col='{i}' data-type='{ty}'>{lab}</th>"
                   for i, (_k, lab, ty) in enumerate(cols))
    rows = []
    for c in sorted(cells, key=lambda c: (c.coord("range_km"), c.coord("launch_alt_m"),
                                          c.coord("launch_speed_ms"), c.coord("target_alt_m"))):
        def num(v, fmt):
            return "-" if v != v else format(v, fmt)

        vals = [c.key, f"{c.coord('range_km'):.0f}", f"{c.coord('launch_alt_m'):.0f}",
                f"{c.coord('launch_speed_ms'):.0f}", f"{c.coord('target_alt_m'):.0f}",
                f"{c.apex:.0f}", f"{c.apex_gain:.0f}", f"{c.v_term:.0f}", f"{c.tof:.1f}",
                f"{c.rmse:.1f}", c.status, num(
                    c.v_min, ".0f"), num(c.nav_min, ".2f"),
                num(c.defect, ".2f"), {None: "-",
                                       True: "yes", False: "no"}[c.reachable],
                c.t_bound or "-", c.flags()]
        cls = {"infeasible": " class='bad'",
               "woven": " class='woven'"}.get(c.status, "")
        tds = []
        for i, v in enumerate(vals):
            extra = f" class='st-{c.status}'" if i == 10 else ""
            tds.append(f"<td{extra}>{v}</td>")
        rows.append(f"<tr{cls}>{''.join(tds)}</tr>")
    return (f"<div class='controls'>"
            f"<input type='text' id='{table_id}-q' placeholder='filter rows (key, flag, health...)'>"
            f"<select id='{table_id}-h'><option value=''>all verdicts</option>"
            f"<option value='ok'>ok</option><option value='woven'>woven</option>"
            f"<option value='infeasible'>infeasible</option>"
            f"<option value='unverified'>no verdict</option></select>"
            f"<span class='note' id='{table_id}-n'></span></div>"
            f"<div class='tblwrap'><table id='{table_id}'>"
            f"<thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>")


HTML_JS = """
(function(){
  var q=document.getElementById('%(tid)s-q'), h=document.getElementById('%(tid)s-h'),
      n=document.getElementById('%(tid)s-n'), t=document.getElementById('%(tid)s'),
      tb=t.tBodies[0], rows=[].slice.call(tb.rows), STATUS_COL=10;
  function status(r){ return r.cells[STATUS_COL].textContent.trim(); }
  function apply(){
    var s=(q && q.value || '').toLowerCase(), hv=(h && h.value) || '', shown=0;
    rows.forEach(function(r){
      var okS=!s || r.textContent.toLowerCase().indexOf(s)>=0;
      var st=status(r);
      var okH=!hv || st===hv || (hv==='unverified' && st==='unknown');
      var vis=okS&&okH; r.style.display=vis?'':'none'; if(vis) shown++;
    });
    if(n) n.textContent=shown+' of '+rows.length+' rows';
  }
  if(q) q.addEventListener('input',apply); if(h) h.addEventListener('change',apply); apply();
  var dir={};
  [].slice.call(t.tHead.rows[0].cells).forEach(function(th){
    th.addEventListener('click',function(){
      var i=+th.dataset.col, ty=th.dataset.type;
      dir[i]=!dir[i];
      rows.sort(function(a,b){
        var x=a.cells[i].textContent.trim(), y=b.cells[i].textContent.trim();
        if(ty==='n'){
          x=parseFloat(x); y=parseFloat(y);
          var xn=isNaN(x), yn=isNaN(y);
          if(xn&&yn) return 0; if(xn) return 1; if(yn) return -1;
        }
        return (x<y?-1:x>y?1:0)*(dir[i]?1:-1);
      });
      rows.forEach(function(r){ tb.appendChild(r); });
      [].slice.call(t.tHead.rows[0].cells).forEach(function(o){
        o.classList.remove('sorted','asc'); o.removeAttribute('aria-sort');
      });
      th.classList.add('sorted'); th.setAttribute('aria-sort', dir[i]?'ascending':'descending');
      if(dir[i]) th.classList.add('asc');
    });
  });
})();
"""


def _plotly_version() -> str:
    try:
        import plotly
        return plotly.__version__
    except Exception:
        return "?"


def plotly_family_div(cells: list[Cell], per_status_cap: int = 60):

    import numpy as np
    import plotly.graph_objects as go
    from plotly.offline import get_plotlyjs

    order = [s for s in STATUS_ORDER]
    chosen, notes = [], []
    for s in order:
        grp = [c for c in cells if c.status == s]
        if len(grp) > per_status_cap:
            notes.append(
                f"{len(grp)} {s} cells, showing the first {per_status_cap}")
            grp = grp[:per_status_cap]
        chosen += grp
    fig = go.Figure()
    colours = {"ok": "#1f77b4", "woven": "#d97706", "infeasible": "#dc2626",
               "unverified": "#9ca3af", "unknown": "#9ca3af"}
    dashes = {"woven": "dash", "infeasible": "dot",
              "unverified": "dot", "unknown": "dot"}
    for c in sorted(chosen, key=lambda c: (c.coord("range_km"), c.coord("launch_alt_m"),
                                           c.coord("launch_speed_ms"))):
        xi = np.linspace(0.0, 1.0, 120)
        fig.add_trace(go.Scatter(
            x=[float(v) * c.xf / 1000.0 for v in xi],
            y=[float(c.alt(v)) / 1000.0 for v in xi],
            mode="lines", name=c.key,
            legendgroup=c.status, legendgrouptitle_text=c.status,
            line=dict(color=colours[c.status], width=1.8,
                      dash=dashes.get(c.status, "solid")),
            customdata=[[c.status, c.v_term, c.apex, c.tof, c.rmse, c.defect]],
            hovertemplate=("<b>%{customdata[0]}</b> " + c.key +
                           "<br>V_term %{customdata[1]:.0f} m/s"
                           "<br>apex %{customdata[2]:.0f} m, tof %{customdata[3]:.1f} s"
                           "<br>rmse %{customdata[4]:.1f} m, defect %{customdata[5]:.2f}"
                           "<extra></extra>"),
        ))
    fig.update_layout(
        template="plotly_white", height=740, margin=dict(l=60, r=20, t=30, b=50),
        xaxis_title="downrange (km)", yaxis_title="altitude (km)",
        legend=dict(groupclick="toggleitem", font=dict(
            size=10), itemsizing="constant"),
        hovermode="closest",
    )
    fig.update_xaxes(gridcolor="rgba(0,0,0,0.08)")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0.08)")
    div = fig.to_html(include_plotlyjs=False, full_html=False,
                      config={"displaylogo": False, "responsive": True})
    return (f"<script>{get_plotlyjs()}</script>{div}", "; ".join(notes))


def write_html(path: Path, figs: list[tuple[str, object]], cells: list[Cell], desc: str, plt,
               args=None):
    interactive = bool(getattr(args, "html_interactive", False)
                       ) if args is not None else False
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>NOLoftFix trajectories - {desc}</title>",
        f"<style>{HTML_CSS}</style></head><body><div class='wrap'>",
        "<h1>NOLoftFix planned trajectories</h1>",
        f"<p class='sub'>{desc}</p>",
        html_summary_tiles(cells),
    ]
    if interactive and cells:
        try:
            div, note = plotly_family_div(cells)
            parts.append("<h2>trajectories (interactive)</h2>")
            if note:
                parts.append(f"<div class='note'>{note}</div>")
            parts.append(f"<div class='card'>{div}</div>")
        except Exception as e:
            print(f"NOTE: --html-interactive could not use plotly "
                  f"({type(e).__name__}: {e}). Install it with `pip install plotly`, or "
                  f"the page keeps the static figures.", file=sys.stderr)
            parts.append(f"<div class='note'>interactive plot unavailable "
                         f"({type(e).__name__}: {e}); static figures follow.</div>")
            interactive = False
    for title, fig in figs:
        if interactive and title.lower().startswith("trajectories"):
            continue
        parts.append(
            f"<h2>{title}</h2><div class='card'>{fig_to_svg(fig, plt)}</div>")
    if cells:
        parts.append("<h2>cells</h2>")
        parts.append("<p class='note'>Click a heading to sort, type to filter. Built from the "
                     "same cells as the figures above.</p>")
        parts.append(html_table(cells))
        parts.append(f"<script>{HTML_JS % {'tid': 'cells'}}</script>")
        parts.append(f"<script id='cells-data' type='application/json'>{cells_json(cells)}"
                     f"</script>")
    parts.append(f"<footer>{summary(cells)}. Generated by plot_trajectories.py"
                 + (f", plotly {_plotly_version()}." if interactive else ".")
                 + "</footer></div></body></html>")
    path.write_text("\n".join(parts))
    print(f"wrote {path}")


def status_counts(cells: list[Cell]) -> dict[str, int]:
    counts = {s: 0 for s in STATUS_ORDER}
    for c in cells:
        counts[c.status] += 1
    return {k: v for k, v in counts.items() if v}


def describe(cells: list[Cell], args) -> str:
    bits = []
    fix = parse_fix(args.fix)
    for a in ORDER:
        if a in fix:
            bits.append(f"{SHORT[a]}={fix[a]:g}")
    for k, v in (("minV", args.min_terminal_speed), ("maxV", args.max_terminal_speed),
                 ("maxRmse", args.max_rmse), ("minValong", args.min_v_along),
                 ("maxDefect", args.max_defect)):
        if v is not None:
            bits.append(f"{k}={v:g}")
    if args.health:
        bits.append("health=" + "/".join(args.health))
    if args.reachable_only:
        bits.append("within envelope")
    if args.unreachable:
        bits.append("beyond envelope")
    if args.hide_flat:
        bits.append("flat hidden")
    if args.only_converged:
        bits.append("converged only")
    return " ".join(bits) + f" -- {summary(cells)}"


def summary(cells: list[Cell]) -> str:
    counts = status_counts(cells)
    bits = [f"{len(cells)} cells"]
    for s in STATUS_ORDER:
        if counts.get(s):
            bits.append(f"{counts[s]} {s}")
    n_env = sum(1 for c in cells if c.beyond_envelope)
    n_env_known = sum(1 for c in cells if c.reachable is not None)
    if n_env_known:
        bits.append(f"{n_env} beyond the envelope of {n_env_known} checked")
    worst = max((c.defect for c in cells if c.defect ==
                c.defect), default=float("nan"))
    if worst == worst:
        bits.append(f"worst defect ratio {worst:.2f}")
    flat = sum(1 for c in cells if c.flat)
    if flat:
        bits.append(f"{flat} never lofted")
    return ", ".join(bits)


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
    sel.add_argument("--health", action="append", default=[], choices=list(HEALTH_STATES),
                     metavar="STATE",
                     help="keep only cells with this --check verdict: ok, woven, infeasible or "
                          "unverified. Repeatable. Cells with no recorded verdict are "
                          "excluded and counted (they are not passes)")
    sel.add_argument("--reachable", dest="reachable_only", action="store_true",
                     help="keep only cells inside the solved envelope (needs a table merged "
                          "with an envelope.json)")
    sel.add_argument("--unreachable", action="store_true",
                     help="keep only cells beyond the solved envelope -- those are not slow "
                          "shots, they are not shots")
    sel.add_argument("--min-v-along", type=float, default=None, metavar="MS",
                     help="keep only cells whose lowest speed along the path is at least this "
                          "(the game loses the missile below its self-destruct speed)")
    sel.add_argument("--max-defect", type=float, default=None, metavar="RATIO",
                     help="keep only cells whose recorded/re-integrated speed ratio is at most "
                          "this (1.0 = the claim flies; grid_dymos.py --check allows 1.25)")
    lay = ap.add_argument_group("layout")
    lay.add_argument("--rows", metavar="AXIS",
                     help="axis to put down the grid")
    lay.add_argument("--cols", metavar="AXIS",
                     help="axis to put across the grid")
    lay.add_argument("--sweep", metavar="AXIS",
                     help="axis that becomes the lines")
    lay.add_argument("--colour", metavar="AXIS",
                     help="axis to colour by in --overview (default launch_alt_m)")
    lay.add_argument("--normalised", action="store_true",
                     help="x axis as %% of Xf instead of km")
    lay.add_argument("--palette", choices=list(PALETTES), default="auto",
                     help="colour scheme. auto (default) is tab10 up to ten lines and "
                          "viridis beyond; okabe-ito is colour-blind safe for the lines, "
                          "cividis is colour-blind safe for --overview's colour scale")
    lay.add_argument("--max-panels", type=int, default=36)
    lay.add_argument("--legend", action="store_true", default=True)
    lay.add_argument("--no-legend", dest="legend", action="store_false")
    lay.add_argument("--annotate", action="store_true", default=True,
                     help="label each curve with its terminal speed (default on)")
    lay.add_argument("--no-annotate", dest="annotate", action="store_false")
    lay.add_argument("--max-legend", type=int, default=12,
                     help="suppress the legend above this many lines (default 12)")
    out = ap.add_argument_group("output")
    out.add_argument("--overview", action="store_true",
                     help="grid health panels instead")
    out.add_argument("--cell", metavar="KEY",
                     help="detail view of one cell (key or index)")
    out.add_argument("--detail-extra", action="store_true",
                     help="in --cell, also plot speed and gamma/alpha when stored")
    out.add_argument("--list", action="store_true",
                     help="print the cells as a table")
    out.add_argument("--compare", action="store_true",
                     help="with two or more --table, label each table's cells separately")
    out.add_argument("--save", metavar="FILE",
                     help="write a PNG here (default: trajectories.png when nothing else is asked)")
    out.add_argument("--html", metavar="FILE",
                     help="write a self-contained HTML page")
    out.add_argument("--html-interactive", action="store_true",
                     help="draw the family plot with plotly instead of as a static SVG, "
                          "so cells can be hovered, zoomed and grouped by verdict. Writes "
                          "an HTML page on its own (trajectories.html) or into --html FILE. "
                          "Needs `pip install plotly`; without it the page keeps the static "
                          "figures and says so, on the page and on stderr")
    out.add_argument("--show", action="store_true",
                     help="open a window (needs a display)")
    out.add_argument("--dpi", type=int, default=140)
    args = ap.parse_args()

    cells: list[Cell] = []
    notes: list[str] = []
    if args.table:
        multi = len(args.table) > 1
        for p in args.table:
            cs, nt = load_table(Path(p), label=(
                Path(p).stem if multi else None))
            cells += cs
            notes += nt
    if args.runs:
        cs, nt = load_runs(Path(args.runs), args.include_attempts)
        cells += cs
        notes += nt
    if not cells and not args.flight:
        for n in notes:
            print(n, file=sys.stderr)
        raise SystemExit(
            "no cells loaded: pass --table FILE or --runs DIR (--help for usage)")
    for n in notes:
        print(n, file=sys.stderr)

    if args.flight:
        plt = import_mpl(args.show)
        figs = [("flight logs", plot_flight([Path(p)
                 for p in args.flight], args, plt))]
        save_figs(figs, args, plt, cells=[])
        return

    cells = select(cells, args)
    if not cells:
        raise SystemExit(
            "selection is empty -- check --fix values, --health and the other filters")
    print(f"plotting {summary(cells)}", file=sys.stderr)

    if args.list:
        cmd_list(cells, args)
        if not (args.save or args.html or args.show):
            return

    plt = import_mpl(args.show)
    desc = describe(cells, args)

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
        figs.append(
            (f"cell {pick[0].key}", plot_detail(pick, args, plt, desc)))
    elif args.overview:
        figs.append(("grid health", plot_overview(cells, args, plt, desc)))
    else:
        multi = args.compare and len(args.table) > 1
        if multi:
            for src in sorted({c.source for c in cells}):
                sub = [c for c in cells if c.source == src]
                figs.append((f"{src} ({len(sub)} cells)",
                             plot_family(sub, args, plt, f"{desc} | {src}")))
            args_sweep = argparse.Namespace(**vars(args))
            args_sweep.sweep = args.sweep
            args_sweep.rows = args.rows
            args_sweep.cols = args.cols
            args_sweep.legend = True
            figs.append(("all sources", plot_family(
                cells, args_sweep, plt, desc)))
        else:
            figs.append(("trajectories", plot_family(cells, args, plt, desc)))

    save_figs(figs, args, plt, cells)


def save_figs(figs, args, plt, cells):
    if args.html_interactive and not args.html:
        args.html = "trajectories.html"
        print(f"NOTE: --html-interactive without --html; writing {args.html}. "
              f"Pass --html FILE to choose the name.", file=sys.stderr)
    if args.html:
        write_html(Path(args.html), figs, cells, describe(
            cells, args) if cells else "", plt, args)
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
