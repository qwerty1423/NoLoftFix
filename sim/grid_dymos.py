"""
grid_dymos.py -- run the whole offline loft grid, resumable after crashes.

    python3 grid_dymos.py --grid grids/scythe.json --jobs 8      # run / resume
    python3 grid_dymos.py --grid grids/scythe.json --status      # what's left
    python3 grid_dymos.py --grid grids/scythe.json --merge       # -> loft_tables.json
    python3 grid_dymos.py --grid grids/scythe.json --retry-failed
    python3 grid_dymos.py --grid grids/scythe.json --limit 20    # smoke test

ALL UNITS ARE METRIC: km, metres, m/s. Imperial axis names (range_nmi,
launch_alt_ft, launch_kt, target_alt_ft, target_kt) are still accepted on input
and converted, with a notice.

WHY IT IS RESUMABLE

Each cell is written to `runs/<name>/cells/<key>.json` via write-temp-then-rename,
so a cell file is either complete or absent -- never half-written. A JSONL index
at `runs/<name>/state.jsonl` records outcomes for fast status; if it is corrupt
(a truncated last line from a kill -9) it is ignored and the index is rebuilt by
scanning cells/.

Concretely, on any exit -- Ctrl-C, OOM, segfault, machine reboot -- just re-run
the same command. Completed cells are skipped. At most the in-flight cells are
lost, and they get redone.

GRID SPEC

    {
      "name":     "scythe",
      "missile":  "AAM2",
      "coeffs":   "coeffs_aam2.json",
      "nodes":    12,
      "axes": {
        "range_km":         [15, 25, 40, 60, 85, 115, 150],
        "launch_alt_m":     [500, 5000, 11000, 17000],
        "launch_speed_ms":  [180, 320, 460],
        "target_alt_m":     [200, 4000, 12000],
        "target_speed_ms":  [0]
      }
    }

`target_speed_ms` is not used by the solver -- the reference profile is generated
against a FIXED predicted impact point, which is how CEAS-GNC-2026-016 does it.
It is carried through to the table so the runtime can index on it, and so you can
generate separate profiles per closing geometry later if you want.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

from pathlib import Path

HERE = Path(__file__).resolve().parent

# Dymos/OpenMDAO write report files into cwd; keep them out of the way.


# --------------------------------------------------------------------------
# grid spec
# --------------------------------------------------------------------------
AXIS_ORDER = ["range_km", "launch_alt_m",
              "launch_speed_ms", "target_alt_m", "target_speed_ms"]

# old imperial axis names -> (metric name, factor). Accepted on input so existing
# grid specs keep working; output is always metric.
LEGACY_AXES = {
    "range_nmi":     ("range_km",         1.852),
    "launch_alt_ft": ("launch_alt_m",     0.3048),
    "launch_kt":     ("launch_speed_ms",  0.514444),
    "target_alt_ft": ("target_alt_m",     0.3048),
    "target_kt":     ("target_speed_ms",  0.514444),
}


def load_grid(path: Path) -> dict:
    g = json.loads(path.read_text())
    for k in ("name", "coeffs", "axes"):
        if k not in g:
            raise SystemExit(f"{path}: missing '{k}'")
    # accept imperial axis names and convert them
    for old, (new, k) in LEGACY_AXES.items():
        if old in g["axes"]:
            if new in g["axes"]:
                raise SystemExit(
                    f"{path}: specify either {old} or {new}, not both")
            g["axes"][new] = [round(v * k, 6) for v in g["axes"].pop(old)]
            print(f"{path}: converted legacy axis {old} -> {new} (x{k})")
    missing = [a for a in AXIS_ORDER if a not in g["axes"]]
    if missing:
        raise SystemExit(f"{path}: axes missing {missing}")
    g["missile"] = g.get("missile", "default")
    g["nodes"] = int(g.get("nodes", 12))
    g.setdefault("order", 7)
    # Minimum load factor the airframe must be ABLE to pull everywhere along the
    # trajectory. Without it the only aero constraint is nload <= gLimit, an
    # UPPER bound, so the optimiser lofts into air too thin to turn in.
    g["min_authority_g"] = float(g.get("min_authority_g", 3.0))
    # Off by default: unconstrained descent. Set it in the grid spec only if a
    # measured flight problem actually calls for it -- it costs terminal speed
    # and nothing observed so far has needed it.
    g["max_dive_deg"] = float(g.get("max_dive_deg", 0.0))
    # SLSQP iteration budget. At the old hardcoded 250 every one of the 252 cells
    # stopped on the cap and only 16 came back `converged`; 60 of the 61 cells that
    # never lofted at all were also non-converged, i.e. they stalled at a flat
    # trajectory rather than choosing one. Raising this is the cheap thing to try
    # first, but it is not sufficient on its own: "stay flat" is a feasible point,
    # so a stalled solve can sit there indefinitely. Recorded per cell so a table
    # can be audited for which budget produced it.
    g["maxiter"] = int(g.get("maxiter", 250))
    return g


def cell_key(c: dict) -> str:
    """Deterministic, filesystem-safe, and stable across runs so resume works."""
    def f(v):
        return f"{v:g}"
    return "_".join(f(c[a]) for a in AXIS_ORDER)


def enumerate_cells(g: dict):
    axes = [list(g["axes"][a]) for a in AXIS_ORDER]
    for combo in itertools.product(*axes):
        yield dict(zip(AXIS_ORDER, combo))


# --------------------------------------------------------------------------
# crash-safe state
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# cell record validation
#
# Store.index() used to wrap ANY json in cells/ as {"ok": True, "cell": <file>}.
# Leftover cells from the pre-metric run (range_nmi / launch_alt_ft / launch_kt)
# therefore reached cmd_merge's sort and died with KeyError: 'range_km' -- or
# worse, would have merged into a table mixing unit systems. Validate instead.
# --------------------------------------------------------------------------
CELL_KEYS = ("range_km", "launch_alt_m", "launch_speed_ms",
             "target_alt_m", "target_speed_ms")
LEGACY_KEYS = ("range_nmi", "launch_alt_ft", "launch_kt",
               "target_alt_ft", "target_kt")


def validate_cell(cell: dict):
    """Returns None if the record is a usable metric cell, else a reason string."""
    if not isinstance(cell, dict):
        return "not a json object"
    missing = [k for k in CELL_KEYS if k not in cell]
    if missing:
        if any(k in cell for k in LEGACY_KEYS):
            return "legacy imperial cell from the pre-metric run"
        return "missing " + ", ".join(missing)
    sp = cell.get("terminal_speed_ms")
    if not isinstance(sp, (int, float)) or not math.isfinite(sp) or sp <= 0.0:
        return f"terminal_speed_ms is not a positive number ({sp!r})"
    pa = cell.get("poly_a")
    if not isinstance(pa, list) or len(pa) < 2 or not all(
            isinstance(v, (int, float)) and math.isfinite(v) for v in pa):
        return "poly_a missing, too short, or non-finite"
    return None


class Store:
    def __init__(self, root: Path):
        self.root = root
        self.cells = root / "cells"
        self.state = root / "state.jsonl"
        self.cells.mkdir(parents=True, exist_ok=True)
        self.rejected: list = []      # [(path, reason)] set by index()

    # -- read ---------------------------------------------------------------
    def index(self) -> dict[str, dict]:
        """key -> record. Prefers state.jsonl, falls back to scanning cells/."""
        out: dict[str, dict] = {}
        if self.state.exists():
            with self.state.open() as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        out[rec["key"]] = rec
                    except Exception:
                        # truncated last line from a hard kill -- ignore it
                        continue
        # always reconcile against the cell files: they are the source of truth
        self.rejected = []
        for p in self.cells.glob("*.json"):
            try:
                cell = json.loads(p.read_text())
            except Exception:
                continue                      # partial file, will be redone
            bad = validate_cell(cell)
            if bad is not None:
                self.rejected.append((p, bad))
                continue
            k = cell.get("_key") or p.stem
            if k not in out or not out[k].get("ok"):
                out[k] = {"key": k, "ok": True, "elapsed_s": cell.get("_elapsed_s", 0.0),
                          "cell": cell}
        return out

    # -- write --------------------------------------------------------------
    def put(self, key: str, rec: dict) -> None:
        cell = rec.get("cell")
        if cell is not None:
            cell = dict(cell)
            cell["_key"] = key
            tmp = self.cells / f".{key}.tmp"
            tmp.write_text(json.dumps(cell))
            os.replace(tmp, self.cells / f"{key}.json")   # atomic on POSIX
        with self.state.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            os.fsync(fh.fileno())


# --------------------------------------------------------------------------
# one cell, in a worker process
# --------------------------------------------------------------------------
def _unused_run_ray(job):
    """Solve a whole ray -- cells sharing every axis except range, range ascending
    -- in one process, warm starting each cell from the previous one.

    Measured on the AAM-29: warm starting is NOT faster (0.92x), but it finds
    materially better optima, because SLSQP is capped at 250 iterations and a
    good start spends that budget better. At 75 km it turned 656 m/s into
    1340 m/s; at 110 km, 790 into 962. Cold starts were landing in bad local
    optima. So rays buy solution QUALITY, and they cost nothing.

    Emits the same (key, record) tuples run_cell did, so the store is unchanged.
    """
    ray_key, entries, coeffs_path, nodes, order = job
    out = []
    guess = None
    # resume inside a ray: reload the last completed cell as the seed
    from_store = entries[0][2]
    if from_store is not None:
        guess = from_store
    for key, geom, _ in entries:
        t0 = time.time()
        rec = _solve_one(key, geom, coeffs_path, nodes, order, guess)
        out.append((key, rec))
        if rec["ok"]:
            guess = rec.pop("_sol", guess)
        yield_ready = True
    return ray_key, out


def _solve_one(key, geom, coeffs_path, nodes, order, guess, auth=3.0, dive=30.0, maxiter=250):
    t0 = time.time()
    try:
        # imported lazily: OpenMDAO is heavy and only workers need it
        sys.path.insert(0, str(HERE))
        from dymos_opt import solve, fit_polynomial, _isolate_workdir
        _isolate_workdir()   # keep the *_out dirs out of the project dir

        sol = solve(coeffs_path,
                    range_km=geom["range_km"], launch_alt_m=geom["launch_alt_m"],
                    target_alt_m=geom["target_alt_m"],
                    launch_speed_ms=geom["launch_speed_ms"],
                    nodes=nodes, verbose=False, guess=guess,
                    min_authority_g=auth, max_dive_deg=dive,
                    maxiter=maxiter)
        a, rmse = fit_polynomial(sol, order=order)
        cell = {
            "maxiter": int(maxiter),
            "range_km": float(geom["range_km"]),
            "launch_alt_m": float(geom["launch_alt_m"]),
            "launch_speed_ms": float(geom["launch_speed_ms"]),
            "target_alt_m": float(geom["target_alt_m"]),
            "target_speed_ms": float(geom["target_speed_ms"]),
            "poly_a": a,
            "apex_alt_m": float(sol["apex"]),
            "apex_downrange_m": float(sol["apex_x"]),
            "terminal_speed_ms": float(sol["V_term"]),
            "tof_s": float(sol["tof"]),
            "fit_rmse_m": float(rmse),
            "converged": bool(sol["converged"]),
            "_elapsed_s": round(time.time() - t0, 1),
        }
        ok = math.isfinite(
            cell["terminal_speed_ms"]) and cell["terminal_speed_ms"] > 0 and cell["fit_rmse_m"] < 1000
        rec = {"key": key, "ok": ok, "cell": cell if ok else None,
               "elapsed_s": cell["_elapsed_s"],
               "error": None if ok else f"rmse probably too high: {cell["fit_rmse_m"]}",
               "converged": cell["converged"],
               "terminal_speed_ms": cell["terminal_speed_ms"]}
        if ok:
            # carry the trajectory forward for the next cell in the ray; stripped
            # before it reaches the store
            rec["_sol"] = {k: sol[k]
                           for k in ("x", "h", "V", "gamma", "mass", "alpha", "tof")}
        return rec
    except Exception as e:
        return {"key": key, "ok": False, "cell": None,
                "elapsed_s": round(time.time() - t0, 1),
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(limit=4)}


def run_cell(job):
    """Single-cell entry point (kept for --no-warm and for ad-hoc use)."""
    key, geom, coeffs_path, nodes, order, _timeout, auth, dive, maxiter = job
    return key, _solve_one(key, geom, coeffs_path, nodes, order, None, auth, dive, maxiter)


def _ray_entry(work, coeffs_path, nodes, order, auth, dive, maxiter=250):
    """Runs one ray in a worker and returns its [(key, record)] list."""
    _rk, entries = work
    out = []
    guess = None
    for (key, geom, _seed) in entries:
        rec = _solve_one(key, geom, coeffs_path, nodes,
                         order, guess, auth, dive, maxiter)
        guess = rec.pop("_sol", None) if rec["ok"] else guess
        out.append((key, rec))
    return out


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
def cmd_run(g, a):
    store = Store(a.runs / g["name"])
    done = store.index()
    warn_rejected(store)
    coeffs_path = str((HERE / g["coeffs"]).resolve()
                      ) if not Path(g["coeffs"]).is_absolute() else g["coeffs"]

    todo = []
    for geom in enumerate_cells(g):
        k = cell_key(geom)
        rec = done.get(k)
        if rec and rec.get("ok"):
            continue
        if rec and not rec.get("ok") and not a.retry_failed:
            continue
        todo.append((k, geom, coeffs_path, g["nodes"], g["order"], a.timeout,
                     a.min_authority, a.max_dive, a.maxiter))
        if a.limit and len(todo) >= a.limit:
            break

    total = sum(1 for _ in enumerate_cells(g))
    ok = sum(1 for r in done.values() if r.get("ok"))
    failed = sum(1 for r in done.values() if not r.get("ok"))
    print(f"{g['name']}: {total} cells total | {ok} done | {failed} failed | {len(todo)} to run "
          f"({a.jobs} jobs, {g['nodes']} nodes)")
    if not todo:
        print("nothing to do. run --merge to emit the table.")
        return

    est = len(todo) * 90.0 / max(a.jobs, 1)
    print(f"rough ETA {est/60:.0f} min at ~90 s/cell\n")

    t0 = time.time()
    n_ok = n_fail = 0

    if a.no_warm:
        # todo entries are 6-tuples; _ray_entry only wants (key, geom, seed)
        work = [(("single", t[0]), [(t[0], t[1], None)]) for t in todo]
        # todo entries carry auth as their 7th element
    else:
        # group by everything except range, then sort each group by range so the
        # ray marches monotonically and warm starts stay valid
        rays: dict = {}
        for t in todo:
            k, gm = t[0], t[1]
            rk = (gm["launch_alt_m"], gm["launch_speed_ms"],
                  gm["target_alt_m"], gm["target_speed_ms"])
            rays.setdefault(rk, []).append((k, gm))
        work = []
        for rk, items in rays.items():
            items.sort(key=lambda t: t[1]["range_km"])
            seeded = [(k, gm, None) for (k, gm) in items]
            work.append((rk, [(k, gm, sd) for (k, gm, sd) in seeded]))
        print(
            f"scheduled {len(work)} rays (warm start on), longest {max(len(w[1]) for w in work)} cells")

    # maxtasksperchild bounds the memory OpenMDAO leaks across solves
    with ProcessPoolExecutor(max_workers=a.jobs, max_tasks_per_child=a.max_tasks) as ex:
        futs = {ex.submit(_ray_entry, w, coeffs_path, g["nodes"], g["order"],
                          a.min_authority, a.max_dive, a.maxiter): w[0] for w in work}
        done_cells = 0
        for fut in as_completed(futs):
            for key, rec in fut.result():
                done_cells += 1
                i = done_cells
                store.put(key, rec)
                if rec["ok"]:
                    n_ok += 1
                    print(f"[{i}/{len(todo)}] {key}  {rec['terminal_speed_ms']:7.1f} m/s  "
                          f"{rec['elapsed_s']:6.1f}s  {'conv' if rec['converged'] else 'ITER-LIMIT'}")
                else:
                    n_fail += 1
                    print(f"[{i}/{len(todo)}] {key}  FAILED: {rec['error']}")
                sys.stdout.flush()

    dt = time.time() - t0
    print(f"\nfinished {n_ok} ok / {n_fail} failed in {dt/60:.1f} min. "
          f"Re-run the same command to retry anything interrupted; --merge to emit the table.")


def warn_rejected(store):
    if not store.rejected:
        return
    print(f"WARNING: {len(store.rejected)} cell file(s) in {store.cells} are not usable "
          f"and were ignored:")
    for p, why in store.rejected[:10]:
        print(f"  {p.name}: {why}")
    if len(store.rejected) > 10:
        print(f"  ... and {len(store.rejected) - 10} more")
    print("  They will not appear in the table. --purge moves them to rejected/ .\n")


def cmd_purge(g, a):
    store = Store(a.runs / g["name"])
    store.index()
    if not store.rejected:
        print(f"{g['name']}: nothing to purge, all cell files are valid")
        return
    dest = store.root / "rejected"
    dest.mkdir(parents=True, exist_ok=True)
    for p, why in store.rejected:
        os.replace(p, dest / p.name)
        print(f"  moved {p.name} -> rejected/  ({why})")
    print(f"{g['name']}: {len(store.rejected)} file(s) moved to {dest}")


def cmd_status(g, a):
    store = Store(a.runs / g["name"])
    done = store.index()
    warn_rejected(store)
    total = sum(1 for _ in enumerate_cells(g))
    ok = {k: r for k, r in done.items() if r.get("ok")}
    failed = {k: r for k, r in done.items() if not r.get("ok")}
    conv = sum(1 for r in ok.values() if r.get("converged"))
    print(f"{g['name']}: {len(ok)}/{total} done ({100*len(ok)/max(total, 1):.1f}%), "
          f"{len(failed)} failed, {conv} of {len(ok)} converged")
    if ok:
        sp = [r["terminal_speed_ms"]
              for r in ok.values() if r.get("terminal_speed_ms")]
        el = [r.get("elapsed_s", 0) for r in ok.values()]
        if sp:
            print(
                f"  terminal speed {min(sp):.0f}..{max(sp):.0f} m/s, mean {sum(sp)/len(sp):.0f}")
        if el:
            rem = total - len(ok) - (0 if a.retry_failed else len(failed))
            print(f"  mean {sum(el)/len(el):.1f} s/cell -> "
                  f"~{rem*sum(el)/len(el)/max(a.jobs, 1)/60:.0f} min left at {a.jobs} jobs")
    for k, r in list(failed.items())[:10]:
        print(f"  FAILED {k}: {r.get('error')}")
    if len(failed) > 10:
        print(f"  ... and {len(failed)-10} more")


def cmd_merge(g, a):
    store = Store(a.runs / g["name"])
    done = store.index()
    warn_rejected(store)
    cells = [r["cell"] for r in done.values() if r.get("ok") and r.get("cell")]
    for c in cells:
        c.setdefault("missile", g["missile"])
        c.pop("_key", None)
    cells.sort(key=lambda c: (c["range_km"], c["launch_alt_m"], c["launch_speed_ms"],
                              c["target_alt_m"], c["target_speed_ms"]))
    out = {
        "poly_order": g["order"],
        "lookahead_s": g.get("lookahead_s", 3.0),
        "missile": g["missile"],
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cells": cells,
    }
    dest = Path(a.out) if a.out else (a.runs / g["name"] / "loft_tables.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    nconv = sum(1 for c in cells if c.get("converged"))
    print(f"wrote {dest}: {len(cells)} cells ({nconv} converged)")
    if len(cells) < sum(1 for _ in enumerate_cells(g)):
        print("  NOTE: grid is incomplete. The mod interpolates over what is there "
              "and falls back to vanilla lofting outside the solved envelope.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--nodes", type=int, help="override the grid spec")
    ap.add_argument("--limit", type=int, default=0,
                    help="run at most N cells (smoke test)")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--max-tasks", type=int, default=8,
                    help="restart each worker after this many cells (OpenMDAO leaks)")
    ap.add_argument("--runs", default=str(HERE / "runs"))
    ap.add_argument("--out", default="")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--min-authority", type=float, default=None,
                    help="minimum AVAILABLE load factor (g) along the trajectory. "
                         "Overrides min_authority_g in the grid spec. 0 disables.")
    ap.add_argument("--max-dive", type=float, default=None,
                    help="steepest allowed descent angle (deg). Overrides max_dive_deg "
                         "in the grid spec. 0 disables.")
    ap.add_argument("--maxiter", type=int, default=None,
                    help="SLSQP iteration budget. Overrides maxiter in the grid spec. "
                         "Default 250, at which every cell so far stopped on the cap.")
    ap.add_argument("--no-warm", action="store_true",
                    help="solve every cell from a cold start (worse optima, same speed)")
    ap.add_argument("--purge", action="store_true",
                    help="move unusable cell files to runs/<name>/rejected/")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    a.runs = Path(a.runs)

    g = load_grid(Path(a.grid))
    if a.nodes:
        g["nodes"] = a.nodes
    a.min_authority = g["min_authority_g"] if a.min_authority is None else a.min_authority
    a.max_dive = g["max_dive_deg"] if a.max_dive is None else a.max_dive
    a.maxiter = g["maxiter"] if a.maxiter is None else a.maxiter

    if a.purge:
        cmd_purge(g, a)
    elif a.status:
        cmd_status(g, a)
    elif a.merge:
        cmd_merge(g, a)
    else:
        cmd_run(g, a)


if __name__ == "__main__":
    main()
