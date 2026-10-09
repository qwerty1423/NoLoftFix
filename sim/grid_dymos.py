"""
examples
python3 grid_dymos.py --grid grids/scythe.json --jobs 8
python3 grid_dymos.py --grid grids/scythe.json --status
python3 grid_dymos.py --grid grids/scythe.json --merge
python3 grid_dymos.py --grid grids/scythe.json --retry-failed
python3 grid_dymos.py --grid grids/scythe.json --limit 20

reoptimise
python3 grid_dymos.py --grid grids/scythe.json --reoptimise --max-rmse 1000
python3 grid_dymos.py --grid grids/scythe.json --reoptimise --min-terminal-speed 800 \
    --reopt-runs 5 --jobs 8
python3 grid_dymos.py --grid grids/scythe.json --reoptimise --max-tof 200 --iterations 50
"""
from __future__ import annotations
from pathlib import Path

import argparse
import itertools
import json
import math
import multiprocessing
import os
import signal
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor as _PoolExecutor, as_completed

_POOLS: "set[ProcessPoolExecutor]" = set()
_HANDLERS_INSTALLED = False


def _drop_workdir():
    """Remove the throwaway OpenMDAO tree this process created, if it made one."""
    import shutil

    d = os.environ.get("OPENMDAO_WORKDIR")
    if d and os.path.basename(d.rstrip("/")).startswith("dymos_work_"):
        shutil.rmtree(d, ignore_errors=True)


def _worker_setup(user_initializer, user_initargs):
    import threading

    def _bye():
        _drop_workdir()
        os._exit(1)

    def _term(_signum, _frame):
        _bye()

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, _term)
    parent = os.getppid()

    def _watch_parent():
        while True:
            if os.getppid() != parent:
                _bye()
            time.sleep(1.0)

    threading.Thread(target=_watch_parent, daemon=True).start()
    if user_initializer is not None:
        user_initializer(*user_initargs)


class ProcessPoolExecutor(_PoolExecutor):
    def __init__(self, *args, **kwargs):
        global _HANDLERS_INSTALLED
        inner = kwargs.pop("initializer", None)
        initargs = tuple(kwargs.pop("initargs", ()))
        kwargs["initializer"] = _worker_setup
        kwargs["initargs"] = (inner, initargs)
        super().__init__(*args, **kwargs)
        _POOLS.add(self)
        if not _HANDLERS_INSTALLED:
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                signal.signal(sig, _stop_now)
            _HANDLERS_INSTALLED = True

    def shutdown(self, wait=True, *, cancel_futures=False):
        _POOLS.discard(self)
        return super().shutdown(wait=wait, cancel_futures=cancel_futures)


def _child_pids(include_dead_parents: bool = True) -> list[int]:
    me = os.getpid()
    found = []
    try:
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                with open(f"/proc/{entry}/stat", "rb") as fh:
                    fields = fh.read().rsplit(b")", 1)[1].split()
                ppid = int(fields[1])
            except Exception:
                continue
            if ppid == me:
                found.append(int(entry))
    except Exception:
        return []
    return found


def _kill_children_now() -> int:
    n = 0
    for pid in _child_pids():
        try:
            os.kill(pid, signal.SIGKILL)
            n += 1
        except ProcessLookupError:
            pass
    return n


def _reap_children(grace: float = 2.0, quiet: bool = True) -> int:
    pids = _child_pids()
    if not pids:
        return 0
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + grace
    while time.time() < deadline:
        alive = [p for p in pids if os.path.exists(f"/proc/{p}")]
        if not alive:
            break
        time.sleep(0.05)
    killed = 0
    for pid in pids:
        if os.path.exists(f"/proc/{pid}"):
            try:
                os.kill(pid, signal.SIGKILL)
                killed += 1
            except ProcessLookupError:
                pass
    if not quiet and killed:
        print(f"NOTE: {killed} child process(es) needed SIGKILL to stop; they are "
              f"gone now.", file=sys.stderr)
    return len(pids)


def _stop_workers():
    import logging

    logging.getLogger("concurrent.futures.process").setLevel(
        logging.CRITICAL + 1)
    procs = [p for ex in list(_POOLS)
             for p in list(getattr(ex, "_processes", {}).values())]
    for p in procs:
        if p.is_alive():
            p.terminate()
    deadline = time.time() + 5.0
    for p in procs:
        if p.is_alive():
            p.join(max(0.0, deadline - time.time()))
    for p in procs:
        if p.is_alive():
            p.kill()

    _reap_children(grace=2.0)
    return len(procs)


_STOPPING = False


def _stop_now(signum, _frame):
    global _STOPPING
    if _STOPPING:
        _kill_children_now()
        os._exit(128 + signum)
    _STOPPING = True
    print(f"\n{signal.Signals(signum).name}: stopping ...")
    sys.stdout.flush()
    try:
        n = _stop_workers()
        print(
            f"  {n} worker process(es) terminated." if n else "  no workers to stop.")
        print("  Finished cells are on disk; re-run the same command to resume.")
        sys.stdout.flush()
    except Exception:
        pass
    _leave(128 + signum)


def _lingering_manager_threads(timeout: float = 2.0,
                               out=sys.stderr) -> list[str]:
    try:
        from concurrent.futures import process as _cfp
        threads = [th for th in list(getattr(_cfp, "_threads_wakeups", {}) or [])
                   if th.is_alive()]
    except Exception:
        return []
    stuck = []
    for th in threads:
        th.join(timeout)
        if th.is_alive():
            stuck.append(getattr(th, "name", "?") or "?")
    if stuck:
        print(f"NOTE: a worker pool's manager thread ({', '.join(stuck)}) did not "
              f"finish shutting down; exiting without waiting for it. Cell files are "
              f"written atomically, so nothing on disk is half-written.",
              file=out)
    return stuck


def _leave(rc: int = 0) -> None:
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    _lingering_manager_threads()
    _reap_children(grace=1.0)
    try:
        _drop_workdir()
    except Exception:
        pass
    os._exit(rc)


HERE = Path(__file__).resolve().parent
AXIS_ORDER = ["range_km", "launch_alt_m",
              "launch_speed_ms", "target_alt_m", "target_speed_ms"]
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
    g["min_authority_g"] = float(g.get("min_authority_g", 3.0))
    g["max_dive_deg"] = float(g.get("max_dive_deg", 0.0))
    g["maxiter"] = int(g.get("maxiter", 250))
    g["store_solutions"] = bool(g.get("store_solutions", False))
    g["min_speed_ms"] = g.get("min_speed_ms", None)
    g["refine_iterations"] = int(g.get("refine_iterations", 0))
    g["refine_tol"] = float(g.get("refine_tol", 1e-4))
    return g


def speed_floor_from_spec(g: dict, cli_value, coeffs_path: str):
    v = cli_value if cli_value is not None else g.get("min_speed_ms")
    if v is None:
        return None
    if isinstance(v, str):
        if v.strip().lower() not in ("self_destruct", "self-destruct"):
            raise SystemExit(f"grid {g['name']}: min_speed_ms must be a number or "
                             f"'self_destruct', not {v!r}")
        return float(json.loads(Path(coeffs_path).read_text())["self_destruct_at_speed"])
    return float(v)


def cell_key(c: dict) -> str:
    def f(v):
        return f"{v:g}"
    return "_".join(f(c[a]) for a in AXIS_ORDER)


def enumerate_cells(g: dict):
    axes = [list(g["axes"][a]) for a in AXIS_ORDER]
    for combo in itertools.product(*axes):
        yield dict(zip(AXIS_ORDER, combo))


CELL_KEYS = ("range_km", "launch_alt_m", "launch_speed_ms",
             "target_alt_m", "target_speed_ms")
LEGACY_KEYS = ("range_nmi", "launch_alt_ft", "launch_kt",
               "target_alt_ft", "target_kt")


def validate_cell(cell: dict):
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


def _read_cell(cells: Path, key: str) -> dict | None:
    try:
        return json.loads((cells / f"{key}.json").read_text())
    except Exception:
        return None


def attempt_path(cells: Path, key: str, n: int) -> Path:
    return cells / f"{key}.attempt{n}.json"


def list_attempts(cells: Path, key: str) -> list[int]:
    nums = []
    for p in cells.glob(f"{key}.attempt*.json"):
        tail = p.stem.rsplit(".attempt", 1)[-1]
        if tail.isdigit():
            nums.append(int(tail))
    return sorted(nums)


def keys_with_attempts(cells: Path) -> set[str]:
    out = set()
    for p in cells.glob("*.attempt*.json"):
        out.add(p.name.rsplit(".attempt", 1)[0])
    return out


def attempts_recorded(cells: Path, key: str, cur: dict | None = None) -> int:
    nums = list_attempts(cells, key)
    made = int((cur or {}).get("_attempts") or 0)
    return max([made, *nums]) if (nums or made) else 0


LGR3 = (0.15505102572168, 0.64494897427832, 1.0)


def _curve_fn(keys):
    ts = [float(k["time"]) for k in keys]
    vs = [float(k["value"]) for k in keys]
    mi = [float(k.get("inSlope", 0.0)) for k in keys]
    mo = [float(k.get("outSlope", 0.0)) for k in keys]

    def f(x: float) -> float:
        if x <= ts[0]:
            return vs[0]
        if x >= ts[-1]:
            return vs[-1]
        i = 0
        while i < len(ts) - 2 and ts[i + 1] <= x:
            i += 1
        dt = ts[i + 1] - ts[i]
        if dt <= 0.0:
            return vs[i]
        u = (x - ts[i]) / dt
        u2 = u * u
        u3 = u2 * u
        return ((2 * u3 - 3 * u2 + 1) * vs[i] + (u3 - 2 * u2 + u) * mo[i] * dt
                + (-2 * u3 + 3 * u2) * vs[i + 1] + (u3 - u2) * mi[i + 1] * dt)

    return f


def load_curves(coeffs_path):
    try:
        d = json.loads(Path(coeffs_path).read_text())
        return {
            "rho": _curve_fn(d["air_density_curve"]["keys"]),
            "cd": _curve_fn(d["drag_curve"]["keys"]),
            "cl": _curve_fn(d["lift_curve"]["keys"]),
            "S": float(d["fin_area"]),
            "dry": float(d["mass_dry"]),
            "motors": [(float(m.get("delay", 0.0)), float(m.get("burn_time", 0.0)),
                        float(m.get("thrust", 0.0)), float(m.get("fuel_mass", 0.0)))
                       for m in d.get("motors", [])],
        }
    except Exception:
        return None


def turning_points(cell, n: int = 2001) -> int:
    a = [float(v) for v in (cell.get("poly_a") or [])
         if isinstance(v, (int, float))]
    if len(a) < 2:
        return 0
    turns = 0
    last = None
    for i in range(n + 1):
        s = _poly_slope(a, i / n)
        sign = (s > 0.0) - (s < 0.0)
        if sign and last and sign != last:
            turns += 1
        if sign:
            last = sign
    return turns


def _control_taus(n_c: int):
    """Time fractions of the control nodes."""
    if n_c >= 3 and n_c % 3 == 0:
        segs = n_c // 3
        return [k / segs + p / segs for k in range(segs) for p in LGR3]
    return [i / (n_c - 1) for i in range(n_c)] if n_c > 1 else [0.0]


def reintegrate(cell: dict, curves: dict, steps: int = 4000) -> dict | None:
    sol = cell.get("_sol")
    if curves is None or not isinstance(sol, dict):
        return None
    try:
        xs = [float(v) for v in sol["x"]]
        hs = [float(v) for v in sol["h"]]
        vs = [float(v) for v in sol["V"]]
        gs = [float(v) for v in sol["gamma"]]
        ms = [float(v) for v in sol["mass"]]
        alpha = [float(v) for v in sol["alpha"]]
        tof = float(sol["tof"])
    except Exception:
        return None
    if len(xs) < 3 or len(alpha) < 2 or tof <= 0.0:
        return None
    tau_c = _control_taus(len(alpha))

    def a_of(t: float) -> float:
        if t <= 0.0:
            return alpha[0]
        if t >= tof:
            return alpha[-1]
        u = t / tof
        i = 0
        while i < len(tau_c) - 2 and tau_c[i + 1] <= u:
            i += 1
        dt = tau_c[i + 1] - tau_c[i]
        w = 0.0 if dt <= 0 else (u - tau_c[i]) / dt
        return alpha[i] * (1.0 - w) + alpha[i + 1] * w

    rho, cd, cl = curves["rho"], curves["cd"], curves["cl"]
    S, g = curves["S"], 9.81
    motors = curves["motors"]

    def motor(t: float):
        T = br = 0.0
        for delay, burn, thrust, fuel in motors:
            if delay <= t < delay + burn:
                T += thrust
                if burn > 0.0:
                    br += fuel / burn
        return T, br

    def deriv(st, t, al):
        x, h, V, gam, m = st
        V = V if V > 1.0 else 1.0
        m = m if m > 1.0 else 1.0
        q = 0.5 * rho(h) * V * V
        D = cd(abs(al)) * q * S
        L = cl(abs(al)) * q * S
        T, br = motor(t)
        return (V * math.cos(gam), V * math.sin(gam),
                (T * math.cos(al) - D) / m - g * math.sin(gam),
                (T * math.sin(al) + L) / (m * V) - g * math.cos(gam) / V, -br)

    st = [0.0, hs[0], vs[0], gs[0], ms[0]]
    dt = tof / steps
    dh_max = dx_max = dV_max = 0.0
    tau_s = [0.0] + tau_c
    t_s = [t * tof for t in tau_s]
    j = 0
    for i in range(steps):
        t = i * dt
        tm = t + 0.5 * dt
        k1 = deriv(st, t, a_of(t))
        k2 = deriv([st[q] + 0.5 * dt * k1[q] for q in range(5)], tm, a_of(tm))
        k3 = deriv([st[q] + 0.5 * dt * k2[q] for q in range(5)], tm, a_of(tm))
        k4 = deriv([st[q] + dt * k3[q]
                   for q in range(5)], t + dt, a_of(t + dt))
        st = [st[q] + dt / 6.0 *
              (k1[q] + 2 * k2[q] + 2 * k3[q] + k4[q]) for q in range(5)]
        while j + 1 < len(t_s) and t_s[j + 1] <= (i + 1) * dt:
            j += 1
            dV_max = max(dV_max, abs(st[2] - vs[j]))
            dh_max = max(dh_max, abs(st[1] - hs[j]) if j < len(hs) else 0.0)
            dx_max = max(dx_max, abs(st[0] - xs[j]) if j < len(xs) else 0.0)
    return {"V_stored": vs[-1], "V_reint": st[2], "dV": dV_max, "dh": dh_max,
            "dx": dx_max, "tof": tof, "steps": steps}


def cell_health(cell: dict, curves, max_weaves: int = 2,
                speed_tol: float = 0.5) -> tuple[str, dict]:
    turns = turning_points(cell)
    if not cell.get("feasible", True):
        return "infeasible", {"turns": turns, "stored": True,
                              "source": "audit",
                              "violation": cell.get("violation")}
    ver = reintegrate(cell, curves) if curves is not None else None
    if ver is not None:
        ratio = ver["V_stored"] / max(ver["V_reint"], 1e-9)
        detail = dict(ver, ratio=ratio, turns=turns, source="reintegration")
        if ratio > 1.0 + speed_tol:
            return "infeasible", detail
        if turns > max_weaves:
            return "woven", detail
        return "ok", detail
    ratio = cell.get("defect_ratio")
    if ratio is not None:
        v_stored = float(cell.get("terminal_speed_ms", 0.0))
        detail = {"turns": turns, "ratio": float(ratio), "stored": True,
                  "source": "recorded-ratio", "V_stored": v_stored,
                  "V_reint": v_stored / max(float(ratio), 1e-9)}
        if float(ratio) > 1.0 + speed_tol:
            return "infeasible", detail
        if turns > max_weaves:
            return "woven", detail
        return "ok", detail
    return (("woven" if turns > max_weaves else "unverified"),
            {"turns": turns, "source": "none"})


def infeasible_reason(d: dict) -> str:
    turns = d.get("turns")
    if d.get("source") == "audit":
        why = d.get("violation") or "the constraints it was given were not met"
        return (f"did not satisfy its own constraint audit: {why}"
                + (f"; {turns} turning points" if turns is not None else ""))
    if "V_stored" in d and "V_reint" in d:
        line = (f"claims {d['V_stored']:.0f} m/s, re-integrating the same alpha "
                f"history arrives at {d['V_reint']:.0f} m/s")
        if d.get("ratio") is not None:
            line += f" ({d['ratio']:.1f}x)"
        if turns is not None:
            line += f", {turns} turning points"
        if d.get("dx") is not None:
            line += f"; paths end {d['dx'] / 1000:.0f} km apart"
        elif d.get("source") == "recorded-ratio":
            line += (" (from the recorded re-integration ratio; "
                     "--store-solutions keeps the paths)")
        return line
    return f"{turns} turning points" if turns is not None else "no detail recorded"


def rank_of(cell: dict, how: str, max_rmse: float = 1500.0,
            health: str | None = None) -> float:
    if validate_cell(cell) is not None:
        return -1e9
    score = float(cell["terminal_speed_ms"])
    if how == "converged" and cell.get("converged"):
        score += 1e6
    if how == "flyable" and health == "infeasible":
        score -= 4e6
    elif how == "flyable" and health == "woven":
        score -= 2e6
    rmse = cell.get("fit_rmse_m")
    if how != "fastest" and (not isinstance(rmse, (int, float))
                             or not math.isfinite(rmse) or rmse > max_rmse):
        score -= 2e6
    return score


RECORD_FIELDS = ("terminal_speed_ms", "fit_rmse_m",
                 "converged", "tof_s", "maxiter")


def _decorate(rec: dict, cell: dict) -> dict:
    for f in RECORD_FIELDS:
        if f in cell:
            rec[f] = cell[f]
    rec["cell"] = cell
    return rec


def _record_for(key: str, cell: dict, elapsed_s=None, ok: bool = True) -> dict:
    rec = {"key": key, "ok": bool(ok)}
    if elapsed_s is not None:
        rec["elapsed_s"] = float(elapsed_s)
    return _decorate(rec, cell)


class Store:
    def __init__(self, root: Path):
        self.root = root
        self.cells = root / "cells"
        self.state = root / "state.jsonl"
        self.cells.mkdir(parents=True, exist_ok=True)
        self.rejected: list = []

    def index(self) -> dict[str, dict]:
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
                        continue
        self.rejected = []
        on_disk = set()
        for p in self.cells.glob("*.json"):
            if ".attempt" in p.name:
                continue
            try:
                cell = json.loads(p.read_text())
            except Exception:
                continue
            bad = validate_cell(cell)
            if bad is not None:
                self.rejected.append((p, bad))
                continue
            k = cell.get("_key") or p.stem
            on_disk.add(k)
            prev = out.get(k) or {}
            out[k] = _record_for(k, cell, ok=True,
                                 elapsed_s=cell.get("_elapsed_s",
                                                    prev.get("elapsed_s", 0.0)))
        for k, rec in out.items():
            if k not in on_disk:
                rec["ok"] = False
                rec.pop("cell", None)
            rec["attempts"] = list_attempts(self.cells, k)
        return out

    def load_attempt(self, key: str, n: int) -> dict | None:
        p = attempt_path(self.cells, key, n)
        try:
            cell = json.loads(p.read_text())
        except Exception:
            return None
        return cell if validate_cell(cell) is None else None

    def archive_current(self, key: str, n: int) -> int:
        cur = _read_cell(self.cells, key)
        if cur is None:
            return 0
        _atomic_write(attempt_path(self.cells, key, n), cur)
        return n

    def put(self, key: str, rec: dict, attempt: int | None = None,
            promote: bool = True) -> None:
        cell = rec.get("cell")
        if cell is not None:
            cell = dict(cell)
            cell["_key"] = key
            if attempt is not None:
                _atomic_write(attempt_path(self.cells, key, attempt), cell)
            if attempt is None or promote:
                _atomic_write(self.cells / f"{key}.json", cell)
        with self.state.open("a") as fh:
            rec = dict(rec)
            if attempt is not None:
                rec["attempt"] = attempt
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def prune(self, key: str) -> tuple[int, int]:
        if _read_cell(self.cells, key) is None:
            return 0, 0
        n = nbytes = 0
        for slot in list_attempts(self.cells, key):
            p = attempt_path(self.cells, key, slot)
            try:
                nbytes += p.stat().st_size
                p.unlink()
                n += 1
            except OSError:
                continue
        return n, nbytes

    def compact(self) -> dict:
        out = {"lines_before": 0, "lines_after": 0,
               "bytes_before": 0, "bytes_after": 0}
        if not self.state.exists():
            return out
        out["bytes_before"] = self.state.stat().st_size
        keep: dict[str, str] = {}
        with self.state.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                out["lines_before"] += 1
                try:
                    key = json.loads(line)["key"]
                except Exception:
                    continue
                keep[key] = line
        out["lines_after"] = len(keep)
        if out["lines_after"] == out["lines_before"]:
            out["bytes_after"] = out["bytes_before"]
            return out
        tmp = self.state.with_name(self.state.name + ".tmp")
        with tmp.open("w") as fh:
            for line in keep.values():
                fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.state)
        out["bytes_after"] = self.state.stat().st_size
        return out


def _atomic_write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def pick_best(cells_dir: Path, key: str, how: str = "best",
              current: dict | None = None,
              max_rmse: float = 1500.0, health_of=None) -> tuple[dict | None, int]:
    if how == "first" or how not in BEST_OF:
        return current, 0
    health = health_of if (
        how == "flyable" and health_of is not None) else (lambda c: None)
    best = current
    best_rank = rank_of(current, how, max_rmse,
                        health(current)) if current else -1e9
    best_n = 0
    for n in list_attempts(cells_dir, key):
        p = attempt_path(cells_dir, key, n)
        try:
            cell = json.loads(p.read_text())
        except Exception:
            continue
        if validate_cell(cell) is not None:
            continue
        r = rank_of(cell, how, max_rmse, health(cell))
        if r > best_rank:
            best, best_rank, best_n = cell, r, n
    return best, best_n


BEST_OF = ("best", "converged", "fastest", "first", "flyable")


def _unused_run_ray(job):
    ray_key, entries, coeffs_path, nodes, order = job
    out = []
    guess = None
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


def _slim_solution(sol: dict, digits: int = 6) -> dict:
    out = {}
    for k in ("x", "h", "V", "gamma", "mass", "alpha"):
        v = sol.get(k)
        if v is None:
            continue
        out[k] = [float(f"{float(x):.{digits}g}") for x in v]
    out["tof"] = float(f"{float(sol.get('tof', 0.0)):.{digits}g}")
    return out


def _solve_opts(a) -> dict:
    opts = {}
    for attr, key in (("refine_iterations", "refine_iterations"),
                      ("refine_tol", "refine_tol"),
                      ("min_speed", "min_speed"),
                      ("max_q", "max_q_kpa"),
                      ("alpha_rate", "alpha_rate_dps")):
        v = getattr(a, attr, None)
        if v is not None:
            opts[key] = v
    return opts


def _solve_one(key, geom, coeffs_path, nodes, order, guess, auth=3.0, dive=30.0,
               maxiter=250, store_sol=False, opts=None):
    t0 = time.time()
    opts = dict(opts or {})
    try:
        sys.path.insert(0, str(HERE))
        from dymos_opt import solve, fit_polynomial, _isolate_workdir
        _isolate_workdir()

        sol = solve(coeffs_path,
                    range_km=geom["range_km"], launch_alt_m=geom["launch_alt_m"],
                    target_alt_m=geom["target_alt_m"],
                    launch_speed_ms=geom["launch_speed_ms"],
                    nodes=nodes, verbose=False, guess=guess,
                    min_authority_g=auth, max_dive_deg=dive,
                    maxiter=maxiter, **opts)
        a, rmse = fit_polynomial(sol, order=order)
        cell = {
            "maxiter": int(maxiter),
            "_settings": {"min_authority_g": float(auth), "max_dive_deg": float(dive),
                          "maxiter": int(maxiter), "nodes": int(nodes),
                          "order": int(order)},
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
        cell["_settings"].update(opts)
        cell["feasible"] = bool(sol.get("feasible", True))
        cell["violation"] = None if cell["feasible"] else str(
            sol.get("msg", ""))
        cell["authority_floor_g"] = float(sol.get("authority_floor", 0.0))
        cell["speed_floor_ms"] = float(sol.get("speed_floor", 0.0))
        cell["v_min_ms"] = float(sol.get("v_min_ms", sol["V_term"]))
        cell["nav_min_g"] = float(sol.get("nav_min_g", 0.0))
        cell["t_bound"] = sol.get("t_at_bound")
        cell["duration_bounds_s"] = sol.get("t_bounds")
        cell["refine_error"] = sol.get("refine_error")
        cell["_sol"] = _slim_solution(sol)
        verdict, detail = cell_health(cell, load_curves(coeffs_path))
        cell["health"] = "infeasible" if not cell["feasible"] else verdict
        cell["defect_ratio"] = (round(float(detail["ratio"]), 3)
                                if detail and "ratio" in detail else None)
        if not store_sol:
            del cell["_sol"]
        ok = math.isfinite(
            cell["terminal_speed_ms"]) and cell["terminal_speed_ms"] > 0
        rec = {"key": key, "ok": ok, "cell": cell if ok else None,
               "elapsed_s": cell["_elapsed_s"],
               "error": None if ok else "solver returned a non-finite speed",
               "converged": cell["converged"],
               "terminal_speed_ms": cell["terminal_speed_ms"]}
        if ok:
            if store_sol:
                cell["_sol"] = _slim_solution(sol)
            rec["_sol"] = {k: sol[k]
                           for k in ("x", "h", "V", "gamma", "mass", "alpha", "tof")}
        return rec
    except Exception as e:
        return {"key": key, "ok": False, "cell": None,
                "elapsed_s": round(time.time() - t0, 1),
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(limit=4)}


def run_cell(job):
    (key, geom, coeffs_path, nodes, order, _timeout, auth, dive, maxiter,
     store_sol, opts) = (list(job) + [False, {}])[:11]
    return key, _solve_one(key, geom, coeffs_path, nodes, order, None, auth, dive,
                           maxiter, store_sol, opts)


def _progress_line(i: int, n: int, key: str, rec: dict) -> str:
    if rec["ok"]:
        return (f"[{i}/{n}] {key}  {rec['terminal_speed_ms']:7.1f} m/s  "
                f"{rec['elapsed_s']:6.1f}s  {'conv' if rec['converged'] else 'ITER-LIMIT'}")
    return f"[{i}/{n}] {key}  FAILED: {rec['error']}"


def _ray_entry(work, coeffs_path, nodes, order, auth, dive, maxiter=250,
               store_sol=False, opts=None, progress=None):
    _rk, entries = work
    out = []
    guess = None
    for (key, geom, _seed) in entries:
        rec = _solve_one(key, geom, coeffs_path, nodes,
                         order, guess, auth, dive, maxiter, store_sol, opts)
        guess = rec.pop("_sol", None) if rec["ok"] else guess
        out.append((key, rec))
        if progress is not None:
            tokens, total = progress
            print(_progress_line(tokens.get(), total, key, rec), flush=True)
    return out


def _poly_eval(c, xi: float) -> float:
    acc = float(c[-1])
    for k in range(len(c) - 2, -1, -1):
        acc = acc * xi + float(c[k])
    return acc


def _poly_slope(c, xi: float) -> float:
    acc = 0.0
    for k in range(len(c) - 1, 0, -1):
        acc = acc * xi + k * float(c[k])
    return acc


def _mass_profile(coeffs_path, tof: float, n: int):
    try:
        d = json.loads(Path(coeffs_path).read_text())
        dry = float(d["mass_dry"])
        motors = d.get("motors", [])
        m0 = dry + sum(float(m.get("fuel_mass", 0.0)) for m in motors)
    except Exception:
        return None, None
    pts, m, t = [], m0, 0.0
    for mo in motors:
        t += float(mo.get("delay", 0.0))
        pts.append((t, m))
        t += float(mo.get("burn_time", 0.0))
        m -= float(mo.get("fuel_mass", 0.0))
        pts.append((t, m))
    pts.append((max(t, 1e-3), m))
    return dry, pts


def seed_from_cell(cell: dict, coeffs_path=None, n: int = 24):
    a = [float(v) for v in (cell.get("poly_a") or [])
         if isinstance(v, (int, float))]
    if len(a) < 2:
        return None
    try:
        Xf = float(cell["range_km"]) * 1000.0
        h0 = float(cell["launch_alt_m"])
        V0 = float(cell["launch_speed_ms"])
        Vt = float(cell.get("terminal_speed_ms", V0))
        tof = float(cell.get("tof_s", 60.0))
    except (KeyError, TypeError, ValueError):
        return None
    if not (Xf > 0 and tof > 0):
        return None

    xi = [i / (n - 1) for i in range(n)]
    h = [_poly_eval(a, x) for x in xi]
    gamma = [math.atan2(_poly_slope(a, x), max(Xf, 1.0)) for x in xi]
    V = [max(50.0, V0 + (Vt - V0) * x) for x in xi]

    _dry, pts = _mass_profile(
        coeffs_path, tof, n) if coeffs_path else (None, None)
    if pts:
        mass = []
        for x in xi:
            t = x * tof
            m = pts[0][1]
            for (t1, m1) in pts:
                if t1 <= t:
                    m = m1
                else:
                    break
            mass.append(m)
    else:
        mass = [max(V0, 1.0)] * n

    return {"t": [x * tof for x in xi], "x": [x * Xf for x in xi], "h": h,
            "V": V, "gamma": gamma, "mass": mass, "alpha": [0.0] * n,
            "tof": tof, "_h0": h0}


def _reopt_entry(job):
    (key, geom, coeffs_path, nodes, order, auth, dive, maxiter,
     attempts, best_of, cand_path, converge_first, store_sol,
     opts) = (list(job) + [{}])[:14]
    out = []
    guess = None
    if cand_path:
        try:
            cand = json.loads(Path(cand_path).read_text())
        except Exception:
            cand = None
        if cand:
            guess = cand.get("_sol") or seed_from_cell(cand, coeffs_path)
    for _i in range(max(1, attempts)):
        rec = _solve_one(key, geom, coeffs_path, nodes, order, guess,
                         auth, dive, maxiter, store_sol, opts)
        rec.pop("_sol", None)
        out.append((key, rec))
        guess = None
        if converge_first and rec["ok"] and rec.get("converged"):
            break
    return out


def _print_floors(g, a) -> None:
    bits = []
    bits.append(f"authority floor {a.min_authority:g} g" if a.min_authority
                else "no authority floor")
    bits.append(f"speed floor {a.min_speed:g} m/s on the PATH" if a.min_speed
                else "no path speed floor (terminal rule only)")
    n = int(getattr(a, "refine_iterations", 0) or 0)
    if n:
        bits.append(f"refine x{n} (tol {getattr(a, 'refine_tol', 1e-4):g})")
    print(f"{g['name']}: {', '.join(bits)}")


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
                     a.min_authority, a.max_dive, a.maxiter, a.store_solutions,
                     _solve_opts(a)))
        if a.limit and len(todo) >= a.limit:
            break

    total = sum(1 for _ in enumerate_cells(g))
    ok = sum(1 for r in done.values() if r.get("ok"))
    failed = sum(1 for r in done.values() if not r.get("ok"))
    print(f"{g['name']}: {total} cells total | {ok} done | {failed} failed | {len(todo)} to run "
          f"({a.jobs} jobs, {g['nodes']} nodes)")
    _print_floors(g, a)
    if not todo:
        print("nothing to do. run --merge to emit the table.")
        return

    est = len(todo) * 90.0 / max(a.jobs, 1)
    print(f"rough ETA {est/60:.0f} min at ~90 s/cell\n")

    t0 = time.time()
    n_ok = n_fail = 0

    if a.no_warm:
        work = [(("single", t[0]), [(t[0], t[1], None)]) for t in todo]
    else:
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

    with multiprocessing.Manager() as mgr:
        tokens = mgr.Queue()
        for k in range(len(todo)):
            tokens.put(k + 1)
        sys.stdout.flush()
        with ProcessPoolExecutor(max_workers=a.jobs, max_tasks_per_child=a.max_tasks) as ex:
            futs = {ex.submit(_ray_entry, w, coeffs_path, g["nodes"], g["order"],
                              a.min_authority, a.max_dive, a.maxiter,
                              a.store_solutions, _solve_opts(a),
                              progress=(tokens, len(todo))): w[0] for w in work}
            for fut in as_completed(futs):
                for key, rec in fut.result():
                    store.put(key, rec)
                    if rec["ok"]:
                        n_ok += 1
                    else:
                        n_fail += 1

    dt = time.time() - t0
    print(f"\nfinished {n_ok} ok / {n_fail} failed in {dt/60:.1f} min. "
          f"Re-run the same command to retry anything interrupted; --merge to emit the table.")


def unusable_reason(cell: dict):
    if cell is None:
        return "no usable cell file"
    why = validate_cell(cell)
    return None if why is None else f"unusable: {why}"


def fails_filter(cell: dict, a, curves=None) -> str | None:
    sp = cell.get("terminal_speed_ms")
    tof = cell.get("tof_s")
    if a.min_terminal_speed is not None and sp is not None and sp < a.min_terminal_speed:
        return f"terminal {sp:.0f} m/s < {a.min_terminal_speed:.0f}"
    if a.max_terminal_speed is not None and sp is not None and sp > a.max_terminal_speed:
        return f"terminal {sp:.0f} m/s > {a.max_terminal_speed:.0f} (solver outlier)"
    if a.max_tof is not None and tof is not None and tof > a.max_tof:
        return f"tof {tof:.0f} s > {a.max_tof:.0f}"
    if a.max_weaves is not None:
        turns = turning_points(cell)
        if turns > a.max_weaves:
            return f"{turns} turning points > {a.max_weaves} (woven profile)"
    if getattr(a, "infeasible", False) and curves is not None:
        verdict, detail = cell_health(
            cell, curves, a.max_weaves or 2, a.speed_tol)
        if verdict == "infeasible":
            return infeasible_reason(detail)
    return None


def select_for_reopt(g, a, done: dict, curves=None) -> list[tuple[str, dict, str]]:
    explicit = (a.min_terminal_speed is not None or a.max_terminal_speed is not None
                or a.max_tof is not None or a.max_weaves is not None
                or getattr(a, "infeasible", False)
                or a.max_rmse is not None and a.rmse_filter)
    out = []
    for geom in enumerate_cells(g):
        k = cell_key(geom)
        rec = done.get(k)
        cell = rec.get("cell") if rec and rec.get("ok") else None

        if cell is None:
            out.append((k, geom, "no usable cell file"))
        elif explicit:
            why = fails_filter(cell, a, curves) or (
                f"fit rmse {float(cell.get('fit_rmse_m', float('nan'))):.0f} m"
                f" > {a.max_rmse:.0f}" if a.rmse_filter and
                (cell.get("fit_rmse_m") or 0.0) > a.max_rmse else None)
            if why is None:
                continue
            out.append((k, geom, why))
        else:
            out.append((k, geom, unusable_reason(cell) or "re-solve"))
        if a.limit and len(out) >= a.limit:
            print(f"(--limit {a.limit}: stopping the selection here)")
            break
    return out


def cmd_reopt(g, a):
    store = Store(a.runs / g["name"])
    done = store.index()
    warn_rejected(store)
    coeffs_path = str((HERE / g["coeffs"]).resolve()
                      ) if not Path(g["coeffs"]).is_absolute() else g["coeffs"]

    if a.best_of_only:
        return cmd_bestof(g, a, store, done)

    curves = load_curves(coeffs_path)
    todo = select_for_reopt(g, a, done, curves)
    total_cells = sum(1 for _ in enumerate_cells(g))

    print(f"{g['name']}: re-optimise -- {len(todo)} of {total_cells} cells selected, "
          f"{a.reopt_runs} attempt(s) each, best-of={a.best_of}")
    reasons: dict = {}
    for _k, _gm, why in todo:
        reasons[why.split("(")[0].strip()] = reasons.get(
            why.split("(")[0].strip(), 0) + 1
    for why, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"    {n:4d}  {why}")
    if not todo:
        print("nothing selected: no cell violates that filter. Drop the filter to "
              "re-solve every cell, or loosen it (--min-terminal-speed etc).")
        return
    if a.dry_run:
        print("\n--dry-run: nothing solved. First 20 selected cells:")
        for k, _gm, why in todo[:20]:
            print(f"    {k}  ({why})")
        if len(todo) > 20:
            print(f"    ... and {len(todo) - 20} more")
        return

    est = len(todo) * a.reopt_runs * 90.0 / max(a.jobs, 1)
    print(f"rough ETA {est/60:.0f} min at ~90 s/cell/attempt\n")

    rows = []
    for k, geom, why in todo:
        rec = done.get(k)
        cur = rec.get("cell") if rec and rec.get("ok") else None
        rows.append((k, geom, why, cur))

    keep = bool(getattr(a, "keep_attempts", False))

    next_slot = {k: attempts_recorded(store.cells, k, cur) + 1
                 for k, _gm, _why, cur in rows}

    jobs = []
    for k, geom, why, cur in rows:
        if keep:
            store.archive_current(k, next_slot[k])
            next_slot[k] += 1
        cand = None
        if (store.cells / f"{k}.json").exists():
            cand = str(store.cells / f"{k}.json")
        jobs.append((k, geom, coeffs_path, g["nodes"], g["order"],
                     a.min_authority, a.max_dive, a.maxiter,
                     a.reopt_runs, a.best_of, cand,
                     getattr(a, "converge_first", False), a.store_solutions,
                     _solve_opts(a)))

    t0 = time.time()
    n_ok = n_fail = 0
    improved = 0
    pruned_files = pruned_bytes = 0
    with ProcessPoolExecutor(max_workers=a.jobs, max_tasks_per_child=a.max_tasks) as ex:
        futs = {ex.submit(_reopt_entry, j): j[0] for j in jobs}
        done_cells = 0
        for fut in as_completed(futs):
            key = futs[fut]
            try:
                results = fut.result()
            except Exception as e:
                print(f"{key}  WORKER CRASHED: {type(e).__name__}: {e}")
                results = []
            before = done.get(key, {}).get("cell")
            before_v = float(before["terminal_speed_ms"]
                             ) if before else float("nan")
            written = []
            for (k, rec) in results:
                attempt = next_slot.get(k, 1)
                next_slot[k] = attempt + 1
                store.put(k, rec, attempt=attempt, promote=keep)
                written.append(attempt)
                if rec["ok"]:
                    n_ok += 1
                    print(f"{k}  attempt {attempt}: {rec['terminal_speed_ms']:7.1f} m/s  "
                          f"{rec['elapsed_s']:6.1f}s  "
                          f"{'conv' if rec['converged'] else 'ITER-LIMIT'}")
                else:
                    n_fail += 1
                    print(f"{k}  attempt {attempt}: FAILED: {rec['error']}")

            win, win_n = pick_best(store.cells, key, a.best_of,
                                   _read_cell(store.cells, key), a.max_rmse,
                                   health_of=_health_fn(a, curves))
            if win is not None:
                win = dict(win)
                win["_key"] = key
                n_total = attempts_recorded(store.cells, key, before)
                win["_attempts"] = n_total
                win["_settings"] = win.get("_settings") or {
                    "min_authority_g": a.min_authority, "max_dive_deg": a.max_dive,
                    "maxiter": a.maxiter, "nodes": g["nodes"], "order": g["order"]}
                _atomic_write(store.cells / f"{key}.json", win)
                if not keep:
                    n_files, n_bytes = store.prune(key)
                    pruned_files += n_files
                    pruned_bytes += n_bytes
                after_v = float(win["terminal_speed_ms"])
                delta = ("" if not before or after_v <= before_v + 1.0
                         else f"  (+{after_v - before_v:.1f} on {before_v:.1f})")
                if after_v > (before_v if before else 0.0) + 1.0:
                    improved += 1
                if win_n == 0:
                    print(f"{key}  -> unchanged, the existing cell is still the best: "
                          f"{after_v:.1f} m/s{delta}")
                elif win_n in written:
                    print(f"{key}  -> kept new attempt {win_n} of {n_total}: "
                          f"{after_v:.1f} m/s{delta}")
                else:
                    print(f"{key}  -> kept attempt {win_n} of {n_total}, from an "
                          f"earlier run: {after_v:.1f} m/s{delta}")
            done_cells += 1
            sys.stdout.flush()

    dt = time.time() - t0
    if keep:
        tail = (f"Attempts are kept in {store.cells}. --best-of-only re-picks "
                f"winners without solving; --merge to emit the table.")
    else:
        tail = (f"Losing attempts deleted: {pruned_files} file(s), "
                f"{pruned_bytes / 1024:.0f} KB. --keep-attempts keeps every "
                f"attempt; --prune-attempts cleans an older runs/ tree; --merge "
                f"to emit the table.")
    print(f"\nre-optimised {len(todo)} cells in {dt/60:.1f} min: "
          f"{n_ok} ok / {n_fail} failed, {improved} cells improved. {tail}")
    _compact_and_report(store)


def _health_fn(a, curves):
    if a.best_of != "flyable":
        return None
    return lambda c: cell_health(c, curves, a.max_weaves or 2, a.speed_tol)[0]


def _compact_and_report(store, quiet=False) -> dict:
    c = store.compact()
    if not quiet and c["lines_after"] < c["lines_before"]:
        print(f"  state.jsonl compacted: {c['lines_before']} -> {c['lines_after']} "
              f"record(s), {c['bytes_before'] / 1024:.0f} -> "
              f"{c['bytes_after'] / 1024:.0f} KB")
    return c


def _repick(g, a, store, done, keys=None, curves=None, quiet=False) -> int:
    changed = 0
    for key in sorted(keys if keys is not None else done):
        rec = done.get(key, {})
        nums = list_attempts(store.cells, key)
        if not nums:
            continue
        cur = rec.get("cell")
        win, win_n = pick_best(store.cells, key, a.best_of, cur, a.max_rmse,
                               health_of=_health_fn(a, curves))
        if win is None:
            continue
        win = dict(win)
        win["_key"] = key
        win["_attempts"] = attempts_recorded(store.cells, key, cur)
        if not win.get("_settings") and rec.get("cell", {}).get("_settings"):
            win["_settings"] = rec["cell"]["_settings"]
        before = float(cur["terminal_speed_ms"]) if cur else float("nan")
        after = float(win["terminal_speed_ms"])
        if win_n or before != after:
            _atomic_write(store.cells / f"{key}.json", win)
            rec = done.setdefault(
                key, {"key": key, "ok": True, "elapsed_s": 0.0})
            _decorate(rec, win)
            rec["ok"] = True
            if after != before:
                changed += 1
                if not quiet:
                    print(f"{key}: {before:.1f} -> {after:.1f} m/s "
                          f"(attempt {win_n} of {len(nums)})")
    return changed


def cmd_bestof(g, a, store=None, done=None):
    store = store or Store(a.runs / g["name"])
    done = done if done is not None else store.index()
    changed = _repick(g, a, store, done, None, load_curves(_coeffs_for(g)))
    print(f"{g['name']}: re-picked winners for cells with attempts on disk, "
          f"{changed} changed. Nothing was solved.")


def _coeffs_for(g) -> str:
    return (str((HERE / g["coeffs"]).resolve())
            if not Path(g["coeffs"]).is_absolute() else g["coeffs"])


def cmd_check(g, a, store=None, done=None) -> int:
    curves = load_curves(_coeffs_for(g))
    rows = []
    if a.table:
        try:
            tab = json.loads(Path(a.table).read_text())
        except Exception as e:
            print(f"cannot read {a.table}: {type(e).__name__}: {e}")
            return 0
        items = tab.get("cells", tab if isinstance(tab, list) else [])
        for cell in items:
            key = cell.get("_key") or "_".join(
                f"{float(cell.get(k, 0.0)):g}" for k in AXIS_ORDER)
            rows.append((key, cell))
    else:
        store = store or Store(a.runs / g["name"])
        done = done if done is not None else store.index()
        warn_rejected(store)
        rows = [(k, r.get("cell"))
                for k, r in sorted(done.items()) if r.get("cell")]

    bad = woven = ok = 0
    for key, cell in rows:
        verdict, d = cell_health(cell, curves, a.max_weaves or 2, a.speed_tol)
        if verdict == "infeasible":
            bad += 1
            print(f"  INFEASIBLE {key}  {infeasible_reason(d)}")
        elif verdict == "woven":
            woven += 1
            print(f"  woven      {key}  {d['turns']} turning points"
                  + (f", claims {d['V_stored']:.0f} m/s -> {d['V_reint']:.0f} m/s"
                     if "V_stored" in d else ""))
        elif verdict == "ok":
            ok += 1
    vmin = [c.get("v_min_ms")
            for _, c in rows if c.get("v_min_ms") is not None]
    nmin = [c.get("nav_min_g")
            for _, c in rows if c.get("nav_min_g") is not None]
    if vmin or nmin:
        print(f"  measured, no floor implied:"
              + (f" lowest speed {min(vmin):.0f} m/s" if vmin else "")
              + (f", lowest available load factor {min(nmin):.2f} g" if nmin else ""))
    src = a.table or f"{a.runs / g['name']}"
    print(f"\n{src}: {len(rows)} cell(s): {ok} ok, {woven} woven, {bad} INFEASIBLE"
          + ("" if curves is not None else " (coeffs file unreadable: no re-integration)"))
    if a.table:
        print("  A table carries no trajectory, so a cell that is not woven can only "
              "be called unverified here. Solve with --store-solutions to have it "
              "re-integrated.")
    return 1 if bad else 0


def cmd_compact_state(g, a, store=None):
    """Rewrite the log with one record per cell."""
    store = store or Store(a.runs / g["name"])
    c = store.compact()
    if not c["lines_before"]:
        print(f"{g['name']}: {store.state.name} is absent or empty, nothing to do")
    elif c["lines_after"] == c["lines_before"]:
        print(f"{g['name']}: state.jsonl already holds one record per cell "
              f"({c['lines_after']} record(s), {c['bytes_after'] / 1024:.0f} KB)")
    else:
        print(f"{g['name']}: state.jsonl compacted, {c['lines_before']} -> "
              f"{c['lines_after']} record(s), {c['bytes_before'] / 1024:.0f} -> "
              f"{c['bytes_after'] / 1024:.0f} KB. Nothing was solved.")


def cmd_prune(g, a, store=None, done=None):
    store = store or Store(a.runs / g["name"])
    done = done if done is not None else store.index()
    keys = set(done) | keys_with_attempts(store.cells)
    curves = load_curves(_coeffs_for(g))
    changed = _repick(g, a, store, done, keys, curves)
    files = nbytes = cells = skipped = 0
    for key in sorted(keys):
        if not list_attempts(store.cells, key):
            continue
        n, b = store.prune(key)
        if n:
            files += n
            nbytes += b
            cells += 1
        else:
            skipped += 1
    print(f"{g['name']}: deleted {files} attempt file(s), {nbytes / 1024:.0f} KB, "
          f"from {cells} cell(s)"
          + (f", promoting the best of them for {changed} cell(s) first" if changed else "")
          + ". Nothing was solved.")
    if skipped:
        print(f"  {skipped} cell(s) kept their attempts: no usable attempt or "
              f"canonical cell to promote, and an attempt may be the only copy "
              f"of a good trajectory. Check them with --status.")
    _compact_and_report(store)


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
    changed = _repick(g, a, store, done, curves=load_curves(_coeffs_for(g)),
                      quiet=True)
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
    with_attempts = [k for k, r in done.items() if len(
        r.get("attempts", [])) > 0]
    if with_attempts:
        extra = sum(len(r.get("attempts", [])) for r in done.values())
        print(f"  {len(with_attempts)} cells have re-optimise attempts on disk "
              f"({extra} attempt files) -- --best-of-only re-picks winners for free")
    else:
        recorded = sum(int((r.get("cell") or {}).get("_attempts") or 0)
                       for r in done.values())
        if recorded:
            print(f"  {recorded} re-optimise attempt(s) recorded, no attempt files "
                  f"on disk (losers are deleted once the winner is picked; "
                  f"--keep-attempts keeps them)")
    if changed:
        print(f"  (re-picked the winner for {changed} cell(s) from the attempts on "
              f"disk; nothing was solved -- --best-of-only does this out loud)")
    onbound = [k for k, r in ok.items()
               if (r.get("cell") or {}).get("t_bound")]
    if onbound:
        shown = ", ".join(sorted(onbound)[:3]) + \
            (" ..." if len(onbound) > 3 else "")
        print(f"  {len(onbound)} cell(s) finished with the flight time ON a duration "
              f"bound ({shown}): the duration is the box's answer, not the "
              f"optimiser's -- see t_bound/duration_bounds_s in the cell")
    for k, r in list(failed.items())[:10]:
        print(f"  FAILED {k}: {r.get('error')}")
    if len(failed) > 10:
        print(f"  ... and {len(failed)-10} more")


def _envelope_one(geom, coeffs_path, nodes, order, maxiter=250, opts=None,
                  auth=3.0):
    t0 = time.time()
    try:
        sys.path.insert(0, str(HERE))
        from dymos_opt import solve, _isolate_workdir
        _isolate_workdir()
        sol = solve(coeffs_path, None,
                    launch_alt_m=geom["launch_alt_m"],
                    target_alt_m=geom["target_alt_m"],
                    launch_speed_ms=geom["launch_speed_ms"],
                    nodes=nodes, verbose=False, maxiter=maxiter,
                    min_authority_g=auth, **(opts or {}))
        return {"ok": True, "max_range_km": float(sol["range_m"]) / 1000.0,
                "terminal_speed_ms": float(sol["V_term"]),
                "tof_s": float(sol["tof"]), "apex_alt_m": float(sol["apex"]),
                "v_min_ms": float(sol.get("v_min_ms", sol["V_term"])),
                "nav_min_g": float(sol.get("nav_min_g", 0.0)),
                "feasible": bool(sol["feasible"]),
                "violation": None if sol["feasible"] else str(sol.get("msg", "")),
                "authority_floor_g": float(sol.get("authority_floor", 0.0)),
                "speed_floor_ms": float(sol.get("speed_floor", 0.0)),
                "elapsed_s": round(time.time() - t0, 1)}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}",
                "elapsed_s": round(time.time() - t0, 1)}


def cmd_envelope(g, a):
    groups = {}
    for geom in enumerate_cells(g):
        key = (float(geom["launch_alt_m"]), float(geom["launch_speed_ms"]),
               float(geom["target_alt_m"]))
        groups.setdefault(key, geom)
    coeffs_path = (str((HERE / g["coeffs"]).resolve())
                   if not Path(g["coeffs"]).is_absolute() else g["coeffs"])
    opts = _solve_opts(a)
    print(f"{g['name']}: reachable set for {len(groups)} launch condition(s), "
          f"{a.jobs} jobs, {g['nodes']} nodes, maxiter {a.maxiter}")
    _print_floors(g, a)
    out, t0 = {}, time.time()
    with ProcessPoolExecutor(max_workers=a.jobs, max_tasks_per_child=a.max_tasks) as ex:
        futs = {ex.submit(_envelope_one, geom, coeffs_path, g["nodes"], g["order"],
                          a.maxiter, opts, a.min_authority): key
                for key, geom in groups.items()}
        for fut in as_completed(futs):
            key = futs[fut]
            rec = fut.result()
            name = f"{key[0]:.0f}_{key[1]:.0f}_{key[2]:.0f}"
            rec.update({"launch_alt_m": key[0], "launch_speed_ms": key[1],
                        "target_alt_m": key[2]})
            out[name] = rec
            if rec.get("ok"):
                note = "" if rec["feasible"] else "  INFEASIBLE: " + \
                    str(rec["violation"])[:70]
                print(f"  {name:>20s}  max range {rec['max_range_km']:7.1f} km  "
                      f"V {rec['terminal_speed_ms']:6.1f} m/s  {rec['elapsed_s']:5.1f}s{note}")
            else:
                print(f"  {name:>20s}  FAILED: {rec['error']}")
            sys.stdout.flush()
    dest = a.runs / g["name"] / "envelope.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(
        {"_settings": {"nodes": g["nodes"], "order": g["order"],
                       "maxiter": a.maxiter, "min_authority_g": a.min_authority,
                       "max_dive_deg": a.max_dive, **opts},
         "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
         "cells": out}, indent=1))
    print(f"\nwrote {dest}: {len(out)} launch condition(s) in "
          f"{(time.time() - t0)/60:.1f} min. --merge writes exp_max_range_km/"
          f"reachable into the table.")


def cmd_merge(g, a):
    store = Store(a.runs / g["name"])
    done = store.index()
    warn_rejected(store)
    changed = _repick(g, a, store, done, curves=load_curves(_coeffs_for(g)),
                      quiet=True)
    cells = [r["cell"] for r in done.values() if r.get("ok") and r.get("cell")]
    for c in cells:
        c.setdefault("missile", g["missile"])
        c.pop("_key", None)
        c.pop("_sol", None)
    env_path = a.runs / g["name"] / "envelope.json"
    n_unreach = 0
    if env_path.exists():
        try:
            env_cells = json.loads(env_path.read_text()).get("cells", {})
        except Exception:
            env_cells = {}
        for c in cells:
            rec = env_cells.get(f"{c['launch_alt_m']:.0f}_{c['launch_speed_ms']:.0f}"
                                f"_{c['target_alt_m']:.0f}")
            if rec and rec.get("ok"):
                c["exp_max_range_km"] = round(float(rec["max_range_km"]), 1)
                c["reachable"] = bool(
                    float(c["range_km"]) <= float(rec["max_range_km"]))
                n_unreach += 0 if c["reachable"] else 1
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
    print(f"wrote {dest}: {len(cells)} cells ({nconv} converged)"
          + (f", {n_unreach} beyond the solved envelope (reachable: false)"
             if env_path.exists() else ""))
    if changed:
        print(f"  (re-picked the winner for {changed} cell(s) from the attempts on "
              f"disk before writing it; nothing was solved)")
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
    ap.add_argument("--max-tasks", type=int, default=10000,
                    help="restart each worker after this many cells (OpenMDAO leaks)")
    ap.add_argument("--store-solutions", dest="store_solutions", action="store_true",
                    default=None,
                    help="write the solved trajectory into each cell file (_sol), so "
                         "plot_trajectories.py can draw speed / flight path angle / "
                         "angle of attack and the polynomial residual for a cell. "
                         "Roughly triples the size of runs/. Off by default; --merge "
                         "never carries it into the table.")
    ap.add_argument("--no-store-solutions", dest="store_solutions", action="store_false",
                    help="override store_solutions=true in the grid spec")
    ap.add_argument("--runs", default=str(HERE / "runs"))
    ap.add_argument("--out", default="")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--refine-iterations", type=int, default=None, metavar="N",
                    help="hp-adaptive grid refinement passes per solve; overrides "
                         "refine_iterations in the grid spec, 0 = off. Expensive "
                         "but this is what makes a first pass good: put it in the "
                         "spec of a run you are leaving overnight")
    ap.add_argument("--refine-tol", type=float, default=None, metavar="E",
                    help="refinement error tolerance; overrides refine_tol in the "
                         "grid spec (default 1e-4)")
    ap.add_argument("--min-speed", type=float, default=None, metavar="MS",
                    help="speed floor along the PATH (m/s), not only at the end. "
                         "Overrides min_speed_ms in the grid spec. Unset = no path "
                         "floor, i.e. only the terminal speed rule, as before")
    ap.add_argument("--max-q", type=float, default=None, metavar="KPA",
                    help="dynamic-pressure ceiling (kPa); off by default, this "
                         "game's cells fly 600-2600 kPa terminal phases by design")
    ap.add_argument("--alpha-rate", type=float, default=None, metavar="DEGS",
                    help="AoA rate limit (deg/s), EuroGNC 2022-040 eqs 5-6: alpha "
                         "becomes a state and its rate the control")
    ap.add_argument("--envelope", action="store_true",
                    help="solve each launch condition for maximum range and write "
                         "envelope.json; --merge then marks cells beyond it")
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
    ap.add_argument("--reoptimise", "--reoptimize", dest="reoptimise",
                    action="store_true",
                    help="re-solve cells and keep the best of the attempts. With no "
                         "filter this is every cell: nothing in a cell identifies it "
                         "as a bad one, and the previous result stays in force unless "
                         "a new attempt beats it. Add --min-terminal-speed / --max-tof "
                         "/ --max-rmse --rmse-filter to target a subset.")
    ap.add_argument("--optimise-all", "--optimize-all", dest="all_cells",
                    action="store_true",
                    help="with --reoptimise, re-solve every cell. This is already "
                         "what a bare --reoptimise does; it is here because it reads "
                         "well.")
    ap.add_argument("--all", dest="all_cells",
                    action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--reopt-runs", type=int, default=1, metavar="N",
                    help="attempts per cell per invocation (default 1). Attempt 1 is "
                         "warm-started from the current solution and is a floor; the "
                         "rest are cold, which is where the variance is.")
    ap.add_argument("--iterations", type=int, default=0, metavar="N",
                    help="re-solve a cell until it converges OR it has had N attempts, "
                         "whichever comes first. Overrides --reopt-runs.")
    ap.add_argument("--check", action="store_true",
                    help="report cells whose trajectory is woven (up-down-up-down) or "
                         "whose stored trajectory does not satisfy its own dynamics; "
                         "solves nothing, exits 1 if any cell is infeasible")
    ap.add_argument("--table", default="", metavar="PATH",
                    help="with --check: check a merged table instead of a run directory")
    ap.add_argument("--max-weaves", type=int, default=None, metavar="N",
                    help="more than N turning points in the altitude profile counts as "
                         "woven (default 2). As a filter, selects those cells for "
                         "--reoptimise; with --best-of flyable it is the threshold")
    ap.add_argument("--speed-tol", type=float, default=0.25, metavar="X",
                    help="with --check/--best-of flyable: a cell may arrive at most X "
                         "faster than re-integrating its own alpha history (default "
                         "0.25: cells that are actually solved land in 0.71-1.10, so "
                         "0.5 was hiding 30-50%% overstatements. Raise it to be "
                         "permissive, lower it to be strict)")
    ap.add_argument("--infeasible", action="store_true",
                    help="select cells whose stored trajectory fails the re-integration "
                         "check (needs --store-solutions to have been used)")
    ap.add_argument("--best-of", choices=list(BEST_OF), default="best",
                    help="which attempt wins: best (highest terminal speed, usable "
                         "cells first), converged (prefer a converged attempt), "
                         "fastest (ignore the fit), first (keep what is there)")
    ap.add_argument("--best-of-only", action="store_true",
                    help="do not solve anything; just re-pick the winner among the "
                         "attempts already on disk (a default pass prunes them, so "
                         "keep attempts with --keep-attempts if you want this to "
                         "have anything to pick from)")
    ap.add_argument("--keep-attempts", dest="keep_attempts", action="store_true",
                    help="keep every attempt file. The default is to delete the "
                         "attempts that lost once a winner has been picked -- "
                         "otherwise a re-optimise pass leaves (runs+1) files per "
                         "cell behind on every pass, forever")
    ap.add_argument("--compact-state", dest="compact_state", action="store_true",
                    help="rewrite state.jsonl with one record per cell and solve "
                         "nothing. It is append-only -- a line per attempt, ever, "
                         "pruned from disk or not -- while index() only reads the "
                         "last line for a key, so the rest is dead weight. "
                         "--reoptimise and --prune-attempts compact it as well")
    ap.add_argument("--prune-attempts", dest="prune_attempts", action="store_true",
                    help="delete the attempts that are not the winner, for every "
                         "cell in this run, and solve nothing; re-picks the winner "
                         "first (--best-of). Cleanup for a runs/ tree that already "
                         "has attempts in it")
    ap.add_argument("--min-terminal-speed", type=float, default=None, metavar="MS",
                    help="select cells whose terminal_speed_ms is below this. Status "
                         "filter too. A low terminal speed is not evidence of a bad "
                         "cell -- the objective is not monotone in the inputs -- so "
                         "this is a deliberate 'give these another go' choice, and a "
                         "big one: 130 of 252 cells in the older shipped table are "
                         "below 800. See --dry-run.")
    ap.add_argument("--max-terminal-speed", type=float, default=None, metavar="MS",
                    help="select cells ABOVE this: the solver sometimes returns an "
                         "outlier (3912 m/s for a 320 m/s launch has been seen)")
    ap.add_argument("--max-tof", type=float, default=None, metavar="S",
                    help="select cells whose time of flight is longer than this")
    ap.add_argument("--max-rmse", type=float, default=1500.0, metavar="M",
                    help="fit-quality threshold (default 1500), used for two things "
                         "only. (1) the winner rule prefers an attempt whose fit is "
                         "within it, because the table ships the polynomial rather "
                         "than the trajectory -- nothing is discarded, a badly "
                         "fitting attempt just loses to a well-fitting one. (2) with "
                         "--rmse-filter it also SELECTS cells whose fit is worse than "
                         "it. It never selects anything on its own, and it never "
                         "decides that a cell is bad.")
    ap.add_argument("--rmse-filter", action="store_true",
                    help="treat --max-rmse as a selection filter: select every cell "
                         "whose fit is worse than it. Only meaningful alongside "
                         "--reoptimise.")
    ap.add_argument("--dry-run", action="store_true",
                    help="with --reoptimise, list the selected cells and reasons "
                         "and stop -- the cheap way to see what a bare --reoptimise "
                         "would cost")
    a = ap.parse_args()
    a.runs = Path(a.runs)

    g = load_grid(Path(a.grid))
    if a.nodes:
        g["nodes"] = a.nodes
    a.min_authority = g["min_authority_g"] if a.min_authority is None else a.min_authority
    a.max_dive = g["max_dive_deg"] if a.max_dive is None else a.max_dive
    a.maxiter = g["maxiter"] if a.maxiter is None else a.maxiter
    a.store_solutions = (g["store_solutions"] if a.store_solutions is None
                         else a.store_solutions)
    a.min_speed = speed_floor_from_spec(g, a.min_speed, _coeffs_for(g))
    slow = min(float(v) for v in g["axes"]["launch_speed_ms"])
    if a.min_speed and float(a.min_speed) > slow:
        print(f"note: the path speed floor is {float(a.min_speed):.0f} m/s but this "
              f"grid launches as slowly as {slow:.0f} m/s, so for those cells the "
              f"floor is capped at {slow:.0f} m/s. Raise the slowest launch above "
              f"the floor (or lower the floor) to have it hold everywhere.")
    a.refine_iterations = (g["refine_iterations"] if a.refine_iterations is None
                           else a.refine_iterations)
    a.refine_tol = g["refine_tol"] if a.refine_tol is None else a.refine_tol

    if a.iterations and a.iterations > 0:
        a.reopt_runs = max(1, a.iterations)
        a.converge_first = True
    else:
        a.converge_first = False

    if a.purge:
        cmd_purge(g, a)
    elif a.envelope:
        cmd_envelope(g, a)
    elif a.status:
        cmd_status(g, a)
    elif a.merge:
        cmd_merge(g, a)
    elif a.check:
        return cmd_check(g, a)
    elif a.compact_state:
        cmd_compact_state(g, a)
    elif a.prune_attempts:
        if a.keep_attempts:
            print("--keep-attempts ignored: --prune-attempts is the explicit one.")
        cmd_prune(g, a)
    elif a.best_of_only:
        cmd_bestof(g, a)
    elif a.reoptimise or a.all_cells:
        cmd_reopt(g, a)
    else:
        cmd_run(g, a)
    return 0


def cli() -> int:
    try:
        rc = main()
    except SystemExit as e:
        code = e.code
        rc = 0 if code is None else (code if isinstance(code, int) else 1)
        return _leave(rc)
    return _leave(int(rc or 0))


if __name__ == "__main__":
    cli()
