from __future__ import annotations
from pathlib import Path
import sys
import os
import math
import json
import io
import contextlib
import argparse
import numpy as np
import openmdao.api as om
import dymos as dm
from missile_sim import Coeffs, Curve

import warnings

from openmdao.utils.om_warnings import DerivativesWarning, OMDeprecationWarning

warnings.filterwarnings(
    "ignore", category=OMDeprecationWarning,
    message=r"The INF_BOUND sentinel in OpenMDAO is deprecated\.")

_LAUNCH_ROWS = (r"(?:  traj\.mid\.(?:navail|V)\[path\], inds=\[\(0, 0\)\]\n)+")
warnings.filterwarnings(
    "ignore", category=DerivativesWarning,
    message=(r"(?s)^The following constraints or objectives cannot be impacted by the "
             r"design variables of the problem at the current design point:\n"
             + _LAUNCH_ROWS + r"$"))

os.environ.setdefault("OPENMDAO_REPORTS", "0")


def _isolate_workdir():
    """Send OpenMDAO's output tree to tmp dir."""
    import atexit
    import shutil
    import tempfile
    if os.environ.get("OPENMDAO_WORKDIR"):
        return os.environ["OPENMDAO_WORKDIR"]
    d = tempfile.mkdtemp(prefix="dymos_work_")
    os.environ["OPENMDAO_WORKDIR"] = d
    atexit.register(shutil.rmtree, d, True)
    return d


sys.path.insert(0, str(Path(__file__).parent))

FT = 0.3048
NMI = 1852.0
G = 9.80665


class MissileODE(om.ExplicitComponent):
    """3-DOF point-mass dynamics using the game's drag/lift curves."""

    def initialize(self):
        self.options.declare("num_nodes", types=int)
        self.options.declare("coeffs", types=Coeffs)
        self.options.declare("verbose", types=bool, default=False)

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("x", val=np.zeros(nn), units="m")
        self.add_input("h", val=np.zeros(nn), units="m")
        self.add_input("V", val=np.zeros(nn), units="m/s")
        self.add_input("gamma", val=np.zeros(nn), units="rad")
        self.add_input("mass", val=np.zeros(nn), units="kg")
        self.add_input("alpha", val=np.zeros(nn), units="rad")
        self.add_input("time", val=np.zeros(nn), units="s")
        self.add_output("xdot", val=np.zeros(nn), units="m/s")
        self.add_output("hdot", val=np.zeros(nn), units="m/s")
        self.add_output("Vdot", val=np.zeros(nn), units="m/s**2")
        self.add_output("gammadot", val=np.zeros(nn), units="rad/s")
        self.add_output("massdot", val=np.zeros(nn), units="kg/s")
        self.add_output("nload", val=np.zeros(nn))
        self.add_output("navail", val=np.zeros(nn))
        self.add_output("qbar", val=np.zeros(nn))
        self.add_input("alphadot", val=np.zeros(nn), units="rad/s")
        self.add_output("alpha_rate", val=np.zeros(nn), units="rad/s")
        self.declare_partials(of="*", wrt="*", method="fd")
        self.declare_coloring(wrt="*", method="fd", form="forward",
                              num_full_jacs=2, tol=1e-20, orders=None,
                              show_summary=self.options["verbose"])

    def _motor(self, time):
        """Mirrors Missile.MotorThrust: sequential stages, each with a delay."""
        c: Coeffs = self.options["coeffs"]
        T = np.zeros_like(np.asarray(time, dtype=float))
        br = np.zeros_like(T)
        for mo in c.motors:
            t0 = getattr(self, "_t0", None)
            start = self._stage_starts[c.motors.index(mo)]
            b0, b1 = start + mo.delay, start + mo.delay + mo.burn_time
            on = (time >= b0) & (time < b1)
            T[on] = mo.thrust
            br[on] = mo.fuel_mass / mo.burn_time
        return T, br

    def _build_stage_starts(self):
        c = self.options["coeffs"]
        starts, t = [], 0.0
        for mo in c.motors:
            starts.append(t)
            t += mo.delay + mo.burn_time
        self._stage_starts = starts

    def compute(self, inputs, outputs):
        c: Coeffs = self.options["coeffs"]
        cl = getattr(self, "_cl_max", None)
        if cl is None:
            cl = self._cl_max = float(max(c.lift_curve.evaluate(a)
                                          for a in np.linspace(0.0, 0.8, 161)))
        self._build_stage_starts()
        x, h, V, gam = (inputs[k] for k in ("x", "h", "V", "gamma"))
        m, al, time = inputs["mass"], inputs["alpha"], inputs["time"]
        T, br = self._motor(time)
        V = np.maximum(V, 1.0)
        m = np.maximum(m, 1.0)

        rho = np.array([c.air_density(hh) for hh in h])
        qbar = 0.5 * rho * V * V
        aoa = np.abs(al)
        Cd = np.array([c.drag_curve.evaluate(a) for a in aoa])
        Cl = np.array([c.lift_curve.evaluate(a) for a in aoa])

        D = Cd * qbar * c.fin_area
        L = Cl * qbar * c.fin_area
        if c.supersonic_drag > 0.0:
            a_snd = np.array([c.speed_of_sound(hh) for hh in h])
            mult = np.ones_like(V)
            hi = V > 1.1 * a_snd
            mid = (~hi) & (V > 0.9 * a_snd)
            mult[hi] = 1.0 + c.supersonic_drag
            if np.any(mid):
                n6 = np.minimum(np.abs(a_snd[mid] - V[mid]) / a_snd[mid], 0.1)
                mult[mid] = 1.0 + ((0.1 - n6) / 0.1) ** 3 * \
                    (c.supersonic_drag + 0.15)
            D = D * mult

        ca, sa = np.cos(al), np.sin(al)
        cg, sg = np.cos(gam), np.sin(gam)

        outputs["xdot"] = V * cg
        outputs["hdot"] = V * sg
        outputs["Vdot"] = (T * ca - D) / m - G * sg
        n_lat = (T * sa + L) / (m * V)
        outputs["gammadot"] = n_lat - G * cg / V
        outputs["nload"] = n_lat * V / G
        outputs["qbar"] = qbar
        outputs["alpha_rate"] = inputs["alphadot"]
        outputs["navail"] = cl * qbar * c.fin_area / (m * G)
        outputs["massdot"] = -br


def thrust_profile(coeffs: Coeffs, t_guess: float, n: int):
    """Constant-thrust-per-stage table sampled over [0, t_guess]."""
    total_burn = sum(m.burn_time + m.delay for m in coeffs.motors)
    ts, T, br = [], [], []
    t = 0.0
    for mo in coeffs.motors:
        ts += [t, t + mo.delay]
        T += [0.0, 0.0]
        br += [0.0, 0.0]
        t += mo.delay
        ts += [t, t + mo.burn_time]
        T += [mo.thrust, mo.thrust]
        br += [mo.fuel_mass / mo.burn_time] * 2
        t += mo.burn_time
    ts += [max(t, total_burn), t_guess * 1.5]
    T += [0.0, 0.0]
    br += [0.0, 0.0]
    tt = np.linspace(0.0, t_guess * 1.2, n)
    return tt, np.interp(tt, ts, T), np.interp(tt, ts, br)


def _clipped(v, lo, hi=None):
    """`v` with every element inside [lo, hi]; None means no bound on that side."""
    v = np.asarray(v, dtype=float)
    if lo is not None:
        v = np.maximum(v, lo)
    if hi is not None:
        v = np.minimum(v, hi)
    return v


def _clamp_start_values(p, bounds, verbose=False) -> int:
    """Put every bounded phase variable back inside its bounds, after the start values have been set."""
    moved, worst, worst_path = 0, 0.0, ""
    for nm, (lo, hi) in bounds.items():
        for group in ("states", "controls"):
            path = f"traj.mid.{group}:{nm}"
            try:
                v = np.asarray(p.get_val(path), dtype=float)
            except (KeyError, RuntimeError):
                continue
            w = v
            if lo is not None:
                w = np.maximum(w, lo)
            if hi is not None:
                w = np.minimum(w, hi)
            if w.shape != v.shape:
                continue
            d = np.zeros_like(v)
            if lo is not None:
                d = np.maximum(d, lo - v)
            if hi is not None:
                d = np.maximum(d, v - hi)
            if d.size and d.max() > worst:
                worst, worst_path = float(d.max()), path
            if not np.array_equal(w, v):
                moved += int(np.count_nonzero(w != v))
                p.set_val(path, w)
    if worst > 1e-9:
        print(f"  start values: {moved} value(s) were outside their bounds "
              f"(by up to {worst:.3g} on {worst_path}); clamped")
    elif moved and verbose:
        print(f"  [start values: {moved} value(s) clamped onto their bound "
              f"(rounding, up to {worst:.1g})]")
    return moved


def _cold_start(ph, Xf, h0, hf, V0, m0, c, t_guess, alpha_state=False):
    ph.set_time_val(initial=0.0, duration=t_guess)
    ph.set_state_val("x", [0.0, Xf])
    ph.set_state_val("h", [h0, hf])
    ph.set_state_val("V", [V0, V0 * 0.6])
    ph.set_state_val("gamma", [0.0, 0.0])
    ph.set_state_val("mass", [m0, c.mass_dry])
    if alpha_state:
        ph.set_state_val("alpha", 0.0)
        ph.set_control_val("alphadot", 0.0)
    else:
        ph.set_control_val("alpha", 0.0)


def _audit(sol, c, min_authority_g=0.0, min_speed=0.0):
    V = np.asarray(sol["V"], dtype=float)
    h = np.asarray(sol["h"], dtype=float)
    m = np.asarray(sol["mass"], dtype=float)
    t = np.asarray(sol["t"], dtype=float)
    viol: dict = {}
    facts: dict = {}
    if not V.size:
        return viol, facts
    facts["v_min_ms"] = float(np.min(V))
    facts["v_min_t_s"] = float(t[int(np.argmin(V))])
    cl_max = float(max(c.lift_curve.evaluate(a)
                       for a in np.linspace(0.0, 0.8, 161)))
    rho = np.array([c.air_density(hh) for hh in h])
    nav = cl_max * 0.5 * rho * V * V * c.fin_area / (np.maximum(m, 1.0) * G)
    facts["nav_min_g"] = float(np.min(nav))
    if min_speed and float(min_speed) > 0.0:
        i = int(np.argmin(V))
        if V[i] < float(min_speed) - 1e-6:
            viol["V"] = f"min V {V[i]:.0f} m/s < {float(min_speed):.0f} at t={t[i]:.0f}s"
    if min_authority_g and float(min_authority_g) > 0.0:
        i = int(np.argmin(nav))
        if nav[i] < float(min_authority_g) - 1e-3:
            viol["navail"] = (f"navail {nav[i]:.2f} g < {float(min_authority_g):.2f} g "
                              f"at h={h[i]/1000:.1f} km, V={V[i]:.0f} m/s")
    return viol, facts


@contextlib.contextmanager
def _in_workdir():
    workdir = _isolate_workdir()
    cwd = os.getcwd()
    try:
        os.chdir(workdir)
        yield
    finally:
        os.chdir(cwd)


def _run_solver(p, ph, refine_iterations, refine_tol, verbose):
    if not refine_iterations or refine_iterations <= 0:
        if verbose:
            p.run_driver()
        else:
            _solver_stdout = io.StringIO()
            try:
                with contextlib.redirect_stdout(_solver_stdout):
                    p.run_driver()
            except Exception as exc:
                _tail = _solver_stdout.getvalue().strip()
                if _tail and hasattr(exc, "add_note"):
                    exc.add_note(
                        "output captured while solving:\n" + _tail[-1000:])
                raise
        return
    ph.set_refine_options(refine=True, tol=float(refine_tol))
    with _in_workdir():
        if verbose:
            dm.run_problem(p, refine_method="hp",
                           refine_iteration_limit=int(refine_iterations),
                           run_driver=True, simulate=False)
        else:
            _solver_stdout = io.StringIO()
            try:
                with contextlib.redirect_stdout(_solver_stdout):
                    dm.run_problem(p, refine_method="hp",
                                   refine_iteration_limit=int(
                                       refine_iterations),
                                   run_driver=True, simulate=False)
            except Exception as exc:
                _tail = _solver_stdout.getvalue().strip()
                if _tail and hasattr(exc, "add_note"):
                    exc.add_note(
                        "output captured while solving:\n" + _tail[-1000:])
                raise


def _refine_metrics(p):
    """max relative error and segments still needing refinement after the run."""
    from dymos.grid_refinement.error_estimation import check_error
    from dymos.load_case import find_phases
    try:
        res = check_error(find_phases(p.model))
    except Exception:
        return None, None
    worst, need = 0.0, 0
    for info in res.values():
        errs = np.asarray(info.get("max_rel_error", []), dtype=float)
        if errs.size:
            worst = max(worst, float(np.max(errs)))
        need += int(np.count_nonzero(info.get("need_refinement", [])))
    return worst, need


def solve(coeffs_path, range_km, launch_alt_m, target_alt_m, launch_speed_ms,
          nodes=40, verbose=True, guess=None, min_authority_g=3.0,
          max_dive_deg=0.0, maxiter=250, refine_iterations=0, refine_tol=1e-4,
          min_speed=None, max_q_kpa=None, alpha_rate_dps=None,
          range_guess_km=None):
    _isolate_workdir()
    c = Coeffs.load(coeffs_path)
    free_range = range_km is None
    Xf = (float(range_guess_km if range_guess_km else 150.0) * 1000.0
          if free_range else range_km * 1000.0)
    h0 = launch_alt_m
    hf = target_alt_m
    V0 = launch_speed_ms
    m0 = c.mass_dry + sum(m.fuel_mass for m in c.motors)
    alpha_state = bool(alpha_rate_dps and alpha_rate_dps > 0.0)

    _cl_max = float(max(c.lift_curve.evaluate(a)
                        for a in np.linspace(0.0, 0.8, 161)))
    auth0 = (_cl_max * 0.5 * c.air_density(h0)
             * V0 * V0 * c.fin_area / (m0 * G))
    auth_floor = min(float(min_authority_g), auth0) if min_authority_g else 0.0
    speed_floor = min(float(min_speed), float(V0)) if min_speed else 0.0

    t_guess = Xf / max(V0 + 400.0, 100.0)

    t_bounds = (t_guess * 0.5, t_guess * 3.0)
    bounds = {
        "x": (0.0, None),
        "h": (0.0, None),
        "V": (50.0, None),
        "gamma": (-1.2, 1.2),
        "mass": (c.mass_dry * 0.95, None),
        "alpha": (-0.35, 0.35),
    }
    if alpha_state:
        bounds["alphadot"] = (-math.radians(float(alpha_rate_dps)),
                              math.radians(float(alpha_rate_dps)))

    p = om.Problem(model=om.Group())
    traj = dm.Trajectory()
    ph = dm.Phase(ode_class=MissileODE,
                  transcription=dm.Radau(num_segments=nodes, order=3),
                  ode_init_kwargs={"coeffs": c, "verbose": verbose})
    traj.add_phase("mid", ph)
    p.model.add_subsystem("traj", traj)

    ph.set_time_options(fix_initial=True, units="s", duration_bounds=t_bounds)
    ph.add_state("x", fix_initial=True, fix_final=not free_range, rate_source="xdot",
                 ref=Xf, lower=bounds["x"][0], units="m")
    ph.add_state("h", fix_initial=True, fix_final=True, rate_source="hdot",
                 ref=max(h0, 1000.0), lower=bounds["h"][0], units="m")
    ph.add_state("V", fix_initial=True, fix_final=False, rate_source="Vdot",
                 ref=1000.0, lower=bounds["V"][0], units="m/s")
    ph.add_state("gamma", fix_initial=True, fix_final=False, rate_source="gammadot",
                 ref=0.1, lower=bounds["gamma"][0], upper=bounds["gamma"][1],
                 units="rad")
    ph.add_state("mass", fix_initial=True, fix_final=False, rate_source="massdot",
                 ref=m0, lower=bounds["mass"][0], units="kg")

    if alpha_state:
        ph.add_state("alpha", fix_initial=True, fix_final=False, rate_source="alpha_rate",
                     ref=0.1, lower=bounds["alpha"][0], upper=bounds["alpha"][1],
                     units="rad")
        ph.add_control("alphadot", opt=True,
                       lower=bounds["alphadot"][0], upper=bounds["alphadot"][1],
                       ref=0.01, units="rad/s")
    else:
        ph.add_control("alpha", opt=True, lower=bounds["alpha"][0],
                       upper=bounds["alpha"][1], ref=0.1, units="rad")

    ph.add_path_constraint("nload", lower=-c.g_limit, upper=c.g_limit)

    if auth_floor > 0.0:
        ph.add_path_constraint("navail", lower=auth_floor)
    if speed_floor > 0.0:
        ph.add_path_constraint("V", lower=speed_floor)
    if max_q_kpa and max_q_kpa > 0.0:
        ph.add_path_constraint("qbar", upper=float(max_q_kpa) * 1000.0)
    # if max_dive_deg and max_dive_deg > 0.0:
    #     ph.add_path_constraint("gamma", lower=-math.radians(max_dive_deg))
    ph.add_boundary_constraint(
        "V", loc="final", lower=c.self_destruct_at_speed)

    if free_range:
        ph.add_objective("x", loc="final", scaler=-1.0 / Xf)
    else:
        ph.add_objective("V", loc="final", scaler=-1.0 / 1000.0)

    p.driver = om.ScipyOptimizeDriver(
        optimizer="SLSQP", tol=1e-6, maxiter=int(maxiter), disp=verbose)
    p.setup()

    ph.set_time_val(initial=0.0, duration=t_guess)
    if guess is not None:
        applied = False
        try:
            tof = float(guess["tof"])
            gx = np.asarray(guess["x"], dtype=float).ravel()
            scale = Xf / max(float(gx[-1]), 1.0)

            duration = float(np.clip(tof * scale, *t_bounds))

            def tv(a):
                return np.linspace(0.0, duration, len(np.asarray(a).ravel()))
            ph.set_time_val(initial=0.0, duration=duration)
            ph.set_state_val("x", _clipped(
                gx * scale, *bounds["x"]), time_vals=tv(gx))
            for nm in ("h", "V", "gamma", "mass"):
                v = _clipped(np.asarray(
                    guess[nm], dtype=float).ravel(), *bounds[nm])
                ph.set_state_val(nm, v, time_vals=tv(v))
            av = _clipped(np.asarray(guess["alpha"], dtype=float).ravel(),
                          *bounds["alpha"])
            if alpha_state:
                ph.set_state_val("alpha", av, time_vals=tv(av))
                ph.set_control_val(
                    "alphadot", np.zeros_like(av), time_vals=tv(av))
            else:
                ph.set_control_val("alpha", av, time_vals=tv(av))
            applied = True
        except Exception as e:
            if verbose:
                print(f"  [warm start rejected: {type(e).__name__}: {e}]")
        if not applied:
            _cold_start(ph, Xf, h0, hf, V0, m0, c, t_guess, alpha_state)
    else:
        _cold_start(ph, Xf, h0, hf, V0, m0, c, t_guess, alpha_state)
    _clamp_start_values(p, bounds, verbose)
    _run_solver(p, ph, refine_iterations, refine_tol, verbose)

    refine_error, refine_segments = (None, None)
    if refine_iterations and refine_iterations > 0:
        refine_error, refine_segments = _refine_metrics(p)

    x = np.asarray(p.get_val("traj.mid.states:x")).ravel()
    h = np.asarray(p.get_val("traj.mid.states:h")).ravel()
    V = np.asarray(p.get_val("traj.mid.states:V")).ravel()

    t = None
    for cand in ("traj.phases.mid.t", "traj.mid.t", "traj.phases.mid.time",
                 "traj.mid.time"):
        try:
            t = np.asarray(p.get_val(cand)).ravel()
            break
        except (KeyError, Exception):
            continue
    if t is None:
        t = np.linspace(0.0, float(p.get_val("traj.phases.mid.t_duration")[-1])
                        if True else 0.0, len(x))
    res = p.driver.result
    gamma = np.asarray(p.get_val("traj.mid.states:gamma")).ravel()
    mass = np.asarray(p.get_val("traj.mid.states:mass")).ravel()
    alpha = np.asarray(p.get_val("traj.mid.controls:alpha") if not alpha_state
                       else "traj.mid.states:alpha").ravel()
    sol = dict(t=t, x=x, h=h, V=V, gamma=gamma, mass=mass, alpha=alpha,
               Xf=Xf, h0=h0, hf=hf, free_range=free_range,
               range_m=float(x[-1]),
               V_term=float(V[-1]), tof=float(t[-1]),
               apex=float(np.max(h)), apex_x=float(x[int(np.argmax(h))]),
               refine_error=refine_error, refine_segments=refine_segments,
               authority_floor=float(auth_floor), speed_floor=float(speed_floor),
               converged=bool(res.success),
               msg=str(getattr(res, "message", getattr(res, "msg", "n/a"))))

    sol["t_bounds"] = [float(t_bounds[0]), float(t_bounds[1])]
    dur = float(t[-1])
    sol["t_at_bound"] = (
        "lower" if dur <= t_bounds[0] * (1.0 + 1e-3) else
        "upper" if dur >= t_bounds[1] * (1.0 - 1e-3) else None)

    viol, facts = _audit(
        sol, c, min_authority_g=auth_floor, min_speed=speed_floor)
    sol["violations"] = viol
    sol.update(facts)
    sol["feasible"] = not viol
    if viol:
        sol["msg"] = "; ".join(f"{k}: {v}" for k, v in viol.items())
    return sol


def fit_polynomial(sol, order=7):
    """Fit h(x) to an order-`order` polynomial in xi = x/Xf"""
    xi = np.clip(sol["x"] / sol["Xf"], 0.0, 1.0)
    h = sol["h"]
    h0, hf = sol["h0"], sol["hf"]
    K = order - 1
    A = np.vstack([(xi * (1 - xi) * xi ** k) for k in range(K)]).T
    rhs = h - (h0 + (hf - h0) * xi)
    coef, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    a = np.zeros(order + 1)
    a[0] = h0
    a[1] = hf - h0
    for k, ck in enumerate(coef):
        a[k + 1] += ck
        a[k + 2] -= ck
    fit = np.polyval(a[::-1], xi)
    rmse = float(np.sqrt(np.mean((fit - h) ** 2)))
    return [float(v) for v in a], rmse


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--coeffs", default=str(Path(__file__).parent / "coeffs_aam2.json"))
    ap.add_argument("--range-km", type=float, default=74.0)
    ap.add_argument("--launch-alt", type=float, default=13716.0, help="metres")
    ap.add_argument("--target-alt", type=float, default=457.0, help="metres")
    ap.add_argument("--launch-speed", type=float, default=350.0, help="m/s")
    ap.add_argument("--nodes", type=int, default=40)
    ap.add_argument("--min-authority", type=float, default=0.1,
                    help="minimum available load factor (g) along the trajectory; "
                         "0 disables. Grid runs take this from the grid spec's "
                         "min_authority_g, which is where the value belongs")
    ap.add_argument("--max-dive", type=float, default=0.0,
                    help="steepest allowed descent angle (deg below horizontal); "
                         "0 = unconstrained (default)")
    ap.add_argument("--refine-iterations", type=int, default=0,
                    help="hp-adaptive grid refinement passes; 0 = off (default). "
                         "Grid runs set it in the grid spec (refine_iterations) "
                         "so an overnight run is configured by its spec")
    ap.add_argument("--refine-tol", type=float, default=1e-4,
                    help="refinement tolerance on the segment error estimate")
    ap.add_argument("--min-speed", type=float, default=None,
                    help="path floor on speed (m/s), off unless given; the game "
                         "rule is a path rule (missile_sim.py:420) but this "
                         "problem only ever constrained the final speed")
    ap.add_argument("--max-q", type=float, default=None,
                    help="kPa; OFF by default, see solve()")
    ap.add_argument("--alpha-rate", type=float, default=None,
                    help="deg/s; bounds the AoA rate by promoting alpha to a state")
    ap.add_argument("--free-range", action="store_true",
                    help="solve for maximum range instead of maximum terminal speed")
    ap.add_argument("--nodes-free", dest="nodes_free", type=float, default=150.0,
                    help="range guess (km) used for the mesh/guess in --free-range")
    ap.add_argument("--maxiter", type=int, default=250)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    s = solve(a.coeffs, None if a.free_range else a.range_km,
              a.launch_alt, a.target_alt, a.launch_speed,
              a.nodes, min_authority_g=a.min_authority, max_dive_deg=a.max_dive,
              refine_iterations=a.refine_iterations, refine_tol=a.refine_tol,
              min_speed=a.min_speed, max_q_kpa=a.max_q, alpha_rate_dps=a.alpha_rate,
              range_guess_km=a.nodes_free, maxiter=a.maxiter)
    what = "max range" if a.free_range else f"{a.range_km:.0f} km"
    print(f"\n=== Dymos solution, {what} ===")
    if a.free_range:
        print(f"  max range      : {s['range_m']/1000:.1f} km")
    print(
        f"  driver         : {'CONVERGED' if s['converged'] else 'NOT CONVERGED -> ' + s['msg']}")
    print(
        f"  terminal speed : {s['V_term']:8.1f} m/s = {s['V_term']/340:.2f} M")
    print(f"  TOF            : {s['tof']:8.1f} s")
    print(
        f"  apex           : {s['apex']:8.0f} m at {s['apex_x']/1000:.1f} km")
    poly, rmse = fit_polynomial(s)
    print(f"  poly a0..a7    : {[round(v, 1) for v in poly]}")
    print(f"  fit RMSE       : {rmse:.1f} m")
    print(f"  floors         : navail >= {s.get('authority_floor', 0):.2f} g, "
          f"V >= {s.get('speed_floor', 0):.0f} m/s")
    print(f"  constraints    : " + ("OK" if s.get("feasible") else
          "VIOLATED -- " + s.get("msg", "")))
    if s.get("refine_error") is not None:
        print(f"  refine error   : {s['refine_error']:.2e} "
              f"({s['refine_segments']} segment(s) still wanting refinement)")
    if a.out:
        Path(a.out).write_text(json.dumps({"poly_order": 7, "poly_a": poly,
                                           "apex_alt_m": s["apex"],
                                           "terminal_speed_ms": s["V_term"],
                                           "tof_s": s["tof"],
                                           "converged": s["converged"],
                                           "fit_rmse_m": rmse}, indent=2))
        print(f"  wrote {a.out}")
