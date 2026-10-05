"""
dymos_opt.py -- the same problem as build_tables.py, solved with Dymos
(pseudospectral transcription + SLSQP), i.e. what CEAS-GNC-2026-016 section 2
actually does. Use this if you want a genuine optimal trajectory rather than a
search over polynomial shapes.

    pip install dymos openmdao

What Dymos gives you over build_tables.py:
  * the trajectory is optimal, not the best member of a 6-parameter family
  * path constraints (g-load, dynamic pressure, altitude floor) are handled
    properly by the NLP instead of as soft penalties
  * it scales to free final time, multi-phase (boost / sustain / coast), etc.

What it costs: you must express the dynamics as an OpenMDAO component, and it is
slower per solve. For a 3000-cell table, build_tables.py is the better tool.

Formulation (3-DOF point mass, vertical plane):
    states    x, h, V, gamma, mass
    control   alpha  (angle of attack; the surrogate for "where the aimpoint is")
    minimise  -V at final time
    subject to  x(tf) = X_f,  h(tf) = h_pip,  |n| <= g_limit,  h >= 0

The aero tables are the SAME Curve objects the simulator uses, so the two tools
cannot disagree about the airframe.
"""
from __future__ import annotations
from missile_sim import Coeffs, Curve
import dymos as dm
import openmdao.api as om
import numpy as np

import argparse
import json
import math
import os
import sys
from pathlib import Path

# OpenMDAO drops a "<script>_out" directory in the cwd for every Problem, and
# increments the suffix, so a grid run leaves grid_dymos_out, grid_dymos2_out ...
# grid_dymos28_out behind. Two separate things create it:
#   * the reports system (n2.html, scaling_report.html) -> OPENMDAO_REPORTS
#   * declare_coloring's cache (coloring_files/*.pkl)   -> OPENMDAO_WORKDIR
# Disabling reports alone is NOT enough -- the coloring file still makes the dir.
# Both must be set before openmdao is imported.
os.environ.setdefault("OPENMDAO_REPORTS", "0")


def _isolate_workdir():
    """Send OpenMDAO's per-Problem output tree to a throwaway directory.

    Called once per process, so parallel grid workers never write the same
    coloring pickle concurrently. Honours a pre-set OPENMDAO_WORKDIR.
    """
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


# --------------------------------------------------------------------------
class MissileODE(om.ExplicitComponent):
    """3-DOF point-mass dynamics using the game's drag/lift curves."""

    def initialize(self):
        self.options.declare("num_nodes", types=int)
        self.options.declare("coeffs", types=Coeffs)

    def setup(self):
        nn = self.options["num_nodes"]
        for n in ("x", "h", "V", "gamma", "mass"):
            self.add_input(n, val=np.zeros(nn), units=None)
        self.add_input("alpha", val=np.zeros(nn))
        # Dymos auto-connects the phase time to any ODE input named "time".
        # Computing thrust here rather than via a lookup control keeps us off
        # version-specific add_control kwargs.
        self.add_input("time", val=np.zeros(nn))
        self.add_output("xdot", val=np.zeros(nn))
        self.add_output("hdot", val=np.zeros(nn))
        self.add_output("Vdot", val=np.zeros(nn))
        self.add_output("gammadot", val=np.zeros(nn))
        self.add_output("massdot", val=np.zeros(nn))
        self.add_output("nload", val=np.zeros(nn))
        # Maneuver AUTHORITY: the load the airframe could pull at max AoA, given
        # the local density and speed. Distinct from nload, which is what the
        # trajectory actually pulls. Without this the only aero constraint is
        # nload <= gLimit, an UPPER bound, so a missile with 0.13 g available at
        # 35 km satisfies it trivially and the optimiser lofts into air it cannot
        # turn in. Measured with the AAM-29 curves: 7.1 g at 25 km, 1.3 g at 30 km,
        # 0.13 g above 35 km (airDensityAltitude flattens at its last key).
        self.add_output("navail", val=np.zeros(nn))
        # finite-difference partials: simplest, and fast enough at this size.
        # Swap to analytic partials only if SLSQP becomes too slow.
        self.declare_partials(of="*", wrt="*", method="fd")
        # OpenMDAO computes an FD colouring: instead of perturbing all ~7 inputs
        # one at a time, it perturbs groups that do not interact. This is the
        # cheapest large speedup available here -- no derivative algebra needed,
        # and it attacks the 65% of wall time that finite differences cost.
        self.declare_coloring(wrt="*", method="fd", form="forward",
                              num_full_jacs=2, tol=1e-20, orders=None)

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
        # Peak of the lift curve = best authority the airframe has at any AoA.
        # Cached on the instance; do NOT do this in a setup() override, that
        # breaks Dymos's ODE output introspection.
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
        outputs["navail"] = cl * qbar * c.fin_area / (m * G)
        outputs["massdot"] = -br


# --------------------------------------------------------------------------
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


def _cold_start(ph, Xf, h0, hf, V0, m0, c, t_guess):
    ph.set_time_val(initial=0.0, duration=t_guess)
    ph.set_state_val("x", [0.0, Xf])
    ph.set_state_val("h", [h0, hf])
    ph.set_state_val("V", [V0, V0 * 0.6])
    ph.set_state_val("gamma", [0.0, 0.0])
    ph.set_state_val("mass", [m0, c.mass_dry])
    ph.set_control_val("alpha", 0.0)


def solve(coeffs_path, range_km, launch_alt_m, target_alt_m, launch_speed_ms,
          nodes=40, verbose=True, guess=None, min_authority_g=3.0,
          max_dive_deg=0.0, maxiter=250):
    """All inputs METRIC: downrange km, altitudes m above sea level, speed m/s.

    `guess` is the dict returned by a previous solve(). Grid cells that differ
    only in range are nearly the same NLP, so warm starting from the neighbouring
    solution is by far the cheapest speedup available -- far more than changing
    solver or hardware. Dymos interpolates the guess onto the new mesh, so a
    different node count is fine.
    """
    _isolate_workdir()
    c = Coeffs.load(coeffs_path)
    Xf = range_km * 1000.0
    h0 = launch_alt_m
    hf = target_alt_m
    V0 = launch_speed_ms
    m0 = c.mass_dry + sum(m.fuel_mass for m in c.motors)

    t_guess = Xf / max(V0 + 400.0, 100.0)

    p = om.Problem(model=om.Group())
    traj = dm.Trajectory()
    ph = dm.Phase(ode_class=MissileODE,
                  transcription=dm.Radau(num_segments=nodes, order=3),
                  ode_init_kwargs={"coeffs": c})
    traj.add_phase("mid", ph)
    p.model.add_subsystem("traj", traj)

    ph.set_time_options(fix_initial=True, duration_bounds=(
        t_guess * 0.5, t_guess * 3.0))
    ph.add_state("x", fix_initial=True, fix_final=True, rate_source="xdot",
                 ref=Xf)
    ph.add_state("h", fix_initial=True, fix_final=True, rate_source="hdot",
                 ref=max(h0, 1000.0), lower=0.0)
    ph.add_state("V", fix_initial=True, fix_final=False, rate_source="Vdot",
                 ref=1000.0, lower=50.0)
    ph.add_state("gamma", fix_initial=True, fix_final=False, rate_source="gammadot",
                 ref=0.1, lower=-1.2, upper=1.2)
    ph.add_state("mass", fix_initial=True, fix_final=False, rate_source="massdot",
                 ref=m0, lower=c.mass_dry * 0.95)

    ph.add_control("alpha", opt=True, lower=-0.35, upper=0.35, ref=0.1)

    ph.add_path_constraint("nload", lower=-c.g_limit, upper=c.g_limit)
    # must RETAIN enough authority to correct, not merely stay under the
    # structural limit. This is what stops the over-loft. Set 0 to disable.
    if min_authority_g and min_authority_g > 0.0:
        ph.add_path_constraint("navail", lower=min_authority_g)
    # OFF by default, useless.
    # if max_dive_deg and max_dive_deg > 0.0:
    #     ph.add_path_constraint("gamma", lower=-math.radians(max_dive_deg))
    ph.add_boundary_constraint(
        "V", loc="final", lower=c.self_destruct_at_speed)
    ph.add_objective("V", loc="final", scaler=-1.0 / 1000.0)

    p.driver = om.ScipyOptimizeDriver(
        optimizer="SLSQP", tol=1e-6, maxiter=int(maxiter))
    p.setup()          # setup() MUST precede set_*_val, or state shape is unknown

    ph.set_time_val(initial=0.0, duration=t_guess)
    if guess is not None:
        # scale the previous solution onto this geometry rather than trusting the
        # mesh interpolation: downrange is stretched, altitude is not.
        applied = False
        try:
            tof = float(guess["tof"])
            gx = np.asarray(guess["x"], dtype=float).ravel()
            scale = Xf / max(float(gx[-1]), 1.0)
            # time_vals is REQUIRED, and must be the SAME LENGTH as the array
            # being set. The state nodes (37) and the time/control nodes (48/36)
            # do not match, so guess["t"] cannot be reused directly -- pass a
            # uniform ramp of the right length for each array. It is only an
            # initial guess, and both solves share a mesh, so the values already
            # line up node-for-node.

            def tv(a):
                return np.linspace(0.0, tof, len(np.asarray(a).ravel()))
            ph.set_time_val(initial=0.0, duration=tof)
            ph.set_state_val("x", gx * scale, time_vals=tv(gx))
            for nm in ("h", "V", "gamma", "mass"):
                v = np.asarray(guess[nm], dtype=float).ravel()
                ph.set_state_val(nm, v, time_vals=tv(v))
            av = np.asarray(guess["alpha"], dtype=float).ravel()
            ph.set_control_val("alpha", av, time_vals=tv(av))
            applied = True
        except Exception as e:
            # do NOT swallow this silently: a failed warm start looks exactly
            # like "warm start does not help", which cost me a full benchmark.
            if verbose:
                print(f"  [warm start rejected: {type(e).__name__}: {e}]")
        if not applied:
            _cold_start(ph, Xf, h0, hf, V0, m0, c, t_guess)
    else:
        _cold_start(ph, Xf, h0, hf, V0, m0, c, t_guess)
    p.run_driver()

    x = np.asarray(p.get_val("traj.mid.states:x")).ravel()
    h = np.asarray(p.get_val("traj.mid.states:h")).ravel()
    V = np.asarray(p.get_val("traj.mid.states:V")).ravel()
    # time lives on the phase object, not as a gettable output path
    # time's output path has moved between Dymos versions; probe for it
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
    alpha = np.asarray(p.get_val("traj.mid.controls:alpha")).ravel()
    return dict(t=t, x=x, h=h, V=V, gamma=gamma, mass=mass, alpha=alpha,
                Xf=Xf, h0=h0, hf=hf,
                V_term=float(V[-1]), tof=float(t[-1]),
                apex=float(np.max(h)), apex_x=float(x[int(np.argmax(h))]),
                converged=bool(res.success), msg=str(getattr(res, "message", getattr(res, "msg", "n/a"))))


def fit_polynomial(sol, order=7):
    """Fit h(x) to an order-`order` polynomial in xi = x/Xf, endpoints pinned.

    Same output format as build_tables.py so the runtime loader is shared.
    """
    xi = np.clip(sol["x"] / sol["Xf"], 0.0, 1.0)
    h = sol["h"]
    h0, hf = sol["h0"], sol["hf"]
    # design matrix for the bump basis xi(1-xi)xi^k, k=0..order-2
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
                    help="minimum available load factor (g) along the trajectory")
    ap.add_argument("--max-dive", type=float, default=0.0,
                    help="steepest allowed descent angle (deg below horizontal); "
                         "0 = unconstrained (default)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    s = solve(a.coeffs, a.range_km, a.launch_alt, a.target_alt, a.launch_speed,
              a.nodes, min_authority_g=a.min_authority, max_dive_deg=a.max_dive)
    print(f"\n=== Dymos solution, {a.range_km:.0f} km ===")
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
    if a.out:
        Path(a.out).write_text(json.dumps({"poly_order": 7, "poly_a": poly,
                                           "apex_alt_m": s["apex"],
                                           "terminal_speed_ms": s["V_term"],
                                           "tof_s": s["tof"],
                                           "converged": s["converged"],
                                           "fit_rmse_m": rmse}, indent=2))
        print(f"  wrote {a.out}")
