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

# ---------------------------------------------------------------------------
# Two of the warnings this script triggers are raised by dymos/OpenMDAO itself,
# not by this model or by anything else in this repository, so neither can be
# made to go away by fixing something here. Each is filtered at the narrowest
# scope that hides it and nothing else; every other warning still prints. Both
# filters go in before `import dymos`, because (1) is raised during that import.
#
# 1. OMDeprecationWarning: "The INF_BOUND sentinel in OpenMDAO is deprecated."
#    It is raised while `import dymos` executes
#        from openmdao.core.constants import INF_BOUND
#    (dymos/trajectory/trajectory.py:12). openmdao >= 3.45 deprecated the
#    sentinel; dymos 1.15.1 is the current release, still imports it, and only
#    requires openmdao >= 3.39 -- so no version pair avoids it and no pin helps.
#    The fix is one import statement, upstream, in dymos. Until then this is one
#    line of noise per process, i.e. one per grid worker.
#
# 2. DerivativesWarning: "the following constraints or objectives cannot be
#    impacted by the design variables ... traj.mid.navail[path], inds=[(0, 0)]".
#    That is the ONLY zero row of the constraint Jacobian, and it is zero by
#    construction: navail depends on h and V alone, and both are FIXED at the
#    launch node, so no design variable can affect the launch condition. The row
#    is structurally degenerate rather than a modelling mistake, and it cannot be
#    removed from the NLP either, because dymos 1.15.1 gives no way to apply a
#    path constraint to a subset of nodes: add_path_constraint() has no `nodes`
#    argument, and its `indices` argument indexes the constrained variable (size
#    1 here) rather than the node -- passing node indices raises
#    "IndexError: index 47 is out of bounds for source dimension of size 1".
#
#    From 0011 there are two such rows, not one: the V >= self-destruct speed
#    floor and the authority floor are both path constraints, and both are
#    evaluated at the launch node, where every state is fixed. Same degeneration,
#    same reason, so the same filter covers the pair.
#
#    The regex below matches ONLY a message whose every entry is one of those two
#    launch rows -- note the closed set of row patterns and the `$`. If any other
#    row or column ever goes singular the message lists it too, does not match,
#    and prints in full -- which is what you want, because a constraint that
#    genuinely cannot be affected by anything is a real modelling bug.
# ---------------------------------------------------------------------------
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

import dymos as dm
import openmdao.api as om
import numpy as np

import argparse
import contextlib
import io
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

# Noise policy for the solver's own output: solve(verbose=...) decides what
# OpenMDAO and SLSQP may print. grid_dymos.py passes verbose=False, because a
# grid run emits ~17 lines per cell of sparsity report and exit summary that
# nobody reads and that buries the run's own per-cell line, and the CLI at the
# bottom of this file leaves it True, because a single solve is something you
# are watching.


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
        # Whether OpenMDAO prints the sparsity-colouring report for this
        # component: a fixed nine lines per solve, near-useless across 480 of
        # them, and genuinely useful when you are solving one cell by hand and
        # wondering where the time went.
        self.options.declare("verbose", types=bool, default=False)

    def setup(self):
        nn = self.options["num_nodes"]
        # Units on every input and every rate output. Every number below is
        # already SI, so nothing converts -- but dymos's collocation constraint
        # is a component whose inputs are the state RATES, in <state units>/s,
        # and it can only warn that it cannot check the connection while this
        # side is unitless:
        #
        #   UnitsWarning: Input '...collocation_constraint.f_computed:x' with
        #   units of '1/s' is connected to output '...rhs_all.xdot' which has no
        #   units.
        #
        # Six of those per solve (five states plus time). Declaring the units is
        # the fix; the warning was the symptom. nload and navail are load
        # factors -- dimensionless ratios -- and stay unitless on purpose.
        self.add_input("x", val=np.zeros(nn), units="m")
        self.add_input("h", val=np.zeros(nn), units="m")
        self.add_input("V", val=np.zeros(nn), units="m/s")
        self.add_input("gamma", val=np.zeros(nn), units="rad")
        self.add_input("mass", val=np.zeros(nn), units="kg")
        self.add_input("alpha", val=np.zeros(nn), units="rad")
        # Dymos auto-connects the phase time to any ODE input named "time".
        # Computing thrust here rather than via a lookup control keeps us off
        # version-specific add_control kwargs.
        self.add_input("time", val=np.zeros(nn), units="s")
        self.add_output("xdot", val=np.zeros(nn), units="m/s")
        self.add_output("hdot", val=np.zeros(nn), units="m/s")
        self.add_output("Vdot", val=np.zeros(nn), units="m/s**2")
        self.add_output("gammadot", val=np.zeros(nn), units="rad/s")
        self.add_output("massdot", val=np.zeros(nn), units="kg/s")
        self.add_output("nload", val=np.zeros(nn))
        # Maneuver AUTHORITY: the load the airframe could pull at max AoA, given
        # the local density and speed. Distinct from nload, which is what the
        # trajectory actually pulls. Without this the only aero constraint is
        # nload <= gLimit, an UPPER bound, so a missile with 0.13 g available at
        # 35 km satisfies it trivially and the optimiser lofts into air it cannot
        # turn in. Measured with the AAM-29 curves: 7.1 g at 25 km, 1.3 g at 30 km,
        # 0.13 g above 35 km (airDensityAltitude flattens at its last key).
        self.add_output("navail", val=np.zeros(nn))
        # Dynamic pressure, and the (optional) AoA rate channel. qbar is what
        # --max-q is written against. alpha_rate exists so that alpha can be
        # promoted from a control to a state when --alpha-rate is used: dymos
        # needs an ODE output to take a state's rate from, and the rate is then
        # the control. Both are inert unless the matching option is set.
        self.add_output("qbar", val=np.zeros(nn))
        self.add_input("alphadot", val=np.zeros(nn), units="rad/s")
        self.add_output("alpha_rate", val=np.zeros(nn), units="rad/s")
        # finite-difference partials: simplest, and fast enough at this size.
        # Swap to analytic partials only if SLSQP becomes too slow.
        self.declare_partials(of="*", wrt="*", method="fd")
        # OpenMDAO computes an FD colouring: instead of perturbing all ~7 inputs
        # one at a time, it perturbs groups that do not interact. This is the
        # cheapest large speedup available here -- no derivative algebra needed,
        # and it attacks the 65% of wall time that finite differences cost.
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
        outputs["qbar"] = qbar
        outputs["alpha_rate"] = inputs["alphadot"]
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


def _clipped(v, lo, hi=None):
    """`v` with every element inside [lo, hi]; None means "no bound that side".

    Used on an initial guess, which comes from a neighbouring cell and is therefore
    only approximately a point of this cell's NLP -- and sometimes not even inside
    the box the phase declares for it.
    """
    v = np.asarray(v, dtype=float)
    if lo is not None:
        v = np.maximum(v, lo)
    if hi is not None:
        v = np.minimum(v, hi)
    return v


def _clamp_start_values(p, bounds, verbose=False) -> int:
    """Put every bounded phase variable back inside its bounds, after the start
    values have been set. Returns how many values had to move.

    Start values are set as short profiles and dymos interpolates them onto the
    node grids. That round trip is not exact at the edges: an array that sits
    exactly on a bound comes back scattered around it by an ULP. Measured on a real solve, an alpha array of 0.35 against a 0.35 bound came
    back between 0.3499999999999999 and 0.35000000000000003 -- 5.6e-17 over at worst
    -- with between none and four of its eighteen values strictly above the bound,
    depending on where the rounding landed. The driver checks the initial values
    against their bounds and reports anything strictly outside, with no tolerance at
    all, so the cells that landed above printed, once per warm-started cell whose
    neighbour had saturated its angle of attack:

        DriverWarning:The following design variable initial conditions are out of
        their specified bounds:  traj.mid.controls:alpha
          val: [0.35 ...]   upper: [0.35 ...]

    The check is `>`, so a value sitting exactly ON the bound is legal. Clamping
    the interpolated values onto the bound is therefore the whole fix, and it is a
    fix rather than a suppression: `invalid_desvar_behavior` stays at its default,
    and a start value that is genuinely outside the bounds is still reported (see
    the tolerance below). Clamping a start value is free in any case: these are
    initial guesses, and the amount that moves is an ULP.
    """
    moved, worst, worst_path = 0, 0.0, ""
    for nm, (lo, hi) in bounds.items():
        for group in ("states", "controls"):
            path = f"traj.mid.{group}:{nm}"
            try:
                v = np.asarray(p.get_val(path), dtype=float)
            except (KeyError, RuntimeError):
                continue                      # the other group holds this name
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
        # Not a round trip: the start value really was outside the box, which is
        # worth saying -- a solution stored under different bounds, say.
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
    """The worst violation of the floors this solve was GIVEN, and the minima.

    SLSQP's success flag is False on almost every cell of these tables -- it stops
    on the iteration cap -- so the flag says nothing about whether the answer is
    usable. The state history does. This is the check that separates "the
    optimiser returned a number" from "the missile can fly that".

    The floors are the caller's, i.e. the grid spec's: an audit that invented a
    floor of its own would be reporting on a different problem from the one that
    was solved. Both arguments default to 0, meaning "no floor was asked for, so
    there is nothing to violate".

    Returns (violations, facts). The facts -- the trajectory's own minimum speed
    and minimum available load factor -- are measured either way, floor or no
    floor, so a floor can be chosen from what real trajectories do instead of from
    taste.
    """
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
        if nav[i] < float(min_authority_g) - 1e-3:   # sitting ON the floor is not a violation
            viol["navail"] = (f"navail {nav[i]:.2f} g < {float(min_authority_g):.2f} g "
                              f"at h={h[i]/1000:.1f} km, V={V[i]:.0f} m/s")
    return viol, facts


@contextlib.contextmanager
def _in_workdir():
    """Run the body from the isolated workdir, then come back.

    Dymos's refinement loop opens its log -- "grid_refinement.out" -- relative to
    the process CWD (grid_refinement/refinement.py), so a grid run with
    --refine-iterations would drop one file per worker into whatever directory
    the run was launched from, and every worker would fight over the same name.
    _isolate_workdir() already owns a throwaway directory for exactly this class
    of output; this is where the refinement log belongs.
    """
    workdir = _isolate_workdir()
    cwd = os.getcwd()
    try:
        os.chdir(workdir)
        yield
    finally:
        os.chdir(cwd)


def _run_solver(p, ph, refine_iterations, refine_tol, verbose):
    """Run the driver, refining the grid when asked to.

    The reference problem for this mod (CEAS-GNC-2026-016, see
    arena_docs/arena/kaist_clean.txt) is solved with GPOPS-II, and the reason its
    answers are trajectories is that its hp-adaptive refinement keeps subdividing
    until the discretisation error is under tolerance. Dymos ships the same loop,
    and it is wired up here -- but it is OFF by default, because on this problem
    it is brutal: two passes on a 12-segment 30 km cell took 470 s (from 8 s) and
    still ended at 1.9e-2 error with 48 segments wanting refinement, since every
    pass re-solves the NLP with finite-difference Jacobians. What is on by
    default instead is the cheap half of the same idea: the achieved refinement
    error is returned with every solution, and _solve_one re-integrates what the
    solver produced, so a solve that did not converge can be *labelled* rather
    than believed (see grid_dymos --check).
    """
    if not refine_iterations or refine_iterations <= 0:
        if verbose:
            p.run_driver()
        else:
            # OpenMDAO prints a three-line failure banner that `disp` does NOT
            # cover -- scipy_optimizer.py:569, where the option guards only the
            # success message. Every cell in the shipped tables stops on the SLSQP
            # iteration cap, so across a 480-cell grid that is ~1500 lines saying
            # the same thing the per-cell report already says. Capture it rather
            # than let it through; nothing is discarded, see the except clause.
            _solver_stdout = io.StringIO()
            try:
                with contextlib.redirect_stdout(_solver_stdout):
                    p.run_driver()
            except Exception as exc:
                # If the solve raised, this buffer may be the only place that says
                # why, so do not lose it.
                _tail = _solver_stdout.getvalue().strip()
                if _tail and hasattr(exc, "add_note"):
                    exc.add_note("output captured while solving:\n" + _tail[-1000:])
                raise
        return
    ph.set_refine_options(refine=True, tol=float(refine_tol))
    # The refinement tables go to stdout; that is solver narration, the same
    # thing 0006 stopped printing for the failure banner.
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
                                   refine_iteration_limit=int(refine_iterations),
                                   run_driver=True, simulate=False)
            except Exception as exc:
                _tail = _solver_stdout.getvalue().strip()
                if _tail and hasattr(exc, "add_note"):
                    exc.add_note("output captured while solving:\n" + _tail[-1000:])
                raise


def _refine_metrics(p):
    """(max relative error, segments still needing refinement) after the run.

    This is the number dymos's refinement loop prints while it works: the
    discretisation error it drives under its tolerance, or fails to. Recorded per
    cell so that a solve which ran out of refinement can never be read as a
    trajectory by accident.
    """
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
    """All inputs METRIC: downrange km, altitudes m above sea level, speed m/s.

    `guess` is the dict returned by a previous solve(). Grid cells that differ
    only in range are nearly the same NLP, so warm starting from the neighbouring
    solution is by far the cheapest speedup available -- far more than changing
    solver or hardware. Dymos interpolates the guess onto the new mesh, so a
    different node count is fine.

    `range_km=None` solves the free-final-range problem instead: maximum range
    that still arrives at the target altitude with V >= min_speed. That is the
    reachability question ("can this shot be made at all"), and it is what
    --envelope uses; the objective becomes the final downrange instead of the
    final speed.

    Two optional floors, both OFF unless the caller asks for them -- the grid spec
    is where they belong, and a spec that sets neither gets the problem this mod
    has always solved:

    * `min_authority_g`: floor on `navail`, the load the airframe could pull at
      Cl_max, i.e. the literature's dynamic-pressure floor ("the dynamic pressure
      cannot be too small", Li 2017) in this model's units. `grid_dymos.py` reads
      it from the spec key of the same name; the standalone default is 0.1 g.
    * `min_speed`: floor on speed along the PATH, not only at the terminal point.
      The game's rule is a path rule -- the missile is lost whenever V drops below
      `selfDestructAtSpeed` after t = 2 s (`sim/missile_sim.py:420`) -- while this
      problem has always constrained only the final speed. A spec that wants the
      game's rule modelled sets `min_speed_ms` (or `"self_destruct"`, meaning the
      missile's own value); unset, nothing changes.

    `alpha_rate_dps` promotes alpha to a state with its rate as the control, which
    is how the dual-pulse AAM problem in EuroGNC 2022-040 bounds control rate.
    """
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

    # Both floors are capped at what the LAUNCH STATE already provides, because
    # the initial state is fixed: a cell launched at 180 m/s from 9 km has 2.39 g
    # of authority and 180 m/s of speed at t=0, so an uncapped 3 g / 200 m/s
    # would make the problem infeasible at the first node and SLSQP would thrash
    # against something the optimiser cannot change. Capped, the constraint still
    # says the useful thing: do not fly anywhere with less control or less speed
    # than the missile was born with.
    _cl_max = float(max(c.lift_curve.evaluate(a)
                        for a in np.linspace(0.0, 0.8, 161)))
    auth0 = (_cl_max * 0.5 * c.air_density(h0) * V0 * V0 * c.fin_area / (m0 * G))
    auth_floor = min(float(min_authority_g), auth0) if min_authority_g else 0.0
    # No min_speed, no floor: the terminal constraint below is the only speed rule
    # this problem has ever had, and it stays that way unless a spec asks for more.
    speed_floor = min(float(min_speed), float(V0)) if min_speed else 0.0

    t_guess = Xf / max(V0 + 400.0, 100.0)

    # Every bound the phase declares, in one place: the same numbers go to
    # add_state/add_control below and to the warm start, which puts its values
    # inside them. An initial value outside a declared bound is not something the
    # solver quietly fixes -- the driver checks it before the first iteration and
    # prints a DriverWarning naming the variable -- and a long ray of cells that
    # warm-start each other prints one of those per cell.
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

    # The phase states and the ODE inputs have to agree on units, or OpenMDAO
    # has nothing to check the connections against. See MissileODE.setup.
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
        # EuroGNC 2022-040 eqs (5)-(6): alpha is a state, its rate is the control,
        # and the rate is bounded. Promoting it is the only way to bound it --
        # Radau has no control-rate constraint.
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
    # must RETAIN enough authority to correct, not merely stay under the
    # structural limit. This is what stops the over-loft. Set 0 to disable. The
    # value is the caller's -- `min_authority_g` in the grid spec, 0.1 g in the
    # grids that ship, which is weak: `nav_min_g` in a cell records what the
    # trajectory actually had, so the floor can be chosen from that.
    if auth_floor > 0.0:
        ph.add_path_constraint("navail", lower=auth_floor)
    # Lost below this speed, any time, not just at the end: in the mod the
    # missile self-destructs. 0 disables.
    if speed_floor > 0.0:
        ph.add_path_constraint("V", lower=speed_floor)
    # Optional structural/aero ceiling. OFF by default: this game's own cells
    # fly their terminal phase at 600-2600 kPa by design (an honest 30 km cell
    # arrives at 686 kPa), so a default q_max would be inventing a limit the
    # model does not have.
    if max_q_kpa and max_q_kpa > 0.0:
        ph.add_path_constraint("qbar", upper=float(max_q_kpa) * 1000.0)
    # OFF by default, useless.
    # if max_dive_deg and max_dive_deg > 0.0:
    #     ph.add_path_constraint("gamma", lower=-math.radians(max_dive_deg))
    ph.add_boundary_constraint(
        "V", loc="final", lower=c.self_destruct_at_speed)
    if free_range:
        # Reachability: how far can it go and still be alive at the target
        # altitude. The scaler is the guess so the number is order 1.
        ph.add_objective("x", loc="final", scaler=-1.0 / Xf)
    else:
        ph.add_objective("V", loc="final", scaler=-1.0 / 1000.0)

    # disp=False unless asked: SLSQP's default is to print its whole exit
    # summary -- current function value, iterations, function and gradient
    # evaluations, "Optimization FAILED." and a rule of dashes -- on every
    # solve. The run already reports the convergence state per cell (ITER-LIMIT
    # vs conv), and verbose still prints it for a one-off solve, where the
    # iteration history is the point.
    p.driver = om.ScipyOptimizeDriver(
        optimizer="SLSQP", tol=1e-6, maxiter=int(maxiter), disp=verbose)
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
            # The neighbour's flight time belongs to the NEIGHBOUR's range: stretch
            # it with the path (x is stretched by the same factor, so the speed
            # profile stays about right), then put it inside THIS cell's duration
            # bounds. It lands outside them routinely, because a solve that ends on
            # a duration bound is exactly what the optimiser does when terminal
            # speed is the objective -- flying the box as fast as it allows -- and
            # the next cell's box is a different box. Copying the raw tof is what
            # printed
            #   DriverWarning: ... design variable initial conditions are out of
            #   their specified bounds: traj.mid.t_duration   val [135.2] vs
            #   lower 172.1
            # once per cell of a long ray.
            duration = float(np.clip(tof * scale, *t_bounds))
            # time_vals is REQUIRED, and must be the SAME LENGTH as the array
            # being set. The state nodes (37) and the time/control nodes (48/36)
            # do not match, so guess["t"] cannot be reused directly -- pass a
            # uniform ramp of the right length for each array. It is only an
            # initial guess, and both solves share a mesh, so the values already
            # line up node-for-node.

            def tv(a):
                return np.linspace(0.0, duration, len(np.asarray(a).ravel()))
            ph.set_time_val(initial=0.0, duration=duration)
            ph.set_state_val("x", _clipped(gx * scale, *bounds["x"]), time_vals=tv(gx))
            for nm in ("h", "V", "gamma", "mass"):
                v = _clipped(np.asarray(guess[nm], dtype=float).ravel(), *bounds[nm])
                ph.set_state_val(nm, v, time_vals=tv(v))
            av = _clipped(np.asarray(guess["alpha"], dtype=float).ravel(),
                          *bounds["alpha"])
            if alpha_state:
                ph.set_state_val("alpha", av, time_vals=tv(av))
                ph.set_control_val("alphadot", np.zeros_like(av), time_vals=tv(av))
            else:
                ph.set_control_val("alpha", av, time_vals=tv(av))
            applied = True
        except Exception as e:
            # do NOT swallow this silently: a failed warm start looks exactly
            # like "warm start does not help", which cost me a full benchmark.
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
    # Where the flight time ended up. On a bound, the duration is the box's choice
    # rather than the optimiser's, and the terminal speed that comes with it is
    # bound-limited -- which is also why the NEXT cell along a ray used to be handed
    # a warm start outside its own box. 1e-3 relative: the solver stops near a
    # bound, not exactly on it (15.005 s against a 15.000 s bound is "on" it).
    sol["t_bounds"] = [float(t_bounds[0]), float(t_bounds[1])]
    dur = float(t[-1])
    sol["t_at_bound"] = (
        "lower" if dur <= t_bounds[0] * (1.0 + 1e-3) else
        "upper" if dur >= t_bounds[1] * (1.0 - 1e-3) else None)
    # Did the answer respect the problem it was given? A returned trajectory that
    # cannot hold 200 m/s is not a slow trajectory, it is a self-destruct.
    viol, facts = _audit(sol, c, min_authority_g=auth_floor, min_speed=speed_floor)
    sol["violations"] = viol
    sol.update(facts)
    sol["feasible"] = not viol
    if viol:
        sol["msg"] = "; ".join(f"{k}: {v}" for k, v in viol.items())
    return sol


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
