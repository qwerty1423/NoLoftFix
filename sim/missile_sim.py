from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

G = 9.80665


class Curve:
    def __init__(self, keys):
        k = [list(map(float, row)) for row in keys]
        if not k:
            raise ValueError("empty curve")
        w = len(k[0])
        if w not in (2, 4):
            raise ValueError("keys must be [t,v] or [t,v,inSlope,outSlope]")
        self.t = np.array([r[0] for r in k])
        self.v = np.array([r[1] for r in k])
        if w == 4:
            self.mi = np.array([r[2] for r in k])
            self.mo = np.array([r[3] for r in k])
        else:
            self.mi = np.gradient(self.v, self.t)
            self.mo = self.mi.copy()
        self._hermite = w == 4

    def evaluate(self, x: float) -> float:
        t = self.t
        if x <= t[0]:
            return float(self.v[0])
        if x >= t[-1]:
            return float(self.v[-1])
        i = int(np.searchsorted(t, x)) - 1
        i = max(0, min(i, len(t) - 2))
        dt = t[i + 1] - t[i]
        if dt <= 0:
            return float(self.v[i])
        if not self._hermite:
            u = (x - t[i]) / dt
            return float(self.v[i] * (1 - u) + self.v[i + 1] * u)
        u = (x - t[i]) / dt
        u2 = u * u
        u3 = u2 * u
        h00 = 2 * u3 - 3 * u2 + 1
        h10 = u3 - 2 * u2 + u
        h01 = -2 * u3 + 3 * u2
        h11 = u3 - u2
        return float(h00 * self.v[i] + h10 * (self.mo[i] * dt)
                     + h01 * self.v[i + 1] + h11 * (self.mi[i + 1] * dt))

    @staticmethod
    def from_json(spec):
        if isinstance(spec, Curve):
            return spec
        if isinstance(spec, dict):
            rows = [[k["time"], k["value"], k.get("inSlope"), k.get("outSlope")]
                    for k in spec["keys"]]
            rows = [r[:2] if r[2] is None else r for r in rows]
            return Curve(rows)
        return Curve(spec)


def air_density_isa(alt_m: float) -> float:
    if alt_m < 11000.0:
        T = 288.15 - 6.5e-3 * alt_m
        p = 101325.0 * (T / 288.15) ** 5.2561
    elif alt_m < 25000.0:
        T = 216.65
        p = 22632.0 * math.exp(-G * (alt_m - 11000.0) / (287.05 * T))
    else:
        T = 216.65 + 1.0e-3 * (alt_m - 25000.0)
        p = 2488.6 * (216.65 / T) ** 34.1632
    return p / (287.05 * T)


def speed_of_sound_isa(alt_m: float) -> float:
    T = 288.15 - 6.5e-3 * min(alt_m, 11000.0)
    return math.sqrt(1.4 * 287.05 * T)


@dataclass
class MotorSpec:
    thrust: float           # N
    burn_time: float        # s
    fuel_mass: float        # kg
    delay: float = 0.0      # s
    top_speed: float = 1e9


@dataclass
class Coeffs:
    # Missile
    mass_dry: float
    fin_area: float
    supersonic_drag: float
    drag_curve: Curve
    lift_curve: Curve
    torque: float
    max_turn_rate_dps: float
    g_limit: float
    motors: list[MotorSpec] = field(default_factory=list)

    # ARHSeeker
    loft_amount: float = 0.2
    self_destruct_at_speed: float = 200.0
    arm_delay: float = 1.0
    guidance_delay: float = 1.0
    max_lead: float = 5.0

    # Environment overrides
    air_density_curve: Curve | None = None
    autopilot_lag: float = 0.0

    def air_density(self, alt_m: float) -> float:
        if self.air_density_curve is not None:
            return self.air_density_curve.evaluate(alt_m * 1e-3)
        return air_density_isa(alt_m)

    def speed_of_sound(self, alt_m: float) -> float:
        return speed_of_sound_isa(alt_m)

    @property
    def q_rate_limit(self):
        return min(math.radians(self.max_turn_rate_dps), 9.81 * self.g_limit)

    @staticmethod
    def load(path) -> "Coeffs":
        d = json.loads(Path(path).read_text())
        return Coeffs(
            mass_dry=d["mass_dry"],
            fin_area=d["fin_area"],
            supersonic_drag=d["supersonic_drag"],
            drag_curve=Curve.from_json(d["drag_curve"]),
            lift_curve=Curve.from_json(d["lift_curve"]),
            torque=d["torque"],
            max_turn_rate_dps=d["max_turn_rate_dps"],
            g_limit=d["g_limit"],
            motors=[MotorSpec(**m) for m in d.get("motors", [])],
            loft_amount=d.get("loft_amount", 0.2),
            self_destruct_at_speed=d.get("self_destruct_at_speed", 200.0),
            arm_delay=d.get("arm_delay", 1.0),
            guidance_delay=d.get("guidance_delay", 1.0),
            max_lead=d.get("max_lead", 5.0),
            air_density_curve=(Curve.from_json(d["air_density_curve"])
                               if "air_density_curve" in d else None),
            autopilot_lag=d.get("autopilot_lag", 0.0),
        )


def loft_bias_vanilla(t_go: float, target_dist: float, loft_amount: float) -> float:
    if loft_amount <= 0.0:
        return 0.0
    return min(t_go * t_go * 4.905 * loft_amount, target_dist * loft_amount)


def t_go_vanilla(rel_range_vec_norm, v_vec, target_dist: float) -> float:
    closing = float(np.dot(rel_range_vec_norm, v_vec))
    return target_dist / max(closing, 10.0)


# Simulator

@dataclass
class Engagement:
    launch_alt: float
    launch_speed: float
    launch_pitch: float
    target_range: float
    target_alt: float
    target_speed: float
    target_accel: float = 0.0


@dataclass
class Result:
    hit: bool
    reason: str
    apex_alt: float
    apex_downrange: float
    apex_fraction: float
    terminal_speed: float
    tof: float
    max_q: float
    max_g: float
    min_speed_after_burnout: float
    history: dict


class MissileSim:
    def __init__(self, coeffs: Coeffs, dt: float = 0.02):
        self.c = coeffs
        self.dt = dt

    def _aero(self, vx, vz, theta, alt, mass, fin_area):
        c = self.c
        V = math.hypot(vx, vz)
        if V < 1e-6:
            return 0.0, 0.0, 0.0

        gamma = math.atan2(vz, vx)
        alpha = theta - gamma
        aoa_rad = abs(self._wrap_pi(alpha))

        rho = c.air_density(alt)
        qbar = 0.5 * rho * V * V

        D = c.drag_curve.evaluate(aoa_rad) * qbar * fin_area
        L = c.lift_curve.evaluate(aoa_rad) * qbar * fin_area

        if c.supersonic_drag > 0.0:
            a = c.speed_of_sound(alt)
            if V > 1.1 * a:
                D *= 1.0 + c.supersonic_drag
            elif V > 0.9 * a:
                num6 = min(abs(a - V) / a, 0.1)
                num7 = (0.1 - num6) / 0.1
                D *= 1.0 + num7 ** 3 * (c.supersonic_drag + 0.15)

        s = math.copysign(1.0, alpha) if alpha != 0.0 else 0.0
        fx = -D * (vx / V) + L * (-(vz / V)) * s
        fz = -D * (vz / V) + L * ((vx / V)) * s
        return fx, fz, qbar

    def _motor(self, state, theta):
        c = self.c
        st = state["motors"]
        if st["stage"] >= len(c.motors):
            return 0.0, 0.0, 0.0
        m = c.motors[st["stage"]]
        if st["delay"] > 0.0:
            st["delay"] -= self.dt
            return 0.0, 0.0, 0.0
        if st["fuel"] <= 0.0:
            st["stage"] += 1
            st["delay"] = c.motors[st["stage"]
                                   ].delay if st["stage"] < len(c.motors) else 0.0
            st["fuel"] = c.motors[st["stage"]
                                  ].fuel_mass if st["stage"] < len(c.motors) else 0.0
            return 0.0, 0.0, 0.0

        burn_rate = m.fuel_mass / m.burn_time
        burn = min(burn_rate * self.dt, st["fuel"])
        st["fuel"] -= burn
        throttle = state["throttle"]
        speed = math.hypot(state["vx"], state["vz"])
        T = m.thrust * throttle if speed < m.top_speed else 0.0
        return T * math.cos(theta), T * math.sin(theta), burn

    @staticmethod
    def _wrap_pi(a):
        while a > math.pi:
            a -= 2 * math.pi
        while a < -math.pi:
            a += 2 * math.pi
        return a

    def run(self, eng: Engagement, aimpoint_fn, t_max: float = 400.0,
            record: bool = True, guards: bool = True):

        c, dt = self.c, self.dt
        st = {
            "x": 0.0, "alt": eng.launch_alt,
            "vx": eng.launch_speed * math.cos(eng.launch_pitch),
            "vz": eng.launch_speed * math.sin(eng.launch_pitch),
            "theta": eng.launch_pitch, "q": 0.0,
            "mass": c.mass_dry + sum(m.fuel_mass for m in c.motors),
            "motors": {"stage": 0, "delay": c.motors[0].delay if c.motors else 0.0,
                       "fuel": c.motors[0].fuel_mass if c.motors else 0.0},
            "throttle": 1.0, "t": 0.0, "q_cmd": 0.0,
            "fin": 0.1 * c.fin_area,
        }

        tx, talt = eng.target_range, eng.target_alt
        apex_alt, apex_x = eng.launch_alt, 0.0
        max_q, max_g, min_speed = 0.0, 0.0, 1e9
        hist = {k: [] for k in ("t", "x", "alt", "V", "gamma_deg", "theta_deg",
                                "q_dps", "ap_alt", "mach", "accel")}
        engine_out_t = None
        n = int(t_max / dt)

        for _ in range(n):
            t = st["t"]
            if t >= c.guidance_delay:
                st["fin"] = c.fin_area
            tx_now = eng.target_range + eng.target_speed * t
            talt_now = eng.target_alt + 0.5 * eng.target_accel * t * t
            state_view = dict(st, target_x=tx_now, target_alt=talt_now)

            ap = aimpoint_fn(t, state_view)
            if ap is None:
                q_cmd = 0.0
            else:
                dx = ap[0] - st["x"]
                dz = ap[1] - st["alt"]
                e = self._wrap_pi(math.atan2(dz, dx) - st["theta"])
                kp = 4.0
                kd = 2.0 * math.sqrt(kp)
                q_cmd = float(np.clip(kp * e - kd * st["q"],
                                      -c.q_rate_limit, c.q_rate_limit))
            if c.autopilot_lag > 0.0:
                st["q_cmd"] += (q_cmd - st["q_cmd"]) * dt / c.autopilot_lag
            else:
                st["q_cmd"] = q_cmd

            V = math.hypot(st["vx"], st["vz"])
            q_max = min(math.radians(c.max_turn_rate_dps),
                        9.81 * c.g_limit / max(V, 1.0))
            q_next = st["q"] + st["q_cmd"] * dt
            excess = max(abs(q_next) - q_max, 0.0)
            q_applied = st["q_cmd"] - \
                math.copysign(
                    excess / dt, st["q_cmd"]) if excess > 0 else st["q_cmd"]

            fx, fz, qbar = self._aero(st["vx"], st["vz"], st["theta"], st["alt"],
                                      st["mass"], st["fin"])
            tfx, tfz, burned = self._motor(st, st["theta"])
            st["mass"] = max(st["mass"] - burned, 1e-3)
            if burned == 0.0 and engine_out_t is None and st["motors"]["stage"] >= len(c.motors):
                engine_out_t = t

            ax = (fx + tfx) / st["mass"]
            az = (fz + tfz) / st["mass"] - G

            st["x"] += st["vx"] * dt
            st["alt"] += st["vz"] * dt
            st["vx"] += ax * dt
            st["vz"] += az * dt
            st["theta"] += st["q"] * dt
            st["q"] += q_applied * dt
            st["t"] += dt

            V = math.hypot(st["vx"], st["vz"])
            gload = abs(st["q"]) * V / 9.81 if V > 1 else 0.0
            max_q = max(max_q, qbar)
            max_g = max(max_g, gload)
            if st["alt"] > apex_alt:
                apex_alt, apex_x = st["alt"], st["x"]

            if record:
                hist["t"].append(t)
                hist["x"].append(st["x"])
                hist["alt"].append(st["alt"])
                hist["V"].append(V)
                hist["gamma_deg"].append(math.degrees(
                    math.atan2(st["vz"], st["vx"])))
                hist["theta_deg"].append(math.degrees(st["theta"]))
                hist["q_dps"].append(math.degrees(st["q"]))
                hist["ap_alt"].append(ap[1] if ap else float("nan"))
                hist["mach"].append(V / c.speed_of_sound(st["alt"]))
                hist["accel"].append(math.hypot(ax, az + G))

            if guards:
                if st["alt"] < 0.0:
                    return self._res(False, "terrain", apex_alt, apex_x, eng, st, V,
                                     max_q, max_g, min_speed, engine_out_t, hist)
                if V < c.self_destruct_at_speed and t > 2.0:
                    return self._res(False, "self_destruct_speed", apex_alt, apex_x,
                                     eng, st, V, max_q, max_g, min_speed, engine_out_t, hist)
                if ap is not None:
                    dvx, dvz = ap[0] - st["x"], ap[1] - st["alt"]
                    if dvx * st["vx"] + dvz * st["vz"] < 0.0 and t > 10.0:
                        return self._res(False, "missed_target", apex_alt, apex_x, eng,
                                         st, V, max_q, max_g, min_speed, engine_out_t, hist)
            r = math.hypot(tx_now - st["x"], talt_now - st["alt"])
            if (r < 50.0 and t > 1.0) or (not guards and st["x"] >= tx_now):
                return self._res(True, "hit", apex_alt, apex_x, eng, st, V,
                                 max_q, max_g, min_speed, engine_out_t, hist)
            if t > 10.0 and st["motors"]["stage"] >= len(c.motors):
                min_speed = min(min_speed, V)

        return self._res(False, "timeout", apex_alt, apex_x, eng, st,
                         math.hypot(st["vx"], st["vz"]
                                    ), max_q, max_g, min_speed,
                         engine_out_t, hist)

    def _res(self, hit, reason, apex_alt, apex_x, eng, st, V, max_q, max_g,
             min_speed, engine_out_t, hist):
        span = eng.target_range if eng.target_range > 0 else 1.0
        return Result(
            hit=hit, reason=reason, apex_alt=apex_alt, apex_downrange=apex_x,
            apex_fraction=apex_x / span, terminal_speed=V, tof=st["t"],
            max_q=max_q, max_g=max_g,
            min_speed_after_burnout=(min_speed if min_speed < 1e8 else V),
            history={k: np.asarray(v) for k, v in hist.items()},
        )
