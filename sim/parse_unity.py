"""
python3 parse_unity.py --prefab ../txt/AAM2.txt \
                        --assets  ../txt/GameAssets.txt \
                        --out     coeffs_aam2.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

KEY_RE = re.compile(
    r"time:\s*(?P<t>[-\d.eE+]+)\s*\n"
    r"\s*value:\s*(?P<v>[-\d.eE+]+)\s*\n"
    r"\s*inSlope:\s*(?P<mi>[-\d.eE+]+)\s*\n"
    r"\s*outSlope:\s*(?P<mo>[-\d.eE+]+)")


def find_curve(text: str, field: str, start: int = 0):
    m = re.search(rf"^  {field}:\s*$", text[start:], re.M)
    if not m:
        raise KeyError(f"curve '{field}' not found")
    begin = start + m.end()
    nxt = re.search(r"^  \w", text[begin:], re.M)
    body = text[begin: begin + nxt.start()] if nxt else text[begin:]
    keys = [[float(k["t"]), float(k["v"]), float(k["mi"]), float(k["mo"])]
            for k in KEY_RE.finditer(body)]
    if not keys:
        raise ValueError(f"no keys parsed for '{field}'")
    return keys


def find_scalar(text: str, field: str, start: int = 0, end: int | None = None):
    seg = text[start:end] if end else text[start:]
    m = re.search(rf"^  {field}:\s*(?P<v>\S+)\s*$", seg, re.M)
    if not m:
        m = re.search(rf"^\s+{field}:\s*(?P<v>\S+)\s*$", seg, re.M)
    if not m:
        raise KeyError(f"scalar '{field}' not found")
    return float(m.group("v"))


def section(text: str, marker: str) -> tuple[int, int]:
    i = text.index(marker)
    b = text.rindex("--- !u!", 0, i)
    nxt = text.find("\n--- !u!", i)
    return b, (nxt if nxt != -1 else len(text))


def locate(text: str, unique_field: str) -> tuple[int, int]:
    i = text.index(f"\n  {unique_field}:")
    b = text.rindex("--- !u!", 0, i)
    nxt = text.find("\n--- !u!", i)
    return b, (nxt if nxt != -1 else len(text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefab", required=True)
    ap.add_argument("--assets", required=True)
    ap.add_argument("--out", default="coeffs_aam2.json")
    a = ap.parse_args()

    P = Path(a.prefab).read_text()
    A = Path(a.assets).read_text()

    mb, me = locate(P, "finArea")
    sb, se = locate(P, "loftAmount")

    def keys(field, txt, lo=0, hi=None):
        return [{"time": k[0], "value": k[1], "inSlope": k[2], "outSlope": k[3]}
                for k in find_curve(txt, field, lo)]

    out = {
        "_provenance": f"parsed from {Path(a.prefab).name} + {Path(a.assets).name}",
        "_mass_wet_prefab": find_scalar(P, "mass", mb, me),
        "fin_area": find_scalar(P, "finArea", mb, me),
        "supersonic_drag": find_scalar(P, "supersonicDrag", mb, me),
        "torque": find_scalar(P, "torque", mb, me),
        "g_limit": find_scalar(P, "gLimit", mb, me),
        "max_turn_rate_dps": find_scalar(P, "maxTurnRate", mb, me),
        "lift_curve": {"keys": keys("liftCurve", P, mb)},
        "drag_curve": {"keys": keys("dragCurve", P, mb)},
        "loft_amount": find_scalar(P, "loftAmount", sb, se),
        "self_destruct_at_speed": find_scalar(P, "selfDestructAtSpeed", sb, se),
        "arm_delay": find_scalar(P, "armDelay", sb, se),
        "guidance_delay": find_scalar(P, "guidanceDelay", sb, se),
        "max_lead": find_scalar(P, "maxLead", sb, se),
        "terminal_range": find_scalar(P, "terminalRange", sb, se),
        "air_density_curve": {"keys": keys("airDensityAltitude", A)},
    }

    mstart = P.index("\n  motors:", mb)
    mend = P.index("\n  mass:", mstart)
    motors = []
    for chunk in re.split(r"\n  - ", P[mstart:mend])[1:]:
        def g(f, _c=chunk):
            return float(re.search(rf"^\s*{f}:\s*(\S+)", _c, re.M).group(1))
        motors.append({"thrust": g("thrust"), "burn_time": g("burnTime"),
                       "fuel_mass": g("fuelMass"), "delay": g("delayTimer"),
                       "top_speed": g("topSpeed")})
    out["motors"] = motors
    out["mass_dry"] = out.pop("_mass_wet_prefab") - \
        sum(m["fuel_mass"] for m in motors)

    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {a.out}\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
