using System;
using System.Runtime.CompilerServices;

using UnityEngine;

namespace NOLoftFix.Core;

/// <summary>Per-missile midcourse state.</summary>
public sealed class LoftState
{
    public GlobalPosition LaunchPos;
    public Vector3 LaunchDir = Vector3.forward;

    /// <summary>
    /// original pip distance
    /// </summary>
    public float Xf0 = 1f;
    /// <summary>
    /// lagged normalisation in use
    /// </summary>
    public float XfUsed = 1f;

    public LoftCell Cell;
    /// <summary>
    /// bump coefficients in use
    /// </summary>
    public double[] Bump = [];
    public double[] BumpTarget = [];
    /// <summary>
    /// launch altitude, metres
    /// </summary>
    public float H0Act;
    /// <summary>
    /// current pip altitude, metres
    /// </summary>
    public float HTNow;
    /// <summary>
    /// lagged target alt
    /// </summary>
    public float HTUsed;
    /// <summary>
    /// seed HTUsed on the first tick
    /// </summary>
    public bool HTInit = true;
    /// <summary>
    /// target alt the cell was picked at
    /// </summary>
    public float HTLookup;
    /// <summary>
    /// launch speed the cell was picked at
    /// </summary>
    public float LookupSpeed;
    public float NextRelookT;
    public bool Engaged;
    public bool NeedLookup;
    public LoftTable Table;
    public LoftSettings Settings;
    public Action<string> Log;
    public bool Rebased;

    // diagnostics

    public int PipRefreshes;
    public float LastPipX, LastPipZ, LastPipY;
    public float MaxTrackErrM;
    public float ApexM;
    public float TermSpeed;
    public float LastLogT = -999f;
    public float LastTermLog = -999f;
    public float LastXfDrift;
    public bool FirstCommandLogged;
    public bool LoftZeroed;
    /// <summary>
    /// unclamped aimpoint deviation, deg
    /// </summary>
    public float WantDev;
    public LineRenderer PathLr;
    public string Id = "";
}

public static class LoftGuidance
{
    private static readonly ConditionalWeakTable<ARHSeeker, LoftState> States = [];

    private static int s_nextId;
    private static bool s_settingsDumped;

    /// <summary>For identifying individual missiles in the log.</summary>
    private static string NextId() => "M" + (++s_nextId);

    public static LoftState Get(ARHSeeker seeker)
    {
        var st = States.GetOrCreateValue(seeker);
        // GetOrCreateValue needs a parameterless ctor, so the tag is created here
        // rather than in a field initializer.
        if (st.Id.Length == 0)
        {
            st.Id = NextId();
        }

        return st;
    }

    private const int MaxEngageReports = 3;
    private static int s_reports;

    private static float FX(GlobalPosition p)
    {
        return p.x;
    }

    private static float FY(GlobalPosition p)
    {
        return p.y;
    }

    private static float FZ(GlobalPosition p)
    {
        return p.z;
    }

    private static Vector3 Flat(GlobalPosition a, GlobalPosition b)
    {
        var v = new Vector3(FX(b) - FX(a), 0f, FZ(b) - FZ(a));
        return v.sqrMagnitude > 1e-6f ? v.normalized : Vector3.forward;
    }

    private static float FlatDist(GlobalPosition a, GlobalPosition b)
    {
        float dx = FX(b) - FX(a), dz = FZ(b) - FZ(a);
        return Mathf.Sqrt((dx * dx) + (dz * dz));
    }

    private static float Distance(GlobalPosition a, GlobalPosition b)
    {
        float dx = FX(b) - FX(a), dy = FY(b) - FY(a), dz = FZ(b) - FZ(a);
        return Mathf.Sqrt((dx * dx) + (dy * dy) + (dz * dz));
    }

    /// <summary>
    /// capture the launch geometry and pick a cell.
    /// </summary>
    public static void OnInitialize(ARHSeeker seeker, LoftTable table,
                                    LoftSettings s, Action<string> log)
    {
        LoftState st = Get(seeker);
        Missile m = seeker.missile;
        if (m == null || table == null || table.Count == 0)
        {
            return;
        }

        st.Engaged = false;
        if (seeker.targetUnit == null)
        {
            if (s_reports++ < MaxEngageReports)
            {
                log?.Invoke($"{st.Id} Not engaged: no target unit at initialize.");
            }

            return;
        }

        GlobalPosition pip0 = seeker.knownPos;
        st.LaunchPos = m.GlobalPosition();
        st.H0Act = FY(st.LaunchPos);
        float range0 = FlatDist(st.LaunchPos, pip0);

        // Detect placeholder
        if (Mathf.Abs(range0 - 100000f) < 250f)
        {
            if (s_reports++ < MaxEngageReports)
            {
                log?.Invoke($"{st.Id} Not engaged: datalink had no track.");
            }

            return;
        }

        st.LaunchDir = Flat(st.LaunchPos, pip0);
        st.Xf0 = Mathf.Max(range0, 500f);
        st.XfUsed = st.Xf0;
        st.HTNow = FY(pip0);
        st.LastPipX = FX(pip0); st.LastPipZ = FZ(pip0); st.LastPipY = FY(pip0);
        st.Rebased = false;

        float targetAltM = st.HTNow;
        float rangeKm = st.Xf0 / 1000f;

        st.Table = table;
        st.Log = log;
        st.Settings = s;
        if (!s_settingsDumped)
        {
            s_settingsDumped = true;
            log?.Invoke($"[settings] TrackGain={s.TrackGain} MaxTrackAngleDeg={s.MaxTrackAngleDeg} " +
                        $"LookaheadS={s.LookaheadSeconds} TargetAltLagRateMPerS={s.TargetAltLagRate} " +
                        $"MaxNormalisedDistance={s.MaxNormDistance} FloorAltM={s.FloorAltM} " +
                        $"DebugDraw={s.DebugDraw} Verbose={s.Verbose} " +
                        $"MinCellTerminalSpeedMs={s.MinCellTerminalSpeedMs:F0} " +
                        $"RejectFlatCells={s.RejectFlatCells} Enabled={s.Enabled} ");
        }
        if (m.speed >= MinLookupSpeed)
        {
            DoLookup(st, table, s, log, m.speed);
        }
        else
        {
            // Missile.LocalStart runs before ignition, so m.speed is 0 here and
            // the table would be queried at a launch speed far outside the grid.
            // Defer to the first Override tick where the missile is moving.
            st.NeedLookup = true;
            if (s.Verbose)
            {
                log?.Invoke($"{st.Id} deferring cell lookup: speed {m.speed:F0} m/s at Initialize " +
                            $"(Xf0={st.Xf0:F0} m captured)");
            }
        }
    }

    /// <summary>
    /// <para>
    /// Zeroing loftAmount also stops ARHSeeker.SlowChecks from writing
    /// targetDist.
    /// So the mod has to keep the field alive itself. Same value SlowChecks
    /// would have computed, just not gated on loftAmount.
    /// </para>
    /// </summary>
    public static void MaintainTargetDist(ARHSeeker seeker)
    {
        Missile m = seeker.missile;
        if (m == null)
        {
            return;
        }

        GlobalPosition pos = m.GlobalPosition();
        GlobalPosition kp = seeker.knownPos;
        float dx = FX(kp) - FX(pos), dy = FY(kp) - FY(pos), dz = FZ(kp) - FZ(pos);
        seeker.targetDist = Mathf.Sqrt((dx * dx) + (dy * dy) + (dz * dz));
    }

    /// <summary>
    /// Below this the launch speed is not yet meaningful for a lookup.
    /// </summary>
    private const float MinLookupSpeed = 50f;

    private static void DoLookup(LoftState st, LoftTable table, LoftSettings s,
                                 Action<string> log, float speed)
    {
        float rangeKm = st.Xf0 / 1000f;
        st.Cell = table.Lookup(rangeKm, st.H0Act, speed, st.HTNow, 0f,
                               s.Neighbours, s.IdwPower, s.MaxNormDistance);
        if (st.Cell == null)
        {
            if (s_reports++ < MaxEngageReports)
            {
                log?.Invoke($"{st.Id} Not engaged at {rangeKm:F1} km / {st.H0Act:F0} m / {speed:F0} m/s " +
                            $"/ tgt {st.HTNow:F0} m -- {table.LastReason}.");
            }

            return;
        }
        st.Bump = st.Cell.BumpCoeffs();
        st.BumpTarget = st.Bump;
        st.HTLookup = st.HTNow;
        st.LookupSpeed = speed;
        st.Engaged = true;
        if (s_reports++ < MaxEngageReports || s.Verbose)
        {
            log?.Invoke($"{st.Id} Engaged at {rangeKm:F1} km / {st.H0Act:F0} m / {speed:F0} m/s " +
                        $"/ tgt {st.HTNow:F0} m (Xf0={st.Xf0:F0} m, nearest cell " +
                        $"{table.LastDistance:F3}, table varies on {table.EffectiveAxes}/5 axes; " +
                        $"fixed: {table.DegenerateAxes})");
        }
    }

    /// <summary>
    /// ARHSeeker.Seek postfix -- runs after the game's own SetAimpoint, so it
    /// overwrites it. Returns false when the vanilla aimpoint should stand.
    /// </summary>
    /// <param name="seeker"></param>
    /// <param name="s"></param>
    /// <param name="log"></param>
    public static bool Override(ARHSeeker seeker, LoftSettings s, Action<string> log)
    {
        LoftState st = Get(seeker);
        Missile m = seeker.missile;
        if (m == null)
        {
            return false;
        }

        if (st.NeedLookup && st.Table != null && m.speed >= MinLookupSpeed)
        {
            st.NeedLookup = false;
            DoLookup(st, st.Table, st.Settings ?? s, st.Log ?? log, m.speed);
        }
        if (!st.Engaged || st.Cell == null)
        {
            return false;
        }

        float dt = Mathf.Max(Time.fixedDeltaTime, 1e-3f);
        GlobalPosition pos = m.GlobalPosition();

        // knownPos is where the target is; GetLeadVectorWithAccel is
        // what ARHSeeker.Seek uses to turn that into an intercept point, so
        // routing through it keeps the lead instead of aiming behind a
        // crossing target.
        GlobalPosition pip = seeker.knownPos;
        try
        {
            Vector3 platformVel = m.timeSinceSpawn < 3f
                ? m.transform.forward * seeker.topSpeed
                : m.rb.velocity;
            pip = seeker.knownPos + TargetCalc.GetLeadVectorWithAccel(
                seeker.knownPos, pos, seeker.knownVel, platformVel,
                seeker.knownAccel, seeker.maxLead);
        }
        catch
        {
            // ignored
        }

        // check PIP movement
        float pipDx = FX(pip) - st.LastPipX, pipDz = FZ(pip) - st.LastPipZ;
        if (Mathf.Abs(pipDx) > s.PipChangeThresholdM || Mathf.Abs(pipDz) > s.PipChangeThresholdM)
        {
            st.LastPipX = FX(pip); st.LastPipZ = FZ(pip); st.LastPipY = FY(pip);
            st.PipRefreshes++;
        }

        // disable in terminal range
        float slant = Distance(pos, pip);
        if (slant <= seeker.terminalRange)
        {
            // log terminal info if verbose logging is enabled
            if (s.Verbose && m.timeSinceSpawn - st.LastTermLog >= s.FlightLogIntervalS)
            {
                st.LastTermLog = m.timeSinceSpawn;
                Vector3 rel = ToV3(pip) - ToV3(pos);
                float closing = -Vector3.Dot(rel.normalized, m.rb ? m.rb.velocity : Vector3.zero);
                log?.Invoke($"{st.Id} [terminal] t={m.timeSinceSpawn:F1}s slant={slant:F0}m " +
                            $"closing={closing:F0}m/s v={m.speed:F0} alt={FY(pos):F0} " +
                            $"tgt={FY(seeker.knownPos):F0} dh={FY(pos) - FY(seeker.knownPos):F0} " +
                            $"act={Mathf.Rad2Deg * Mathf.Asin(Mathf.Clamp(m.transform.forward.y, -1f, 1f)):F1}deg " +
                            $"mode={m.seekerMode}");
            }
            return false;
        }

        Vector3 dirNow = Flat(pos, pip);
        float xfNow = Mathf.Max(FlatDist(st.LaunchPos, pip), 500f);
        // use terminal altitude from the track instead of the lead point
        // otherwise large oscillations will result
        st.HTNow = FY(seeker.knownPos);
        float hTNow = st.HTNow;
        if (st.HTInit) { st.HTUsed = hTNow; st.HTInit = false; }

        // One-shot rebase. SlowChecks refreshes knownPos at 1 Hz, so the value
        // at Initialize can be noticeably off.
        if (!st.Rebased && m.timeSinceSpawn > 1.5f && m.timeSinceSpawn < s.RebaseWindowS)
        {
            st.Rebased = true;
            if (Mathf.Abs(xfNow - st.Xf0) > 0.25f * st.Xf0)
            {
                if (s.Verbose)
                {
                    log?.Invoke($"[rebase] Xf0 {st.Xf0:F0} m -> {xfNow:F0} m from the live track");
                }

                st.Xf0 = xfNow;
                st.XfUsed = xfNow;
                st.LaunchDir = Flat(st.LaunchPos, pip);
            }
        }

        // lag the normalisation
        // Without this, a target turning away increases xfNow, decreases s, and the
        // missile is suddenly told it is earlier in the profile and must climb.
        float maxStep = s.XfLagRate * dt;
        st.XfUsed += Mathf.Clamp(xfNow - st.XfUsed, -maxStep, maxStep);
        st.LastXfDrift = xfNow - st.XfUsed;

        // Cross-fade to a new profile shape if one was queued.
        // fade bump -> bumpTarget; lengths can differ between cells, so only
        // fade the overlap and snap anything beyond it.
        if (st.BumpTarget?.Length > 0)
        {
            float a = Mathf.Clamp01(dt / Mathf.Max(s.BumpFadeS, 1e-3f));
            if (st.Bump == null || st.Bump.Length != st.BumpTarget.Length)
            {
                st.Bump = (double[])st.BumpTarget.Clone();
            }
            else
            {
                for (int i = 0; i < st.Bump.Length; i++)
                {
                    st.Bump[i] += (st.BumpTarget[i] - st.Bump[i]) * a;
                }
            }
        }

        // lag the altitude axis too
        // Guard 1 above lags the range, because an unlagged xfNow turns a
        // datalink refresh into a step command. The altitude axis has exactly
        // the same problem and was never lagged: Ref() uses
        //     h0Act + (hTNow - h0Act) * s + B(s)
        // with hTNow = FY(pip), and knownPos refreshes at 1 Hz. A step in the
        // track therefore moves the reference by dHt * s -- half the step
        // mid-flight -- and pure pursuit chases it. The table is not the
        // source: the 4-cell IDW blend for a logged shot gives a smooth bump
        float hStep = s.TargetAltLagRate * dt;
        st.HTUsed += Mathf.Clamp(hTNow - st.HTUsed, -hStep, hStep);

        // progress along the frozen launch axis
        //
        // This used to be x = xf0 - FlatDist(pos, pip). That is only a measure
        // of distance closed if pip means the same point at both ends of the
        // subtraction, and it does not: pip = knownPos + lead, and the lead
        // grows from about nothing at ignition to kilometres once the missile
        // is fast. In a logged shot xf0 was 42.7 km while launch->pip was
        // 47.1 km one second later -- a 4.4 km offset purely from lead -- so
        // x stayed clamped at 0 for ELEVEN seconds. The reference froze at
        // launch altitude while the missile climbed 800 m past it, putting the
        // aimpoint below the missile and commanding the dive.
        //
        // Projecting onto launchDir fixes both problems at once: it is monotone
        // by construction, it needs no consistency between two pip samples, and
        // because the axis is captured once at launch it cannot collapse when
        // the target turns -- which was the failure this guard was written to
        // avoid in the first place.
        Vector3 fromLaunch = new(FX(pos) - FX(st.LaunchPos), 0f,
                                         FZ(pos) - FZ(st.LaunchPos));
        float x = Mathf.Clamp(Vector3.Dot(fromLaunch, st.LaunchDir), 0f, st.Xf0);

        // reference here, and at the lookahead point
        float d = Mathf.Clamp(s.LookaheadSeconds * Mathf.Max(m.speed, 1f),
                              s.MinLookaheadM, s.MaxLookaheadM);
        float xAp = Mathf.Min(x + d, st.Xf0);
        double sNow = Mathf.Clamp01(x / st.XfUsed);
        double sAp = Mathf.Clamp01(xAp / st.XfUsed);

        double hRef = Ref(st, sNow);
        double hAp = Ref(st, sAp);

        // tracking-error correction
        float trackErr = (float)hRef - FY(pos);
        st.MaxTrackErrM = Mathf.Max(st.MaxTrackErrM, Mathf.Abs(trackErr));
        // Plain proportional trim. Its angular authority is bounded below, by
        // the total deviation clamp on the aimpoint -- clamping it here as well
        // would be redundant, and clamping it here ALONE was not enough, because
        // the steep commands came from the altitude error itself rather than
        // from this term.
        hAp += s.TrackGain * trackErr;

        // build the aimpoint
        // Aim along the line to the CURRENT pip, so the horizontal command
        // tracks a turning target too.
        GlobalPosition ap = pos + (dirNow * d);
        float apY = Mathf.Max((float)hAp, s.FloorAltM);

        // Bound the commanded angle relative to the profile's own slope.
        //
        // The aimpoint sits d = clamp(3*V, 800, 15000) metres ahead, so ANY
        // altitude difference between the missile and the reference at that
        // point becomes atan(dh / d) of commanded flight path angle -- and d
        // shrinks as the missile slows.
        //
        // Clamping the TrackGain term did not touch this, because the
        // steepness comes from the altitude error itself and not from the
        // correction. This clamps the total deviation instead, so the missile
        // converges on the profile at a bounded rate whatever its speed.
        float apHoriz = Mathf.Sqrt(((FX(ap) - FX(pos)) * (FX(ap) - FX(pos))) +
                                   ((FZ(ap) - FZ(pos)) * (FZ(ap) - FZ(pos))));
        apHoriz = Mathf.Max(apHoriz, 1f);
        float slope = Mathf.Atan2((float)(hAp - hRef), Mathf.Max(d, 1f));
        float want = Mathf.Atan2(apY - FY(pos), apHoriz);
        st.WantDev = Mathf.Rad2Deg * (want - slope);
        if (s.MaxTrackAngleDeg > 0f)
        {
            float lim = s.MaxTrackAngleDeg * Mathf.Deg2Rad;
            float cmdAng = slope + Mathf.Clamp(want - slope, -lim, lim);
            apY = FY(pos) + (Mathf.Tan(cmdAng) * apHoriz);
            ap.y = apY;
        }

        // Never command into the ground, and never put the aimpoint behind 
        // the missile -- that trips ARHSeeker.MissedTarget()
        // and self-destructs the round.
        float minAhead = Mathf.Max(300f, 1.5f * m.speed);
        Vector3 fwd = m.rb.velocity.sqrMagnitude > 1f ? m.rb.velocity.normalized : m.transform.forward;
        Vector3 toAp = new(FX(ap) - FX(pos), apY - FY(pos), FZ(ap) - FZ(pos));
        float ahead = Vector3.Dot(toAp, fwd);
        if (ahead < minAhead)
        {
            ap = pos + toAp + (fwd * (minAhead - ahead));
        }

        ap.y = apY;

        if (s.DebugDraw)
        {
            Draw(st, m, pos, pip, sNow);
        }

        m.SetAimpoint(ap, seeker.knownVel);
        if (!st.FirstCommandLogged)
        {
            st.FirstCommandLogged = true;
            if (s.Verbose)
            {
                log?.Invoke($"{st.Id} [law] first aimpoint command at t={m.timeSinceSpawn:F1}s: " +
                        $"missile {FY(pos):F0} m, aimpoint {apY:F0} m, " +
                        $"ref here {(float)hRef:F0} m, progress {x / 1000f:F1}/{st.XfUsed / 1000f:F1} km");
            }

            st.LastLogT = m.timeSinceSpawn;
        }

        st.ApexM = Mathf.Max(st.ApexM, FY(pos));
        st.TermSpeed = m.speed;

        if (s.Verbose && m.timeSinceSpawn - st.LastLogT > s.FlightLogIntervalS)
        {
            st.LastLogT = m.timeSinceSpawn;
            // Every term that can move the reference:
            //   s      progress along the profile
            //   B      the bump, i.e. everything that is not the endpoint chord
            //   hT     live pip altitude vs the lagged one actually used
            //   xf     live range vs the lagged one actually used
            //   d      lookahead distance actually applied to the aimpoint
            // slope = the profile's own flight path angle over the lookahead.
            // cmd   = what the aimpoint actually commands, after the clamp.
            // act   = what the missile is doing.
            // dev   = how far the unclamped aimpoint wanted to deviate.
            // cmd far from act -> the airframe is not following the aimpoint.
            // dev pinned at the limit -> the missile is off the profile and
            //     recovering as fast as it is allowed to.
            // Note: do not compare cmd against slope directly -- cmd is measured
            // from the missile's actual altitude and slope from the reference, so
            // they differ by the altitude error even with a perfect aimpoint.
            float cmdDeg = Mathf.Rad2Deg * Mathf.Atan2(apY - FY(pos), apHoriz);
            float actDeg = Mathf.Rad2Deg * Mathf.Asin(Mathf.Clamp(fwd.y, -1f, 1f));
            float refDeg = Mathf.Rad2Deg * slope;
            log?.Invoke($"{st.Id} [flight] t={m.timeSinceSpawn:F1}s s={sNow:F3} " +
                        $"alt={FY(pos):F0} ref={(float)hRef:F0} B={Poly.Eval(st.Bump, sNow):F0} " +
                        $"hT={hTNow:F0}/{st.HTUsed:F0} xf={xfNow / 1000f:F1}/{st.XfUsed / 1000f:F1}km " +
                        $"v={m.speed:F0} d={d:F0} err={trackErr:F0} pip={st.PipRefreshes} " +
                        $"cell={st.Cell.Range_km:F0}km | slope={refDeg:F1} cmd={cmdDeg:F1} " +
                        $"act={actDeg:F1} dev={st.WantDev:F1} deg");
        }
        return true;
    }

    private static readonly Color ProfileCol = Color.magenta;

    private static Vector3 ToV3(GlobalPosition p)
    {
        return new Vector3(FX(p), FY(p), FZ(p));
    }

    private static bool EnsurePath(LoftState st)
    {
        try
        {
            var go = new GameObject("NOLoftFix_Path");
            UnityEngine.Object.DontDestroyOnLoad(go);
            var lr = go.AddComponent<LineRenderer>();
            lr.useWorldSpace = true;
            lr.startWidth = 1.5f;
            lr.endWidth = 1.5f;
            lr.numCapVertices = 2;
            lr.positionCount = 0;
            lr.startColor = ProfileCol;
            lr.endColor = ProfileCol;
            Shader sh = Shader.Find("Sprites/Default");
            if (sh != null)
            {
                lr.material = new Material(sh);
            }

            st.PathLr = lr;
            return true;
        }
        catch { return false; }
    }

    private static void Draw(LoftState st, Missile m, GlobalPosition pos,
                             GlobalPosition pip, double sNow)
    {
        Vector3 axis = new(FX(pip) - FX(st.LaunchPos), 0f,
                                   FZ(pip) - FZ(st.LaunchPos));
        float len = axis.magnitude;

        if (len > 1f)
        {
            if (st.PathLr == null && !EnsurePath(st))
            {
                return;
            }

            if (st.PathLr == null)
            {
                return;
            }

            Vector3 anchor = m.transform.position;
            Vector3 u = st.LaunchDir;
            const int n = 28;
            Vector3[] pts = new Vector3[n + 1];
            for (int i = 0; i <= n; i++)
            {
                double sf = sNow + ((1.0 - sNow) * i / n);
                float along = (float)(sf - sNow) * st.XfUsed;
                pts[i] = anchor + new Vector3(u.x * along,
                                              (float)Ref(st, sf) - FY(pos),
                                              u.z * along);
            }
            st.PathLr.positionCount = n + 1;
            st.PathLr.SetPositions(pts);
        }
    }


    /// <summary>
    /// h(s) = h_launch + (h_pip_now - h_launch)*s + B(s), evaluated
    /// with the endpoints held by the caller in hTNow. Kept inline because it
    /// needs the live target altitude.
    /// </summary>
    /// <param name="st"></param>
    /// <param name="sFrac"></param>
    private static double Ref(LoftState st, double sFrac)
    {
        return st.H0Act + ((st.HTUsed - st.H0Act) * sFrac) + Poly.Eval(st.Bump, sFrac);
    }
}
