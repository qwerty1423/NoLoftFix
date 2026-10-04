using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

using UnityEngine;

namespace NOLoftFix
{
    /// <summary>Per-missile midcourse state. Lives as long as the ARHSeeker does.</summary>
    public sealed class LoftState
    {
        public GlobalPosition launchPos;
        public Vector3 launchDir = Vector3.forward;   // unit horizontal, launch -> original PIP
        public float xf0 = 1f;                        // launch -> ORIGINAL pip distance, frozen
        public float xfUsed = 1f;                     // lagged normalisation actually in use

        public LoftCell cell;
        public double[] bump = new double[0];         // bump coefficients currently in force
        public double[] bumpTarget = new double[0];   // where they are fading to
        public float h0Act;                           // launch altitude, metres
        public float hTNow;                           // LIVE pip altitude, metres
        public float hTUsed;                          // LAGGED target alt the ref is built from
        public bool hTInit = true;                    // seed hTUsed on the first tick
        public float hTLookup;                        // target alt the cell was picked at
        public float lookupSpeed;                     // launch speed the cell was picked at
        public float nextRelookT;
        public bool engaged;
        public bool needLookup;                       // cell lookup deferred past ignition
        public LoftTable table;
        public LoftSettings settings;
        public Action<string> log;
        public bool rebased;                          // xf0 re-derived once, post-launch

        // diagnostics
        public int pipRefreshes;
        public float lastPipX, lastPipZ, lastPipY;
        public float maxTrackErrM;
        public float apexM;
        public float termSpeed;
        public float lastLogT = -999f;
        public float lastTermLog = -999f;   // separate clock: terminal starts at -999 so the handover itself is logged
        public float lastXfDrift;
        public bool firstCommandLogged;
        public bool loftZeroed;
        public float wantDev;                         // unclamped aimpoint deviation, deg
        public LineRenderer pathLr;
        public string id = "";                        // short per-round tag for the log
    }

    /// <summary>
    /// The actual guidance law.
    ///
    /// The game gives us exactly one control surface: the aimpoint.
    /// Missile.Steering() does PURE PURSUIT toward it (direction to aimPoint ->
    /// body frame -> PID2D -> torque). There is no acceleration command anywhere,
    /// so CEAS-GNC-2026-016's APG law a_com = N*V*(lambda_ref - lambda) is NOT
    /// implemented literally. The reference profile is followed by PLACEMENT
    /// instead: the aimpoint goes at a lookahead point on the profile, plus a
    /// proportional altitude-error correction. Pure pursuit toward a lookahead
    /// point on a reference path IS trajectory following.
    ///
    /// ---- TARGET MOTION ----------------------------------------------------
    /// The solved profile is an absolute polynomial pinned to the launch and
    /// target altitudes it was generated for, so it cannot be used directly once
    /// the target moves. It is decomposed into a BUMP -- the polynomial minus the
    /// straight line between its own endpoints -- which vanishes at both ends.
    /// The runtime reference is then rebuilt as
    ///
    ///     h(s) = h_launch + (h_pip_now - h_launch) * s + B(s)
    ///
    /// so the offline SHAPE survives while both endpoints track the live PIP
    /// exactly. Three guards keep that continuous:
    ///
    ///   1. `s` is normalised by `xfUsed`, a first-order lag on the true
    ///      launch->PIP distance, so a target turning away cannot step the
    ///      reference.
    ///   2. progress is measured as (xf0 - remaining range), which stays monotone
    ///      even if the target turns 90 degrees and the along-axis projection
    ///      collapses.
    ///   3. when the range drifts far enough to warrant a different table cell,
    ///      the bump coefficients are cross-faded rather than swapped.
    /// </summary>
    public static class LoftGuidance
    {
        private static readonly ConditionalWeakTable<ARHSeeker, LoftState> States = new();

        private static int s_nextId;
        private static bool s_settingsDumped;

        /// <summary>Every log line carries this so interleaved rounds can be told apart.</summary>
        private static string NextId() => "M" + (++s_nextId);

        public static LoftState Get(ARHSeeker seeker)
        {
            var st = States.GetOrCreateValue(seeker);
            // GetOrCreateValue needs a parameterless ctor, so the tag is minted here
            // on first touch rather than in a field initializer.
            if (st.id.Length == 0) st.id = NextId();
            return st;
        }

        private const int MaxEngageReports = 12;
        private static int s_reports;

        private static float FX(GlobalPosition p) => (float)p.x;
        private static float FY(GlobalPosition p) => (float)p.y;
        private static float FZ(GlobalPosition p) => (float)p.z;

        private static Vector3 Flat(GlobalPosition a, GlobalPosition b)
        {
            var v = new Vector3(FX(b) - FX(a), 0f, FZ(b) - FZ(a));
            return v.sqrMagnitude > 1e-6f ? v.normalized : Vector3.forward;
        }

        private static float FlatDist(GlobalPosition a, GlobalPosition b)
        {
            float dx = FX(b) - FX(a), dz = FZ(b) - FZ(a);
            return Mathf.Sqrt(dx * dx + dz * dz);
        }

        private static float Distance(GlobalPosition a, GlobalPosition b)
        {
            float dx = FX(b) - FX(a), dy = FY(b) - FY(a), dz = FZ(b) - FZ(a);
            return Mathf.Sqrt(dx * dx + dy * dy + dz * dz);
        }

        /// <summary>
        /// ARHSeeker.Initialize postfix: capture the launch geometry and pick a cell.
        ///
        /// IMPORTANT: the `aimpoint` argument is NOT the target. Missile.LocalStart
        /// does `aimPoint = ownerPos + ownerForward * 100000f` and passes that, and
        /// ARHSeeker.Initialize ignores the parameter entirely -- it sets
        /// `knownPos = missilePos + forward * 100000f` and then overwrites it from
        /// NetworkHQ.TryGetKnownPosition when there is a target. So the only usable
        /// geometry at this point is seeker.knownPos, which the game has already
        /// populated by the time this postfix runs. Using the parameter instead
        /// made xf0 = 100 km for every shot, which put the missile at s=0.55 on the
        /// profile at launch and destroyed the guidance.
        /// </summary>
        public static void OnInitialize(ARHSeeker seeker, LoftTable table,
                                        LoftSettings s, Action<string> log)
        {
            var st = Get(seeker);
            var m = seeker.missile;
            if (m == null || table == null || table.Count == 0) return;

            st.engaged = false;
            if (seeker.targetUnit == null)
            {
                if (s_reports++ < MaxEngageReports)
                    log?.Invoke($"{st.id} NOT ENGAGED: no target unit at Initialize, so knownPos is the " +
                                "100 km forward placeholder and the range is unknown.");
                return;
            }

            GlobalPosition pip0 = seeker.knownPos;
            st.launchPos = m.GlobalPosition();
            st.h0Act = FY(st.launchPos);
            float range0 = FlatDist(st.launchPos, pip0);

            // Detect the untouched placeholder: exactly 100 km ahead along forward.
            if (Mathf.Abs(range0 - 100000f) < 250f)
            {
                if (s_reports++ < MaxEngageReports)
                    log?.Invoke($"{st.id} NOT ENGAGED: knownPos is still the 100 km forward placeholder " +
                                "(datalink had no track). Vanilla lofting stays.");
                return;
            }

            st.launchDir = Flat(st.launchPos, pip0);
            st.xf0 = Mathf.Max(range0, 500f);
            st.xfUsed = st.xf0;
            st.hTNow = FY(pip0);
            st.lastPipX = FX(pip0); st.lastPipZ = FZ(pip0); st.lastPipY = FY(pip0);
            st.rebased = false;

            float targetAltM = st.hTNow;
            float rangeKm = st.xf0 / 1000f;

            st.table = table;
            st.log = log;
            st.settings = s;
            if (!s_settingsDumped)
            {
                // BepInEx Bind returns the value already in the config file, so a
                // default changed in code silently does NOT apply to an installed
                // config. Print what is actually in force, once.
                s_settingsDumped = true;
                log?.Invoke($"[settings] TrackGain={s.trackGain} MaxTrackAngleDeg={s.maxTrackAngleDeg} " +
                            $"LookaheadS={s.lookaheadSeconds} TargetAltLagRateMPerS={s.targetAltLagRate} " +
                            $"MaxNormalisedDistance={s.maxNormDistance} FloorAltM={s.floorAltM} " +
                            $"DebugDraw={s.debugDraw} Verbose={s.verbose} (these are LIVE -- edit " +
                            $"BepInEx/config/NOLoftFix.cfg and they apply on the next tick) | " +
                            $"MinCellTerminalSpeedMs={s.minCellTerminalSpeedMs:F0} " +
                            $"RejectFlatCells={s.rejectFlatCells} Enabled={s.enabled} " +
                            $"(these three need a restart)");
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
                st.needLookup = true;
                if (s.verbose)
                    log?.Invoke($"{st.id} deferring cell lookup: speed {m.speed:F0} m/s at Initialize " +
                                $"(Xf0={st.xf0:F0} m captured)");
            }
        }

        /// <summary>
        /// Zeroing loftAmount also stops ARHSeeker.SlowChecks from writing
        /// targetDist, and the GAME reads that field in two places that have
        /// nothing to do with lofting:
        ///
        ///   line 207:  returnStrength == 0 &amp;&amp; targetDist &lt; 5000f &amp;&amp; ECM &gt; 2f
        ///              -> with targetDist 0 this is true at ANY range
        ///   line 301:  jinkEvasion.amount &gt; 0 &amp;&amp; multipleInbound &amp;&amp; targetDist &gt; terminalRange
        ///              -> with targetDist 0 this is never true, so ApplyJink is dead
        ///              (the AAM-29 prefab sets jinkEvasion.amount = 0.2)
        ///
        /// So the mod has to keep the field alive itself. Same value SlowChecks
        /// would have computed, just not gated on loftAmount.
        /// </summary>
        public static void MaintainTargetDist(ARHSeeker seeker)
        {
            var m = seeker.missile;
            if (m == null) return;
            var pos = m.GlobalPosition();
            var kp = seeker.knownPos;
            float dx = FX(kp) - FX(pos), dy = FY(kp) - FY(pos), dz = FZ(kp) - FZ(pos);
            seeker.targetDist = Mathf.Sqrt(dx * dx + dy * dy + dz * dz);
        }

        /// <summary>Below this the launch speed is not yet meaningful for a lookup.</summary>
        private const float MinLookupSpeed = 50f;

        private static void DoLookup(LoftState st, LoftTable table, LoftSettings s,
                                     Action<string> log, float speed)
        {
            float rangeKm = st.xf0 / 1000f;
            st.cell = table.Lookup(rangeKm, st.h0Act, speed, st.hTNow, 0f,
                                   s.neighbours, s.idwPower, s.maxNormDistance);
            if (st.cell == null)
            {
                if (s_reports++ < MaxEngageReports)
                    log?.Invoke($"{st.id} NOT ENGAGED at {rangeKm:F1} km / {st.h0Act:F0} m / {speed:F0} m/s " +
                                $"/ tgt {st.hTNow:F0} m -- {table.LastReason}. Vanilla lofting stays.");
                return;
            }
            st.bump = st.cell.BumpCoeffs();
            st.bumpTarget = st.bump;
            st.hTLookup = st.hTNow;
            st.lookupSpeed = speed;
            st.engaged = true;
            if (s_reports++ < MaxEngageReports || s.verbose)
                log?.Invoke($"{st.id} ENGAGED at {rangeKm:F1} km / {st.h0Act:F0} m / {speed:F0} m/s " +
                            $"/ tgt {st.hTNow:F0} m (Xf0={st.xf0:F0} m, nearest cell " +
                            $"{table.LastDistance:F3}, table varies on {table.EffectiveAxes}/5 axes; " +
                            $"fixed: {table.DegenerateAxes})");
        }

        /// <summary>
        /// ARHSeeker.Seek postfix -- runs AFTER the game's own SetAimpoint, so it
        /// overwrites it. Returns false when the vanilla aimpoint should stand.
        /// </summary>
        public static bool Override(ARHSeeker seeker, LoftTable table, LoftSettings s, Action<string> log)
        {
            var st = Get(seeker);
            var m = seeker.missile;
            if (m == null) return false;

            if (st.needLookup && st.table != null && m.speed >= MinLookupSpeed)
            {
                st.needLookup = false;
                DoLookup(st, st.table, st.settings ?? s, st.log ?? log, m.speed);
            }
            if (!st.engaged || st.cell == null) return false;

            float dt = Mathf.Max(Time.fixedDeltaTime, 1e-3f);
            var pos = m.GlobalPosition();

            // ---- live PIP, with the same lead the game uses -----------------
            // knownPos is where the target IS; GetLeadVectorWithAccel is exactly
            // what ARHSeeker.Seek uses to turn that into an intercept point, so
            // routing through it keeps the lead instead of aiming behind a
            // crossing target.
            GlobalPosition pip = seeker.knownPos;
            try
            {
                Vector3 platformVel = (m.timeSinceSpawn < 3f)
                    ? m.transform.forward * seeker.topSpeed
                    : m.rb.velocity;
                pip = seeker.knownPos + TargetCalc.GetLeadVectorWithAccel(
                    seeker.knownPos, pos, seeker.knownVel, platformVel,
                    seeker.knownAccel, seeker.maxLead);
            }
            catch (Exception)
            {
                // lead unavailable -- fall back to the raw track rather than
                // giving up guidance entirely
            }

            // ---- has the PIP moved? -----------------------------------------
            float pipDx = FX(pip) - st.lastPipX, pipDz = FZ(pip) - st.lastPipZ;
            if (Mathf.Abs(pipDx) > s.pipChangeThresholdM || Mathf.Abs(pipDz) > s.pipChangeThresholdM)
            {
                st.lastPipX = FX(pip); st.lastPipZ = FZ(pip); st.lastPipY = FY(pip);
                st.pipRefreshes++;
            }

            // Terminal phase: inside terminalRange the seeker is radar-locked on
            // the real target with its own lead calculation, so get out of the way.
            //
            // DO NOT use seeker.targetDist here. ARHSeeker.SlowChecks computes it
            // INSIDE `if (loftAmount > 0f)` -- the same block this mod disables by
            // zeroing loftAmount -- so with the mod active it is never written and
            // sits at its default 0f. Gating on it made Override return false on
            // every tick: no loft, and apex/trackErr never updated. Compute the
            // slant range from the live PIP instead, which is also fresher than a
            // 1 Hz cached value.
            float slant = Distance(pos, pip);
            if (slant <= seeker.terminalRange)
            {
                // Hand back to the game's own seeker -- but keep logging. Every
                // dud so far has died in this window, and returning silently here
                // meant the last 12 km of every engagement was invisible: the
                // [flight] line just stopped and the next thing in the log was the
                // self-destruct record. This is where the miss actually happens, so
                // this is the part that has to be measured.
                if (s.verbose && m.timeSinceSpawn - st.lastTermLog >= s.flightLogIntervalS)
                {
                    st.lastTermLog = m.timeSinceSpawn;
                    Vector3 rel = ToV3(pip) - ToV3(pos);
                    float closing = -Vector3.Dot(rel.normalized, m.rb ? m.rb.velocity : Vector3.zero);
                    log?.Invoke($"{st.id} [terminal] t={m.timeSinceSpawn:F1}s slant={slant:F0}m " +
                                $"closing={closing:F0}m/s v={m.speed:F0} alt={FY(pos):F0} " +
                                $"tgt={FY(seeker.knownPos):F0} dh={FY(pos) - FY(seeker.knownPos):F0} " +
                                $"act={Mathf.Rad2Deg * Mathf.Asin(Mathf.Clamp(m.transform.forward.y, -1f, 1f)):F1}deg " +
                                $"mode={m.seekerMode}");
                }
                return false;
            }

            Vector3 dirNow = Flat(pos, pip);
            float xfNow = Mathf.Max(FlatDist(st.launchPos, pip), 500f);
            // Terminal altitude comes from the TRACK, not from the lead point.
            //
            // pip = knownPos + GetLeadVectorWithAccel(...), and with the AAM-29's
            // maxLead of 10 s against a 500-1400 m/s closing speed the lead's
            // VERTICAL component is huge and noisy. Taking FY(pip) as the target
            // altitude made it swing roughly 5 000 m in a single second -- a logged
            // round went +4400 / +577 / +38 / -320 / -468 / -793 / +218 / -829 --
            // which is lead noise, not target motion. That drove the reference
            // endpoint and churned the cell re-selection ~30 times per flight.
            //
            // Lead belongs in the horizontal aim (dirNow above and the aimpoint),
            // where it is doing the right job. The altitude the profile is built
            // against should be where the target actually is.
            st.hTNow = FY(seeker.knownPos);
            float hTNow = st.hTNow;
            if (st.hTInit) { st.hTUsed = hTNow; st.hTInit = false; }

            // One-shot rebase. SlowChecks refreshes knownPos at 1 Hz, so the value
            // at Initialize can be noticeably off; if the first live track
            // disagrees badly, adopt it once and then freeze xf0 for good.
            if (!st.rebased && m.timeSinceSpawn > 1.5f && m.timeSinceSpawn < s.rebaseWindowS)
            {
                st.rebased = true;
                if (Mathf.Abs(xfNow - st.xf0) > 0.25f * st.xf0)
                {
                    if (s.verbose)
                        log?.Invoke($"[rebase] Xf0 {st.xf0:F0} m -> {xfNow:F0} m from the live track");
                    st.xf0 = xfNow;
                    st.xfUsed = xfNow;
                    st.launchDir = Flat(st.launchPos, pip);
                }
            }

            // ---- guard 1: lag the normalisation ------------------------------
            // Without this, a target turning away grows xfNow, shrinks s, and the
            // missile is suddenly told it is earlier in the profile and must
            // climb -- a step command that pure pursuit chases hard.
            float maxStep = s.xfLagRate * dt;
            st.xfUsed += Mathf.Clamp(xfNow - st.xfUsed, -maxStep, maxStep);
            st.lastXfDrift = xfNow - st.xfUsed;

            // Cross-fade to a new profile shape if one was queued.
            // fade bump -> bumpTarget; lengths can differ between cells, so only
            // fade the overlap and snap anything beyond it.
            if (st.bumpTarget != null && st.bumpTarget.Length > 0)
            {
                float a = Mathf.Clamp01(dt / Mathf.Max(s.bumpFadeS, 1e-3f));
                if (st.bump == null || st.bump.Length != st.bumpTarget.Length)
                {
                    st.bump = (double[])st.bumpTarget.Clone();
                }
                else
                {
                    for (int i = 0; i < st.bump.Length; i++)
                        st.bump[i] += (st.bumpTarget[i] - st.bump[i]) * a;
                }
            }

            // ---- guard 1b: lag the ALTITUDE axis too ------------------------
            // Guard 1 above lags the range, because an unlagged xfNow turns a
            // datalink refresh into a step command. The altitude axis has exactly
            // the same problem and was never lagged: Ref() uses
            //     h0Act + (hTNow - h0Act) * s + B(s)
            // with hTNow = FY(pip), and knownPos refreshes at 1 Hz. A step in the
            // track therefore moves the reference by dHt * s -- half the step
            // mid-flight -- and pure pursuit chases it. The table is not the
            // source: the 4-cell IDW blend for a logged shot gives a smooth
            // monotone bump (0 / 10 669 / 14 093 / 11 666 / 0 m).
            float hStep = s.targetAltLagRate * dt;
            st.hTUsed += Mathf.Clamp(hTNow - st.hTUsed, -hStep, hStep);

            // ---- guard 2: progress along the FROZEN launch axis -------------
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
            Vector3 fromLaunch = new Vector3(FX(pos) - FX(st.launchPos), 0f,
                                             FZ(pos) - FZ(st.launchPos));
            float x = Mathf.Clamp(Vector3.Dot(fromLaunch, st.launchDir), 0f, st.xf0);

            // ---- reference here, and at the lookahead point ------------------
            float d = Mathf.Clamp(s.lookaheadSeconds * Mathf.Max(m.speed, 1f),
                                  s.minLookaheadM, s.maxLookaheadM);
            float xAp = Mathf.Min(x + d, st.xf0);
            double sNow = Mathf.Clamp01(x / st.xfUsed);
            double sAp = Mathf.Clamp01(xAp / st.xfUsed);

            double hRef = Ref(st, sNow);
            double hAp = Ref(st, sAp);

            // ---- tracking-error correction ----------------------------------
            float trackErr = (float)hRef - FY(pos);
            st.maxTrackErrM = Mathf.Max(st.maxTrackErrM, Mathf.Abs(trackErr));
            // Plain proportional trim. Its angular authority is bounded below, by
            // the total deviation clamp on the aimpoint -- clamping it here as well
            // would be redundant, and clamping it here ALONE was not enough, because
            // the steep commands came from the altitude error itself rather than
            // from this term.
            hAp += s.trackGain * trackErr;

            // ---- build the aimpoint -----------------------------------------
            // Aim along the line to the CURRENT pip, so the horizontal command
            // tracks a turning target too.
            GlobalPosition ap = pos + dirNow * d;
            float apY = Mathf.Max((float)hAp, s.floorAltM);

            // Bound the commanded ANGLE relative to the profile's own slope.
            //
            // The aimpoint sits d = clamp(3*V, 800, 15000) metres ahead, so ANY
            // altitude difference between the missile and the reference at that
            // point becomes atan(dh / d) of commanded flight path angle -- and d
            // shrinks as the missile slows. Measured on a 67 km round: 1115 m high
            // with 843 m of lookahead commanded -60 deg against a -22 deg profile,
            // which deepened the dive, bled more speed, and shortened d further.
            //
            // Clamping the TrackGain term did not touch this, because the
            // steepness comes from the altitude error itself and not from the
            // correction. This clamps the total deviation instead, so the missile
            // converges on the profile at a bounded rate whatever its speed.
            float apHoriz = Mathf.Sqrt((FX(ap) - FX(pos)) * (FX(ap) - FX(pos)) +
                                       (FZ(ap) - FZ(pos)) * (FZ(ap) - FZ(pos)));
            apHoriz = Mathf.Max(apHoriz, 1f);
            float slope = Mathf.Atan2((float)(hAp - hRef), Mathf.Max(d, 1f));
            float want = Mathf.Atan2(apY - FY(pos), apHoriz);
            st.wantDev = Mathf.Rad2Deg * (want - slope);
            if (s.maxTrackAngleDeg > 0f)
            {
                float lim = s.maxTrackAngleDeg * Mathf.Deg2Rad;
                float cmdAng = slope + Mathf.Clamp(want - slope, -lim, lim);
                apY = FY(pos) + Mathf.Tan(cmdAng) * apHoriz;
                ap.y = apY;
            }

            // Never command into the ground, and never put the aimpoint behind
            // or on top of the missile -- that trips ARHSeeker.MissedTarget()
            // and self-destructs the round.
            float minAhead = Mathf.Max(300f, 1.5f * m.speed);
            Vector3 fwd = m.rb.velocity.sqrMagnitude > 1f ? m.rb.velocity.normalized : m.transform.forward;
            Vector3 toAp = new Vector3(FX(ap) - FX(pos), apY - FY(pos), FZ(ap) - FZ(pos));
            float ahead = Vector3.Dot(toAp, fwd);
            if (ahead < minAhead) ap = pos + toAp + fwd * (minAhead - ahead);
            ap.y = apY;

            if (s.debugDraw)
                Draw(st, m, pos, ap, pip, sNow, fwd, s);

            m.SetAimpoint(ap, seeker.knownVel);
            if (!st.firstCommandLogged)
            {
                st.firstCommandLogged = true;
                if (s.verbose) log?.Invoke($"{st.id} [law] first aimpoint command at t={m.timeSinceSpawn:F1}s: " +
                            $"missile {FY(pos):F0} m, aimpoint {apY:F0} m, " +
                            $"ref here {(float)hRef:F0} m, progress {x / 1000f:F1}/{st.xfUsed / 1000f:F1} km");
                st.lastLogT = m.timeSinceSpawn;
            }

            st.apexM = Mathf.Max(st.apexM, FY(pos));
            st.termSpeed = m.speed;

            if (s.verbose && m.timeSinceSpawn - st.lastLogT > s.flightLogIntervalS)
            {
                st.lastLogT = m.timeSinceSpawn;
                // Every term that can move the reference, so an oscillation can be
                // attributed instead of guessed at:
                //   s      progress along the profile
                //   B      the bump, i.e. everything that is not the endpoint chord
                //   hT     live pip altitude vs the lagged one actually used
                //   xf     live range vs the lagged one actually used
                //   d      lookahead distance actually applied to the aimpoint
                // slope = the profile's own flight path angle over the lookahead.
                // cmd   = what the aimpoint actually commands, after the clamp.
                // act   = what the missile is doing.
                // dev   = how far the UNCLAMPED aimpoint wanted to deviate.
                // cmd far from act -> the airframe is not following the aimpoint.
                // dev pinned at the limit -> the missile is off the profile and
                //     recovering as fast as it is allowed to.
                // NOTE: do NOT compare cmd against slope directly -- cmd is measured
                // from the missile's actual altitude and slope from the reference, so
                // they differ by the altitude error even with a perfect aimpoint.
                float cmdDeg = Mathf.Rad2Deg * Mathf.Atan2(apY - FY(pos), apHoriz);
                float actDeg = Mathf.Rad2Deg * Mathf.Asin(Mathf.Clamp(fwd.y, -1f, 1f));
                float refDeg = Mathf.Rad2Deg * slope;
                log?.Invoke($"{st.id} [flight] t={m.timeSinceSpawn:F1}s s={sNow:F3} " +
                            $"alt={FY(pos):F0} ref={(float)hRef:F0} B={Poly.Eval(st.bump, sNow):F0} " +
                            $"hT={hTNow:F0}/{st.hTUsed:F0} xf={xfNow / 1000f:F1}/{st.xfUsed / 1000f:F1}km " +
                            $"v={m.speed:F0} d={d:F0} err={trackErr:F0} pip={st.pipRefreshes} " +
                            $"cell={st.cell.range_km:F0}km | slope={refDeg:F1} cmd={cmdDeg:F1} " +
                            $"act={actDeg:F1} dev={st.wantDev:F1} deg");
            }
            return true;
        }

        // Colours chosen to be distinguishable from the game's own green aim line.
        private static readonly Color ProfileCol = Color.cyan;     // where the law wants to be
        private static readonly Color AimCol = Color.magenta;      // what was handed to the seeker
        private static readonly Color VelCol = Color.yellow;       // where the missile is going

        private static Vector3 ToV3(GlobalPosition p)
            => new Vector3(FX(p), FY(p), FZ(p));

        /// <summary>
        /// Debug.DrawLine renders through the gizmo system, which a standalone player
        /// does not draw -- that is why the earlier version showed nothing while the
        /// game's own aim line, a LineRenderer, was visible. The game's DebugVis class
        /// is a prefab/marker manager (Enabled =&gt; PlayerSettings.debugVis), not a line
        /// drawer, so the path needs its own LineRenderer. Sprites/Default is used
        /// because it ships with essentially every Unity build.
        ///
        /// Width is 0.5 m: this is a diagnostic overlay read against the missile's
        /// own track, and a wide ribbon hides the very deviation it exists to show.
        /// </summary>
        private static bool EnsurePath(LoftState st)
        {
            try
            {
                var go = new GameObject("NOLoftFix_Path");
                UnityEngine.Object.DontDestroyOnLoad(go);
                var lr = go.AddComponent<LineRenderer>();
                lr.useWorldSpace = true;
                lr.startWidth = 0.5f;
                lr.endWidth = 0.5f;
                lr.numCapVertices = 2;
                lr.positionCount = 0;
                lr.startColor = ProfileCol;
                lr.endColor = ProfileCol;
                var sh = Shader.Find("Sprites/Default");
                if (sh != null) lr.material = new Material(sh);
                st.pathLr = lr;
                return true;
            }
            catch { return false; }
        }

        /// <summary>
        /// Draws the remaining reference profile in world space so it can be compared
        /// directly against the game's own aim line and the missile's actual track.
        ///
        /// The profile is laid out AHEAD OF THE MISSILE along the current launch->PIP
        /// axis, at (sFrac - sNow) * xfUsed metres, at altitude Ref(sFrac). That is the
        /// same mapping the law commands, so if the cyan line and the missile diverge
        /// the disagreement is in the airframe or the aimpoint plumbing, not in the
        /// reference.
        /// </summary>
        private static void Draw(LoftState st, Missile m, GlobalPosition pos, GlobalPosition ap,
                                 GlobalPosition pip, double sNow, Vector3 fwd,
                                 LoftSettings s)
        {
            Vector3 here = ToV3(pos);
            Vector3 axis = new Vector3(FX(pip) - FX(st.launchPos), 0f,
                                       FZ(pip) - FZ(st.launchPos));
            float len = axis.magnitude;

            if (len > 1f)
            {
                if (st.pathLr == null && !EnsurePath(st)) return;
                if (st.pathLr == null) return;

                // LineRenderer needs UNITY world positions, and GlobalPosition is the
                // game's own large-world coordinate. Anchor on the missile's transform
                // and apply profile offsets as deltas, which sidesteps the offset
                // entirely -- the profile only spans tens of km.
                Vector3 anchor = m.transform.position;
                // Lay the polyline along launchDir, the SAME axis progress is
                // measured on. Using the live launch->PIP direction instead shears
                // the line sideways whenever the target has moved.
                Vector3 u = st.launchDir;
                const int N = 28;
                var pts = new Vector3[N + 1];
                // Start on the REFERENCE, not on the missile. Anchoring pts[0] at the
                // missile made the first segment span the whole tracking error, which
                // read as a kink or zigzag at the apex; starting on the reference makes
                // the missile's offset from the line directly visible instead.
                for (int i = 0; i <= N; i++)
                {
                    double sf = sNow + (1.0 - sNow) * i / N;
                    float along = (float)(sf - sNow) * st.xfUsed;
                    pts[i] = anchor + new Vector3(u.x * along,
                                                  (float)Ref(st, sf) - FY(pos),
                                                  u.z * along);
                }
                st.pathLr.positionCount = N + 1;
                st.pathLr.SetPositions(pts);
            }
        }

        
        /// <summary>h(s) = h_launch + (h_pip_now - h_launch)*s + B(s), evaluated
        /// with the endpoints held by the caller in hTNow. Kept inline because it
        /// needs the live target altitude.</summary>
        private static double Ref(LoftState st, double sFrac)
        {
            return st.h0Act + (st.hTUsed - st.h0Act) * sFrac + Poly.Eval(st.bump, sFrac);
        }
    }
}
