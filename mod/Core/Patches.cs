using System;
using System.Collections.Generic;

using HarmonyLib;

using UnityEngine;

namespace NOLoftFix
{
    /// <summary>Picks which loaded table applies to this missile. Prefers an exact
    /// name match, then "default", then the only table if exactly one is loaded.</summary>
    internal static class TableRouter
    {
        public static LoftTable For(Dictionary<string, LoftTable> tables, ARHSeeker seeker)
        {
            if (tables == null || tables.Count == 0) return null;
            string nm = seeker?.missile != null ? seeker.missile.name : "";
            foreach (var kv in tables)
                if (kv.Key != "default" && nm.IndexOf(kv.Key, StringComparison.OrdinalIgnoreCase) >= 0)
                    return kv.Value;
            if (tables.TryGetValue("default", out var d)) return d;
            if (tables.Count == 1) { foreach (var kv in tables) return kv.Value; }
            return null;
        }
    }

    /// <summary>
    /// Kills the vanilla loft law.
    ///
    /// ARHSeeker.Seek() has `if (loftAmount > 0f) { ... leadVectorWithAccel += num * Vector3.up; }`
    /// and ARHSeeker.SlowChecks() gates its own loft bookkeeping on the same field.
    /// Setting it to 0 disables BOTH from one hook, which is why this is a
    /// one-line postfix rather than a transpiler over Seek().
    /// </summary>
    [HarmonyPatch(typeof(ARHSeeker), nameof(ARHSeeker.Initialize))]
    internal static class SeekerInitializePatch
    {
        // No `aimpoint` parameter here on purpose: Missile.LocalStart passes
        // ownerPos + forward*100000f, a placeholder the game itself ignores.
        private static void Postfix(ARHSeeker __instance)
        {
            if (!Plugin.Ready) return;
            try
            {
                // do this even when no table matched: a zeroed loftAmount plus no
                // engaged profile is exactly "vanilla aimpoint minus the bias",
                // and Override() returns false so the game's flat pursuit stays.
                // We only want that when we actually have a profile, so gate it.
                var table = TableRouter.For(Plugin.Tables, __instance);
                if (table == null) return;

                __instance.loftAmount = 0f;      // private, visible via Publicizer
                LoftGuidance.Get(__instance).loftZeroed = true;
                LoftGuidance.OnInitialize(__instance, table, Plugin.Settings, Plugin.Log);
            }
            catch (Exception e)
            {
                Plugin.Fail("Initialize", e);
            }
        }
    }

    /// <summary>
    /// Replaces the aimpoint. Runs AFTER Seek() has finished, so it overwrites
    /// whatever missile.SetAimpoint() was last called with.
    /// </summary>
    [HarmonyPatch(typeof(ARHSeeker), nameof(ARHSeeker.Seek))]
    internal static class SeekerSeekPatch
    {
        private static void Postfix(ARHSeeker __instance)
        {
            if (!Plugin.Ready) return;
            try
            {
                // Must run whether or not the profile engaged: zeroing loftAmount
                // stops SlowChecks computing targetDist, and the game reads that
                // field for jamming and jink-evasion decisions.
                LoftGuidance.MaintainTargetDist(__instance);
                LoftGuidance.Override(__instance, TableRouter.For(Plugin.Tables, __instance),
                                      Plugin.Settings, Plugin.Log);
            }
            catch (Exception e)
            {
                Plugin.Fail("Seek", e);
            }
        }
    }

    /// <summary>
    /// Telemetry. Postfix on the self-destruct path so every dud gets recorded
    /// with the reason -- this is how the acceptance criteria in the plan
    /// (no MissedTarget/LosingGround duds while above profile) get measured.
    /// Off by default; costs a Debug.Log per detonation.
    /// </summary>
    [HarmonyPatch(typeof(ARHSeeker), nameof(ARHSeeker.SlowChecks))]
    internal static class SlowChecksTelemetryPatch
    {
        private static void Postfix(ARHSeeker __instance)
        {
            if (!Plugin.Ready || !Plugin.Settings.verbose) return;
            try
            {
                var st = LoftGuidance.Get(__instance);
                if (!st.engaged) return;
                var m = __instance.missile;
                if (m == null) return;
                Plugin.Log($"[telemetry] t={m.timeSinceSpawn:F1}s alt={(float)m.GlobalPosition().y:F0}m " +
                           $"v={m.speed:F0}m/s apex={st.apexM:F0}m " +
                           $"maxTrackErr={st.maxTrackErrM:F0}m pipRefreshes={st.pipRefreshes}");
            }
            catch { /* telemetry must never break the shot */ }
        }
    }
}
