using System;
using System.Collections.Generic;

using HarmonyLib;

using JetBrains.Annotations;

namespace NOLoftFix.Core;

/// <summary>
/// Picks which loft table applies to the missile. Prefers an exact
/// name match, then "default", then the only table if exactly one is loaded.
/// </summary>
internal static class TableRouter
{
    public static LoftTable For(Dictionary<string, LoftTable> tables, ARHSeeker seeker)
    {
        if (tables == null || tables.Count == 0)
        {
            return null;
        }

        string nm = seeker?.missile != null ? seeker.missile.name : "";
        foreach (KeyValuePair<string, LoftTable> kv in tables)
        {
            if (kv.Key != "default" && nm.IndexOf(kv.Key, StringComparison.OrdinalIgnoreCase) >= 0)
            {
                return kv.Value;
            }
        }

        if (tables.TryGetValue("default", out LoftTable d))
        {
            return d;
        }

        return null;
    }
}

/// <summary>
/// Removes vanilla lofting if a loft table is available.
/// </summary>
[HarmonyPatch(typeof(ARHSeeker), nameof(ARHSeeker.Initialize))]
internal static class SeekerInitializePatch
{
    [UsedImplicitly]
    private static void Postfix(ARHSeeker __instance)
    {
        if (!Plugin.Ready || __instance.missile?.LocalSim != true)
        {
            return;
        }

        try
        {
            LoftTable table = TableRouter.For(Plugin.Tables, __instance);
            if (table == null)
            {
                return;
            }

            // OnInitialize disables vanilla loft only after finding a cell
            LoftGuidance.OnInitialize(__instance, table, Plugin.Settings, Plugin.Log);
        }
        catch (Exception e)
        {
            LoftGuidance.RestoreVanilla(__instance);
            Plugin.Fail("Initialize", e);
        }
    }
}

/// <summary>
/// Replaces the aimpoint.
/// </summary>
[HarmonyPatch(typeof(ARHSeeker), nameof(ARHSeeker.Seek))]
internal static class SeekerSeekPatch
{
    [UsedImplicitly]
    [HarmonyPostfix]
    private static void Postfix(ARHSeeker __instance)
    {
        if (!Plugin.Ready || __instance.missile?.LocalSim != true)
            return;

        try
        {
            LoftGuidance.Override(__instance, Plugin.Settings, Plugin.Log);
        }
        catch (Exception e)
        {
            LoftGuidance.RestoreVanilla(__instance);
            Plugin.Fail("Seek.Postfix", e);
        }
    }
}

[HarmonyPatch(typeof(ARHSeeker), nameof(ARHSeeker.SlowChecks))]
internal static class SlowChecksTelemetryPatch
{
    [UsedImplicitly]
    [HarmonyPrefix]
    private static void Prefix(ARHSeeker __instance)
    {
        if (__instance.missile?.LocalSim != true ||
            !LoftGuidance.TryGet(__instance, out LoftState st) || !st.LoftZeroed)
            return;

        if (!Plugin.Ready)
        {
            LoftGuidance.RestoreVanilla(__instance);
            return;
        }

        try
        {
            // Vanilla computes targetDist from SlowChecks once a second,
            // but only while loftAmount > 0. Retain that cadence.
            LoftGuidance.MaintainTargetDist(__instance);
        }
        catch (Exception e)
        {
            LoftGuidance.RestoreVanilla(__instance);
            Plugin.Fail("SlowChecks.Prefix", e);
        }
    }

    [UsedImplicitly]
    [HarmonyPostfix]
    private static void Postfix(ARHSeeker __instance)
    {
        if (!Plugin.Ready || !Plugin.Settings.Verbose ||
            __instance.missile?.LocalSim != true)
        {
            return;
        }

        try
        {
            if (!LoftGuidance.TryGet(__instance, out LoftState st) || !st.Engaged)
            {
                return;
            }

            Missile m = __instance.missile;
            if (m == null)
            {
                return;
            }

            Plugin.Log($"[telemetry] t={m.timeSinceSpawn:F1}s alt={m.GlobalPosition().y:F0}m " +
                       $"v={m.speed:F0}m/s apex={st.ApexM:F0}m " +
                       $"maxTrackErr={st.MaxTrackErrM:F0}m pipRefreshes={st.PipRefreshes}");
        }
        catch
        {
            // ignored
        }
    }
}
