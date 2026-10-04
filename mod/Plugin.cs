using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;

using BepInEx;
using BepInEx.Logging;

using HarmonyLib;

namespace NOLoftFix
{
    /// <summary>
    /// Reference-profile midcourse lofting for Nuclear Option ARH missiles.
    ///
    /// Problem: ARHSeeker.Seek() computes its loft bias as
    ///     min(t_go^2 * 4.905 * loftAmount, targetDist * loftAmount)
    /// with loftAmount = 0.7 on the AAM-29. That bias is a function of time-to-go
    /// ALONE, so it swings from "aim below the launch altitude" to "aim 40 km up"
    /// as range changes, and it is recomputed from a t_go that is refreshed at
    /// 1 Hz and then decremented every tick. The result is the behaviour reported
    /// in game: no loft under ~45 nmi, a working loft in the 50-65 nmi band, and
    /// over-lofting beyond it.
    ///
    /// Fix: replace the bias with an offline-optimised reference trajectory
    /// (CEAS-GNC-2026-016, augmented pursuit guidance), followed by placing the
    /// aimpoint on the profile. See README.md and ../PLAN-apg-loft-mod.md.
    /// </summary>
    [BepInPlugin(Guid, Name, Version)]
    public class Plugin : BaseUnityPlugin
    {
        public const string Guid = MyPluginInfo.PLUGIN_GUID;
        public const string Name = MyPluginInfo.PLUGIN_NAME;
        public const string Version = MyPluginInfo.PLUGIN_VERSION;

        /// <summary>Shadows BaseUnityPlugin.Logger (an instance property) so the
        /// static patch code can log without a reference to the plugin object.
        /// Same trick the reference NOAutopilot mod uses.</summary>
        internal static new ManualLogSource Logger;

        private const int MaxFailures = 5;
        private static int s_failures;
        private static string s_lastError = "";

        /// <summary>False once the mod has disabled itself. Every patch checks this
        /// first, so a broken table degrades to "vanilla behaviour" rather than
        /// to "missiles stop guiding".</summary>
        public static bool Ready { get; private set; }

        public static LoftSettings Settings { get; private set; }
        public static Dictionary<string, LoftTable> Tables { get; private set; }
            = new(StringComparer.OrdinalIgnoreCase);

        private static Harmony s_harmony;

        public static void Log(string msg) =>
            Logger?.LogInfo($"[NOLoftFix] {msg}");

        /// <summary>Counts failures and disarms the mod rather than throwing every
        /// tick from inside the guidance loop.</summary>
        public static void Fail(string where, Exception e)
        {
            s_failures++;
            string key = e.GetType().Name;
            if (key != s_lastError)
            {
                s_lastError = key;
                Logger?.LogError($"[NOLoftFix] {where} failed: {e}");
            }
            if (s_failures >= MaxFailures && Ready)
            {
                Ready = false;
                Logger?.LogError($"[NOLoftFix] {s_failures} consecutive failures -- " +
                                 "DISABLING ITSELF. Vanilla lofting resumes. Check the errors above.");
            }
        }

        private void Awake()
        {
            try
            {
                Logger = base.Logger;
                Settings = LoftSettings.Bind(Config);
                if (!Settings.enabled)
                {
                    Logger.LogInfo("[NOLoftFix] disabled in config; not patching.");
                    return;
                }

                string dir = Path.Combine(
                    Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location) ?? ".", "Tables");
                Tables = LoftTable.LoadAll(dir, m => Logger.LogInfo($"[NOLoftFix] {m}"), Settings.minCellTerminalSpeedMs, Settings.rejectFlatCells);

                if (Tables.Count == 0)
                {
                    Logger.LogWarning(
                        "[NOLoftFix] no loft tables loaded -- the mod will not engage. " +
                        $"Expected *.json under {dir}. Generate them with sim/grid_dymos.py " +
                        "or sim/build_tables.py.");
                    return;
                }

                s_harmony = new Harmony(Guid);
                s_harmony.PatchAll(Assembly.GetExecutingAssembly());
                Ready = true;
                StartConfigWatch();

                int total = 0;
                foreach (var kv in Tables)
                {
                    total += kv.Value.Count;
                    Logger.LogInfo($"[NOLoftFix]   table \"{kv.Key}\": {kv.Value.Count} cells, " +
                                   $"varies on {kv.Value.EffectiveAxes}/5 axes " +
                                   $"(fixed: {kv.Value.DegenerateAxes})");
                }
                Logger.LogInfo($"[NOLoftFix] {Version} active: {Tables.Count} table(s), {total} cells, patched. " +
                               "Look for ENGAGED / NOT ENGAGED lines when you fire.");
            }
            catch (Exception e)
            {
                Ready = false;
                Logger.LogError($"[NOLoftFix] failed to initialise, leaving the game alone: {e}");
            }
        }

        // ---- live config reload -------------------------------------------
        //
        // Every tunable used to be snapshotted into plain fields at Awake, so
        // editing NOLoftFix.cfg meant restarting the game to see any effect. The
        // settings are now read through their ConfigEntry, so all that is needed is
        // for something to notice the file changed and re-read it.

        private FileSystemWatcher _cfgWatcher;
        private volatile bool _reloadPending;

        private void StartConfigWatch()
        {
            try
            {
                string file = Config.ConfigFilePath;
                string dir = Path.GetDirectoryName(file);
                if (string.IsNullOrEmpty(dir) || !Directory.Exists(dir)) return;

                _cfgWatcher = new FileSystemWatcher(dir, Path.GetFileName(file))
                {
                    NotifyFilter = NotifyFilters.LastWrite | NotifyFilters.Size,
                    EnableRaisingEvents = true,
                };
                _cfgWatcher.Changed += (_, __) => _reloadPending = true;
                Logger.LogInfo($"[NOLoftFix] watching {file} -- edits apply without a restart.");
            }
            catch (Exception e)
            {
                Logger.LogWarning($"[NOLoftFix] config auto-reload unavailable: {e.Message}");
            }
        }

        private void Update()
        {
            if (!_reloadPending) return;
            _reloadPending = false;

            // The watcher fires on a thread-pool thread, while the guidance law
            // reads ConfigEntry.Value on the main thread -- so reload here, not in
            // the callback. Editors also write in several bursts; coalescing to one
            // reload per frame is fine and a redundant reload is harmless.
            try
            {
                Config.Reload();
                Logger.LogInfo("[NOLoftFix] config reloaded from disk; new values are live.");
            }
            catch (Exception e)
            {
                Logger.LogWarning($"[NOLoftFix] config reload failed: {e.Message}");
            }
        }
    }
}
