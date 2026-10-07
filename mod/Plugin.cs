using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;

using BepInEx;
using BepInEx.Logging;

using HarmonyLib;

using JetBrains.Annotations;

using NOLoftFix.Core;

using NuclearOption.Networking;

namespace NOLoftFix;

/// <summary>
/// Reference-profile midcourse lofting for Nuclear Option ARH missiles.
/// </summary>
[BepInPlugin(Guid, Name, Version)]
public class Plugin : BaseUnityPlugin
{
    public const string Guid = MyPluginInfo.PLUGIN_GUID;
    public const string Name = MyPluginInfo.PLUGIN_NAME;
    public const string Version = MyPluginInfo.PLUGIN_VERSION;

    internal static new ManualLogSource Logger;

    private const int MaxFailures = 5;
    private static int s_failures;
    private static string s_lastError = "";

    public static bool Ready { get; private set; }

    public static LoftSettings Settings { get; private set; }
    public static Dictionary<string, LoftTable> Tables { get; private set; }
        = new(StringComparer.OrdinalIgnoreCase);

    private static Harmony s_harmony;

    public static void Log(string msg) =>
        Logger?.LogInfo($"{msg}");

    /// <summary>Counts failures and disables the mod.</summary>
    /// <param name="where"></param>
    /// <param name="e"></param>
    public static void Fail(string where, Exception e)
    {
        s_failures++;
        string key = e.GetType().Name;
        if (key != s_lastError)
        {
            s_lastError = key;
            Logger?.LogError($"{where} failed: {e}");
        }
        if (s_failures >= MaxFailures && Ready)
        {
            Ready = false;
            Logger?.LogError($"{s_failures} failures, the mod will now disable itself.");
        }
    }

    [UsedImplicitly]
    private void Awake()
    {
        try
        {
            Logger = base.Logger;
            Settings = LoftSettings.Bind(Config);
            if (!Settings.Enabled)
            {
                Logger.LogInfo("disabled in config.");
                return;
            }

            string dir = Path.Combine(
                Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location) ?? ".", "Tables");
            Tables = LoftTable.LoadAll(dir, static m => Logger.LogInfo($"{m}"), Settings.MinCellTerminalSpeedMs, Settings.RejectFlatCells);

            if (Tables.Count == 0)
            {
                Logger.LogWarning(
                    "no loft tables loaded -- the mod will not engage. " +
                    $"Expected *.json under {dir}. Generate them with sim/grid_dymos.py " +
                    "or sim/build_tables.py.");
                return;
            }

            s_harmony = new Harmony(Guid);
            s_harmony.PatchAll(Assembly.GetExecutingAssembly());
            Ready = true;
            StartConfigWatch();

            int total = 0;
            foreach (KeyValuePair<string, LoftTable> kv in Tables)
            {
                total += kv.Value.Count;
                Logger.LogInfo($"table \"{kv.Key}\": {kv.Value.Count} cells, " +
                               $"varies on {kv.Value.EffectiveAxes}/5 axes " +
                               $"(fixed: {kv.Value.DegenerateAxes})");
            }
            Logger.LogInfo($"{Version} active: {Tables.Count} table(s), {total} cells");
        }
        catch (Exception e)
        {
            Ready = false;
            Logger.LogError($"failed to initialise: {e}");
        }
    }

    // live config reload
    private FileSystemWatcher _cfgWatcher;
    private volatile bool _reloadPending;
    private bool _moddedFlagSet;

    private void StartConfigWatch()
    {
        try
        {
            string file = Config.ConfigFilePath;
            string dir = Path.GetDirectoryName(file);
            if (string.IsNullOrEmpty(dir) || !Directory.Exists(dir))
            {
                return;
            }

            _cfgWatcher = new FileSystemWatcher(dir, Path.GetFileName(file))
            {
                NotifyFilter = NotifyFilters.LastWrite | NotifyFilters.Size,
                EnableRaisingEvents = true,
            };
            _cfgWatcher.Changed += (_, __) => _reloadPending = true;
        }
        catch (Exception e)
        {
            Logger.LogWarning($"config auto-reload error: {e.Message}");
        }
    }

    [UsedImplicitly]
    private void Update()
    {
        if (!_moddedFlagSet && Ready)
        {
            var networkManager = FindObjectOfType<NetworkManagerNuclearOption>();
            if (networkManager != null)
            {
                networkManager.SetModdedServer(true);
                _moddedFlagSet = true;
            }
        }

        if (!_reloadPending)
        {
            return;
        }

        _reloadPending = false;

        try
        {
            Config.Reload();
            Logger.LogInfo("config reloaded from disk.");
        }
        catch (Exception e)
        {
            Logger.LogWarning($"config reload failed: {e.Message}");
        }
    }
}
