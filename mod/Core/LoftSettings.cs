using BepInEx.Configuration;

namespace NOLoftFix.Core;

public sealed class LoftSettings
{
    private ConfigEntry<bool> _enabled;
    private ConfigEntry<bool> _verbose;
    private ConfigEntry<float> _maxNormDistance;
    private ConfigEntry<float> _lookaheadSeconds;
    private ConfigEntry<float> _trackGain;
    private ConfigEntry<float> _maxTrackAngleDeg;
    private ConfigEntry<float> _floorAltM;
    private ConfigEntry<float> _targetAltLagRate;
    private ConfigEntry<float> _flightLogIntervalS;
    private ConfigEntry<bool> _debugDraw;

    public bool Enabled => _enabled.Value;
    public bool Verbose => _verbose.Value;
    public float MaxNormDistance => _maxNormDistance.Value;
    public float LookaheadSeconds => _lookaheadSeconds.Value;
    public float TrackGain => _trackGain.Value;
    public float MaxTrackAngleDeg => _maxTrackAngleDeg.Value;
    public float FloorAltM => _floorAltM.Value;
    public float TargetAltLagRate => _targetAltLagRate.Value;
    public float FlightLogIntervalS => _flightLogIntervalS.Value;
    public bool DebugDraw => _debugDraw.Value;

    public float MinCellTerminalSpeedMs { get; private set; }
    public bool RejectFlatCells { get; private set; }

    // constants
    /// <summary>Cells blended when a shot falls between grid points.</summary>
    public readonly int Neighbours = 8;
    /// <summary>Inverse-distance exponent for the cell blend.</summary>
    public readonly float IdwPower = 2f;
    /// <summary>Aimpoint lookahead is clamped to this range, metres.</summary>
    public readonly float MinLookaheadM = 800f;
    public readonly float MaxLookaheadM = 15000f;
    /// <summary>PIP motion beyond this counts as a refresh (diagnostic).</summary>
    public readonly float PipChangeThresholdM = 1000f;
    /// <summary>How fast the downrange normalisation may chase the live track, m/s.</summary>
    public readonly float XfLagRate = 1500f;
    /// <summary>Cross-fade time constant for a profile shape change, s.</summary>
    public readonly float BumpFadeS = 1.5f;
    /// <summary>Window in which the launch track may correct Xf0 once, s.</summary>
    public readonly float RebaseWindowS = 6f;

    public static LoftSettings Bind(ConfigFile cfg)
    {
        LoftSettings s = new();
        const string gen = "General", look = "Lookup", fol = "Following";

        s._enabled = cfg.Bind(gen, "Enabled", true,
            "Checked before patching, therefore it needs a restart to apply.");

        s._verbose = cfg.Bind(gen, "VerboseLogging", false,
            "It will log more if you enable this.");

        s._maxNormDistance = cfg.Bind(look, "MaxNormalisedDistance", 0.5f,
            "Beyond this normalised distance from a " +
            "solved cell, the mod stops engaging.");

        s._lookaheadSeconds = cfg.Bind(fol, "LookaheadSeconds", 3f,
            "Aimpoint is placed this many seconds of flight ahead along the " +
            "reference profile.");

        s._trackGain = cfg.Bind(fol, "TrackGain", 0.5f,
            "Proportional gain on altitude tracking error, 0..1.");

        s._maxTrackAngleDeg = cfg.Bind(fol, "MaxTrackAngleDeg", 12f,
            "How many degrees the aimpoint may deviate from the profile's " +
            "slope.");

        s._floorAltM = cfg.Bind(fol, "FloorAltM", 60f,
            "The aimpoint is never placed below this altitude (m).");

        s._targetAltLagRate = cfg.Bind(fol, "TargetAltLagRateMPerS", 250f,
            "How fast the target altitude used to build the reference may move, m/s.");

        s._flightLogIntervalS = cfg.Bind(fol, "FlightLogIntervalS", 1f,
            "Seconds between [flight] and [terminal] lines.");

        s._debugDraw = cfg.Bind(fol, "DebugDraw", false,
            "Draw the remaining reference profile with a LineRenderer. Quite janky.");

        s.MinCellTerminalSpeedMs = cfg.Bind(gen, "MinCellTerminalSpeedMs", 0f,
            "Table cells predicting a terminal speed below this are dropped at " +
            "load. Restart required to apply.").Value;

        s.RejectFlatCells = cfg.Bind(gen, "RejectFlatCells", false,
            "This is really dumb, don't enable. Restart required to apply.").Value;

        return s;
    }
}
