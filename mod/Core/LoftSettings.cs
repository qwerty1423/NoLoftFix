using BepInEx.Configuration;

namespace NOLoftFix
{
    /// <summary>
    /// Tunables. The ones worth tuning are bound to BepInEx config and read live
    /// through their ConfigEntry, so editing BepInEx/config/NOLoftFix.cfg takes
    /// effect on the next tick without restarting the game.
    ///
    /// Everything else is a plain constant. It used to be configurable too, which
    /// meant 24 keys in the file and no way to tell the two knobs that matter from
    /// the internals nobody had ever changed. Two exceptions worth knowing about:
    ///
    ///   - MinCellTerminalSpeedMs and RejectFlatCells are read once when the table
    ///     loads, so those two DO need a restart. ReloadTables() is the hook if
    ///     that ever needs to change.
    ///   - `Enabled` is checked before patching, so it also needs a restart.
    /// </summary>
    public sealed class LoftSettings
    {
        // ---- live, config-backed ------------------------------------------
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

        /// <summary>Checked before patching, so changing it needs a restart.</summary>
        public bool enabled => _enabled.Value;
        public bool verbose => _verbose.Value;
        public float maxNormDistance => _maxNormDistance.Value;
        public float lookaheadSeconds => _lookaheadSeconds.Value;
        public float trackGain => _trackGain.Value;
        public float maxTrackAngleDeg => _maxTrackAngleDeg.Value;
        public float floorAltM => _floorAltM.Value;
        public float targetAltLagRate => _targetAltLagRate.Value;
        public float flightLogIntervalS => _flightLogIntervalS.Value;
        public bool debugDraw => _debugDraw.Value;

        /// <summary>Read at table-load time only; changing it needs a restart.</summary>
        public float minCellTerminalSpeedMs { get; private set; }
        /// <summary>Read at table-load time only; changing it needs a restart.</summary>
        public bool rejectFlatCells { get; private set; }

        // ---- constants: real, but never worth a config key ----------------
        /// <summary>Cells blended when a shot falls between grid points.</summary>
        public readonly int neighbours = 8;
        /// <summary>Inverse-distance exponent for the cell blend.</summary>
        public readonly float idwPower = 2f;
        /// <summary>Aimpoint lookahead is clamped to this range, metres.</summary>
        public readonly float minLookaheadM = 800f;
        public readonly float maxLookaheadM = 15000f;
        /// <summary>PIP motion beyond this counts as a refresh (diagnostic).</summary>
        public readonly float pipChangeThresholdM = 1000f;
        /// <summary>How fast the downrange normalisation may chase the live track, m/s.</summary>
        public readonly float xfLagRate = 1500f;
        /// <summary>Cross-fade time constant for a profile shape change, s.</summary>
        public readonly float bumpFadeS = 1.5f;
        /// <summary>Window in which the launch track may correct Xf0 once, s.</summary>
        public readonly float rebaseWindowS = 6f;

        public static LoftSettings Bind(ConfigFile cfg)
        {
            var s = new LoftSettings();
            const string gen = "General", look = "Lookup", fol = "Following";

            s._enabled = cfg.Bind(gen, "Enabled", true,
                "Master kill switch, checked before patching. Changing it needs a restart.");

            s._verbose = cfg.Bind(gen, "VerboseLogging", false,
                "Emit the [flight], [terminal] and [law] lines. Live.");

            s._maxNormDistance = cfg.Bind(look, "MaxNormalisedDistance", 0.5f,
                "Beyond this normalised distance from any solved cell the mod declines " +
                "to engage and vanilla lofting stays. Raise it if lookups start failing " +
                "after cells are filtered out. Live.");

            s._lookaheadSeconds = cfg.Bind(fol, "LookaheadSeconds", 3f,
                "Aimpoint is placed this many seconds of flight ahead along the " +
                "reference profile. THE main tuning knob: too small and the missile " +
                "cuts corners, too large and it lags the profile. Live.");

            s._trackGain = cfg.Bind(fol, "TrackGain", 0.5f,
                "Proportional gain on altitude tracking error, 0..1. 0 = pure " +
                "lookahead following, 1 = command the full error away. Live.");

            s._maxTrackAngleDeg = cfg.Bind(fol, "MaxTrackAngleDeg", 12f,
                "How many degrees the aimpoint may deviate from the profile's own " +
                "slope. The aimpoint sits d = clamp(3*V, 800, 15000) m ahead, so ANY " +
                "altitude error becomes atan(dh/d) of commanded angle and d shrinks " +
                "as the missile slows -- measured: 1115 m high with 843 m of lookahead " +
                "commanded -60 deg against a -22 deg profile, and the round decayed to " +
                "193 m/s. This bounds the recovery rate instead. Smaller is gentler but " +
                "converges slower; 0 disables. Live.");

            s._floorAltM = cfg.Bind(fol, "FloorAltM", 60f,
                "The aimpoint is never placed below this absolute altitude (m). The " +
                "mod has no terrain query. Live.");

            s._targetAltLagRate = cfg.Bind(fol, "TargetAltLagRateMPerS", 250f,
                "How fast the target altitude used to build the reference may move, " +
                "m/s. It must stay ABOVE the target's own vertical rate or the " +
                "reference endpoint falls permanently behind: on a target diving at " +
                "~122 m/s the old 60 m/s default let hTUsed drift 2 400 m off the real " +
                "target. 0 disables. Live.");

            s._flightLogIntervalS = cfg.Bind(fol, "FlightLogIntervalS", 1f,
                "Seconds between [flight] and [terminal] lines. Live.");

            s._debugDraw = cfg.Bind(fol, "DebugDraw", false,
                "Draw the remaining reference profile with a LineRenderer. Live.");

            s.minCellTerminalSpeedMs = cfg.Bind(gen, "MinCellTerminalSpeedMs", 0f,
                "Table cells predicting a terminal speed below this are dropped at " +
                "load. 130 of the 252 cells in the current table predict under " +
                "800 m/s. 0 disables. RESTART REQUIRED -- the table loads once.").Value;

            s.rejectFlatCells = cfg.Bind(gen, "RejectFlatCells", false,
                "Drop cells whose apex equals their launch altitude: they never " +
                "lofted, and their polynomial fit looks perfect because a flat " +
                "reference fits a flat answer. 61 of 252 cells are like this. " +
                "RESTART REQUIRED -- the table loads once.").Value;

            return s;
        }
    }
}
