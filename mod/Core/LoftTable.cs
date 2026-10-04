using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;

using UnityEngine;

namespace NOLoftFix
{
    /// <summary>
    /// One solved cell of the offline grid. Field names MUST match the JSON keys
    /// produced by sim/build_tables.py and sim/grid_dymos.py -- JsonUtility binds
    /// by name and silently leaves unmatched fields at their default.
    /// </summary>
    [Serializable]
    public sealed class LoftCell
    {
        // --- grid coordinates, ALL METRIC ------------------------------------
        public float range_km;
        public float launch_alt_m;
        public float launch_speed_ms;
        public float target_alt_m;
        public float target_speed_ms;

        // --- legacy imperial keys, read-only compatibility --------------------
        // Tables generated before the metric switch still load. Normalise() moves
        // these into the metric fields; they are never written back out.
        public float range_nmi;
        public float launch_alt_ft;
        public float launch_kt;
        public float target_alt_ft;
        public float target_kt;

        /// <summary>Converts legacy imperial fields into the metric ones. Called
        /// once per cell right after parsing.</summary>
        public void Normalise()
        {
            if (range_km <= 0f && range_nmi > 0f) range_km = range_nmi * 1.852f;
            if (launch_alt_m <= 0f && launch_alt_ft > 0f) launch_alt_m = launch_alt_ft * 0.3048f;
            if (launch_speed_ms <= 0f && launch_kt > 0f) launch_speed_ms = launch_kt * 0.514444f;
            if (target_alt_m <= 0f && target_alt_ft > 0f) target_alt_m = target_alt_ft * 0.3048f;
            if (target_speed_ms <= 0f && target_kt > 0f) target_speed_ms = target_kt * 0.514444f;
            // target_alt 0 is legitimate (sea level), so only migrate when the
            // legacy key was actually present in the file.
        }

        // --- the reference profile -------------------------------------------
        /// <summary>Coefficients of h(xi) = sum_k poly_a[k] * xi^k, xi = x / X_f.
        /// Altitude in METRES, absolute (not AGL). poly_order is always 7.</summary>
        public double[] poly_a;

        // --- diagnostics, informational only ---------------------------------
        public double apex_alt_m;
        public double apex_downrange_m;
        public double terminal_speed_ms;
        public double tof_s;
        public double fit_rmse_m;
        public bool converged;
        public string missile = "";

        /// <summary>
        /// The profile as a pure BUMP: poly(xi) minus the straight line joining its
        /// own two endpoints. B(0) = B(1) = 0 by construction, because the offline
        /// solve pins both ends.
        ///
        /// This is what makes midcourse retargeting possible. The runtime reference
        /// is rebuilt as
        ///
        ///     h(xi) = h_launch + (h_pip_now - h_launch) * xi + B(xi)
        ///
        /// so the solved SHAPE is kept while the endpoints follow the live PIP.
        /// Both ends stay exact no matter where the target has moved to.
        /// </summary>
        public double[] BumpCoeffs()
        {
            if (poly_a == null || poly_a.Length == 0) return new double[0];
            double h0 = poly_a[0];
            double h1 = 0.0;
            for (int i = 0; i < poly_a.Length; i++) h1 += poly_a[i];
            var b = (double[])poly_a.Clone();
            b[0] = 0.0;
            if (b.Length > 1) b[1] = poly_a[1] - (h1 - h0);
            return b;
        }

        /// <summary>Evaluate the reference altitude (metres) at normalised downrange.</summary>
        public double AltitudeAt(double xi)
        {
            if (poly_a == null || poly_a.Length == 0) return 0.0;
            // Horner; xi outside [0,1] is clamped by the caller.
            double acc = poly_a[poly_a.Length - 1];
            for (int i = poly_a.Length - 2; i >= 0; i--) acc = acc * xi + poly_a[i];
            return acc;
        }
    }

    [Serializable]
    public sealed class LoftTableFile
    {
        public int poly_order = 7;
        public float lookahead_s = 3.0f;
        public string missile = "";
        public LoftCell[] cells = Array.Empty<LoftCell>();
    }

    internal static class Poly
    {
        /// <summary>Horner evaluation of sum c[k] * xi^k.</summary>
        public static double Eval(double[] c, double xi)
        {
            if (c == null || c.Length == 0) return 0.0;
            double acc = c[c.Length - 1];
            for (int i = c.Length - 2; i >= 0; i--) acc = acc * xi + c[i];
            return acc;
        }
    }

    internal static class Json
    {
        /// <summary>
        /// Newtonsoft first (it is in the game's Managed folder, it supports double,
        /// and it THROWS with a real message instead of silently dropping fields).
        /// JsonUtility only as a last resort, and when it is used we say so, because
        /// it will quietly ignore anything it cannot handle.
        /// </summary>
        public static LoftTableFile Parse(string text, string name, Action<string> log)
        {
            try
            {
                return Newtonsoft.Json.JsonConvert.DeserializeObject<LoftTableFile>(text);
            }
            catch (Exception e)
            {
                log?.Invoke($"{name}: Newtonsoft failed ({e.GetType().Name}: {e.Message}); " +
                            "falling back to JsonUtility, which may silently drop fields");
            }
            return UnityEngine.JsonUtility.FromJson<LoftTableFile>(text);
        }
    }

    /// <summary>
    /// Holds the loaded grid and answers "give me a profile for this shot".
    ///
    /// Interpolation is inverse-distance weighted over the k nearest cells in
    /// normalised axis space. That was a deliberate choice over trilinear: the
    /// grid produced by the offline solver will be sparse and irregular (failed
    /// cells get dropped), and trilinear needs every corner of the bracketing
    /// box to exist. IDW degrades gracefully to nearest-neighbour when it does
    /// not, and is still C0-continuous across cell boundaries.
    /// </summary>
    public sealed class LoftTable
    {
        private readonly LoftCell[] _cells;

        // axis spans, used to normalise the distance metric
        private readonly float[] _min = new float[5];
        private readonly float[] _span = new float[5];
        private readonly bool[] _degenerate = new bool[5];

        /// <summary>Axes the table actually varies on. Degenerate axes are excluded
        /// from the distance metric, so a 3-cell table indexed only on range still
        /// works as a range lookup.</summary>
        public int EffectiveAxes
        {
            get { int n = 0; for (int i = 0; i < 5; i++) if (!_degenerate[i]) n++; return n; }
        }

        public string DegenerateAxes
        {
            get
            {
                string[] names = { "range_km", "launch_alt_m", "launch_speed_ms", "target_alt_m", "target_speed_ms" };
                var l = new System.Collections.Generic.List<string>();
                for (int i = 0; i < 5; i++) if (_degenerate[i]) l.Add(names[i] + "=" + _min[i].ToString("0.#"));
                return l.Count == 0 ? "none" : string.Join(", ", l.ToArray());
            }
        }

        public int Count => _cells.Length;
        public string Missile { get; }
        public float DefaultLookahead { get; }

        /// <summary>Cells dropped at load for never having lofted.</summary>
        public int RejectedFlat { get; private set; }
        /// <summary>Cells dropped at load for an implausible terminal speed.</summary>
        public int RejectedSlow { get; private set; }

        public LoftTable(LoftTableFile f, float minTerminalSpeedMs = 0f, bool rejectFlatCells = false)
        {
            var list = new List<LoftCell>();
            foreach (var c in f.cells ?? Array.Empty<LoftCell>())
            {
                if (c?.poly_a == null || c.poly_a.Length == 0) continue;

                // A cell whose apex equals its launch altitude never lofted: the
                // optimiser handed back a flat trajectory. Because a flat reference
                // fits a flat answer exactly, these report fit_rmse_m = 0 and look
                // like the BEST cells in the table -- so nothing downstream rejects
                // them. 61 of the 252 cells in the current table are like this, some
                // "arriving" at 108 m/s. Left in, IDW blends one into a good
                // neighbourhood and drags the whole reference down with it.
                if (rejectFlatCells && Math.Abs(c.apex_alt_m - c.launch_alt_m) < 1.0)
                {
                    RejectedFlat++;
                    continue;
                }

                // Terminal speed is not monotone in the inputs -- at 100 km from
                // 4000 m the table says 692 m/s for a 180 m/s launch, 639 for 320
                // and 1809 for 460, because every cell hit the solver's iteration
                // cap and only 16 of 252 actually converged. 130 of 252 predict
                // under 800 m/s. In flight the split was clean: rounds handing over
                // above ~950 m/s hit, rounds handing over near 600 m/s did not.
                if (minTerminalSpeedMs > 0f && c.terminal_speed_ms < minTerminalSpeedMs)
                {
                    RejectedSlow++;
                    continue;
                }

                list.Add(c);
            }
            _cells = list.ToArray();
            Missile = f.missile ?? "";
            DefaultLookahead = f.lookahead_s > 0.01f ? f.lookahead_s : 3.0f;

            for (int a = 0; a < 5; a++) { _min[a] = float.MaxValue; }
            float[] max = { float.MinValue, float.MinValue, float.MinValue, float.MinValue, float.MinValue };
            foreach (var c in _cells)
            {
                float[] v = Coords(c);
                for (int a = 0; a < 5; a++)
                {
                    if (v[a] < _min[a]) _min[a] = v[a];
                    if (v[a] > max[a]) max[a] = v[a];
                }
            }
            for (int a = 0; a < 5; a++)
            {
                // A degenerate axis (one distinct value, or none) carries no
                // information and MUST contribute zero distance. Clamping the
                // span to 1e-6 instead makes (q-min)/span explode to ~1e8 for any
                // query that is not exactly on that value, so every lookup fails
                // and the mod silently never engages. That bug made the shipped
                // example table unusable.
                float sp = max[a] - _min[a];
                _span[a] = sp > 1e-4f ? sp : 0f;
                if (sp <= 1e-4f) _degenerate[a] = true;
            }
        }

        private static float[] Coords(LoftCell c) => new[]
        {
            c.range_km, c.launch_alt_m, c.launch_speed_ms, c.target_alt_m, c.target_speed_ms
        };

        /// <summary>
        /// Blend the k nearest cells' polynomials. Returns null when the shot is
        /// outside the solved envelope by more than <paramref name="maxNormDist"/>.
        /// </summary>
        /// <summary>Set by the last Lookup: distance to the nearest cell, and a
        /// human-readable reason when it returned null. This is what makes "the
        /// mod is doing nothing" diagnosable instead of silent.</summary>
        public float LastDistance { get; private set; }
        public string LastReason { get; private set; } = "";

        /// <summary>All arguments metric: km, metres, m/s.</summary>
        public LoftCell Lookup(float rangeKm, float launchAltM, float launchSpeedMs,
                               float targetAltM, float targetSpeedMs,
                               int k, float power, float maxNormDist)
        {
            if (_cells.Length == 0) { LastReason = "table has no cells"; return null; }

            float[] q = { rangeKm, launchAltM, launchSpeedMs, targetAltM, targetSpeedMs };

            // range matters most: the loft law is a strong function of t_go.
            float[] w = { 2.0f, 1.0f, 0.5f, 1.0f, 0.5f };

            int n = Mathf.Min(k, _cells.Length);
            var best = new (LoftCell cell, float d)[n];
            for (int i = 0; i < n; i++) best[i] = (null, float.MaxValue);

            foreach (var c in _cells)
            {
                float[] v = Coords(c);
                float dist = 0f;
                for (int a = 0; a < 5; a++)
                {
                    if (_degenerate[a]) continue;      // axis carries no information
                    float t = (v[a] - q[a]) / _span[a];
                    dist += w[a] * t * t;
                }
                dist = Mathf.Sqrt(dist);
                if (dist < best[n - 1].d)
                {
                    best[n - 1] = (c, dist);
                    // insertion sort, keeps `best` ascending without allocating
                    for (int i = n - 1; i > 0 && best[i].d < best[i - 1].d; i--)
                        (best[i], best[i - 1]) = (best[i - 1], best[i]);
                }
            }

            LastDistance = best[0].cell == null ? float.NaN : best[0].d;
            if (best[0].cell == null) { LastReason = "no cells at all"; return null; }
            if (best[0].d > maxNormDist)
            {
                LastReason = $"nearest cell is {best[0].d:F3} away, beyond MaxNormalisedDistance={maxNormDist:F2}";
                return null;
            }

            // Exact hit: no blending, avoids IDW noise on a dense grid.
            if (best[0].d < 1e-4f) return best[0].cell;

            int m = 0;
            double wsum = 0;
            for (int i = 0; i < n && best[i].cell != null; i++)
            {
                double wi = 1.0 / Math.Pow(Math.Max(best[i].d, 1e-6), power);
                wsum += wi; m++;
            }
            if (m == 0 || wsum <= 0) return best[0].cell;

            int order = best[0].cell.poly_a.Length;
            var blended = new double[order];
            for (int i = 0; i < m; i++)
            {
                double wi = 1.0 / Math.Pow(Math.Max(best[i].d, 1e-6), power) / wsum;
                var pa = best[i].cell.poly_a;
                for (int j = 0; j < order; j++)
                    blended[j] += wi * (j < pa.Length ? pa[j] : 0.0);
            }

            return new LoftCell
            {
                poly_a = blended,
                range_km = q[0], launch_alt_m = q[1], launch_speed_ms = q[2],
                target_alt_m = q[3], target_speed_ms = q[4],
                missile = "blended",
            };
        }

        /// <summary>Loads every *.json under <paramref name="dir"/> into one table
        /// per distinct `missile` value. Never throws.</summary>
        public static Dictionary<string, LoftTable> LoadAll(string dir, Action<string> log,
                                                            float minTerminalSpeedMs = 0f,
                                                            bool rejectFlatCells = false)
        {
            var byMissile = new Dictionary<string, List<LoftCell>>(StringComparer.OrdinalIgnoreCase);
            float lookahead = 3.0f;

            if (!Directory.Exists(dir))
            {
                log?.Invoke($"no table directory at {dir}");
                return new Dictionary<string, LoftTable>(StringComparer.OrdinalIgnoreCase);
            }

            foreach (var path in Directory.GetFiles(dir, "*.json", SearchOption.AllDirectories))
            {
                try
                {
                    string json = File.ReadAllText(path);
                    LoftTableFile f = Json.Parse(json, Path.GetFileName(path), log);
                    if (f == null) continue;
                    if (f.cells == null || f.cells.Length == 0)
                    {
                        log?.Invoke($"{Path.GetFileName(path)}: parsed OK but contained 0 cells");
                        continue;
                    }
                    lookahead = f.lookahead_s > 0.01f ? f.lookahead_s : 3.0f;
                    string key = string.IsNullOrEmpty(f.missile) ? "default" : f.missile;
                    if (!byMissile.TryGetValue(key, out var l)) byMissile[key] = l = new List<LoftCell>();
                    foreach (var cc in f.cells) cc.Normalise();
                    l.AddRange(f.cells);
                    log?.Invoke($"{Path.GetFileName(path)}: {f.cells.Length} cells (missile=\"{key}\")");
                }
                catch (Exception e)
                {
                    log?.Invoke($"{Path.GetFileName(path)}: PARSE FAILED - {e.GetType().Name}: {e.Message}");
                }
            }

            var outp = new Dictionary<string, LoftTable>(StringComparer.OrdinalIgnoreCase);
            foreach (var kv in byMissile)
            {
                var f = new LoftTableFile { missile = kv.Key, lookahead_s = lookahead, cells = kv.Value.ToArray() };
                outp[kv.Key] = new LoftTable(f, minTerminalSpeedMs, rejectFlatCells);
                var t0 = outp[kv.Key];
                log?.Invoke($"table \"{kv.Key}\": {t0.Count} usable cells" +
                            (t0.RejectedFlat + t0.RejectedSlow > 0
                                ? $" (rejected {t0.RejectedFlat} that never lofted, {t0.RejectedSlow} below {minTerminalSpeedMs:F0} m/s terminal)"
                                : ""));
            }
            return outp;
        }
    }
}
