using System;
using System.Collections.Generic;
using System.IO;

using JetBrains.Annotations;

using UnityEngine;

namespace NOLoftFix.Core;

/// <summary>
/// One solved cell of the offline grid.
/// </summary>
[Serializable]
public sealed class LoftCell
{
    // grid coordinates
    public float Range_km;
    public float Launch_alt_m;
    public float Launch_speed_ms;
    public float Target_alt_m;
    public float Target_speed_ms;

    // legacy imperial keys
    public float Range_nmi;
    public float Launch_alt_ft;
    public float Launch_kt;
    public float Target_alt_ft;
    public float Target_kt;

    /// <summary>
    /// Converts legacy imperial fields into metric.
    /// </summary>
    public void Normalise()
    {
        if (Range_km <= 0f && Range_nmi > 0f)
        {
            Range_km = Range_nmi * 1.852f;
        }

        if (Launch_alt_m <= 0f && Launch_alt_ft > 0f)
        {
            Launch_alt_m = Launch_alt_ft * 0.3048f;
        }

        if (Launch_speed_ms <= 0f && Launch_kt > 0f)
        {
            Launch_speed_ms = Launch_kt * 0.514444f;
        }

        if (Target_alt_m <= 0f && Target_alt_ft > 0f)
        {
            Target_alt_m = Target_alt_ft * 0.3048f;
        }

        if (Target_speed_ms <= 0f && Target_kt > 0f)
        {
            Target_speed_ms = Target_kt * 0.514444f;
        }
    }

    /// <summary>
    /// <para>the reference profile</para>
    /// <para>Coefficients of h(xi) = sum_k poly_a[k] * xi^k, xi = x / X_f.</para>
    /// </summary>
    public double[] Poly_a;

    // diagnostics, informational only
    public double Apex_alt_m;
    public double Apex_downrange_m;
    public double Terminal_speed_ms;
    public double Tof_s;
    public double Fit_rmse_m;
    public bool Converged;
    public string Missile = "";

    /// <summary>
    /// <para>
    /// The profile as poly(xi) minus the straight line joining its
    /// own two endpoints. B(0) = B(1) = 0 by construction, because the offline
    /// solve pins both ends.
    /// </para>
    /// <para>
    /// This is what makes midcourse retargeting possible.
    /// The runtime reference is rebuilt as:
    /// </para>
    /// <para>
    /// h(xi) = h_launch + (h_pip_now - h_launch) * xi + B(xi)
    /// </para>
    /// <para>
    /// so the solved shape is kept while the endpoints follow the live PIP.
    /// Both ends stay exact no matter where the target has moved to.
    /// </para>
    /// </summary>
    public double[] BumpCoeffs()
    {
        if (Poly_a == null || Poly_a.Length == 0)
        {
            return [];
        }

        double h0 = Poly_a[0];
        double h1 = 0.0;
        for (int i = 0; i < Poly_a.Length; i++)
        {
            h1 += Poly_a[i];
        }

        double[] b = (double[])Poly_a.Clone();
        b[0] = 0.0;
        if (b.Length > 1)
        {
            b[1] = Poly_a[1] - (h1 - h0);
        }

        return b;
    }

    /// <summary>Evaluate the reference altitude at normalised downrange.</summary>
    /// <param name="xi"></param>
    public double AltitudeAt(double xi)
    {
        if (Poly_a == null || Poly_a.Length == 0)
        {
            return 0.0;
        }
        // Horner; xi outside [0,1] is clamped by the caller.
        double acc = Poly_a[^1];
        for (int i = Poly_a.Length - 2; i >= 0; i--)
        {
            acc = (acc * xi) + Poly_a[i];
        }

        return acc;
    }
}

[Serializable]
public sealed class LoftTableFile
{
    public int Poly_order = 7;
    public float Lookahead_s = 3.0f;
    public string Missile = "";
    public LoftCell[] Cells = [];
}

internal static class Poly
{
    /// <summary>Horner evaluation of sum c[k] * xi^k.</summary>
    public static double Eval(double[] c, double xi)
    {
        if (c == null || c.Length == 0)
        {
            return 0.0;
        }

        double acc = c[^1];
        for (int i = c.Length - 2; i >= 0; i--)
        {
            acc = (acc * xi) + c[i];
        }

        return acc;
    }
}

internal static class Json
{
    public static LoftTableFile Parse(string text, string name, Action<string> log)
    {
        try
        {
            return Newtonsoft.Json.JsonConvert.DeserializeObject<LoftTableFile>(text);
        }
        catch (Exception e)
        {
            log?.Invoke($"{name}: Newtonsoft failed ({e.GetType().Name}: {e.Message}); " +
                        "falling back to JsonUtility");
        }
        return JsonUtility.FromJson<LoftTableFile>(text);
    }
}

/// <summary>
/// <para>Holds the loaded grid and gives a profile for the shot.</para>
/// <para>
/// Interpolation is inverse-distance weighted over the k nearest cells in
/// normalised axis space. That was a deliberate choice over trilinear: the
/// grid produced by the offline solver will be sparse and irregular (failed
/// cells get dropped), and trilinear needs every corner of the bracketing
/// box to exist. IDW degrades gracefully to nearest-neighbour when it does
/// not, and is still C0-continuous across cell boundaries.
/// </para>
/// </summary>
public sealed class LoftTable
{
    private readonly LoftCell[] _cells;

    /// <summary>
    /// axis spans, used to normalise the distance metric
    /// </summary>
    private readonly float[] _min = new float[5];
    private readonly float[] _span = new float[5];
    private readonly bool[] _degenerate = new bool[5];

    /// <summary>
    /// Axes the table actually varies on. Degenerate axes are excluded
    /// from the distance metric, so a 3-cell table indexed only on range still
    /// works as a range lookup.
    /// </summary>
    public int EffectiveAxes
    {
        get
        {
            int n = 0;
            for (int i = 0; i < 5; i++)
            {
                if (!_degenerate[i])
                {
                    n++;
                }
            }
            return n;
        }
    }

    public string DegenerateAxes
    {
        get
        {
            string[] names = ["range_km", "launch_alt_m", "launch_speed_ms", "target_alt_m", "target_speed_ms"];
            List<string> l = [];
            for (int i = 0; i < 5; i++)
            {
                if (_degenerate[i])
                {
                    l.Add(names[i] + "=" + _min[i].ToString("0.#"));
                }
            }

            return l.Count == 0 ? "none" : string.Join(", ", [.. l]);
        }
    }

    public int Count => _cells.Length;
    public string Missile { get; }
    public float DefaultLookahead { get; }

    /// <summary>Cells dropped at load for never having lofted.</summary>
    public int RejectedFlat { get; }
    /// <summary>Cells dropped at load for an implausible terminal speed.</summary>
    public int RejectedSlow { get; }

    public LoftTable(LoftTableFile f, float minTerminalSpeedMs = 0f, bool rejectFlatCells = false)
    {
        List<LoftCell> list = [];
        foreach (LoftCell c in f.Cells ?? [])
        {
            if (c?.Poly_a == null || c.Poly_a.Length == 0)
            {
                continue;
            }

            if (rejectFlatCells && Math.Abs(c.Apex_alt_m - c.Launch_alt_m) < 1.0)
            {
                RejectedFlat++;
                continue;
            }

            if (minTerminalSpeedMs > 0f && c.Terminal_speed_ms < minTerminalSpeedMs)
            {
                RejectedSlow++;
                continue;
            }

            list.Add(c);
        }
        _cells = [.. list];
        Missile = f.Missile ?? "";
        DefaultLookahead = f.Lookahead_s > 0.01f ? f.Lookahead_s : 3.0f;

        for (int a = 0; a < 5; a++) { _min[a] = float.MaxValue; }
        float[] max = [float.MinValue, float.MinValue, float.MinValue, float.MinValue, float.MinValue];
        foreach (LoftCell c in _cells)
        {
            float[] v = Coords(c);
            for (int a = 0; a < 5; a++)
            {
                if (v[a] < _min[a])
                {
                    _min[a] = v[a];
                }

                if (v[a] > max[a])
                {
                    max[a] = v[a];
                }
            }
        }
        for (int a = 0; a < 5; a++)
        {
            // A degenerate axis (one distinct value, or none) carries no
            // information and must contribute zero distance.
            float sp = max[a] - _min[a];
            _span[a] = sp > 1e-4f ? sp : 0f;
            if (sp <= 1e-4f)
            {
                _degenerate[a] = true;
            }
        }
    }

    private static float[] Coords(LoftCell c)
    {
        return [
        c.Range_km, c.Launch_alt_m, c.Launch_speed_ms, c.Target_alt_m, c.Target_speed_ms
    ];
    }

    /// <summary>
    /// Blend the k nearest cells' polynomials. Returns null when the shot is
    /// outside the solved envelope by more than <paramref name="maxNormDist"/>.
    /// </summary>
    /// <summary>
    /// Set by the last Lookup: distance to the nearest cell, and a
    /// human-readable reason when it returned null. This is what makes "the
    /// mod is doing nothing" diagnosable instead of silent.
    /// </summary>
    public float LastDistance { get; private set; }
    public string LastReason { get; private set; } = "";

    public LoftCell Lookup(float rangeKm, float launchAltM, float launchSpeedMs,
                           float targetAltM, float targetSpeedMs,
                           int k, float power, float maxNormDist)
    {
        if (_cells.Length == 0) { LastReason = "table has no cells"; return null; }

        float[] q = [rangeKm, launchAltM, launchSpeedMs, targetAltM, targetSpeedMs];

        // range matters most: the loft law is a strong function of t_go.
        float[] w = [2.0f, 1.0f, 0.5f, 1.0f, 0.5f];

        int n = Mathf.Min(k, _cells.Length);
        (LoftCell cell, float d)[] best = new (LoftCell cell, float d)[n];
        for (int i = 0; i < n; i++)
        {
            best[i] = (null, float.MaxValue);
        }

        foreach (LoftCell c in _cells)
        {
            float[] v = Coords(c);
            float dist = 0f;
            for (int a = 0; a < 5; a++)
            {
                if (_degenerate[a])
                {
                    continue; // axis carries no information
                }

                float t = (v[a] - q[a]) / _span[a];
                dist += w[a] * t * t;
            }
            dist = Mathf.Sqrt(dist);
            if (dist < best[n - 1].d)
            {
                best[n - 1] = (c, dist);
                // insertion sort, keeps `best` ascending without allocating
                for (int i = n - 1; i > 0 && best[i].d < best[i - 1].d; i--)
                {
                    (best[i], best[i - 1]) = (best[i - 1], best[i]);
                }
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
        if (best[0].d < 1e-4f)
        {
            return best[0].cell;
        }

        int m = 0;
        double wsum = 0;
        for (int i = 0; i < n && best[i].cell != null; i++)
        {
            double wi = 1.0 / Math.Pow(Math.Max(best[i].d, 1e-6), power);
            wsum += wi; m++;
        }
        if (m == 0 || wsum <= 0)
        {
            return best[0].cell;
        }

        int order = best[0].cell.Poly_a.Length;
        double[] blended = new double[order];
        for (int i = 0; i < m; i++)
        {
            double wi = 1.0 / Math.Pow(Math.Max(best[i].d, 1e-6), power) / wsum;
            double[] pa = best[i].cell.Poly_a;
            for (int j = 0; j < order; j++)
            {
                blended[j] += wi * (j < pa.Length ? pa[j] : 0.0);
            }
        }

        return new LoftCell
        {
            Poly_a = blended,
            Range_km = q[0],
            Launch_alt_m = q[1],
            Launch_speed_ms = q[2],
            Target_alt_m = q[3],
            Target_speed_ms = q[4],
            Missile = "blended",
        };
    }

    /// <summary>Loads every json under <paramref name="dir"/> into one table
    /// per `missile` value.</summary>
    /// <param name="dir"></param>
    /// <param name="log"></param>
    /// <param name="minTerminalSpeedMs"></param>
    /// <param name="rejectFlatCells"></param>
    public static Dictionary<string, LoftTable> LoadAll(string dir, Action<string> log,
                                                        float minTerminalSpeedMs = 0f,
                                                        bool rejectFlatCells = false)
    {
        Dictionary<string, List<LoftCell>> byMissile = new(StringComparer.OrdinalIgnoreCase);
        float lookahead = 3.0f;

        if (!Directory.Exists(dir))
        {
            log?.Invoke($"no table directory at {dir}");
            return new Dictionary<string, LoftTable>(StringComparer.OrdinalIgnoreCase);
        }

        foreach (string path in Directory.GetFiles(dir, "*.json", SearchOption.AllDirectories))
        {
            try
            {
                string json = File.ReadAllText(path);
                LoftTableFile f = Json.Parse(json, Path.GetFileName(path), log);
                if (f == null)
                {
                    continue;
                }

                if (f.Cells == null || f.Cells.Length == 0)
                {
                    log?.Invoke($"{Path.GetFileName(path)}: parsed OK but contained 0 cells");
                    continue;
                }
                lookahead = f.Lookahead_s > 0.01f ? f.Lookahead_s : 3.0f;
                string key = string.IsNullOrEmpty(f.Missile) ? "default" : f.Missile;
                if (!byMissile.TryGetValue(key, out List<LoftCell> l))
                {
                    byMissile[key] = l = [];
                }

                foreach (LoftCell cc in f.Cells)
                {
                    cc.Normalise();
                }

                l.AddRange(f.Cells);
                log?.Invoke($"{Path.GetFileName(path)}: {f.Cells.Length} cells (missile=\"{key}\")");
            }
            catch (Exception e)
            {
                log?.Invoke($"{Path.GetFileName(path)}: Parse failed - {e.GetType().Name}: {e.Message}");
            }
        }

        Dictionary<string, LoftTable> outp = new(StringComparer.OrdinalIgnoreCase);
        foreach (KeyValuePair<string, List<LoftCell>> kv in byMissile)
        {
            LoftTableFile f = new() { Missile = kv.Key, Lookahead_s = lookahead, Cells = [.. kv.Value] };
            outp[kv.Key] = new LoftTable(f, minTerminalSpeedMs, rejectFlatCells);
            LoftTable t0 = outp[kv.Key];
            log?.Invoke($"table \"{kv.Key}\": {t0.Count} usable cells" +
                        (t0.RejectedFlat + t0.RejectedSlow > 0
                            ? $" (rejected {t0.RejectedFlat} with apex lower than launch altitude, {t0.RejectedSlow} below {minTerminalSpeedMs:F0} m/s terminal)"
                            : ""));
        }
        return outp;
    }
}
