namespace Ape.Module.Maths.Algebra;

/// <summary>
/// Log-gamma (Lanczos), the chi-square CDF (regularized gamma), and the inverse normal (Acklam, then Newton).
/// </summary>
internal static class Special
{
    private static readonly double[] Lanczos =
    [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    public static double ChiSquareCdf(int degreesOfFreedom, double x)
    {
        if (x < 0 || !double.IsFinite(x))
            return 0;
        if (x == 0)
            return 0;
        var a = 0.5 * degreesOfFreedom;
        return x < a + 1
            ? GammaPSeries(a, 0.5 * x)
            : 1 - GammaQContinued(a, 0.5 * x);
    }

    public static double ChiSquarePdf(int degreesOfFreedom, double x)
    {
        if (x <= 0 || !double.IsFinite(x))
            return 0;
        var half = 0.5 * degreesOfFreedom;
        var log = -0.5 * x
            + (half - 1) * Math.Log(x)
            - half * Math.Log(2)
            - LogGamma(half);
        return Math.Exp(log);
    }

    public static double InverseNormal(double probability)
    {
        var z = Acklam(probability);
        var scale = Math.Sqrt(2);
        var norm = Math.Sqrt(2 * Math.PI);
        for (var i = 0; i < 3; i++)
        {
            var cdf = 0.5 * (1 + Erf(z / scale));
            var density = Math.Exp(-0.5 * z * z) / norm;
            if (density < 1e-300)
                break;
            var step = (cdf - probability) / density;
            z -= step;
            if (Math.Abs(step) < 1e-14)
                break;
        }

        return z;
    }

    private static double Erf(double x)
    {
        if (x == 0)
            return 0;
        var sign = x < 0 ? -1 : 1;
        var x2 = x * x;
        var probability = x2 < 1.5 ? GammaPSeries(0.5, x2) : 1 - GammaQContinued(0.5, x2);
        return sign * probability;
    }

    private static double LogGamma(double z)
    {
        if (z < 0.5)
            return Math.Log(Math.PI / Math.Sin(Math.PI * z)) - LogGamma(1 - z);

        z -= 1;
        var sum = Lanczos[0];
        for (var i = 1; i < Lanczos.Length; i++)
            sum += Lanczos[i] / (z + i);

        var t = z + 7.5;
        return 0.5 * Math.Log(2 * Math.PI) + (z + 0.5) * Math.Log(t) - t + Math.Log(sum);
    }

    private static double GammaPSeries(double a, double x)
    {
        var term = 1.0 / a;
        var sum = term;
        var ap = a;
        for (var n = 0; n < 200; n++)
        {
            ap += 1;
            term *= x / ap;
            sum += term;
            if (Math.Abs(term) < Math.Abs(sum) * 1e-15)
                break;
        }

        return sum * Math.Exp(-x + a * Math.Log(x) - LogGamma(a));
    }

    private static double GammaQContinued(double a, double x)
    {
        const double tiny = 1e-30;
        var b = x + 1 - a;
        var c = 1 / tiny;
        var d = 1 / b;
        var h = d;
        for (var i = 1; i <= 200; i++)
        {
            var an = -i * (i - a);
            b += 2;
            d = an * d + b;
            if (Math.Abs(d) < tiny)
                d = tiny;
            c = b + an / c;
            if (Math.Abs(c) < tiny)
                c = tiny;
            d = 1 / d;
            var delta = d * c;
            h *= delta;
            if (Math.Abs(delta - 1) < 1e-14)
                break;
        }

        return Math.Exp(-x + a * Math.Log(x) - LogGamma(a)) * h;
    }

    private static double Acklam(double probability)
    {
        const double plow = 0.02425;
        const double phigh = 1 - plow;
        double q;
        if (probability < plow)
        {
            q = Math.Sqrt(-2 * Math.Log(probability));
            return (((((C0 * q + C1) * q + C2) * q + C3) * q + C4) * q + C5)
                / ((((D0 * q + D1) * q + D2) * q + D3) * q + 1);
        }

        if (probability > phigh)
        {
            q = Math.Sqrt(-2 * Math.Log(1 - probability));
            return -(((((C0 * q + C1) * q + C2) * q + C3) * q + C4) * q + C5)
                / ((((D0 * q + D1) * q + D2) * q + D3) * q + 1);
        }

        q = probability - 0.5;
        var r = q * q;
        return (((((A0 * r + A1) * r + A2) * r + A3) * r + A4) * r + A5) * q
            / (((((B0 * r + B1) * r + B2) * r + B3) * r + B4) * r + 1);
    }

    private const double A0 = -3.969683028665376e+01;
    private const double A1 = 2.209460984245205e+02;
    private const double A2 = -2.759285104469687e+02;
    private const double A3 = 1.383577518672690e+02;
    private const double A4 = -3.066479806614716e+01;
    private const double A5 = 2.506628277459239e+00;
    private const double B0 = -5.447609879822406e+01;
    private const double B1 = 1.615858368580409e+02;
    private const double B2 = -1.556989798598866e+02;
    private const double B3 = 6.680131188771972e+01;
    private const double B4 = -1.328068155288572e+01;
    private const double C0 = -7.784894002430293e-03;
    private const double C1 = -3.223964580411365e-01;
    private const double C2 = -2.400758277161838e+00;
    private const double C3 = -2.549732539343734e+00;
    private const double C4 = 4.374664141464968e+00;
    private const double C5 = 2.938163982698783e+00;
    private const double D0 = 7.784695709041462e-03;
    private const double D1 = 3.224671290700398e-01;
    private const double D2 = 2.445134137142996e+00;
    private const double D3 = 3.754408661907416e+00;
}
