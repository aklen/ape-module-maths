using Ape.Module.Maths.Algebra;

namespace Ape.Module.Maths.Estimation;

/// <summary>
/// Chi-square quantile. <see cref="Threshold"/> is the value a squared Mahalanobis distance is compared with.
/// The probability is an argument: this type does not store a gate.
/// </summary>
public static class ChiSquare
{
    /// <summary>
    /// Inverse CDF. <paramref name="probability"/> is the chance a draw falls at or below the returned value.
    /// </summary>
    public static double Threshold(int degreesOfFreedom, double probability)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(degreesOfFreedom, 1);
        if (!double.IsFinite(probability) || probability <= 0 || probability >= 1)
            throw new ArgumentOutOfRangeException(nameof(probability), "Probability must be strictly between 0 and 1.");

        if (degreesOfFreedom == 1)
        {
            var z = Special.InverseNormal(0.5 * (1 + probability));
            return z * z;
        }

        if (degreesOfFreedom == 2)
            return -2 * Math.Log(1 - probability);

        var guess = WilsonHilferty(degreesOfFreedom, probability);
        var x = guess > 0 ? guess : 1e-6;
        for (var i = 0; i < 30; i++)
        {
            var cdf = Special.ChiSquareCdf(degreesOfFreedom, x);
            var pdf = Special.ChiSquarePdf(degreesOfFreedom, x);
            if (pdf < 1e-30 || !double.IsFinite(pdf))
                break;
            var step = (cdf - probability) / pdf;
            x -= step;
            if (x < 0)
                x = 1e-12;
            if (Math.Abs(step) <= 1e-12 * Math.Max(1, x))
                return x;
        }

        var residual = Math.Abs(Special.ChiSquareCdf(degreesOfFreedom, x) - probability);
        if (residual > 1e-8)
            throw new InvalidOperationException("Chi-square inverse did not converge.");
        return x;
    }

    private static double WilsonHilferty(int degreesOfFreedom, double probability)
    {
        var z = Special.InverseNormal(probability);
        var term = 2.0 / (9 * degreesOfFreedom);
        var cube = 1 - term + z * Math.Sqrt(term);
        return degreesOfFreedom * cube * cube * cube;
    }
}
