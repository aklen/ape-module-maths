using Ape.Module.Maths.Algebra;
using Ape.Module.Maths.Estimation;

namespace Ape.Module.Maths.Tests;

public class ChiSquareTests
{
    [Theory]
    [InlineData(1, 0.95, 3.841)]
    [InlineData(2, 0.95, 5.991)]
    [InlineData(3, 0.95, 7.815)]
    [InlineData(4, 0.95, 9.488)]
    [InlineData(6, 0.95, 12.592)]
    [InlineData(1, 0.99, 6.635)]
    [InlineData(3, 0.99, 11.345)]
    [InlineData(6, 0.99, 16.812)]
    [InlineData(1, 0.5, 0.455)]
    public void Threshold_matches_the_published_table(int degreesOfFreedom, double probability, double table)
    {
        var threshold = ChiSquare.Threshold(degreesOfFreedom, probability);
        Assert.InRange(threshold, table - 5e-4, table + 5e-4);
        MathsAssert.Near(probability, Special.ChiSquareCdf(degreesOfFreedom, threshold), 1e-9);
    }

    [Fact]
    public void Two_degrees_of_freedom_is_the_exponential_quantile()
    {
        const double probability = 0.95;
        MathsAssert.Near(-2 * Math.Log(1 - probability), ChiSquare.Threshold(2, probability), 1e-12);
    }

    [Fact]
    public void Probability_outside_the_open_unit_interval_is_rejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => ChiSquare.Threshold(3, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => ChiSquare.Threshold(3, 1));
    }
}
