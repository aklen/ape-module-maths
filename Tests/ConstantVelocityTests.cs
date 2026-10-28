using Ape.Module.Maths.Estimation;

namespace Ape.Module.Maths.Tests;

public class ConstantVelocityTests
{
    [Fact]
    public void Transition_adds_velocity_times_time_on_the_diagonal_blocks()
    {
        var transition = ConstantVelocity.Transition(2, spatialDimension: 1);
        MathsAssert.Matrix(transition, 0, 1, 2, 0, 1);
    }

    [Fact]
    public void Process_noise_matches_the_white_acceleration_blocks()
    {
        var noise = ConstantVelocity.ProcessNoise(2, noiseDensity: 3, spatialDimension: 1);
        MathsAssert.Matrix(noise, 1e-12, 8, 6, 6, 6);
    }

    [Fact]
    public void Axes_do_not_share_process_noise()
    {
        var noise = ConstantVelocity.ProcessNoise(1, noiseDensity: 4, spatialDimension: 2);
        Assert.Equal(0, noise[0, 1]);
        Assert.Equal(0, noise[0, 3]);
        MathsAssert.Near(4.0 / 3.0, noise[0, 0]);
        MathsAssert.Near(2, noise[0, 2]);
        MathsAssert.Near(4, noise[2, 2]);
    }

    [Fact]
    public void Zero_time_is_identity_and_zero_noise()
    {
        var transition = ConstantVelocity.Transition(0, spatialDimension: 2);
        var noise = ConstantVelocity.ProcessNoise(0, noiseDensity: 9, spatialDimension: 2);
        MathsAssert.Matrix(transition, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
        for (var row = 0; row < noise.Rows; row++)
        {
            for (var column = 0; column < noise.Columns; column++)
                Assert.Equal(0, noise[row, column]);
        }
    }

    [Fact]
    public void Negative_time_is_rejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => ConstantVelocity.Transition(-1, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => ConstantVelocity.ProcessNoise(1, -0.1, 1));
    }
}
