using Ape.Module.Maths.Algebra;

namespace Ape.Module.Maths.Estimation;

/// <summary>
/// Constant-velocity transition for independent axes.
/// State order is position of each axis, then velocity of each axis.
/// Process noise is continuous white acceleration: the density is the caller's scale, in acceleration² per second.
/// </summary>
public static class ConstantVelocity
{
    public static int StateLength(int spatialDimension)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(spatialDimension, 1);
        return spatialDimension * 2;
    }

    public static Matrix Transition(double seconds, int spatialDimension)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(seconds);
        var axes = spatialDimension;
        var n = StateLength(axes);
        var values = new double[n * n];
        for (var i = 0; i < n; i++)
            values[i * n + i] = 1;
        for (var axis = 0; axis < axes; axis++)
            values[axis * n + axes + axis] = seconds;
        return Matrix.RowMajor(n, n, values);
    }

    public static Matrix ProcessNoise(double seconds, double noiseDensity, int spatialDimension)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(seconds);
        ArgumentOutOfRangeException.ThrowIfNegative(noiseDensity);
        var axes = spatialDimension;
        var n = StateLength(axes);
        var values = new double[n * n];
        var dt2 = seconds * seconds;
        var position = noiseDensity * dt2 * seconds / 3.0;
        var cross = noiseDensity * dt2 / 2.0;
        var velocity = noiseDensity * seconds;
        for (var axis = 0; axis < axes; axis++)
        {
            var velocityIndex = axes + axis;
            values[axis * n + axis] = position;
            values[axis * n + velocityIndex] = cross;
            values[velocityIndex * n + axis] = cross;
            values[velocityIndex * n + velocityIndex] = velocity;
        }

        return Matrix.RowMajor(n, n, values);
    }
}
