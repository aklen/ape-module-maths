using Ape.Module.Maths.Algebra;

namespace Ape.Module.Maths.Tests;

internal static class MathsAssert
{
    public static void Near(double expected, double actual, double tolerance = 1e-12) =>
        Assert.InRange(actual, expected - tolerance, expected + tolerance);

    public static void Matrix(Matrix matrix, double tolerance, params double[] expected)
    {
        Assert.Equal(expected.Length, matrix.Rows * matrix.Columns);
        for (var row = 0; row < matrix.Rows; row++)
        {
            for (var column = 0; column < matrix.Columns; column++)
                Near(expected[row * matrix.Columns + column], matrix[row, column], tolerance);
        }
    }
}
