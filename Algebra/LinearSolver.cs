namespace Ape.Module.Maths.Algebra;

/// <summary>Cholesky factor and solve for a symmetric positive-definite matrix.</summary>
internal static class LinearSolver
{
    public static double[] Cholesky(Matrix matrix)
    {
        ArgumentNullException.ThrowIfNull(matrix);
        if (matrix.Rows != matrix.Columns)
            throw new ArgumentException("Matrix must be square.", nameof(matrix));

        var n = matrix.Rows;
        var lower = new double[n * n];
        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j <= i; j++)
            {
                var sum = matrix[i, j];
                for (var k = 0; k < j; k++)
                    sum -= lower[i * n + k] * lower[j * n + k];

                if (i == j)
                {
                    if (sum <= 0 || !double.IsFinite(sum))
                        throw new InvalidOperationException("Matrix is not positive definite.");
                    lower[i * n + j] = Math.Sqrt(sum);
                }
                else
                {
                    lower[i * n + j] = sum / lower[j * n + j];
                }
            }
        }

        return lower;
    }

    /// <summary>Solves L Lᵀ x = b in place. <paramref name="lower"/> is row-major, lower triangle only.</summary>
    public static void Solve(double[] lower, int size, Span<double> rightHandSide)
    {
        if (rightHandSide.Length != size)
            throw new ArgumentException("Right-hand side length must match the factor.", nameof(rightHandSide));

        for (var i = 0; i < size; i++)
        {
            var sum = rightHandSide[i];
            for (var k = 0; k < i; k++)
                sum -= lower[i * size + k] * rightHandSide[k];
            rightHandSide[i] = sum / lower[i * size + i];
        }

        for (var i = size - 1; i >= 0; i--)
        {
            var sum = rightHandSide[i];
            for (var k = i + 1; k < size; k++)
                sum -= lower[k * size + i] * rightHandSide[k];
            rightHandSide[i] = sum / lower[i * size + i];
        }
    }
}
