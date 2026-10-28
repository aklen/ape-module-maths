using Ape.Module.Maths.Algebra;

namespace Ape.Module.Maths.Estimation;

/// <summary>Mahalanobis distance of a residual under a covariance.</summary>
public static class Mahalanobis
{
    public static double Squared(ReadOnlySpan<double> residual, Matrix covariance)
    {
        ArgumentNullException.ThrowIfNull(covariance);
        if (residual.Length == 0)
            throw new ArgumentException("Residual is empty.", nameof(residual));
        if (covariance.Rows != residual.Length || covariance.Columns != residual.Length)
            throw new ArgumentException("Covariance size must match the residual.", nameof(covariance));

        var solved = new double[residual.Length];
        residual.CopyTo(solved);
        var factor = LinearSolver.Cholesky(covariance.Symmetrize());
        LinearSolver.Solve(factor, residual.Length, solved);

        var distance = 0.0;
        for (var i = 0; i < residual.Length; i++)
            distance += residual[i] * solved[i];
        return distance;
    }

    public static double Distance(ReadOnlySpan<double> residual, Matrix covariance) =>
        Math.Sqrt(Squared(residual, covariance));
}
