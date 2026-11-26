using Ape.Module.Maths.Algebra;

namespace Ape.Module.Maths.Estimation;

/// <summary>
/// Causal linear-Gaussian factor-graph filter for one state trajectory.
/// Factors are a unary belief (prior or filtered marginal) and, per step, a binary constant-velocity
/// transition factor plus optional unary measurement factors. Inference is forward-only: the
/// current-state marginal given the measurements available so far. Re-estimating earlier states
/// from a full batch is smoothing and is out of scope for this type.
/// For linear-Gaussian models the filtered mean and covariance match <see cref="KalmanFilter"/>.
/// </summary>
public sealed class CausalFactorGraphFilter
{
    private double[] _information;
    private Matrix _precision;

    private CausalFactorGraphFilter(double[] information, Matrix precision)
    {
        _information = information;
        _precision = precision;
    }

    public int Size => _information.Length;

    /// <summary>Create a filter whose initial unary factor is the given Gaussian prior.</summary>
    public static CausalFactorGraphFilter FromPrior(KalmanState prior)
    {
        ArgumentNullException.ThrowIfNull(prior);
        var precision = Invert(prior.Covariance);
        var information = precision.Multiply(prior.MeanSpan);
        return new CausalFactorGraphFilter(information, precision);
    }

    /// <summary>
    /// Apply a binary transition factor xₖ₊₁ = F xₖ + w, w ~ N(0, Q), and eliminate xₖ
    /// so only the causal marginal on xₖ₊₁ remains.
    /// Elimination uses the covariance-form Schur complement
    /// (P′ = F P Fᵀ + Q, μ′ = F μ), which is algebraically equivalent to the information-form
    /// Schur when Q is SPD and remains well-defined for Q = 0 (deterministic transition) and
    /// for small dt where Q⁻¹ would otherwise blow up.
    /// </summary>
    public void ApplyTransitionFactor(Matrix transition, Matrix processNoise)
    {
        ArgumentNullException.ThrowIfNull(transition);
        ArgumentNullException.ThrowIfNull(processNoise);
        RequireSquare(transition, Size, nameof(transition));
        RequireSquare(processNoise, Size, nameof(processNoise));

        var belief = CurrentMarginal();
        var mean = transition.Multiply(belief.MeanSpan);
        var covariance = transition
            .Multiply(belief.Covariance)
            .Multiply(transition.Transpose())
            .Add(processNoise)
            .Symmetrize();
        _precision = Invert(covariance);
        _information = _precision.Multiply(mean);
    }

    /// <summary>Apply a unary linear measurement factor z = H x + v, v ~ N(0, R).</summary>
    public void ApplyMeasurementFactor(ReadOnlySpan<double> measurement, Matrix observation, Matrix measurementNoise)
    {
        ArgumentNullException.ThrowIfNull(observation);
        ArgumentNullException.ThrowIfNull(measurementNoise);
        if (measurement.Length == 0)
            throw new ArgumentException("Measurement is empty.", nameof(measurement));
        if (observation.Rows != measurement.Length || observation.Columns != Size)
            throw new ArgumentException("Observation matrix must be measurements by state.", nameof(observation));
        if (measurementNoise.Rows != measurement.Length || measurementNoise.Columns != measurement.Length)
            throw new ArgumentException("Measurement noise must be square in the measurement dimension.", nameof(measurementNoise));

        var rInv = Invert(measurementNoise);
        var htRi = observation.Transpose().Multiply(rInv);
        _precision = _precision.Add(htRi.Multiply(observation)).Symmetrize();
        var infoAdd = htRi.Multiply(measurement);
        for (var i = 0; i < _information.Length; i++)
            _information[i] += infoAdd[i];
    }

    /// <summary>Current-state marginal (mean and covariance) after the factors applied so far.</summary>
    public KalmanState CurrentMarginal()
    {
        var covariance = Invert(_precision).Symmetrize();
        var mean = covariance.Multiply(_information);
        return KalmanState.Create(mean, covariance);
    }

    private static Matrix Invert(Matrix matrix)
    {
        var n = matrix.Rows;
        if (matrix.Columns != n)
            throw new ArgumentException("Matrix must be square.", nameof(matrix));
        var factor = LinearSolver.Cholesky(matrix.Symmetrize());
        var values = new double[n * n];
        var column = new double[n];
        for (var j = 0; j < n; j++)
        {
            Array.Clear(column);
            column[j] = 1;
            LinearSolver.Solve(factor, n, column);
            for (var i = 0; i < n; i++)
                values[i * n + j] = column[i];
        }

        return Matrix.RowMajor(n, n, values).Symmetrize();
    }

    private static void RequireSquare(Matrix matrix, int size, string name)
    {
        if (matrix.Rows != size || matrix.Columns != size)
            throw new ArgumentException($"Matrix must be {size} by {size}.", name);
    }
}
