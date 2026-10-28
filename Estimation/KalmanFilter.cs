using Ape.Module.Maths.Algebra;

namespace Ape.Module.Maths.Estimation;

/// <summary>
/// Mean and covariance of a linear state. The mean array is owned by this instance.
/// </summary>
public sealed class KalmanState
{
    private readonly double[] _mean;

    private KalmanState(double[] mean, Matrix covariance)
    {
        _mean = mean;
        Covariance = covariance;
    }

    public int Size => _mean.Length;

    public Matrix Covariance { get; }

    public double Mean(int index)
    {
        if ((uint)index >= (uint)_mean.Length)
            throw new ArgumentOutOfRangeException(nameof(index));
        return _mean[index];
    }

    public double[] CopyMean()
    {
        var copy = new double[_mean.Length];
        _mean.AsSpan().CopyTo(copy);
        return copy;
    }

    public static KalmanState Create(ReadOnlySpan<double> mean, Matrix covariance)
    {
        ArgumentNullException.ThrowIfNull(covariance);
        if (mean.Length == 0)
            throw new ArgumentException("State is empty.", nameof(mean));
        if (covariance.Rows != mean.Length || covariance.Columns != mean.Length)
            throw new ArgumentException("Covariance size must match the state.", nameof(covariance));

        var copy = new double[mean.Length];
        mean.CopyTo(copy);
        return new KalmanState(copy, covariance);
    }

    internal ReadOnlySpan<double> MeanSpan => _mean;
}

/// <summary>Result of a Kalman update. <see cref="MahalanobisSquared"/> is yᵀ S⁻¹ y.</summary>
public readonly record struct KalmanUpdate(KalmanState State, double MahalanobisSquared);

/// <summary>
/// Discrete linear Kalman filter. The caller supplies every matrix.
/// Predict is x ← F x and P ← F P Fᵀ + Q. Update uses the Joseph form and then symmetrizes P.
/// </summary>
public static class KalmanFilter
{
    public static KalmanState Predict(KalmanState state, Matrix transition, Matrix processNoise)
    {
        ArgumentNullException.ThrowIfNull(state);
        RequireSquare(transition, state.Size, nameof(transition));
        RequireSquare(processNoise, state.Size, nameof(processNoise));

        var mean = transition.Multiply(state.MeanSpan);
        var covariance = transition
            .Multiply(state.Covariance)
            .Multiply(transition.Transpose())
            .Add(processNoise)
            .Symmetrize();
        return KalmanState.Create(mean, covariance);
    }

    public static KalmanUpdate Update(
        KalmanState state,
        ReadOnlySpan<double> measurement,
        Matrix observation,
        Matrix measurementNoise)
    {
        ArgumentNullException.ThrowIfNull(state);
        ArgumentNullException.ThrowIfNull(observation);
        ArgumentNullException.ThrowIfNull(measurementNoise);
        if (measurement.Length == 0)
            throw new ArgumentException("Measurement is empty.", nameof(measurement));
        if (observation.Rows != measurement.Length || observation.Columns != state.Size)
            throw new ArgumentException("Observation matrix must be measurements by state.", nameof(observation));
        if (measurementNoise.Rows != measurement.Length || measurementNoise.Columns != measurement.Length)
            throw new ArgumentException("Measurement noise must be square in the measurement dimension.", nameof(measurementNoise));

        var predicted = observation.Multiply(state.MeanSpan);
        var innovation = new double[measurement.Length];
        for (var i = 0; i < innovation.Length; i++)
            innovation[i] = measurement[i] - predicted[i];

        var cross = state.Covariance.Multiply(observation.Transpose());
        var innovationCovariance = observation.Multiply(cross).Add(measurementNoise).Symmetrize();
        var factor = LinearSolver.Cholesky(innovationCovariance);

        var solvedInnovation = innovation.AsSpan().ToArray();
        LinearSolver.Solve(factor, innovation.Length, solvedInnovation);
        var mahalanobis = 0.0;
        for (var i = 0; i < innovation.Length; i++)
            mahalanobis += innovation[i] * solvedInnovation[i];

        var inverse = SolveInverse(factor, innovation.Length);
        var gain = cross.Multiply(inverse);
        var corrected = state.CopyMean();
        var correction = gain.Multiply(innovation);
        for (var i = 0; i < corrected.Length; i++)
            corrected[i] += correction[i];

        var identity = Matrix.Identity(state.Size);
        var residualGain = identity.Subtract(gain.Multiply(observation));
        var covariance = residualGain
            .Multiply(state.Covariance)
            .Multiply(residualGain.Transpose())
            .Add(gain.Multiply(measurementNoise).Multiply(gain.Transpose()))
            .Symmetrize();

        return new KalmanUpdate(KalmanState.Create(corrected, covariance), mahalanobis);
    }

    private static Matrix SolveInverse(double[] factor, int size)
    {
        var values = new double[size * size];
        var column = new double[size];
        for (var j = 0; j < size; j++)
        {
            Array.Clear(column);
            column[j] = 1;
            LinearSolver.Solve(factor, size, column);
            for (var i = 0; i < size; i++)
                values[i * size + j] = column[i];
        }

        return Matrix.RowMajor(size, size, values);
    }

    private static void RequireSquare(Matrix matrix, int size, string name)
    {
        ArgumentNullException.ThrowIfNull(matrix);
        if (matrix.Rows != size || matrix.Columns != size)
            throw new ArgumentException($"Matrix must be {size} by {size}.", name);
    }
}
