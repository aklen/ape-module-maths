using Ape.Module.Maths.Algebra;
using Ape.Module.Maths.Estimation;

namespace Ape.Module.Maths.Tests;

/// <summary>Slice 1 gate: causal LG factor-graph filter matches Kalman on the same measurement prefix.</summary>
public class CausalFactorGraphFilterTests
{
    /// <summary>Documented absolute tolerance for mean and covariance entries (Joseph Kalman vs info-form FG).</summary>
    public const double EquivalenceEpsilon = 1e-9;

    [Fact]
    public void Prior_marginal_matches_kalman_state()
    {
        var prior = KalmanState.Create([1, 2, 0, 0], Matrix.Diagonal(4, 4, 9, 9));
        var fg = CausalFactorGraphFilter.FromPrior(prior);
        AssertBeliefClose(prior, fg.CurrentMarginal(), EquivalenceEpsilon);
    }

    [Fact]
    public void Cv_predict_and_position_update_match_kalman_within_epsilon()
    {
        var prior = KalmanState.Create([0, 0, 0, 0, 0, 0], Matrix.Diagonal(15 * 15, 15 * 15, 15 * 15, 30 * 30, 30 * 30, 30 * 30));
        var dt = 1.0;
        var q = 4.0;
        var F = ConstantVelocity.Transition(dt, 3);
        var Q = ConstantVelocity.ProcessNoise(dt, q, 3);
        var H = Matrix.RowMajor(3, 6, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0);
        var R = Matrix.Diagonal(15 * 15, 15 * 15, 15 * 15);
        double[] z = [10, 2, 5];

        var kalman = KalmanFilter.Update(KalmanFilter.Predict(prior, F, Q), z, H, R).State;

        var fg = CausalFactorGraphFilter.FromPrior(prior);
        fg.ApplyTransitionFactor(F, Q);
        fg.ApplyMeasurementFactor(z, H, R);

        AssertBeliefClose(kalman, fg.CurrentMarginal(), EquivalenceEpsilon);
    }

    [Fact]
    public void Causal_prefix_multi_step_matches_kalman_not_full_batch_smoothing()
    {
        var prior = KalmanState.Create([0, 0, 0, 1, 0, 0], Matrix.Diagonal(25, 25, 25, 16, 16, 16));
        var H = Matrix.RowMajor(3, 6, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0);
        var R = Matrix.Diagonal(4, 4, 4);
        var measurements = new double[][]
        {
            [1.0, 0.1, 0.0],
            [2.1, 0.0, 0.2],
            [2.9, -0.1, 0.1],
            [4.0, 0.2, -0.1],
        };

        var kalman = prior;
        var fg = CausalFactorGraphFilter.FromPrior(prior);
        for (var t = 0; t < measurements.Length; t++)
        {
            var dt = 1.0;
            var F = ConstantVelocity.Transition(dt, 3);
            var Q = ConstantVelocity.ProcessNoise(dt, noiseDensity: 1.0, 3);
            kalman = KalmanFilter.Update(KalmanFilter.Predict(kalman, F, Q), measurements[t], H, R).State;
            fg.ApplyTransitionFactor(F, Q);
            fg.ApplyMeasurementFactor(measurements[t], H, R);
            AssertBeliefClose(kalman, fg.CurrentMarginal(), EquivalenceEpsilon);
        }
    }

    [Fact]
    public void Coast_only_transition_matches_kalman_predict()
    {
        var prior = KalmanState.Create([3, 4, 5, 1, -1, 0.5], Matrix.Diagonal(1, 2, 3, 0.5, 0.5, 0.5));
        var F = ConstantVelocity.Transition(0.25, 3);
        var Q = ConstantVelocity.ProcessNoise(0.25, 2.0, 3);
        var kalman = KalmanFilter.Predict(prior, F, Q);
        var fg = CausalFactorGraphFilter.FromPrior(prior);
        fg.ApplyTransitionFactor(F, Q);
        AssertBeliefClose(kalman, fg.CurrentMarginal(), EquivalenceEpsilon);
    }

    [Theory]
    [InlineData(0.1)]
    [InlineData(0.001)]
    public void Small_dt_cv_birth_style_prior_matches_kalman_within_epsilon(double dt)
    {
        // Baseline kinematic birth: position σ²=225, velocity σ²=900; ProcessNoise density=4.
        var prior = KalmanState.Create(
            [0, 0, 0, 0, 0, 0],
            Matrix.Diagonal(225, 225, 225, 900, 900, 900));
        var F = ConstantVelocity.Transition(dt, 3);
        var Q = ConstantVelocity.ProcessNoise(dt, noiseDensity: 4.0, 3);
        var H = Matrix.RowMajor(3, 6, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0);
        var R = Matrix.Diagonal(225, 225, 225);
        double[] z = [1.5, -0.25, 12.0];

        var kalman = KalmanFilter.Update(KalmanFilter.Predict(prior, F, Q), z, H, R).State;
        var fg = CausalFactorGraphFilter.FromPrior(prior);
        fg.ApplyTransitionFactor(F, Q);
        fg.ApplyMeasurementFactor(z, H, R);
        AssertBeliefClose(kalman, fg.CurrentMarginal(), EquivalenceEpsilon);
    }

    [Fact]
    public void Zero_process_noise_deterministic_transition_matches_kalman()
    {
        var prior = KalmanState.Create([1, 2, 3, 4, 5, 6], Matrix.Diagonal(225, 225, 225, 900, 900, 900));
        var F = ConstantVelocity.Transition(0.5, 3);
        var Q = ConstantVelocity.ProcessNoise(0.5, noiseDensity: 0, 3);
        var kalman = KalmanFilter.Predict(prior, F, Q);
        var fg = CausalFactorGraphFilter.FromPrior(prior);
        fg.ApplyTransitionFactor(F, Q);
        AssertBeliefClose(kalman, fg.CurrentMarginal(), EquivalenceEpsilon);
    }

    private static void AssertBeliefClose(KalmanState expected, KalmanState actual, double epsilon)
    {
        Assert.Equal(expected.Size, actual.Size);
        for (var i = 0; i < expected.Size; i++)
            MathsAssert.Near(expected.Mean(i), actual.Mean(i), epsilon);
        for (var r = 0; r < expected.Size; r++)
            for (var c = 0; c < expected.Size; c++)
                MathsAssert.Near(expected.Covariance[r, c], actual.Covariance[r, c], epsilon);
    }
}
