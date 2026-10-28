using Ape.Module.Maths.Algebra;
using Ape.Module.Maths.Estimation;

namespace Ape.Module.Maths.Tests;

public class KalmanFilterTests
{
    [Fact]
    public void Predict_applies_transition_and_adds_process_noise()
    {
        var state = KalmanState.Create([1, 2], Matrix.Identity(2));
        var transition = Matrix.RowMajor(2, 2, 1, 0.5, 0, 1);
        var noise = Matrix.Diagonal(0.25, 0);

        var predicted = KalmanFilter.Predict(state, transition, noise);

        MathsAssert.Near(1.0 + 2.0 * 0.5, predicted.Mean(0));
        MathsAssert.Near(2, predicted.Mean(1));
        MathsAssert.Matrix(predicted.Covariance, 1e-12, 1.5, 0.5, 0.5, 1);
    }

    [Fact]
    public void Update_position_matches_joseph_form()
    {
        var state = KalmanState.Create([0, 0], Matrix.Identity(2));
        var observation = Matrix.RowMajor(1, 2, 1, 0);
        var noise = Matrix.RowMajor(1, 1, 1);

        var updated = KalmanFilter.Update(state, [2], observation, noise);

        MathsAssert.Near(1, updated.State.Mean(0));
        MathsAssert.Near(0, updated.State.Mean(1));
        MathsAssert.Matrix(updated.State.Covariance, 1e-12, 0.5, 0, 0, 1);
        MathsAssert.Near(2, updated.MahalanobisSquared);
    }

    [Fact]
    public void Predict_with_constant_velocity_moves_each_axis()
    {
        var state = KalmanState.Create([1, 2, 3, 4], Matrix.Diagonal(1, 1, 1, 1));
        var transition = ConstantVelocity.Transition(0.5, spatialDimension: 2);
        var noise = ConstantVelocity.ProcessNoise(0.5, noiseDensity: 0, spatialDimension: 2);

        var predicted = KalmanFilter.Predict(state, transition, noise);

        MathsAssert.Near(2.5, predicted.Mean(0));
        MathsAssert.Near(4, predicted.Mean(1));
        MathsAssert.Near(3, predicted.Mean(2));
        MathsAssert.Near(4, predicted.Mean(3));
    }
}
