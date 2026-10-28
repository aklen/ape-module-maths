using Ape.Module.Maths.Algebra;
using Ape.Module.Maths.Estimation;

namespace Ape.Module.Maths.Tests;

public class MahalanobisTests
{
    [Fact]
    public void Distance_on_a_scaled_identity_is_euclidean_over_sigma()
    {
        var covariance = Matrix.Diagonal(4, 4);
        MathsAssert.Near(2.5, Mahalanobis.Distance([3, 4], covariance));
    }

    [Fact]
    public void Squared_distance_uses_the_correlation()
    {
        var covariance = Matrix.RowMajor(2, 2, 2, 1, 1, 2);
        MathsAssert.Near(2, Mahalanobis.Squared([1, -1], covariance));
    }

    [Fact]
    public void Singular_covariance_is_rejected()
    {
        var covariance = Matrix.RowMajor(2, 2, 1, 0, 0, 0);
        Assert.Throws<InvalidOperationException>(() => Mahalanobis.Squared([1, 0], covariance));
    }
}
