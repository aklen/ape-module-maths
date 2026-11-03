namespace Ape.Module.Maths.Tests;

public class SingleMeasurementAssignmentTests
{
    [Fact]
    public void Single_row_does_not_allocate_a_square_cost_matrix()
    {
        double[] costs=Enumerable.Range(0,128).Select(i=>(double)i).ToArray();
        Hungarian.Minimize([1d],1,1);
        long start=GC.GetAllocatedBytesForCurrentThread();
        var assignment=Assert.Single(Hungarian.Minimize(costs,1,costs.Length));
        long allocated=GC.GetAllocatedBytesForCurrentThread()-start;
        Assert.Equal(new Assignment(0,0),assignment);
        Assert.True(allocated<4096,$"One measurement allocated {allocated} bytes for 128 tracks.");
    }
    [Fact]
    public void Ties_keep_first_column_and_nonfinite_costs_are_absent()
    {
        Assert.Equal(new Assignment(0,1),Assert.Single(Hungarian.Minimize([double.NaN,-1d,-1d,0d,double.NegativeInfinity],1,5)));
        Assert.Empty(Hungarian.Minimize([double.NaN,double.PositiveInfinity],1,2));
        Assert.Equal(new Assignment(0,1),Assert.Single(Hungarian.Minimize([double.MaxValue,1d],1,2)));
    }
}
