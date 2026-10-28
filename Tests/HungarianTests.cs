using Ape.Module.Maths;

namespace Ape.Module.Maths.Tests;

public class HungarianTests
{
    [Fact]
    public void Minimize_matches_brute_force_on_a_square_cost()
    {
        double[,] cost =
        {
            { 7, 5, 9, 8 },
            { 6, 4, 2, 3 },
            { 5, 8, 7, 4 },
            { 9, 6, 3, 1 },
        };

        var assigned = Hungarian.Minimize(Flatten(cost), 4, 4);

        Assert.Equal(4, assigned.Count);
        Assert.Equal(assigned.Select(pair => pair.Column).Distinct().Count(), assigned.Count);
        Assert.Equal(Brute(cost), Sum(cost, assigned));
    }

    [Fact]
    public void Minimize_matches_brute_force_when_a_pair_is_absent()
    {
        double[,] cost =
        {
            { 4, double.NaN, 1 },
            { 2, 0, 5 },
        };

        var assigned = Hungarian.Minimize(Flatten(cost), 2, 3);

        Assert.Equal(Brute(cost), Sum(cost, assigned));
        Assert.Contains(assigned, pair => pair.Row == 0 && pair.Column == 2);
        Assert.Contains(assigned, pair => pair.Row == 1 && pair.Column == 1);
    }

    [Fact]
    public void Extra_row_stays_unmatched()
    {
        var assigned = Hungarian.Minimize([5, 1], rows: 2, columns: 1);

        var only = Assert.Single(assigned);
        Assert.Equal(1, only.Row);
        Assert.Equal(0, only.Column);
    }

    [Fact]
    public void Absent_matrix_assigns_nothing()
    {
        var assigned = Hungarian.Minimize([double.NaN, double.PositiveInfinity], rows: 1, columns: 2);
        Assert.Empty(assigned);
    }

    [Fact]
    public void Unique_diagonal_is_chosen()
    {
        var assigned = Hungarian.Minimize([1, 4, 3, 2], rows: 2, columns: 2);
        Assert.Equal(2, assigned.Count);
        Assert.Contains(assigned, pair => pair.Row == 0 && pair.Column == 0);
        Assert.Contains(assigned, pair => pair.Row == 1 && pair.Column == 1);
    }

    private static double Brute(double[,] cost)
    {
        var bestCount = -1;
        var bestSum = 0.0;
        var used = new bool[cost.GetLength(1)];
        Search(cost, used, row: 0, count: 0, sum: 0, ref bestCount, ref bestSum);
        return bestCount < 0 ? 0 : bestSum;

        static void Search(
            double[,] cost,
            bool[] used,
            int row,
            int count,
            double sum,
            ref int bestCount,
            ref double bestSum)
        {
            if (row == cost.GetLength(0))
            {
                if (count > bestCount || (count == bestCount && sum < bestSum))
                {
                    bestCount = count;
                    bestSum = sum;
                }

                return;
            }

            Search(cost, used, row + 1, count, sum, ref bestCount, ref bestSum);
            for (var column = 0; column < cost.GetLength(1); column++)
            {
                if (used[column] || !double.IsFinite(cost[row, column]))
                    continue;
                used[column] = true;
                Search(cost, used, row + 1, count + 1, sum + cost[row, column], ref bestCount, ref bestSum);
                used[column] = false;
            }
        }
    }

    private static double Sum(double[,] cost, IReadOnlyList<Assignment> assigned)
    {
        var sum = 0.0;
        foreach (var pair in assigned)
            sum += cost[pair.Row, pair.Column];
        return sum;
    }

    private static double[] Flatten(double[,] cost)
    {
        var rows = cost.GetLength(0);
        var columns = cost.GetLength(1);
        var flat = new double[rows * columns];
        for (var row = 0; row < rows; row++)
        {
            for (var column = 0; column < columns; column++)
                flat[row * columns + column] = cost[row, column];
        }

        return flat;
    }
}
