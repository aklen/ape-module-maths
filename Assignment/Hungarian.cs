namespace Ape.Module.Maths;

/// <summary>One selected pair. Row and column are indexes into the cost matrix.</summary>
public readonly record struct Assignment(int Row, int Column);

/// <summary>
/// Minimum-cost assignment. A non-finite cost is an absent pair and is left unmatched.
/// When a row or a column has no finite pair, it is omitted from the result.
/// Among matchings that use as many finite pairs as possible, the solver minimizes the sum.
/// </summary>
public static class Hungarian
{
    public static IReadOnlyList<Assignment> Minimize(ReadOnlySpan<double> cost, int rows, int columns)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(rows);
        ArgumentOutOfRangeException.ThrowIfNegative(columns);
        if (cost.Length != rows * columns)
            throw new ArgumentException("Cost length must equal rows times columns.", nameof(cost));
        if (rows == 0 || columns == 0)
            return [];

        double minimum = 0, maximum = 0;
        var anyFinite = false;
        for (var i = 0; i < cost.Length; i++)
        {
            var value = cost[i];
            if (!double.IsFinite(value))
                continue;
            if (!anyFinite)
            {
                minimum = maximum = value;
                anyFinite = true;
            }
            else
            {
                minimum = Math.Min(minimum, value);
                maximum = Math.Max(maximum, value);
            }
        }

        if (!anyFinite)
            return [];

        var span = Math.Max(maximum, 0) - Math.Min(minimum, 0);
        var n = Math.Max(rows, columns);
        var forbidden = span * n + 1;
        var square = new double[n, n];
        for (var r = 0; r < n; r++)
        {
            for (var c = 0; c < n; c++)
            {
                if (r < rows && c < columns && double.IsFinite(cost[r * columns + c]))
                    square[r, c] = cost[r * columns + c];
                else
                    square[r, c] = forbidden;
            }
        }

        var columnOfRow = MinimizeSquare(square);
        var matched = new List<Assignment>();
        for (var row = 0; row < rows; row++)
        {
            var column = columnOfRow[row];
            if (column < 0 || column >= columns)
                continue;
            if (!double.IsFinite(cost[row * columns + column]))
                continue;
            matched.Add(new Assignment(row, column));
        }

        return matched;
    }

    /// <summary>Square min-cost assignment. Returns the column chosen for each row.</summary>
    private static int[] MinimizeSquare(double[,] cost)
    {
        var n = cost.GetLength(0);
        var u = new double[n + 1];
        var v = new double[n + 1];
        var parent = new int[n + 1];
        var way = new int[n + 1];

        for (var i = 1; i <= n; i++)
        {
            parent[0] = i;
            var column = 0;
            var minimumSlack = new double[n + 1];
            var used = new bool[n + 1];
            Array.Fill(minimumSlack, double.PositiveInfinity);

            do
            {
                used[column] = true;
                var row = parent[column];
                var delta = double.PositiveInfinity;
                var nextColumn = 0;
                for (var j = 1; j <= n; j++)
                {
                    if (used[j])
                        continue;
                    var reduced = cost[row - 1, j - 1] - u[row] - v[j];
                    if (reduced < minimumSlack[j])
                    {
                        minimumSlack[j] = reduced;
                        way[j] = column;
                    }

                    if (minimumSlack[j] < delta)
                    {
                        delta = minimumSlack[j];
                        nextColumn = j;
                    }
                }

                for (var j = 0; j <= n; j++)
                {
                    if (used[j])
                    {
                        u[parent[j]] += delta;
                        v[j] -= delta;
                    }
                    else
                    {
                        minimumSlack[j] -= delta;
                    }
                }

                column = nextColumn;
            } while (parent[column] != 0);

            do
            {
                var previous = way[column];
                parent[column] = parent[previous];
                column = previous;
            } while (column != 0);
        }

        var columnOfRow = new int[n];
        Array.Fill(columnOfRow, -1);
        for (var column = 1; column <= n; column++)
        {
            if (parent[column] != 0)
                columnOfRow[parent[column] - 1] = column - 1;
        }

        return columnOfRow;
    }
}
