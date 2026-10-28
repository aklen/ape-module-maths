namespace Ape.Module.Maths.Algebra;

/// <summary>Dense real matrix, row-major. Values are copied in and are not shared with the caller.</summary>
public sealed class Matrix
{
    private readonly double[] _values;

    private Matrix(int rows, int columns, double[] values)
    {
        Rows = rows;
        Columns = columns;
        _values = values;
    }

    public int Rows { get; }

    public int Columns { get; }

    public double this[int row, int column]
    {
        get
        {
            if ((uint)row >= (uint)Rows)
                throw new ArgumentOutOfRangeException(nameof(row));
            if ((uint)column >= (uint)Columns)
                throw new ArgumentOutOfRangeException(nameof(column));
            return _values[row * Columns + column];
        }
    }

    public static Matrix Identity(int size)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(size, 1);
        var values = new double[size * size];
        for (var i = 0; i < size; i++)
            values[i * size + i] = 1;
        return new Matrix(size, size, values);
    }

    public static Matrix Diagonal(params ReadOnlySpan<double> diagonal)
    {
        if (diagonal.Length == 0)
            throw new ArgumentException("Diagonal is empty.", nameof(diagonal));
        var n = diagonal.Length;
        var values = new double[n * n];
        for (var i = 0; i < n; i++)
            values[i * n + i] = diagonal[i];
        return new Matrix(n, n, values);
    }

    public static Matrix RowMajor(int rows, int columns, params ReadOnlySpan<double> values)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(rows, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(columns, 1);
        if (values.Length != rows * columns)
            throw new ArgumentException("Value count must equal rows times columns.", nameof(values));

        var copy = new double[values.Length];
        values.CopyTo(copy);
        return new Matrix(rows, columns, copy);
    }

    public Matrix Transpose()
    {
        var values = new double[Columns * Rows];
        for (var r = 0; r < Rows; r++)
        {
            for (var c = 0; c < Columns; c++)
                values[c * Rows + r] = _values[r * Columns + c];
        }

        return new Matrix(Columns, Rows, values);
    }

    public Matrix Multiply(Matrix right)
    {
        ArgumentNullException.ThrowIfNull(right);
        if (Columns != right.Rows)
            throw new ArgumentException("Left column count must equal right row count.", nameof(right));

        var values = new double[Rows * right.Columns];
        for (var i = 0; i < Rows; i++)
        {
            for (var k = 0; k < Columns; k++)
            {
                var left = _values[i * Columns + k];
                if (left == 0)
                    continue;
                var rightRow = k * right.Columns;
                var outRow = i * right.Columns;
                for (var j = 0; j < right.Columns; j++)
                    values[outRow + j] += left * right._values[rightRow + j];
            }
        }

        return new Matrix(Rows, right.Columns, values);
    }

    public double[] Multiply(ReadOnlySpan<double> vector)
    {
        if (vector.Length != Columns)
            throw new ArgumentException("Vector length must equal the column count.", nameof(vector));

        var result = new double[Rows];
        for (var i = 0; i < Rows; i++)
        {
            var sum = 0.0;
            var row = i * Columns;
            for (var j = 0; j < Columns; j++)
                sum += _values[row + j] * vector[j];
            result[i] = sum;
        }

        return result;
    }

    public Matrix Add(Matrix other) => Combine(other, 1);

    public Matrix Subtract(Matrix other) => Combine(other, -1);

    /// <summary>Average of this matrix and its transpose. Requires a square matrix.</summary>
    public Matrix Symmetrize()
    {
        if (Rows != Columns)
            throw new InvalidOperationException("Only a square matrix can be symmetrized.");

        var values = new double[_values.Length];
        for (var i = 0; i < Rows; i++)
        {
            values[i * Rows + i] = _values[i * Rows + i];
            for (var j = i + 1; j < Rows; j++)
            {
                var averaged = 0.5 * (_values[i * Rows + j] + _values[j * Rows + i]);
                values[i * Rows + j] = averaged;
                values[j * Rows + i] = averaged;
            }
        }

        return new Matrix(Rows, Rows, values);
    }

    private Matrix Combine(Matrix other, double sign)
    {
        ArgumentNullException.ThrowIfNull(other);
        if (Rows != other.Rows || Columns != other.Columns)
            throw new ArgumentException("Matrix sizes must match.", nameof(other));

        var values = new double[_values.Length];
        for (var i = 0; i < values.Length; i++)
            values[i] = _values[i] + sign * other._values[i];
        return new Matrix(Rows, Columns, values);
    }
}
