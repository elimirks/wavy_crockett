use nalgebra::*;

pub struct Spline {
    curve_starts: Vec<f64>,
    coefficients: Vec<f64>,
    x_min: f64,
    x_max: f64,
}

impl Spline {
    /// Assumes the points are in increasing X order
    pub fn new(points: &[(f64, f64)]) -> Self {
        let curve_starts = points.iter()
            .take(points.len() - 1)
            .map(|(x, _)| *x)
            .collect::<Vec<_>>();
        let coefficients = spline_coefficients(points);
        let x_min = points[0].0;
        let x_max = points.last().unwrap().0;
        Spline {
            curve_starts,
            coefficients,
            x_min,
            x_max,
        }
    }

    /// t is an percentage (from 0.0 to 1.0) of point along the spline curve
    pub fn interpolate(&self, t: f64) -> f64 {
        let x = self.x_min + t * (self.x_max - self.x_min);
        // Curve index
        let mut ci = 0;
        while ci < self.curve_starts.len() - 1 {
            if self.curve_starts[ci + 1] > x {
                break;
            }
            ci += 1;
        }
        let coeff = &self.coefficients[ci * 4..(ci + 1) * 4];
        coeff[0] * x.powi(3) + coeff[1] * x.powi(2) + coeff[2] * x + coeff[3]
    }
}

/// See https://pythonnumericalmethods.berkeley.edu/notebooks/chapter17.03-Cubic-Spline-Interpolation.html#:~:text=In%20cubic%20spline%20interpolation%20(as,%2C%E2%80%A6%2Cn%E2%88%921.
/// Finds coefficients for a cubic polynomial spline functions
#[allow(clippy::needless_range_loop)]
fn spline_coefficients(points: &[(f64, f64)]) -> Vec<f64> {
    let row_len = 4 * (points.len() - 1);
    let mut spline_matrix = vec![];

    let mut ys = vec![];

    // s_i(x_i) = y_i constraint
    for i in 0..(points.len() - 1) {
        let mut row = vec![0.0; i * 4];
        let (x, y) = points[i];
        row.push(x.powi(3));
        row.push(x.powi(2));
        row.push(x.powi(1));
        row.push(1.0);
        row.resize(row_len, 0.0);
        spline_matrix.push(row);
        ys.push(y);
    }
    // s_i(x_{i+1}) = y_{i+1} constraint
    for i in 0..(points.len() - 1) {
        let mut row = vec![0.0; i * 4];
        let (x, y) = points[i + 1];
        row.push(x.powi(3));
        row.push(x.powi(2));
        row.push(x.powi(1));
        row.push(1.0);
        row.resize(row_len, 0.0);
        spline_matrix.push(row);
        ys.push(y);
    }
    // s'_i(x_{i+1}) = s'_{i+1}(x_{i+1}) constraint
    for i in 0..(points.len() - 2) {
        let mut row = vec![0.0; (i.max(1) - 1) * 4];
        let x = points[i + 1].0;
        row.push(3.0 * x.powi(2));
        row.push(2.0 * x);
        row.push(1.0);
        row.push(0.0);
        row.push(-3.0 * x.powi(2));
        row.push(-2.0 * x);
        row.push(-1.0);
        row.push(0.0);
        row.resize(row_len, 0.0);
        spline_matrix.push(row);
        ys.push(0.0);
    }
    // s''_i(x_{i+1}) = s''_{i+1}(x_{i+1}) constraint
    for i in 0..(points.len() - 2) {
        let mut row = vec![0.0; (i.max(1) - 1) * 4];
        let x = points[i + 1].0;
        row.push(6.0 * x);
        row.push(2.0);
        row.push(0.0);
        row.push(0.0);
        row.push(-6.0 * x);
        row.push(-2.0);
        row.push(0.0);
        row.push(0.0);
        row.resize(row_len, 0.0);
        spline_matrix.push(row);
        ys.push(0.0);
    }
    // s''_0(x_0) = 0 constraint
    {
        let x = points[0].0;
        let mut row = vec![6.0 * x, 2.0];
        row.resize(row_len, 0.0);
        spline_matrix.push(row);
        ys.push(0.0);
    }
    // s''_{n-2}(x_{n-1}) = 0 constraint
    {
        let mut row = vec![0.0; row_len - 4];
        let x = points[points.len() - 1].0;
        row.push(6.0 * x);
        row.push(2.0);
        row.push(0.0);
        row.push(0.0);
        spline_matrix.push(row);
        ys.push(0.0);
    }
    print_matrix(&spline_matrix);
    solve_linear_equation(&spline_matrix, &ys)
}

// Checks if the points are sorted by the x coordinate
pub fn are_points_sorted(points: &[(f64, f64)]) -> bool {
    let mut previous = points[0].0;
    for (x, _) in points.iter().skip(1) {
        if *x <= previous {
            return false;
        }
        previous = *x;
    }
    true
}

/// Finds z for Mz = y (where M is a matrix of size n x n, y is a vector of size n)
fn solve_linear_equation(poro_matrix: &[Vec<f64>], ys: &[f64]) -> Vec<f64> {
    // "PlainOldRustObject" :]
    let n = poro_matrix.len();
    assert!(n == ys.len());
    assert!(poro_matrix.iter().all(|row| row.len() == n));
    let matrix = DMatrix::from_vec(n, n, poro_matrix.iter().flatten().cloned().collect::<Vec<_>>()).transpose();
    let ys_vec = DVector::from_column_slice(ys);
    let inv = matrix.try_inverse().unwrap();

    let result = inv * ys_vec;
    result.iter().cloned().collect::<Vec<_>>()
}

#[allow(unused)]
fn print_matrix(matrix: &[Vec<f64>]) {
    for row in matrix.iter() {
        for value in row.iter() {
            print!("{value: >6.1} ");
        }
        println!();
    }
    println!("{} x {} (rows x cols)", matrix.len(), matrix[0].len());
}

fn factorial(mut n: i64) -> i64 {
    let mut result = 1;
    while n >= 2 {
        result *= n;
        n -= 1;
    }
    result
}

fn binomial(n: i64, k: i64) -> i64 {
    factorial(n) / (factorial(k) * factorial(n - k))
}

// Bezier function for the given time step
// points: List of (x, y) pairs
// t: Time parameter, from 0 to 1
#[allow(unused)]
fn bezier(points: &[(f64, f64)], t: f64) -> (f64, f64) {
    let n = points.len() as i64;
    let mut result = (0.0, 0.0);
    for (i, point) in points.iter().enumerate() {
        let coefficient = binomial(n, i as i64) as f64 *
            (1.0 - t).powi(n as i32 - i as i32) *
            t.powi(i as i32);
        result.0 += point.0 * coefficient;
        result.1 += point.1 * coefficient;
    }
    result
}
