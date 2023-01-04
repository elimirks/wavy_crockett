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
