from numbers import Real


def _check_bounds(bounds, dims=1, min_seperation=1e-5):
    if bounds is None:
        return None

    if not isinstance(bounds, list):
        raise TypeError("Bounds must be specified as a list of tuples.")

    bounds = bounds.copy()

    if len(bounds) != dims:
        raise ValueError("Number of bounds must match the dimensions of input data")

    for i in range(len(bounds)):
        if not isinstance(bounds[i], tuple):
            raise TypeError("Bounds must be specified as a list of tuples, got a '%s'" % type(bounds[i]))

        lower, upper = bounds[i]

        if not isinstance(lower, Real) or not isinstance(upper, Real) or lower > upper:
            raise ValueError("For each feature bound, lower bound must be smaller than upper bound"
                             "(error found in bound %s" % str(bounds[i]))

        if upper - lower <= min_seperation:
            bounds[i] = (lower - min_seperation, upper + min_seperation)

    return bounds
