"""
Basic functions and other utilities for machine learning models in the differential privacy library
"""
from numbers import Real


def _check_bounds(bounds, dims=1, min_separation=1e-5):
    if bounds is None:
        return None

    if not isinstance(bounds, list):
        raise TypeError("Bounds must be specified as a list of tuples.")

    new_bounds = list()

    if len(bounds) != dims:
        raise ValueError("Number of bounds must match the dimensions of input data")

    for bound in bounds:
        if not isinstance(bound, tuple):
            raise TypeError("Bounds must be specified as a list of tuples, got a '%s'" % type(bound))

        lower, upper = bound

        if not isinstance(lower, Real) or not isinstance(upper, Real) or lower > upper:
            raise ValueError("For each feature bound, lower bound must be smaller than upper bound"
                             "(error found in bound %s" % str(bound))

        if upper - lower <= min_separation:
            bound = (lower - min_separation, upper + min_separation)

        new_bounds.append(bound)

    return new_bounds
