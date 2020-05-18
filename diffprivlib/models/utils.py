# MIT License
#
# Copyright (C) IBM Corporation 2019
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Basic functions and other utilities for machine learning models in the differential privacy library
"""
from numbers import Real, Integral

import numpy as np


def _check_bounds(bounds, shape=1, min_separation=1e-5):
    """Input validation for the ``bounds`` parameter.

    Checks that ``bounds`` is composed of a list of tuples of the form (lower, upper), where lower <= upper and both
    are numeric.  Also checks that ``bounds`` contains the appropriate number of dimensions, and that there is a
    ``min_separation`` between the bounds.

    Parameters
    ----------
    bounds : tuple or None
        Tuple of bounds of the form (min, max).

    shape : tuple or int, default: 1
        Number of dimensions to be expected in ``bounds``.

    min_separation : float, default: 1e-5
        The minimum separation between `lower` and `upper` of each dimension.  This separation is enforced if not
        already satisfied.

    Returns
    -------
    bounds : tuple

    """
    if bounds is None:
        return None

    if not isinstance(bounds, tuple):
        raise TypeError("Bounds must be specified as a tuple of (min, max), got {}.".format(type(bounds)))

    if isinstance(shape, Integral):
        shape = (shape, )

    lower, upper = bounds

    if isinstance(lower, Real) and isinstance(upper, Real):
        lower = np.ones(shape=shape, dtype=type(lower)) * lower
        upper = np.ones(shape=shape, dtype=type(upper)) * upper
    else:
        lower = np.asarray(lower)
        upper = np.asarray(upper)

        if lower.shape != shape or upper.shape != shape:
            raise ValueError("Shape of min/max bounds must match input data {}, got min: {}, max: {}.".format(
                shape, lower.shape, upper.shape
            ))

    iterator = np.nditer(lower, flags=['multi_index'])

    while not iterator.finished:
        _lower = lower[iterator.multi_index]
        _upper = upper[iterator.multi_index]

        if not isinstance(_lower, Real) or not isinstance(_upper, Real):
            raise TypeError("Each bound must be numeric, got {} ({}) and {} ({}).".format(_lower, type(_lower),
                                                                                          _upper, type(_upper)))

        if _lower > _upper:
            raise ValueError("For each bound, lower bound must be smaller than upper bound, got {}, {})".format(
                lower, upper))

        if _upper - _lower < min_separation:
            mid = (upper + lower) / 2
            lower[iterator.multi_index] = mid - min_separation / 2
            upper[iterator.multi_index] = mid + min_separation / 2

        iterator.iternext()

    return lower, upper
