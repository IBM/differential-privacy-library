# MIT License
#
# Copyright (C) IBM Corporation 2020
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
Validation functions for the differential privacy library
"""
from numbers import Real, Integral

import numpy as np


def check_epsilon_delta(epsilon, delta, allow_zero=False):
    """Checks that epsilon and delta are valid values for differential privacy.  Throws an error if checks fail,
    otherwise returns nothing.

    As well as the requirements of epsilon and delta separately, both cannot be simultaneously zero, unless
    ``allow_zero`` is set to ``True``.

    Parameters
    ----------
    epsilon : float
        Epsilon parameter for differential privacy.  Must be non-negative.

    delta : float
        Delta parameter for differential privacy.  Must be on the unit interval, [0, 1].

    allow_zero : bool, default: False
        Allow epsilon and delta both be zero.

    """
    if not isinstance(epsilon, Real) or not isinstance(delta, Real):
        raise TypeError("Epsilon and delta must be numeric")

    if epsilon < 0:
        raise ValueError("Epsilon must be non-negative")

    if not 0 <= delta <= 1:
        raise ValueError("Delta must be in [0, 1]")

    if not allow_zero and epsilon + delta == 0:
        raise ValueError("Epsilon and Delta cannot both be zero")


def check_bounds(bounds, shape=1, min_separation=1e-5):
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


def clip_to_norm(array, clip):
    """Clips the examples of a 2-dimensional array to a given maximum norm.

    Parameters
    ----------
    array : np.ndarray
        Array to be clipped.  After clipping, all examples have a 2-norm of at most `clip`.

    clip : float
        Norm at which to clip each example

    Returns
    -------
    array : np.ndarray
        The clipped array.

    """
    if not isinstance(array, np.ndarray):
        raise TypeError("Input array must be a numpy array, got {}.".format(type(array)))
    if array.ndim != 2:
        raise ValueError("input array must be 2-dimensional, got {} dimensions.".format(array.ndim))
    if not isinstance(clip, Real):
        raise TypeError("Clip value must be numeric, got {}.".format(type(clip)))
    if clip <= 0:
        raise ValueError("Clip value must be strictly positive, got {}.".format(clip))

    norms = np.linalg.norm(array, axis=1) / clip
    norms[norms < 1] = 1

    return array / norms[:, np.newaxis]


def clip_to_bounds(array, bounds):
    """Clips the examples of a 2-dimensional array to given bounds.

    Parameters
    ----------
    array : np.ndarray
        Array to be clipped.  After clipping, all examples have a 2-norm of at most `clip`.

    bounds : tuple
        Tuple of bounds of the form (min, max) which the array is to be clipped to. `min` and `max` must be scalar,
        unless array is 2-dimensional.

    Returns
    -------
    array : np.ndarray
        The clipped array.

    """
    if not isinstance(array, np.ndarray):
        raise TypeError("Input array must be a numpy array, got {}.".format(type(array)))

    if np.shape(bounds[0]) != np.shape(bounds[1]):
        raise ValueError("Bounds must be of the same shape, got {} and {}.".format(np.shape(bounds[0]),
                                                                                   np.shape(bounds[1])))

    lower, upper = check_bounds(bounds, np.shape(bounds[0]) or 1, min_separation=0)
    clipped_array = array.copy()

    if np.allclose(lower, np.min(lower)) and np.allclose(upper, np.max(upper)):
        clipped_array = np.clip(clipped_array, np.min(lower), np.max(upper))
    else:
        if array.ndim != 2:
            raise ValueError("For non-scalar bounds, input array must be 2-dimensional. Got %d dimensions." %
                             array.ndim)

        for feature in range(array.shape[1]):
            clipped_array[:, feature] = np.clip(array[:, feature], lower[feature], upper[feature])

    return clipped_array
