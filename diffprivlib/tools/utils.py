"""
General utilities and tools for performing differentially private operations on data.
"""
from numbers import Real
import numpy as np

from diffprivlib.mechanisms import Laplace

_range = range


def mean(a, epsilon, range, axis=None, dtype=None, out=None, keepdims=np._NoValue):
    """
    Calculates the differentially private mean of a numpy array. Same functionality as numpy's mean.

    :param a: Numpy array for which the mean is sought.
    :param epsilon: Differential privacy parameter epsilon.
    :param range: Range of each dimension of the returned mean.
    :type range: float or array-like, same shape as mean
    :param axis: See np.mean.
    :param dtype: See np.mean.
    :param out: See np.mean.
    :param keepdims: See np.mean.
    :return:
    """
    if isinstance(axis, tuple):
        temp_axis = axis
    elif axis is not None:
        try:
            temp_axis = tuple(axis)
        except TypeError:
            temp_axis = (axis,)
    else:
        temp_axis = tuple(_range(len(a.shape)))

    n = 1
    for i in temp_axis:
        n *= a.shape[i]

    actual_mean = np.mean(a, axis, dtype, out, keepdims)

    if isinstance(range, Real):
        ranges = np.ones_like(actual_mean) * range
    else:
        ranges = np.array(range)

    if not (ranges > 0).all():
        raise ValueError("Ranges must be specified for each value returned by np.mean(), and must be non-negative")

    if isinstance(actual_mean, np.ndarray):
        dp_mean = np.zeros_like(actual_mean)
        iterator = np.nditer(actual_mean, flags=['multi_index'])

        while not iterator.finished:
            try:
                dp_mech = Laplace().set_epsilon(epsilon).set_sensitivity(ranges[iterator.multi_index] / n)
            except IndexError:
                raise ValueError("Shape of range must be same as shape of np.mean")

            dp_mean[iterator.multi_index] = dp_mech.randomise(float(iterator[0]))
            iterator.iternext()

        return dp_mean

    range = np.ravel(ranges)[0]
    dp_mech = Laplace().set_epsilon(epsilon).set_sensitivity(range / n)
    return dp_mech.randomise(actual_mean)


def var():
    pass
