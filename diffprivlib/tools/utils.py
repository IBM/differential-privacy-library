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
#
#
# Copyright (c) 2005-2019, NumPy Developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice, this list of conditions and the following
#       disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
#       following disclaimer in the documentation and/or other materials provided with the distribution.
#
#     * Neither the name of the NumPy Developers nor the names of any contributors may be used to endorse or promote
#       products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
General utilities and tools for performing differentially private operations on data.
"""
import warnings
from numbers import Real
import numpy as np
from numpy.core import multiarray as mu
from numpy.core import umath as um

from diffprivlib.mechanisms import Laplace, LaplaceBoundedDomain
from diffprivlib.utils import PrivacyLeakWarning

_range = range


def mean(a, epsilon=1.0, range=None, axis=None, dtype=None, out=None, keepdims=np._NoValue):
    r"""
    Compute the differentially private arithmetic mean along the specified axis.

    Returns the average of the array elements with differential privacy.  The average is taken over the flattened array
    by default, otherwise over the specified axis. Noise is added using :class:`.Laplace` to satisfy differential
    privacy, where sensitivity is calculated using `range`.  Users are advised to consult the documentation of
    :obj:`numpy.mean` for further details, as the behaviour of `mean` closely follows its Numpy variant.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose mean is desired. If `a` is not an array, a conversion is attempted.

    epsilon : float
        Privacy parameter :math:`\epsilon`.

    range : array_like
        Range of each dimension of the returned mean. Same shape as np.mean(a)

    axis : None or int or tuple of ints, optional
        Axis or axes along which the means are computed. The default is to compute the mean of the flattened array.

        If this is a tuple of ints, a mean is performed over multiple axes, instead of a single axis or all the axes as
        before.

    dtype : data-type, optional
        Type to use in computing the mean.  For integer inputs, the default is `float64`; for floating point inputs, it
        is the same as the input dtype.

    out : ndarray, optional
        Alternate output array in which to place the result.  The default is ``None``; if provided, it must have the
        same shape as the expected output, but the type will be cast if necessary.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this
        option, the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be passed through to the `mean` method of sub-classes
        of `ndarray`, however any non-default value will be.  If the sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    Returns
    -------
    m : ndarray, see dtype parameter above
        If `out=None`, returns a new array containing the mean values, otherwise a reference to the output array is
        returned.

    See Also
    --------
    std, var, nanmean

    """
    return _mean(a, epsilon, range, axis, dtype, out, keepdims, False)


def nanmean(a, epsilon=1.0, range=None, axis=None, dtype=None, out=None, keepdims=np._NoValue):
    r"""
    Compute the differentially private arithmetic mean along the specified axis, ignoring NaNs.

    Returns the average of the array elements with differential privacy.  The average is taken over the flattened array
    by default, otherwise over the specified axis. Noise is added using :class:`.Laplace` to satisfy differential
    privacy, where sensitivity is calculated using `range`.  Users are advised to consult the documentation of
    :obj:`numpy.mean` for further details, as the behaviour of `mean` closely follows its Numpy variant.

    For all-NaN slices, NaN is returned and a `RuntimeWarning` is raised.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose mean is desired. If `a` is not an array, a conversion is attempted.

    epsilon : float
        Privacy parameter :math:`\epsilon`.

    range : array_like
        Range of each dimension of the returned mean. Same shape as np.mean(a)

    axis : None or int or tuple of ints, optional
        Axis or axes along which the means are computed. The default is to compute the mean of the flattened array.

        If this is a tuple of ints, a mean is performed over multiple axes, instead of a single axis or all the axes as
        before.

    dtype : data-type, optional
        Type to use in computing the mean.  For integer inputs, the default is `float64`; for floating point inputs, it
        is the same as the input dtype.

    out : ndarray, optional
        Alternate output array in which to place the result.  The default is ``None``; if provided, it must have the
        same shape as the expected output, but the type will be cast if necessary.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this
        option, the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be passed through to the `mean` method of sub-classes
        of `ndarray`, however any non-default value will be.  If the sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    Returns
    -------
    m : ndarray, see dtype parameter above
        If `out=None`, returns a new array containing the mean values, otherwise a reference to the output array is
        returned.

    See Also
    --------
    std, var, mean

    """
    return _mean(a, epsilon, range, axis, dtype, out, keepdims, True)


def _mean(a, epsilon=1.0, range=None, axis=None, dtype=None, out=None, keepdims=np._NoValue, nan=False):
    if isinstance(axis, tuple):
        temp_axis = axis
    elif axis is not None:
        try:
            temp_axis = tuple(axis)
        except TypeError:
            temp_axis = (axis,)
    else:
        temp_axis = tuple(_range(len(a.shape)))

    num_datapoints = 1
    for i in temp_axis:
        num_datapoints *= a.shape[i]

    if nan:
        actual_mean = np.nanmean(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
    else:
        actual_mean = np.mean(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

    if range is None:
        warnings.warn("Range parameter hasn't been specified, so falling back to determining range from the data.\n"
                      "This will result in additional privacy leakage. To ensure differential privacy with no "
                      "additional privacy loss, specify `range` for each valued returned by np.mean().",
                      PrivacyLeakWarning)

        ranges = np.maximum(np.ptp(a, axis=axis), 1e-5)
    elif isinstance(range, Real):
        ranges = np.ones_like(actual_mean) * range
    else:
        ranges = np.array(range)

    if not (ranges > 0).all():
        raise ValueError("Ranges must be specified for each value returned by np.mean(), and must be non-negative")
    if ranges.shape != actual_mean.shape:
        raise ValueError("Shape of range must be same as shape of np.mean")

    if isinstance(actual_mean, np.ndarray):
        dp_mean = np.zeros_like(actual_mean)
        iterator = np.nditer(actual_mean, flags=['multi_index'])

        while not iterator.finished:
            dp_mech = Laplace().set_epsilon(epsilon).set_sensitivity(ranges[iterator.multi_index] / num_datapoints)

            dp_mean[iterator.multi_index] = dp_mech.randomise(float(iterator[0]))
            iterator.iternext()

        return dp_mean

    range = np.ravel(ranges)[0]
    dp_mech = Laplace().set_epsilon(epsilon).set_sensitivity(range / num_datapoints)

    return dp_mech.randomise(actual_mean)


def var(a, epsilon=1.0, range=None, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue):
    r"""
    Compute the differentially private variance along the specified axis.

    Returns the variance of the array elements, a measure of the spread of a distribution, with differential privacy.
    The variance is computer for the flattened array by default, otherwise over the specified axis. Noise is added using
    :class:`.LaplaceBoundedDomain` to satisfy differential privacy, where sensitivity is calculated using `range`. Users
    are advised to consult the documentation of :obj:`numpy.var` for further details, as the behaviour of `var` closely
    follows its Numpy variant.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose variance is desired.  If `a` is not an array, a conversion is attempted.

    epsilon : float
        Privacy parameter :math:`\epsilon`.

    range : array_like
        Range of each dimension of the returned var. Same shape as np.var(a)

    axis : None or int or tuple of ints, optional
        Axis or axes along which the variance is computed.  The default is to compute the variance of the flattened
        array.

        If this is a tuple of ints, a variance is performed over multiple axes, instead of a single axis or all the axes
        as before.

    dtype : data-type, optional
        Type to use in computing the variance.  For arrays of integer type the default is `float32`; for arrays of float
        types it is the same as the array type.

    out : ndarray, optional
        Alternate output array in which to place the result.  It must have the same shape as the expected output, but
        the type is cast if necessary.

    ddof : int, optional
        "Delta Degrees of Freedom": the divisor used in the calculation is ``N - ddof``, where ``N`` represents the
        number of elements. By default `ddof` is zero.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this
        option, the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be passed through to the `var` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    Returns
    -------
    variance : ndarray, see dtype parameter above
        If ``out=None``, returns a new array containing the variance; otherwise, a reference to the output array is
        returned.

    See Also
    --------
    std , mean, nanvar

    """
    return _var(a, epsilon, range, axis, dtype, out, ddof, keepdims, False)


def nanvar(a, epsilon=1.0, range=None, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue):
    r"""
    Compute the differentially private variance along the specified axis, ignoring NaNs.

    Returns the variance of the array elements, a measure of the spread of a distribution, with differential privacy.
    The variance is computer for the flattened array by default, otherwise over the specified axis. Noise is added using
    :class:`.LaplaceBoundedDomain` to satisfy differential privacy, where sensitivity is calculated using `range`. Users
    are advised to consult the documentation of :obj:`numpy.var` for further details, as the behaviour of `var` closely
    follows its Numpy variant.

    For all-NaN slices, NaN is returned and a `RuntimeWarning` is raised.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose variance is desired.  If `a` is not an array, a conversion is attempted.

    epsilon : float
        Privacy parameter :math:`\epsilon`.

    range : array_like
        Range of each dimension of the returned var. Same shape as np.var(a)

    axis : None or int or tuple of ints, optional
        Axis or axes along which the variance is computed.  The default is to compute the variance of the flattened
        array.

        If this is a tuple of ints, a variance is performed over multiple axes, instead of a single axis or all the axes
        as before.

    dtype : data-type, optional
        Type to use in computing the variance.  For arrays of integer type the default is `float32`; for arrays of float
        types it is the same as the array type.

    out : ndarray, optional
        Alternate output array in which to place the result.  It must have the same shape as the expected output, but
        the type is cast if necessary.

    ddof : int, optional
        "Delta Degrees of Freedom": the divisor used in the calculation is ``N - ddof``, where ``N`` represents the
        number of elements. By default `ddof` is zero.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this
        option, the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be passed through to the `var` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    Returns
    -------
    variance : ndarray, see dtype parameter above
        If ``out=None``, returns a new array containing the variance; otherwise, a reference to the output array is
        returned.

    See Also
    --------
    std , mean, var

    """
    return _var(a, epsilon, range, axis, dtype, out, ddof, keepdims, True)


def _var(a, epsilon=1.0, range=None, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue, nan=False):
    if isinstance(axis, tuple):
        temp_axis = axis
    elif axis is not None:
        try:
            temp_axis = tuple(axis)
        except TypeError:
            temp_axis = (axis,)
    else:
        temp_axis = tuple(_range(len(a.shape)))

    num_datapoints = 1
    for i in temp_axis:
        num_datapoints *= a.shape[i]

    if nan:
        actual_var = np.nanvar(a, axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)
    else:
        actual_var = np.var(a, axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)

    if range is None:
        warnings.warn("Range parameter hasn't been specified, so falling back to determining range from the data.\n"
                      "This will result in additional privacy leakage. To ensure differential privacy with no "
                      "additional privacy loss, specify `range` for each valued returned by np.mean().",
                      PrivacyLeakWarning)

        ranges = np.maximum(np.ptp(a, axis=axis), 1e-5)
    elif isinstance(range, Real):
        ranges = np.ones_like(actual_var) * range
    else:
        ranges = np.array(range)

    if not (ranges > 0).all():
        raise ValueError("Ranges must be specified for each value returned by np.var(), and must be non-negative")
    if ranges.shape != actual_var.shape:
        raise ValueError("Shape of range must be same as shape of np.var()")

    if isinstance(actual_var, np.ndarray):
        dp_var = np.zeros_like(actual_var)
        iterator = np.nditer(actual_var, flags=['multi_index'])

        while not iterator.finished:
            dp_mech = LaplaceBoundedDomain().set_epsilon(epsilon).set_bounds(0, float("inf")) \
                .set_sensitivity((ranges[iterator.multi_index] / num_datapoints) ** 2 * (num_datapoints - 1))

            dp_var[iterator.multi_index] = dp_mech.randomise(float(iterator[0]))
            iterator.iternext()

        return dp_var

    range = np.ravel(ranges)[0]
    dp_mech = LaplaceBoundedDomain().set_epsilon(epsilon).set_bounds(0, float("inf")). \
        set_sensitivity(range ** 2 / num_datapoints)

    return dp_mech.randomise(actual_var)


def std(a, epsilon=1.0, range=None, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue):
    r"""
    Compute the standard deviation along the specified axis.

    Returns the standard deviation of the array elements, a measure of the spread of a distribution, with differential
    privacy.  The standard deviation is computed for the flattened array by default, otherwise over the specified axis.
    Noise is added using :class:`.LaplaceBoundedDomain` to satisfy differential privacy, where sensitivity is
    calculated using `range`.  Users are advised to consult the documentation of :obj:`numpy.std` for further details,
    as the behaviour of `std` closely follows its Numpy variant.

    Parameters
    ----------
    a : array_like
        Calculate the standard deviation of these values.

    epsilon : float
        Privacy parameter :math:`\epsilon`.

    range : array_like
        Range of each dimension of the returned var. Same shape as np.var(a)

    axis : None or int or tuple of ints, optional
        Axis or axes along which the standard deviation is computed. The default is to compute the standard deviation of
        the flattened array.

        If this is a tuple of ints, a standard deviation is performed over multiple axes, instead of a single axis or
        all the axes as before.

    dtype : dtype, optional
        Type to use in computing the standard deviation. For arrays of integer type the default is float64, for arrays
        of float types it is the same as the array type.

    out : ndarray, optional
        Alternative output array in which to place the result. It must have the same shape as the expected output but
        the type (of the calculated values) will be cast if necessary.

    ddof : int, optional
        Means Delta Degrees of Freedom.  The divisor used in calculations is ``N - ddof``, where ``N`` represents the
        number of elements. By default `ddof` is zero.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this
        option, the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be passed through to the `std` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    Returns
    -------
    standard_deviation : ndarray, see dtype parameter above.
        If `out` is None, return a new array containing the standard deviation, otherwise return a reference to the
        output array.

    See Also
    --------
    var, mean, nanstd

    """
    return _std(a, epsilon, range, axis, dtype, out, ddof, keepdims, False)


def nanstd(a, epsilon=1.0, range=None, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue):
    r"""
    Compute the standard deviation along the specified axis, ignoring NaNs.

    Returns the standard deviation of the array elements, a measure of the spread of a distribution, with differential
    privacy.  The standard deviation is computed for the flattened array by default, otherwise over the specified axis.
    Noise is added using :class:`.LaplaceBoundedDomain` to satisfy differential privacy, where sensitivity is
    calculated using `range`.  Users are advised to consult the documentation of :obj:`numpy.std` for further details,
    as the behaviour of `std` closely follows its Numpy variant.

    For all-NaN slices, NaN is returned and a `RuntimeWarning` is raised.

    Parameters
    ----------
    a : array_like
        Calculate the standard deviation of these values.

    epsilon : float
        Privacy parameter :math:`\epsilon`.

    range : array_like
        Range of each dimension of the returned var. Same shape as np.var(a)

    axis : None or int or tuple of ints, optional
        Axis or axes along which the standard deviation is computed. The default is to compute the standard deviation of
        the flattened array.

        If this is a tuple of ints, a standard deviation is performed over multiple axes, instead of a single axis or
        all the axes as before.

    dtype : dtype, optional
        Type to use in computing the standard deviation. For arrays of integer type the default is float64, for arrays
        of float types it is the same as the array type.

    out : ndarray, optional
        Alternative output array in which to place the result. It must have the same shape as the expected output but
        the type (of the calculated values) will be cast if necessary.

    ddof : int, optional
        Means Delta Degrees of Freedom.  The divisor used in calculations is ``N - ddof``, where ``N`` represents the
        number of elements. By default `ddof` is zero.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this
        option, the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be passed through to the `std` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    Returns
    -------
    standard_deviation : ndarray, see dtype parameter above.
        If `out` is None, return a new array containing the standard deviation, otherwise return a reference to the
        output array.

    See Also
    --------
    var, mean, std

    """
    return _std(a, epsilon, range, axis, dtype, out, ddof, keepdims, True)


def _std(a, epsilon=1.0, range=None, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue, nan=False):
    ret = _var(a, epsilon=epsilon, range=range, axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims, nan=nan)

    if isinstance(ret, mu.ndarray):
        ret = um.sqrt(ret, out=ret)
    elif hasattr(ret, 'dtype'):
        ret = ret.dtype.type(um.sqrt(ret))
    else:
        ret = um.sqrt(ret)

    return ret
