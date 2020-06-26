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
from numbers import Integral
import numpy as np
from numpy.core import multiarray as mu
from numpy.core import umath as um

from diffprivlib.accountant import BudgetAccountant
from diffprivlib.mechanisms import LaplaceBoundedDomain, GeometricTruncated, LaplaceTruncated
from diffprivlib.utils import PrivacyLeakWarning, warn_unused_args
from diffprivlib.validation import check_bounds

_sum_ = sum


def count_nonzero(array, epsilon=1.0, accountant=None, axis=None, keepdims=False):
    r"""Counts the number of non-zero values in the array ``array`` with differential privacy.

    The word "non-zero" is in reference to the Python 2.x built-in method ``__nonzero__()`` (renamed ``__bool__()`` in
    Python 3.x) of Python objects that tests an object's "truthfulness".  For example, any number is considered truthful
    if it is nonzero, whereas any string is considered truthful if it is not the empty string.  Thus, this function
    (recursively) counts how many elements in ``array`` (and in sub-arrays thereof) have their ``__nonzero__()`` or
    ``__bool__()`` method evaluated to ``True``.

    Parameters
    ----------
    array : array_like
        The array for which to count non-zeros.

    epsilon : float, default: 1.0
        Privacy parameter :math:`\epsilon`.

    accountant : BudgetAccountant, optional
        Accountant to keep track of privacy budget.

    axis : int or tuple, optional
        Axis or tuple of axes along which to count non-zeros.  Default is None, meaning that non-zeros will be counted
        along a flattened version of ``array``.

    keepdims : bool, optional
        If this is set to True, the axes that are counted are left in the result as dimensions with size one. With this
        option, the result will broadcast correctly against the input array.

    Returns
    -------
    count : int or array of int
        Differentially private number of non-zero values in the array along a given axis.  Otherwise, the total number
        of non-zero values in the array is returned.

    """
    array = np.asanyarray(array)

    if np.issubdtype(array.dtype, np.character):
        array_bool = array != array.dtype.type()
    else:
        array_bool = array.astype(np.bool_, copy=False)

    return sum(array_bool, axis=axis, dtype=np.intp, bounds=(0, 1), epsilon=epsilon, accountant=accountant,
               keepdims=keepdims)


def mean(array, epsilon=1.0, bounds=None, axis=None, dtype=None, keepdims=np._NoValue, accountant=None, **unused_args):
    r"""
    Compute the differentially private arithmetic mean along the specified axis.

    Returns the average of the array elements with differential privacy.  The average is taken over the flattened array
    by default, otherwise over the specified axis.  Noise is added using :class:`.Laplace` to satisfy differential
    privacy, where sensitivity is calculated using `bounds`.  Users are advised to consult the documentation of
    :obj:`numpy.mean` for further details, as the behaviour of `mean` closely follows its Numpy variant.

    Parameters
    ----------
    array : array_like
        Array containing numbers whose mean is desired.  If `array` is not an array, a conversion is attempted.

    epsilon : float, default: 1.0
        Privacy parameter :math:`\epsilon`.

    bounds : tuple, optional
        Bounds of the values of the array, of the form (min, max).

    axis : int or tuple of ints, optional
        Axis or axes along which the means are computed.  The default is to compute the mean of the flattened array.

        If this is a tuple of ints, a mean is performed over multiple axes, instead of a single axis or all the axes as
        before.

    dtype : data-type, optional
        Type to use in computing the mean.  For integer inputs, the default is `float64`; for floating point inputs, it
        is the same as the input dtype.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the result as dimensions with size one.  With
        this option, the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be passed through to the `mean` method of sub-classes
        of `ndarray`, however any non-default value will be.  If the sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    accountant : BudgetAccountant, optional
        Accountant to keep track of privacy budget.

    Returns
    -------
    m : ndarray, see dtype parameter above
        Returns a new array containing the mean values.

    See Also
    --------
    std, var, nanmean

    """
    warn_unused_args(unused_args)

    return _mean(array, epsilon=epsilon, bounds=bounds, axis=axis, dtype=dtype, keepdims=keepdims,
                 accountant=accountant, nan=False)


def nanmean(array, epsilon=1.0, bounds=None, axis=None, dtype=None, keepdims=np._NoValue, accountant=None,
            **unused_args):
    r"""
    Compute the differentially private arithmetic mean along the specified axis, ignoring NaNs.

    Returns the average of the array elements with differential privacy.  The average is taken over the flattened array
    by default, otherwise over the specified axis.  Noise is added using :class:`.Laplace` to satisfy differential
    privacy, where sensitivity is calculated using `bounds`.  Users are advised to consult the documentation of
    :obj:`numpy.mean` for further details, as the behaviour of `mean` closely follows its Numpy variant.

    For all-NaN slices, NaN is returned and a `RuntimeWarning` is raised.

    Parameters
    ----------
    array : array_like
        Array containing numbers whose mean is desired.  If `array` is not an array, a conversion is attempted.

    epsilon : float, default: 1.0
        Privacy parameter :math:`\epsilon`.

    bounds : tuple, optional
        Bounds of the values of the array, of the form (min, max).

    axis : int or tuple of ints, optional
        Axis or axes along which the means are computed.  The default is to compute the mean of the flattened array.

        If this is a tuple of ints, a mean is performed over multiple axes, instead of a single axis or all the axes as
        before.

    dtype : data-type, optional
        Type to use in computing the mean.  For integer inputs, the default is `float64`; for floating point inputs, it
        is the same as the input dtype.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the result as dimensions with size one.  With
        this option, the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be passed through to the `mean` method of sub-classes
        of `ndarray`, however any non-default value will be.  If the sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    accountant : BudgetAccountant, optional
        Accountant to keep track of privacy budget.

    Returns
    -------
    m : ndarray, see dtype parameter above
        Returns a new array containing the mean values.

    See Also
    --------
    std, var, mean

    """
    warn_unused_args(unused_args)

    return _mean(array, epsilon=epsilon, bounds=bounds, axis=axis, dtype=dtype, keepdims=keepdims,
                 accountant=accountant, nan=True)


def _mean(array, epsilon=1.0, bounds=None, axis=None, dtype=None, keepdims=np._NoValue, accountant=None, nan=False):
    accountant = BudgetAccountant.load_default(accountant)
    accountant.check(epsilon, 0)

    _func = np.nanmean if nan else np.mean
    output_form = _func(np.zeros_like(array), axis=axis, keepdims=keepdims)
    vector_out = (np.ndim(output_form) == 1)
    n_datapoints = np.sum(np.ones_like(array), axis=axis, keepdims=keepdims).flat[0]

    if bounds is None:
        warnings.warn("Bounds have not been specified and will be calculated on the data provided. This will "
                      "result in additional privacy leakage. To ensure differential privacy and no additional "
                      "privacy leakage, specify bounds for each dimension.", PrivacyLeakWarning)
        if np.ndim(output_form) <= 1:
            bounds = (np.min(array, axis=axis, keepdims=keepdims), np.max(array, axis=axis, keepdims=keepdims))
        else:
            bounds = (np.min(array), np.max(array))

    lower, upper = check_bounds(bounds, output_form.shape[0] if vector_out else 1, dtype=dtype or float)
    array = np.clip(array, lower, upper)

    actual_mean = _func(array, axis=axis, dtype=dtype, keepdims=keepdims)

    if isinstance(actual_mean, np.ndarray):
        dp_mean = np.zeros_like(actual_mean)
        iterator = np.nditer(actual_mean, flags=['multi_index'])

        while not iterator.finished:
            idx = iterator.multi_index
            _lower, _upper = (lower[idx], upper[idx]) if vector_out else (lower[0], upper[0])
            local_diam = _upper - _lower
            dp_mech = LaplaceTruncated().set_epsilon(epsilon).set_sensitivity(local_diam / n_datapoints).\
                set_bounds(_lower, _upper)

            dp_mean[iterator.multi_index] = dp_mech.randomise(actual_mean[idx])
            iterator.iternext()

        accountant.spend(epsilon, 0)

        return dp_mean

    local_diam = upper[0] - lower[0]
    dp_mech = LaplaceTruncated().set_epsilon(epsilon).set_sensitivity(local_diam / n_datapoints).\
        set_bounds(lower[0], upper[0])

    accountant.spend(epsilon, 0)

    return dp_mech.randomise(actual_mean)


def var(array, epsilon=1.0, bounds=None, axis=None, dtype=None, keepdims=np._NoValue, accountant=None, **unused_args):
    r"""
    Compute the differentially private variance along the specified axis.

    Returns the variance of the array elements, a measure of the spread of a distribution, with differential privacy.
    The variance is computer for the flattened array by default, otherwise over the specified axis.  Noise is added
    using :class:`.LaplaceBoundedDomain` to satisfy differential privacy, where sensitivity is calculated using
    `bounds`.  Users are advised to consult the documentation of :obj:`numpy.var` for further details, as the behaviour
    of `var` closely follows its Numpy variant.

    Parameters
    ----------
    array : array_like
        Array containing numbers whose variance is desired.  If `array` is not an array, a conversion is attempted.

    epsilon : float, default: 1.0
        Privacy parameter :math:`\epsilon`.

    bounds : tuple, optional
        Bounds of the values of the array, of the form (min, max).

    axis : int or tuple of ints, optional
        Axis or axes along which the variance is computed.  The default is to compute the variance of the flattened
        array.

        If this is a tuple of ints, a variance is performed over multiple axes, instead of a single axis or all the axes
        as before.

    dtype : data-type, optional
        Type to use in computing the variance.  For arrays of integer type the default is `float32`; for arrays of float
        types it is the same as the array type.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the result as dimensions with size one.  With
        this option, the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be passed through to the `var` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    accountant : BudgetAccountant, optional
        Accountant to keep track of privacy budget.

    Returns
    -------
    variance : ndarray, see dtype parameter above
        Returns a new array containing the variance.

    See Also
    --------
    std , mean, nanvar

    """
    warn_unused_args(unused_args)

    return _var(array, epsilon=epsilon, bounds=bounds, axis=axis, dtype=dtype, keepdims=keepdims, accountant=accountant,
                nan=False)


def nanvar(array, epsilon=1.0, bounds=None, axis=None, dtype=None, keepdims=np._NoValue, accountant=None,
           **unused_args):
    r"""
    Compute the differentially private variance along the specified axis, ignoring NaNs.

    Returns the variance of the array elements, a measure of the spread of a distribution, with differential privacy.
    The variance is computer for the flattened array by default, otherwise over the specified axis.  Noise is added
    using :class:`.LaplaceBoundedDomain` to satisfy differential privacy, where sensitivity is calculated using
    `bounds`.  Users are advised to consult the documentation of :obj:`numpy.var` for further details, as the behaviour
    of `var` closely follows its Numpy variant.

    For all-NaN slices, NaN is returned and a `RuntimeWarning` is raised.

    Parameters
    ----------
    array : array_like
        Array containing numbers whose variance is desired.  If `array` is not an array, a conversion is attempted.

    epsilon : float, default: 1.0
        Privacy parameter :math:`\epsilon`.

    bounds : tuple, optional
        Bounds of the values of the array, of the form (min, max).

    axis : int or tuple of ints, optional
        Axis or axes along which the variance is computed.  The default is to compute the variance of the flattened
        array.

        If this is a tuple of ints, a variance is performed over multiple axes, instead of a single axis or all the axes
        as before.

    dtype : data-type, optional
        Type to use in computing the variance.  For arrays of integer type the default is `float32`; for arrays of float
        types it is the same as the array type.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the result as dimensions with size one.  With
        this option, the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be passed through to the `var` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    accountant : BudgetAccountant, optional
        Accountant to keep track of privacy budget.

    Returns
    -------
    variance : ndarray, see dtype parameter above
        If ``out=None``, returns a new array containing the variance; otherwise, a reference to the output array is
        returned.

    See Also
    --------
    std , mean, var

    """
    warn_unused_args(unused_args)

    return _var(array, epsilon=epsilon, bounds=bounds, axis=axis, dtype=dtype, keepdims=keepdims, accountant=accountant,
                nan=True)


def _var(array, epsilon=1.0, bounds=None, axis=None, dtype=None, keepdims=np._NoValue, accountant=None, nan=False):
    accountant = BudgetAccountant.load_default(accountant)
    accountant.check(epsilon, 0)

    _func = np.nanvar if nan else np.var
    output_form = _func(np.zeros_like(array), axis=axis, keepdims=keepdims)
    vector_out = (np.ndim(output_form) == 1)
    n_datapoints = np.sum(np.ones_like(array, dtype=int), axis=axis, keepdims=keepdims).flat[0]

    if bounds is None:
        warnings.warn("Bounds have not been specified and will be calculated on the data provided. This will "
                      "result in additional privacy leakage. To ensure differential privacy and no additional "
                      "privacy leakage, specify bounds for each dimension.", PrivacyLeakWarning)
        if np.ndim(output_form) <= 1:
            bounds = (np.min(array, axis=axis, keepdims=keepdims), np.max(array, axis=axis, keepdims=keepdims))
        else:
            bounds = (np.min(array), np.max(array))

    lower, upper = check_bounds(bounds, output_form.shape[0] if vector_out else 1, dtype=dtype or float)
    array = np.clip(array, lower, upper)

    actual_var = _func(array, axis=axis, dtype=dtype, keepdims=keepdims)

    if isinstance(actual_var, np.ndarray):
        dp_var = np.zeros_like(actual_var)
        iterator = np.nditer(actual_var, flags=['multi_index'])

        while not iterator.finished:
            idx = iterator.multi_index
            local_diam = upper[idx] - lower[idx] if vector_out else upper[0] - lower[0]
            dp_mech = LaplaceBoundedDomain().set_epsilon(epsilon).set_bounds(0, float("inf")) \
                .set_sensitivity((local_diam / n_datapoints) ** 2 * (n_datapoints - 1))

            dp_var[iterator.multi_index] = np.minimum(dp_mech.randomise(actual_var[idx]), local_diam ** 2)
            iterator.iternext()

        accountant.spend(epsilon, 0)

        return dp_var

    local_diam = upper[0] - lower[0]
    dp_mech = LaplaceBoundedDomain().set_epsilon(epsilon).set_bounds(0, float("inf")). \
        set_sensitivity((local_diam / n_datapoints) ** 2 * (n_datapoints - 1))

    accountant.spend(epsilon, 0)

    return np.minimum(dp_mech.randomise(actual_var), local_diam ** 2)


def std(array, epsilon=1.0, bounds=None, axis=None, dtype=None, keepdims=np._NoValue, accountant=None, **unused_args):
    r"""
    Compute the standard deviation along the specified axis.

    Returns the standard deviation of the array elements, a measure of the spread of a distribution, with differential
    privacy.  The standard deviation is computed for the flattened array by default, otherwise over the specified axis.
    Noise is added using :class:`.LaplaceBoundedDomain` to satisfy differential privacy, where sensitivity is
    calculated using `bounds`.  Users are advised to consult the documentation of :obj:`numpy.std` for further details,
    as the behaviour of `std` closely follows its Numpy variant.

    Parameters
    ----------
    array : array_like
        Calculate the standard deviation of these values.

    epsilon : float, default: 1.0
        Privacy parameter :math:`\epsilon`.

    bounds : tuple, optional
        Bounds of the values of the array, of the form (min, max).

    axis : int or tuple of ints, optional
        Axis or axes along which the standard deviation is computed.  The default is to compute the standard deviation
        of the flattened array.

        If this is a tuple of ints, a standard deviation is performed over multiple axes, instead of a single axis or
        all the axes as before.

    dtype : dtype, optional
        Type to use in computing the standard deviation.  For arrays of integer type the default is float64, for arrays
        of float types it is the same as the array type.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the result as dimensions with size one.  With
        this option, the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be passed through to the `std` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    accountant : BudgetAccountant, optional
        Accountant to keep track of privacy budget.

    Returns
    -------
    standard_deviation : ndarray, see dtype parameter above.
        Return a new array containing the standard deviation.

    See Also
    --------
    var, mean, nanstd

    """
    warn_unused_args(unused_args)

    return _std(array, epsilon=epsilon, bounds=bounds, axis=axis, dtype=dtype, keepdims=keepdims,
                accountant=accountant, nan=False)


def nanstd(array, epsilon=1.0, bounds=None, axis=None, dtype=None, keepdims=np._NoValue, accountant=None,
           **unused_args):
    r"""
    Compute the standard deviation along the specified axis, ignoring NaNs.

    Returns the standard deviation of the array elements, a measure of the spread of a distribution, with differential
    privacy.  The standard deviation is computed for the flattened array by default, otherwise over the specified axis.
    Noise is added using :class:`.LaplaceBoundedDomain` to satisfy differential privacy, where sensitivity is
    calculated using `bounds`.  Users are advised to consult the documentation of :obj:`numpy.std` for further details,
    as the behaviour of `std` closely follows its Numpy variant.

    For all-NaN slices, NaN is returned and a `RuntimeWarning` is raised.

    Parameters
    ----------
    array : array_like
        Calculate the standard deviation of these values.

    epsilon : float, default: 1.0
        Privacy parameter :math:`\epsilon`.

    bounds : tuple, optional
        Bounds of the values of the array, of the form (min, max).

    axis : int or tuple of ints, optional
        Axis or axes along which the standard deviation is computed.  The default is to compute the standard deviation
        of the flattened array.

        If this is a tuple of ints, a standard deviation is performed over multiple axes, instead of a single axis or
        all the axes as before.

    dtype : dtype, optional
        Type to use in computing the standard deviation.  For arrays of integer type the default is float64, for arrays
        of float types it is the same as the array type.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the result as dimensions with size one.  With
        this option, the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be passed through to the `std` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    accountant : BudgetAccountant, optional
        Accountant to keep track of privacy budget.

    Returns
    -------
    standard_deviation : ndarray, see dtype parameter above.
        Return a new array containing the standard deviation.

    See Also
    --------
    var, mean, std

    """
    warn_unused_args(unused_args)

    return _std(array, epsilon=epsilon, bounds=bounds, axis=axis, dtype=dtype, keepdims=keepdims, accountant=accountant,
                nan=True)


def _std(array, epsilon=1.0, bounds=None, axis=None, dtype=None, keepdims=np._NoValue, accountant=None, nan=False):
    ret = _var(array, epsilon=epsilon, bounds=bounds, axis=axis, dtype=dtype, keepdims=keepdims, accountant=accountant,
               nan=nan)

    if isinstance(ret, mu.ndarray):
        ret = um.sqrt(ret)
    elif hasattr(ret, 'dtype'):
        ret = ret.dtype.type(um.sqrt(ret))
    else:
        ret = um.sqrt(ret)

    return ret


def sum(array, epsilon=1.0, bounds=None, accountant=None, axis=None, dtype=None, keepdims=np._NoValue, **unused_args):
    r"""Sum of array elements over a given axis with differential privacy.

    Parameters
    ----------
    array : array_like
        Elements to sum.

    epsilon : float, default: 1.0
        Privacy parameter :math:`\epsilon`.

    bounds : tuple, optional
        Bounds of the values of the array, of the form (min, max).

    accountant : BudgetAccountant, optional
        Accountant to keep track of privacy budget.

    axis : None or int or tuple of ints, optional
        Axis or axes along which a sum is performed.  The default, axis=None, will sum all of the elements of the input
        array.  If axis is negative it counts from the last to the first axis.

        If axis is a tuple of ints, a sum is performed on all of the axes specified in the tuple instead of a single
        axis or all the axes as before.

    dtype : dtype, optional
        The type of the returned array and of the accumulator in which the elements are summed.  The dtype of `array` is
        used by default unless `array` has an integer dtype of less precision than the default platform integer.  In
        that case, if `array` is signed then the platform integer is used while if `array` is unsigned then an unsigned
        integer of the same precision as the platform integer is used.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the result as dimensions with size one.  With
        this option, the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be passed through to the `sum` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    Returns
    -------
    sum_along_axis : ndarray
        An array with the same shape as `array`, with the specified axis removed.   If `array` is a 0-d array, or if
        `axis` is None, a scalar is returned.

    See Also
    --------
    ndarray.sum : Equivalent non-private method.

    mean, nansum

    """
    warn_unused_args(unused_args)

    return _sum(array, epsilon=epsilon, bounds=bounds, accountant=accountant, axis=axis, dtype=dtype, keepdims=keepdims,
                nan=False)


def nansum(array, epsilon=1.0, bounds=None, accountant=None, axis=None, dtype=None, keepdims=np._NoValue,
           **unused_args):
    r"""Sum of array elements over a given axis with differential privacy, ignoring NaNs.

    Parameters
    ----------
    array : array_like
        Elements to sum.

    epsilon : float, default: 1.0
        Privacy parameter :math:`\epsilon`.

    bounds : tuple, optional
        Bounds of the values of the array, of the form (min, max).

    accountant : BudgetAccountant, optional
        Accountant to keep track of privacy budget.

    axis : None or int or tuple of ints, optional
        Axis or axes along which a sum is performed.  The default, axis=None, will sum all of the elements of the input
        array.  If axis is negative it counts from the last to the first axis.

        If axis is a tuple of ints, a sum is performed on all of the axes specified in the tuple instead of a single
        axis or all the axes as before.

    dtype : dtype, optional
        The type of the returned array and of the accumulator in which the elements are summed.  The dtype of `array` is
        used by default unless `array` has an integer dtype of less precision than the default platform integer.  In
        that case, if `array` is signed then the platform integer is used while if `array` is unsigned then an unsigned
        integer of the same precision as the platform integer is used.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the result as dimensions with size one.  With
        this option, the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be passed through to the `sum` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    Returns
    -------
    sum_along_axis : ndarray
        An array with the same shape as `array`, with the specified axis removed.   If `array` is a 0-d array, or if
        `axis` is None, a scalar is returned.  If an output array is specified, a reference to `out` is returned.

    See Also
    --------
    ndarray.sum : Equivalent non-private method.

    mean, sum

    """
    warn_unused_args(unused_args)

    return _sum(array, epsilon=epsilon, bounds=bounds, accountant=accountant, axis=axis, dtype=dtype,
                keepdims=keepdims, nan=True)


def _sum(array, epsilon=1.0, bounds=None, accountant=None, axis=None, dtype=None, keepdims=np._NoValue, nan=False):
    accountant = BudgetAccountant.load_default(accountant)
    accountant.check(epsilon, 0)

    _func = np.nansum if nan else np.sum
    output_form = _func(np.zeros_like(array), axis=axis, keepdims=keepdims)
    vector_out = (np.ndim(output_form) == 1)
    n_datapoints = np.sum(np.ones_like(array, dtype=int), axis=axis, keepdims=keepdims).flat[0]

    if bounds is None:
        warnings.warn("Bounds have not been specified and will be calculated on the data provided. This will "
                      "result in additional privacy leakage. To ensure differential privacy and no additional "
                      "privacy leakage, specify bounds for each dimension.", PrivacyLeakWarning)
        if np.ndim(output_form) <= 1:
            bounds = (np.min(array, axis=axis, keepdims=keepdims), np.max(array, axis=axis, keepdims=keepdims))
        else:
            bounds = (np.min(array), np.max(array))

    lower, upper = check_bounds(bounds, output_form.shape[0] if vector_out else 1, dtype=dtype or float)
    array = np.clip(array, lower, upper)

    actual_sum = _func(array, axis=axis, dtype=dtype, keepdims=keepdims)

    dp_mech = GeometricTruncated if dtype is not None and issubclass(dtype, Integral) else LaplaceTruncated

    if isinstance(actual_sum, np.ndarray):
        dp_sum = np.zeros_like(actual_sum, dtype=dtype)
        iterator = np.nditer(actual_sum, flags=['multi_index'])

        while not iterator.finished:
            idx = iterator.multi_index
            _lower, _upper = (lower[idx], upper[idx]) if vector_out else (lower[0], upper[0])
            local_diam = _upper - _lower
            mech = dp_mech().set_epsilon(epsilon).set_sensitivity(local_diam).\
                set_bounds(_lower * n_datapoints, _upper * n_datapoints)

            dp_sum[idx] = mech.randomise(actual_sum[idx])
            iterator.iternext()

        accountant.spend(epsilon, 0)

        return dp_sum

    local_diam = upper[0] - lower[0]
    mech = dp_mech().set_epsilon(epsilon).set_sensitivity(local_diam).set_bounds(lower[0] * n_datapoints,
                                                                                 upper[0] * n_datapoints)

    accountant.spend(epsilon, 0)

    return mech.randomise(actual_sum)
