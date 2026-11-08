# MIT License
#
# Copyright (C) IBM Corporation 2026
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
Functions for estimating the bounds and norm of data with differential privacy
"""
from numbers import Integral, Real

import numpy as np

from diffprivlib.accountant import BudgetAccountant
from diffprivlib.mechanisms import Geometric
from diffprivlib.utils import warn_unused_args, check_random_state


def approx_bounds(array, epsilon=1.0, confidence=1 - 1e-9, min_exponent=-64, max_exponent=64, random_state=None,
                  accountant=None, **unused_args):
    r"""
    Estimate the bounds of the data with differential privacy.

    Implements the ``APPROX_BOUNDS`` algorithm of [WZLDSG20]_.  Each value is counted in a logarithmic histogram
    of base 2, with one bin for each magnitude :math:`[2^k, 2^{k+1})` and sign, for :math:`k` from `min_exponent` to
    `max_exponent`, and a single bin for magnitudes below ``2 ** min_exponent``, whose edges are
    ``-2 ** min_exponent`` and ``2 ** min_exponent``.  Magnitudes of ``2 ** max_exponent``
    and above are counted in the largest bin of their sign.  Noise is added to each count with the :class:`.Geometric`
    mechanism, and the bounds are the outer edges of the most extreme bins whose noisy count exceeds the threshold:

    .. math::
        t = \frac{1}{\epsilon} \log\left(\frac{1}{1 - P^{1 / (B - 1)}}\right),

    where :math:`B` is the number of bins and :math:`P` is `confidence`.  The bounds therefore contain the bulk of the
    data, but outliers may fall outside them.

    Each record changes one count of each column's histogram by one, so each column is estimated with
    :math:`\epsilon`-differential privacy.  For 2-dimensional input, `epsilon` is split evenly between the columns by
    sequential composition.

    Parameters
    ----------
    array : array_like
        Input data, of shape (n_samples,) or (n_samples, n_features).  Bounds are estimated for each feature.  The data
        must not contain NaN.

    epsilon : float, default: 1.0
        Privacy parameter :math:`\epsilon`, split evenly between the features.

    confidence : float, default: 1 - 1e-9
        Lower bound on the probability that none of :math:`B - 1` empty bins has a noisy count above the threshold.
        Higher values make false bounds from noise less likely, but need more data.  Must be in (0, 1).

    min_exponent : int, default: -64
        Exponent of the smallest non-zero magnitude bin, at least -1074.  Must be independent of the data.

    max_exponent : int, default: 64
        Exponent of the largest magnitude bin, at most 1022.  Values of magnitude ``2 ** (max_exponent + 1)`` or more
        are counted in this bin, so the bounds do not cover them.  Must be independent of the data.

    random_state : int or RandomState, optional
        Controls the randomness of the algorithm.  To obtain a deterministic behaviour during randomisation,
        ``random_state`` has to be fixed to an integer.

    accountant : BudgetAccountant, optional
        Accountant to keep track of privacy budget.

    Returns
    -------
    bounds : tuple
        Tuple of the form (min, max), where `min` and `max` are floats for 1-dimensional input, and arrays of shape
        (n_features,) for 2-dimensional input.  The tuple can be passed as the `bounds` parameter of models and tools.

    Raises
    ------
    ValueError
        If no noisy count exceeds the threshold, for example when there are too few samples for the chosen `epsilon`.
        The budget is spent in this case, as the outcome depends on the data.

    See Also
    --------
    approx_data_norm

    References
    ----------
    .. [WZLDSG20] Wilson, Royce J., Celia Yuxin Zhang, William Lam, Damien Desfontaines, Daniel Simmons-Marengo, and
        Bryant Gipson. "Differentially Private SQL with Bounded User Contribution." arXiv preprint arXiv:1909.01917
        (2019).

    """
    warn_unused_args(unused_args)

    random_state = check_random_state(random_state)
    array = _check_array(array, ndims=(1, 2))
    _check_params(confidence, min_exponent, max_exponent)

    accountant = BudgetAccountant.load_default(accountant)
    accountant.check(epsilon, 0)

    columns = array.T if array.ndim == 2 else array[np.newaxis]
    column_epsilon = epsilon / columns.shape[0]
    column_bounds = [_log_histogram_bounds(column, column_epsilon, confidence, min_exponent, max_exponent, random_state,
                                           signed=True) for column in columns]

    accountant.spend(epsilon, 0)
    _raise_if_not_found(column_bounds)

    if array.ndim == 1:
        return column_bounds[0]

    lower, upper = zip(*column_bounds)
    return np.array(lower), np.array(upper)


def approx_data_norm(array, epsilon=1.0, confidence=1 - 1e-9, min_exponent=-64, max_exponent=64, random_state=None,
                     accountant=None, **unused_args):
    r"""
    Estimate the maximum Euclidean norm of the samples of the data with differential privacy.

    The L2 norm of each sample is counted in a logarithmic histogram of base 2, and the estimate is the upper edge of
    the largest bin whose noisy count exceeds the threshold, as described in :obj:`.approx_bounds` and [WZLDSG20b]_.
    Each sample changes one count by one, so the estimate satisfies :math:`\epsilon`-differential privacy.

    Parameters
    ----------
    array : array_like
        Input data, of shape (n_samples, n_features).  The data must not contain NaN.

    epsilon : float, default: 1.0
        Privacy parameter :math:`\epsilon`.

    confidence : float, default: 1 - 1e-9
        Lower bound on the probability that none of :math:`B - 1` empty bins has a noisy count above the threshold.
        Higher values make false bounds from noise less likely, but need more data.  Must be in (0, 1).

    min_exponent : int, default: -64
        Exponent of the smallest non-zero magnitude bin, at least -1074.  Must be independent of the data.

    max_exponent : int, default: 64
        Exponent of the largest magnitude bin, at most 1022.  Values of magnitude ``2 ** (max_exponent + 1)`` or more
        are counted in this bin, so the bounds do not cover them.  Must be independent of the data.

    random_state : int or RandomState, optional
        Controls the randomness of the algorithm.  To obtain a deterministic behaviour during randomisation,
        ``random_state`` has to be fixed to an integer.

    accountant : BudgetAccountant, optional
        Accountant to keep track of privacy budget.

    Returns
    -------
    data_norm : float
        Estimate of the maximum norm of the samples, which can be passed as the `data_norm` parameter of models.

    Raises
    ------
    ValueError
        If no noisy count exceeds the threshold, for example when there are too few samples for the chosen `epsilon`.
        The budget is spent in this case, as the outcome depends on the data.

    See Also
    --------
    approx_bounds

    References
    ----------
    .. [WZLDSG20b] Wilson, Royce J., Celia Yuxin Zhang, William Lam, Damien Desfontaines, Daniel Simmons-Marengo, and
        Bryant Gipson. "Differentially Private SQL with Bounded User Contribution." arXiv preprint arXiv:1909.01917
        (2019).

    """
    warn_unused_args(unused_args)

    random_state = check_random_state(random_state)
    array = _check_array(array, ndims=(2,))
    _check_params(confidence, min_exponent, max_exponent)

    accountant = BudgetAccountant.load_default(accountant)
    accountant.check(epsilon, 0)

    with np.errstate(over="ignore"):
        norms = np.linalg.norm(array, axis=1)
    bounds = _log_histogram_bounds(norms, epsilon, confidence, min_exponent, max_exponent, random_state, signed=False)

    accountant.spend(epsilon, 0)
    _raise_if_not_found([bounds])

    return bounds[1]


def _check_array(array, ndims):
    array = np.asarray(array, dtype=float)

    if array.ndim not in ndims:
        raise ValueError(f"Input array must have {' or '.join(map(str, ndims))} dimensions, got {array.ndim}.")
    if array.size == 0:
        raise ValueError("Input array must not be empty.")
    if np.isnan(array).any():
        raise ValueError("Input array must not contain NaN.")

    return array


def _check_params(confidence, min_exponent, max_exponent):
    if not isinstance(confidence, Real):
        raise TypeError(f"Confidence must be numeric, got {type(confidence)}.")
    if not 0 < confidence < 1:
        raise ValueError(f"Confidence must be in (0, 1), got {confidence}.")
    if not isinstance(min_exponent, Integral) or not isinstance(max_exponent, Integral):
        raise TypeError("Exponents must be integers.")
    if not -1074 <= min_exponent <= max_exponent <= 1022:
        raise ValueError("Exponents must satisfy -1074 <= min_exponent <= max_exponent <= 1022, got "
                         f"{min_exponent} and {max_exponent}.")


def _raise_if_not_found(bounds):
    if any(column_bounds is None for column_bounds in bounds):
        raise ValueError("No histogram bin has a noisy count above the threshold, so the bounds cannot be estimated. "
                         "Increase epsilon, decrease confidence or provide more data.")


def _threshold(epsilon, n_bins, confidence):
    return -np.log(-np.expm1(np.log(confidence) / (n_bins - 1))) / epsilon


def _log_histogram_bounds(values, epsilon, confidence, min_exponent, max_exponent, random_state, *, signed):
    powers = 2.0 ** np.arange(min_exponent, max_exponent + 2)
    lower_edges = np.concatenate(([0.0], powers[:-1]))
    upper_edges = powers.copy()

    magnitudes = np.minimum(np.abs(values), powers[-2])
    indices = np.where(magnitudes < powers[0], 0, np.frexp(magnitudes)[1] - min_exponent)

    if signed:
        indices = lower_edges.size - 1 + np.sign(values).astype(int) * indices
        lower_edges, upper_edges = (np.concatenate((-upper_edges[::-1], lower_edges[1:])),
                                    np.concatenate((-lower_edges[:0:-1], upper_edges)))

    counts = np.bincount(indices, minlength=lower_edges.size)
    mech = Geometric(epsilon=epsilon, sensitivity=1, random_state=random_state)
    noisy_counts = np.array([mech.randomise(int(count)) for count in counts])

    exceeding = np.flatnonzero(noisy_counts > _threshold(epsilon, counts.size, confidence))

    if exceeding.size == 0:
        return None

    return float(lower_edges[exceeding[0]]), float(upper_edges[exceeding[-1]])
