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
Basic functions and other utilities for the differential privacy library
"""
import secrets
import warnings

import numpy as np
from sklearn.utils import check_random_state as skl_check_random_state


def copy_docstring(source):
    """Decorator function to copy a docstring from a `source` function to a `target` function.

    The docstring is only copied if a docstring is present in `source`, and if none is present in `target`.  Takes
    inspiration from similar in `matplotlib`.

    Parameters
    ----------
    source : method
        Source function from which to copy the docstring.  If ``source.__doc__`` is empty, do nothing.

    Returns
    -------
    target : method
        Target function with new docstring.

    """
    def copy_func(target):
        if source.__doc__ and not target.__doc__:
            target.__doc__ = source.__doc__
        return target
    return copy_func


def warn_unused_args(args):
    """Warn the user about supplying unused `args` to a diffprivlib model.

    Arguments can be supplied as a string, a list of strings, or a dictionary as supplied to kwargs.

    Parameters
    ----------
    args : str or list or dict
        Arguments for which warnings should be thrown.

    Returns
    -------
    None

    """
    if isinstance(args, str):
        args = [args]

    for arg in args:
        warnings.warn(f"Parameter '{arg}' is not functional in diffprivlib.  Remove this parameter to suppress this "
                      "warning.", DiffprivlibCompatibilityWarning)


def check_random_state(seed, secure=False):
    """Turn seed into a np.random.RandomState or secrets.SystemRandom instance.

    If secure=True, and seed is None (or was generated from a previous None seed), then secrets is used.  Otherwise a
    np.random.RandomState is used.

    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None and secure is False, return the RandomState singleton used by np.random.
        If seed is None and secure is True, return a SystemRandom instance from secrets.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState or SystemRandom instance, return it.
        Otherwise raise ValueError.

    secure : bool, default: False
        Specifies if a secure random number generator from secrets can be used.
    """
    if secure:
        if isinstance(seed, secrets.SystemRandom):
            return seed

        if seed is None or seed is np.random.mtrand._rand:  # pylint: disable=protected-access
            return secrets.SystemRandom()
    elif isinstance(seed, secrets.SystemRandom):
        raise ValueError("secrets.SystemRandom instance cannot be passed when secure is False.")

    return skl_check_random_state(seed)


class Budget(tuple):
    """Custom tuple subclass for privacy budgets of the form (epsilon, delta).

    The ``Budget`` class allows for correct comparison/ordering of privacy budget, ensuring that both epsilon and delta
    satisfy the comparison (tuples are compared lexicographically).  Additionally, tuples are represented with added
    verbosity, labelling epsilon and delta appropriately.

    Examples
    --------

    >>> from diffprivlib.utils import Budget
    >>> Budget(1, 0.5)
    (epsilon=1, delta=0.5)
    >>> Budget(2, 0) >= Budget(1, 0.5)
    False
    >>> (2, 0) >= (1, 0.5) # Tuples are compared with lexicographic ordering
    True

    """
    def __new__(cls, epsilon, delta):
        if epsilon < 0:
            raise ValueError("Epsilon must be non-negative")

        if not 0 <= delta <= 1:
            raise ValueError("Delta must be in [0, 1]")

        return tuple.__new__(cls, (epsilon, delta))

    def __gt__(self, other):
        if self.__ge__(other) and not self.__eq__(other):
            return True
        return False

    def __ge__(self, other):
        if self[0] >= other[0] and self[1] >= other[1]:
            return True
        return False

    def __lt__(self, other):
        if self.__le__(other) and not self.__eq__(other):
            return True
        return False

    def __le__(self, other):
        if self[0] <= other[0] and self[1] <= other[1]:
            return True
        return False

    def __repr__(self):
        return f"(epsilon={self[0]}, delta={self[1]})"


class BudgetError(ValueError):
    """Custom exception to capture the privacy budget being exceeded, typically controlled by a
    :class:`.BudgetAccountant`.

    For example, this exception may be raised when the user:

        - Attempts to execute a query which would exceed the privacy budget of the accountant.
        - Attempts to change the slack of the accountant in such a way that the existing budget spends would exceed the
          accountant's budget.

    """


class PrivacyLeakWarning(RuntimeWarning):
    """Custom warning to capture privacy leaks resulting from incorrect parameter setting.

    For example, this warning may occur when the user:

        - fails to specify the bounds or range of data to a model where required (e.g., `bounds=None` to
          :class:`.GaussianNB`).
        - inputs data to a model that falls outside the bounds or range originally specified.

    """


class DiffprivlibCompatibilityWarning(RuntimeWarning):
    """Custom warning to capture inherited class arguments that are not compatible with diffprivlib.

    The purpose of the warning is to alert the user of the incompatibility, but to continue execution having fixed the
    incompatibility at runtime.

    For example, this warning may occur when the user:

        - passes a parameter value that is not compatible with diffprivlib (e.g., `solver='liblinear'` to
          :class:`.LogisticRegression`)
        - specifies a non-default value for a parameter that is ignored by diffprivlib (e.g., `intercept_scaling=0.5`
          to :class:`.LogisticRegression`.

    """


warnings.simplefilter('always', PrivacyLeakWarning)
