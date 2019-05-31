"""
Basic functions and other utilities for the differential privacy library
"""
import abc
import sys
import warnings

import numpy as np

# Ensure compatibility with Python 2 and 3 when using ABCMeta
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


def global_seed(seed):
    """
    Sets the seed for all random number generators, to guarantee reproducibility in experiments.

    :param seed: The seed value for the random number generators.
    :type seed: `int`
    :return: None
    """
    np.random.seed(seed)


def copy_docstring(source):
    """Decorator function to copy a docstring from a `source` function to a `target` function.

    The docstring is only copied if a docstring is present in `source`, and if none is present in `target`.  Takes
    inspiration from similar in ``matplotlib``.

    Parameters
    ----------
    source : method
        Source function from which to copy the docstring.  If `source.__doc__` is empty, do nothing.

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


def warn_unused_args(args_dict):
    """Warn the user about supplying unused args to a diffprivlib model."""

    for key in args_dict:
        warnings.warn("Parameter '%s' is not functional in diffprivlib.  Remove this parameter to suppress this "
                      "warning." % key, DiffprivlibCompatibilityWarning)


class PrivacyLeakWarning(RuntimeWarning):
    """Custom warning to capture privacy leaks resulting from incorrect parameter setting.

    For example, this warning may occur when the user:

        - fails to specify the bounds or range of data to a model where required (e.g., `bounds=None` to
          :class:`.GaussianNB`).
        - inputs data to a model that falls outside the bounds or range originally specified.

    """


class DiffprivlibCompatibilityWarning(RuntimeWarning):
    """Custom warning to capture inherited class arguments that are not compatible with diffprivlib.

    For example, this warning may occur when the user:

        - passes a parameter value that is not compatible with diffprivlib (e.g., `solver='liblinear'` to
          :class:`.LogisticRegression`)
        - specifies a non-default value for a parameter that is ignored by diffprivlib (e.g., `intercept_scaling=0.5`
          to :class:`.LogisticRegression`.

    """


warnings.simplefilter('always', PrivacyLeakWarning)
