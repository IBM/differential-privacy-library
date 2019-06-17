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
import warnings

import numpy as np


def global_seed(seed):
    """Sets the seed for all random number generators, to guarantee reproducibility in experiments.

    Parameters
    ----------
    seed : int
        The seed value for the random number generators.

    Returns
    -------
    None

    """
    np.random.seed(seed)


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

    if not isinstance(args, (dict, list)):
        raise ValueError("args must be a string, a list of strings or a dictionary, got type '%s'." % type(args))

    for arg in args:
        warnings.warn("Parameter '%s' is not functional in diffprivlib.  Remove this parameter to suppress this "
                      "warning." % arg, DiffprivlibCompatibilityWarning)


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
