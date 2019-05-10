"""
Basic functions and other utilities for the library
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


class PrivacyLeakWarning(RuntimeWarning):
    pass


warnings.simplefilter('always', PrivacyLeakWarning)
