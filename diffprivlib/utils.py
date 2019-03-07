import abc
import numpy as np
import sys
from copy import copy, deepcopy

# Ensure compatibility with Python 2 and 3 when using ABCMeta
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


class DPMachine(ABC):
    """
    Parent class for :class:`.DPMechanism` and :class:`.DPTransformer`, providing and specifying basic functionality.
    """
    @abc.abstractmethod
    def randomise(self, value):
        """
        Randomise the given value using the :class:`.DPMachine`.

        :param value: Value to be randomised.
        :type value: `int` or `float` or `string`
        :return: Randomised value, same type as value.
        :rtype: `int` or `float` or `string`
        """
        pass

    def copy(self):
        """
        Copies the given class.

        :return: A copy of the input class.
        :rtype: `DPMachine`
        """
        return copy(self)

    def deepcopy(self):
        """
        Produces a deep copy of the given class.

        :return: A deepcopy of the input class.
        :rtype: `DPMachine`
        """
        return deepcopy(self)

    @abc.abstractmethod
    def set_epsilon(self, epsilon):
        """
        Sets the value of epsilon to be used by the mechanism.

        :param epsilon: Epsilon value for differential privacy.
        :type epsilon: `float`
        :return: self
        """
        pass


def global_seed(seed):
    """
    Sets the seed for all random number generators, to guarantee reproducibility in experiments.

    :param seed: The seed value for the random number generators.
    :type seed: `int`
    :return: None
    """
    np.random.seed(seed)
