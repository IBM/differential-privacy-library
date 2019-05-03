"""
Core utilities for differential privacy mechanisms.
"""
import abc
import sys
from copy import copy, deepcopy
from numbers import Real

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

    def set_epsilon(self, epsilon):
        """
        Sets the value of epsilon to be used by the mechanism.

        :param epsilon: Epsilon value for differential privacy.
        :type epsilon: `float`
        :return: self
        """
        return self.set_epsilon_delta(epsilon, 0.0)

    @abc.abstractmethod
    def set_epsilon_delta(self, epsilon, delta):
        """
        Set the privacy parameters epsilon and delta for the mechanism.

        Epsilon must be non-negative, epsilon >= 0. Delta must be on the unit interval, 0 <= delta <= 1. At least
        one or epsilon and delta must be non-zero.

        Pure (strict) differential privacy is given when delta = 0. Approximate (relaxed) differential privacy is given
        when delta > 0.

        :param epsilon: Epsilon value of the mechanism.
        :type epsilon: `float`
        :param delta: Delta value of the mechanism.
        :type delta: `float`
        :return: self
        """
        pass


class DPMechanism(DPMachine, ABC):
    """
    Base class for all mechanisms. Instantiated from :class:`.DPMachine`.
    """
    def __init__(self):
        self._epsilon = None
        self._delta = None

    def __repr__(self):
        output = str(self.__module__) + "." + str(self.__class__.__name__) + "()"

        if self._epsilon is not None and self._delta is not None:
            output += ".set_epsilon_delta(" + str(self._epsilon) + "," + str(self._delta) + ")"
        elif self._epsilon is not None:
            output += ".set_epsilon(" + str(self._epsilon) + ")"

        return output

    @abc.abstractmethod
    def randomise(self, value):
        """
        Randomise the given value using the mechanism.

        :param value: Value to be randomised.
        :type value: `int` or `float` or `string`
        :return: Randomised value, same type as `value`.
        :rtype: `int` or `float` or `string`
        """
        pass

    def get_bias(self, value):
        """
        Returns the bias of the mechanism at a given `value`.

        :param value: The value at which the bias of the mechanism is sought.
        :type value: `int` or `float`
        :return: The bias of the mechanism at `value`. `None` if bias is not implemented or not defined.
        :rtype: Union[None, float]
        """
        pass

    def get_variance(self, value):
        """
        Returns the variance of the mechanism at a given `value`.

        :param value: The value at which the variance of the mechanism is sought.
        :type value: Union[int, float]
        :return: The variance of the mechanism at `value`. `None` if the variance is not implemented or not defined.
        :rtype: Union[None, float]
        """
        pass

    def get_mse(self, value):
        """
        Gives the mean squared error (MSE) of the mechanism, if its bias and variance is defined.

        :param value: The value at which the MSE is sought.
        :type value: Union[int, float]
        :return: The MSE of the mechanism at `value`. `None` if either :func:`get_bias` or :func:`get_variance` is None.
        """
        if self.get_variance(value) is None or self.get_bias(value) is None:
            pass

        return self.get_variance(value) + (self.get_bias(value)) ** 2

    def set_epsilon(self, epsilon):
        """
        Sets the value of epsilon to be used by the mechanism.

        Epsilon be strictly positive, epsilon > 0.

        :param epsilon: Epsilon value for differential privacy.
        :type epsilon: `float`
        :return: self
        """
        if epsilon <= 0:
            raise ValueError("Epsilon must be strictly positive")

        return self.set_epsilon_delta(epsilon, 0.0)

    def set_epsilon_delta(self, epsilon, delta):
        """
        Set the privacy parameters epsilon and delta for the mechanism.

        Epsilon must be non-negative, epsilon >= 0. Delta must be on the unit interval, 0 <= delta <= 1. At least
        one or epsilon and delta must be non-zero.

        Pure (strict) differential privacy is given when delta = 0. Approximate (relaxed) differential privacy is given
        when delta > 0.

        :param epsilon: Epsilon value of the mechanism.
        :type epsilon: `float`
        :param delta: Delta value of the mechanism.
        :type delta: `float`
        :return: self
        """
        if not isinstance(epsilon, Real) or not isinstance(delta, Real):
            raise ValueError("Epsilon and delta must be numeric")

        if epsilon < 0:
            raise ValueError("Epsilon must be non-negative")

        if not 0 <= delta <= 1:
            raise ValueError("Delta must be in [0, 1]")

        if epsilon + delta == 0:
            raise ValueError("Epsilon and Delta cannot both be zero")

        self._epsilon = float(epsilon)
        self._delta = float(delta)

        return self

    def check_inputs(self, value):
        """
        Check that all parameters of the mechanism have been initialised correctly, and that the mechanism is ready
        to be used.

        :param value: Value to be checked.
        :type value: Union[float, int, str]
        :return: True if the mechanism is ready to be used.
        :rtype: `bool`
        """
        del value
        if self._epsilon is None:
            raise ValueError("Epsilon must be set")
        return True


class TruncationAndFoldingMachine:
    """
    Base class for truncating or folding the outputs of a mechanism. Must be instantiated with a :class:`.DPMechanism`.
    """
    def __init__(self):
        if not isinstance(self, DPMechanism):
            raise TypeError("TruncationAndFoldingMachine must be implemented alongside a :class:`.DPMechanism`")

        self._lower_bound = None
        self._upper_bound = None

    def __repr__(self):
        output = ".setBounds(" + str(self._lower_bound) + ", " + str(self._upper_bound) + ")" \
            if self._lower_bound is not None else ""

        return output

    def set_bounds(self, lower, upper):
        """
        Set the lower and upper bounds of the mechanism. Must satisfy lower <= upper.

        :param lower: Lower bound value.
        :type lower: float
        :param upper: Upper bound value.
        :type upper: float
        :return: self
        """
        if not isinstance(lower, Real) or not isinstance(upper, Real):
            raise TypeError("Bounds must be numeric")

        if lower > upper:
            raise ValueError("Lower bound must not be greater than upper bound")

        self._lower_bound = float(lower)
        self._upper_bound = float(upper)

        return self

    def check_inputs(self, value):
        """
        Check that all parameters of the mechanism have been initialised correctly, and that the mechanism is ready
        to be used.

        :param value: Value to be checked.
        :type value: `float`
        :return: True if the mechanism is ready to be used.
        :rtype: `bool`
        """
        del value
        if (self._lower_bound is None) or (self._upper_bound is None):
            raise ValueError("Upper and lower bounds must be set")
        return True

    def _truncate(self, value):
        if value > self._upper_bound:
            return self._upper_bound
        elif value < self._lower_bound:
            return self._lower_bound

        return value

    def _fold(self, value):
        if value < self._lower_bound:
            return self._fold(2 * self._lower_bound - value)
        if value > self._upper_bound:
            return self._fold(2 * self._upper_bound - value)

        return value
