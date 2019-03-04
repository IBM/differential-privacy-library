import abc
import sys
from numbers import Real

from .. import DPMachine

# Ensure compatibility with Python 2 and 3 when using ABCMeta
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


class DPMechanism(DPMachine, ABC):
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
        pass

    def get_bias(self, value):
        """

        :param value:
        :return:
        :rtype: Union[None, float]
        """
        return None

    def get_variance(self, value):
        return None

    def get_mse(self, value):
        if self.get_variance(value) is None or self.get_bias(value) is None:
            return None

        return self.get_variance(value) + (self.get_bias(value)) ** 2

    def set_epsilon(self, epsilon):
        if epsilon <= 0:
            raise ValueError("Epsilon must be strictly positive")

        return self.set_epsilon_delta(epsilon, 0.0)

    def set_epsilon_delta(self, epsilon, delta):
        if not isinstance(epsilon, Real) or not isinstance(delta, Real):
            raise ValueError("Epsilon and delta must be numeric")

        if epsilon < 0:
            raise ValueError("Epsilon must be non-negative")

        # noinspection PyTypeChecker
        if not (0 <= delta <= 1):
            raise ValueError("Delta must be in [0, 1]")

        if epsilon + delta == 0:
            raise ValueError("Epsilon and Delta cannot both be zero")

        self._epsilon = float(epsilon)
        self._delta = float(delta)

        return self

    def check_inputs(self, value):
        """

        :param value:
        :return:
        :rtype: `bool`
        """
        if self._epsilon is None:
            raise ValueError("Epsilon must be set")
        return True


class TruncationAndFoldingMachine:
    def __init__(self):
        self._lower_bound = None
        self._upper_bound = None

    def __repr__(self):
        output = ".setBounds(" + str(self._lower_bound) + ", " + str(self._upper_bound) + ")" \
            if self._lower_bound is not None else ""

        return output

    def set_bounds(self, lower, upper):
        if not isinstance(lower, Real) or not isinstance(upper, Real):
            raise TypeError("Bounds must be numeric")

        if lower > upper:
            raise ValueError("Lower bound must not be greater than upper bound")

        self._lower_bound = float(lower)
        self._upper_bound = float(upper)

        return self

    def check_inputs(self, value):
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
