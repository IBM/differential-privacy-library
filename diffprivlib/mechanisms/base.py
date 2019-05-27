"""
Base classes for differential privacy mechanisms.
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
        """Randomise `value` with the mechanism.

        Parameters
        ----------
        value : int or float or str or method
            The value to be randomised.

        Returns
        -------
        int or float or str or method
            The randomised value, same type as `value`.

        """
        pass

    def copy(self):
        """Produces a copy of the class.

        Returns
        -------
        self : class
            Returns the copy.

        """
        return copy(self)

    def deepcopy(self):
        """Produces a deep copy of the class.

        Returns
        -------
        self : class
            Returns the deep copy.

        """
        return deepcopy(self)

    def set_epsilon(self, epsilon):
        """Sets the value of epsilon to be used by the mechanism.

        Parameters
        ----------
        epsilon : float
            The value of epsilon for achieving :math:`\epsilon`-differential privacy with the mechanism.  Must have
            `epsilon > 0`.

        Returns
        -------
        self : class

        """
        return self.set_epsilon_delta(epsilon, 0.0)

    @abc.abstractmethod
    def set_epsilon_delta(self, epsilon, delta):
        """Sets the value of epsilon and delta to be used by the mechanism.

        `epsilon` and `delta` cannot both be zero.

        Parameters
        ----------
        epsilon : float
            The value of epsilon for achieving :math:`(\epsilon,\delta)`-differential privacy with the mechanism. Must
            have `epsilon >= 0`.
        delta : float
            The value of delta for achieving :math:`(\epsilon,\delta)`-differential privacy with the mechanism.
            Must have `0 <= delta <= 1`.

            `delta=0` gives strict (pure) differential privacy (:math:`\epsilon`-differential privacy).  `delta > 0`
            gives relaxed (approximate) differential privacy.

        Returns
        -------
        self : class

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
        """Randomise `value` with the mechanism.

        Parameters
        ----------
        value : int or float or str or method
            The value to be randomised.

        Returns
        -------
        int or float or str or method
            The randomised value, same type as `value`.

        """
        pass

    def get_bias(self, value):
        """Returns the bias of the mechanism at a given `value`.

        Parameters
        ----------
        value : int or float
            The value at which the bias of the mechanism is sought.

        Returns
        -------
        bias : float or None
            The bias of the mechanism at `value` if defined, `None` otherwise.

        """
        pass

    def get_variance(self, value):
        """Returns the variance of the mechanism at a given `value`.

        Parameters
        ----------
        value : int or float
            The value at which the variance of the mechanism is sought.

        Returns
        -------
        bias : float or None
            The variance of the mechanism at `value` if defined, `None` otherwise.

        """
        pass

    def get_mse(self, value):
        """Returns the mean squared error (MSE) of the mechanism at a given `value`.

        Parameters
        ----------
        value : int or float
            The value at which the MSE of the mechanism is sought.

        Returns
        -------
        bias : float or None
            The MSE of the mechanism at `value` if defined, `None` otherwise.

        """
        if self.get_variance(value) is None or self.get_bias(value) is None:
            pass

        return self.get_variance(value) + (self.get_bias(value)) ** 2

    def set_epsilon(self, epsilon):
        """Sets the value of epsilon to be used by the mechanism.

        Parameters
        ----------
        epsilon : float
            The value of epsilon for achieving :math:`\epsilon`-differential privacy with the mechanism.  Must have
            `epsilon > 0`.

        Returns
        -------
        self : class

        """

        return self.set_epsilon_delta(epsilon, 0.0)

    def set_epsilon_delta(self, epsilon, delta):
        """Sets the value of epsilon and delta to be used by the mechanism.

        `epsilon` and `delta` cannot both be zero.

        Parameters
        ----------
        epsilon : float
            The value of epsilon for achieving :math:`(\epsilon,\delta)`-differential privacy with the mechanism. Must
            have `epsilon >= 0`.
        delta : float
            The value of delta for achieving :math:`(\epsilon,\delta)`-differential privacy with the mechanism.
            Must have `0 <= delta <= 1`.

            `delta=0` gives strict (pure) differential privacy (:math:`\epsilon`-differential privacy).  `delta > 0`
            gives relaxed (approximate) differential privacy.

        Returns
        -------
        self : class

        Raises
        ------
        ValueError
            If `epsilon` is negative, or if `delta` falls outside [0,1], or if `epsilon` and `delta` are both zero.

        """
        if not isinstance(epsilon, Real) or not isinstance(delta, Real):
            raise TypeError("Epsilon and delta must be numeric")

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
        """Checks that all parameters of the mechanism have been initialised correctly, and that the mechanism is ready
        to be used.

        Parameters
        ----------
        value : int or float or str or method

        Returns
        -------
        True if the mechanism is ready to be used.

        Raises
        ------
        Exception
            If parameters have not been set correctly, or if `value` falls outside the domain of the mechanism.

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
        """Sets the lower and upper bounds of the mechanism.

        Must have lower <= upper.

        Parameters
        ----------
        lower : float
            The lower bound of the mechanism.
        upper : float
            The upper bound of the mechanism.

        Returns
        -------
        self : class

        """
        if not isinstance(lower, Real) or not isinstance(upper, Real):
            raise TypeError("Bounds must be numeric")

        if lower > upper:
            raise ValueError("Lower bound must not be greater than upper bound")

        self._lower_bound = float(lower)
        self._upper_bound = float(upper)

        return self

    def check_inputs(self, value):
        """Checks that all parameters of the mechanism have been initialised correctly, and that the mechanism is ready
        to be used.

        Parameters
        ----------
        value : float

        Returns
        -------
        True if the mechanism is ready to be used.

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
