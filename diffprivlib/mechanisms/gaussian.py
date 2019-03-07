from numbers import Real

from numpy import log, sqrt, cos, pi
from numpy.random import random

from . import DPMechanism


class Gaussian(DPMechanism):
    """
    The Gaussian mechanism in differential privacy.

    As first proposed by Dwork and Roth in "The algorithmic foundations of differential privacy".
    Paper link: https://www.nowpublishers.com/article/DownloadSummary/TCS-042
    """
    def __init__(self):
        super().__init__()
        self._sensitivity = None
        self._scale = None

    def set_epsilon_delta(self, epsilon, delta):
        """
        Set privacy parameters epsilon and delta for the mechanism.

        For the Gaussian mechanism, epsilon cannot be greater than 1.

        :param epsilon: Epsilon value of the mechanism.
        :type epsilon: `float`
        :param delta: Delta value of the mechanism.
        :type delta: `float`
        :return: self
        :rtype: :class:`.Gaussian`
        """
        if epsilon == 0 or delta == 0:
            raise ValueError("Neither Epsilon nor Delta can be zero")

        if isinstance(epsilon, Real) and epsilon > 1.0:
            raise ValueError("Epsilon cannot be greater than 1")

        self._scale = None
        return super().set_epsilon_delta(epsilon, delta)

    def set_sensitivity(self, sensitivity):
        """
        Set the sensitivity of the mechanism.

        :param sensitivity: The sensitivity of the function being considered, must be > 0.
        :type sensitivity: `float`
        :return: self
        :rtype: :class:`.Uniform`
        """
        if not isinstance(sensitivity, Real):
            raise TypeError("Sensitivity must be numeric")

        if sensitivity <= 0:
            raise ValueError("Sensitivity must be strictly positive")

        self._scale = None
        self._sensitivity = sensitivity
        return self

    def check_inputs(self, value):
        """
        Checks that all parameters of the mechanism have been initialised correctly, and that the mechanism is ready
        to be used.

        :param value: Value to be checked.
        :type value: `float`
        :return: True if the mechanism is ready to be used.
        :rtype: `bool`
        """
        super().check_inputs(value)

        if self._delta is None:
            raise ValueError("Delta must be set")

        if self._sensitivity is None:
            raise ValueError("Sensitivity must be set")

        if self._scale is None:
            self._scale = sqrt(2 * log(1.25 / self._delta)) * self._sensitivity / self._epsilon

        if not isinstance(value, Real):
            raise TypeError("Value to be randomised must be a number")

        return True

    def get_bias(self, value):
        """
        Get the bias of the mechanism at `value`.

        :param value: The value at which the bias of the mechanism is sought.
        :type value: `float`
        :return: The bias of the mechanism at `value`.
        :rtype: `float`
        """
        return 0.0

    def get_variance(self, value):
        """
        Get the variance of the mechanism at `value`.

        :param value: The value at which the variance is sought.
        :type value: `float`
        :return: The variance of the mechanism at `value`.
        :rtype: `float`
        """
        self.check_inputs(0)

        return self._scale ** 2

    def randomise(self, value):
        """
        Randomise the given value.

        :param value: Value to be randomised.
        :type value: `float`
        :return: Randomised value.
        :rtype: `float`
        """
        self.check_inputs(value)

        u1 = random()
        u2 = random()

        return sqrt(- 2 * log(u1)) * cos(2 * pi * u2) * self._scale + value
