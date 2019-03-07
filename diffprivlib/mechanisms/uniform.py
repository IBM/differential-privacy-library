from numbers import Real

from numpy.random import random

from . import DPMechanism


class Uniform(DPMechanism):
    """
    The Uniform mechanism in differential privacy.

    This emerges as a special case of the :class:`.LaplaceBoundedNoise` mechanism when epsilon = 0.
    Paper link: https://arxiv.org/pdf/1810.00877.pdf
    """
    def __init__(self):
        super().__init__()
        self._sensitivity = None

    def set_epsilon_delta(self, epsilon, delta):
        """
        Set privacy parameters epsilon and delta for the mechanism.

        For the uniform mechanism, epsilon must be strictly zero and delta must satisfy 0 < delta <= 0.5.

        :param epsilon: Epsilon value of the mechanism.
        :type epsilon: `float`
        :param delta: Delta value of the mechanism.
        :type delta: `float`
        :return: self
        :rtype: :class:`.Uniform`
        """
        if not epsilon == 0:
            raise ValueError("Epsilon must be strictly zero.")

        if not (0 < delta <= 0.5):
            raise ValueError("Delta must satisfy 0 < delta <= 0.5")

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

        self._sensitivity = float(sensitivity)
        return self

    def get_bias(self, value):
        """
        Get the bias of the mechanism at `value`.

        :param value: The value at which the bias of the mechanism is sought.
        :type value: `float`
        :return: The bias of the mechanism at `value`.
        :rtype: `float`
        """
        return 0.0

    def check_inputs(self, value=None):
        """
        Checks that all parameters of the mechanism have been initialised correctly, and that the mechanism is ready
        to be used.

        :param value: Value to be checked.
        :type value: `float`
        :return: True if the mechanism is ready to be used.
        :rtype: `bool`
        """
        super().check_inputs(value)

        if (value is not None) and not isinstance(value, Real):
            raise TypeError("Value to be randomised must be a number")

        if self._sensitivity is None:
            raise ValueError("Sensitivity must be set")

        return True

    def randomise(self, value):
        """
        Randomise the given value.

        :param value: Value to be randomised.
        :type value: `float`
        :return: Randomised value.
        :rtype: `float`
        """
        self.check_inputs(value)

        u = 2 * random() - 1
        u *= self._sensitivity / self._delta / 2

        return value + u
