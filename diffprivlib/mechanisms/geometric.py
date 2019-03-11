from numbers import Integral

import numpy as np
from numpy.random import random

from . import DPMechanism, TruncationAndFoldingMachine


class Geometric(DPMechanism):
    def __init__(self):
        super().__init__()
        self._sensitivity = None
        self._scale = None

    def __repr__(self):
        output = super().__repr__()
        output += ".set_sensitivity(" + str(self._sensitivity) + ")" if self._sensitivity is not None else ""

        return output

    def set_sensitivity(self, sensitivity):
        """

        :param sensitivity:
        :type sensitivity `float`
        :return:
        """
        if not isinstance(sensitivity, Integral):
            raise TypeError("Sensitivity must be an integer")

        if sensitivity <= 0:
            raise ValueError("Sensitivity must be strictly positive")

        self._sensitivity = sensitivity
        return self

    def check_inputs(self, value):
        """
        Check that all parameters of the mechanism have been initialised correctly, and that the mechanism is ready
        to be used.

        :param value: Value to be checked.
        :type value: `int`
        :return: True if the mechanism is ready to be used.
        :rtype: `bool`
        """
        super().check_inputs(value)

        if not isinstance(value, Integral):
            raise TypeError("Value to be randomised must be an integer")

        if self._sensitivity is None:
            raise ValueError("Sensitivity must be set")

    def set_epsilon_delta(self, epsilon, delta):
        """
        Set the privacy parameters epsilon and delta for the mechanism.

        For the geometric mechanism, delta must be zero. Epsilon must be strictly positive, epsilon >= 0.

        :param epsilon: Epsilon value of the mechanism.
        :type epsilon: `float`
        :param delta: Delta value of the mechanism. For the geometric mechanism, this must be zero.
        :type delta: `float`
        :return: self
        :rtype: :class:`.Geometric`
        """
        if not delta == 0:
            raise ValueError("Delta must be zero")

        return super().set_epsilon_delta(epsilon, delta)

    def randomise(self, value):
        """
        Randomise the given value using the mechanism.

        :param value: Value to be randomised.
        :type value: `int`
        :return: Randomised value.
        :rtype: `int`
        """
        self.check_inputs(value)

        if self._scale is None:
            self._scale = - self._epsilon / self._sensitivity

        # Need to account for overlap of 0-value between distributions of different sign
        unif_rv = random() - 0.5
        unif_rv *= 1 + np.exp(self._scale)
        sgn = -1 if unif_rv < 0 else 1

        # Use formula for geometric distribution, with ratio of exp(-epsilon/sensitivity)
        return int(np.round(value + sgn * np.floor(np.log(sgn * unif_rv) / self._scale)))


class GeometricTruncated(Geometric, TruncationAndFoldingMachine):
    def __init__(self):
        super().__init__()
        TruncationAndFoldingMachine.__init__(self)

    def __repr__(self):
        output = super().__repr__()
        output += TruncationAndFoldingMachine.__repr__(self)

        return output

    def set_bounds(self, lower, upper):
        """
        Set the lower and upper bounds of the mechanism. Must satisfy lower <= upper.

        For the geometric mechanism, the bounds must be integer-valued.

        :param lower: Lower bound value.
        :type lower: `int`
        :param upper: Upper bound value.
        :type upper: `int`
        :return: self.
        :rtype: :class:`.GeometricTruncated`
        """
        if not isinstance(lower, Integral) or not isinstance(upper, Integral):
            raise TypeError("Bounds must be integers")

        return super().set_bounds(lower, upper)

    def randomise(self, value):
        """
        Randomise the given value using the mechanism.

        :param value: Value to be randomised.
        :type value: `int`
        :return: Randomised value.
        :rtype: `int`
        """
        TruncationAndFoldingMachine.check_inputs(self, value)

        noisy_value = super().randomise(value)
        return int(np.round(self._truncate(noisy_value)))


class GeometricFolded(Geometric, TruncationAndFoldingMachine):
    def __init__(self):
        super().__init__()
        TruncationAndFoldingMachine.__init__(self)

    def __repr__(self):
        output = super().__repr__()
        output += TruncationAndFoldingMachine.__repr__(self)

        return output

    def set_bounds(self, lower, upper):
        """
        Set the lower and upper bounds of the mechanism. Must satisfy lower <= upper.

        For the folded geometric mechanism, the bounds must be integer- or half-integer- valued.

        :param lower: Lower bound value.
        :type lower: `float`
        :param upper: Upper bound value.
        :type upper: `float`
        :return: self.
        :rtype: :class:`.GeometricFolded`
        """
        if not np.isclose(2 * lower, np.round(2 * lower)) or not np.isclose(2 * upper, np.round(2 * upper)):
            raise ValueError("Bounds must be integer or half-integer floats")

        return super().set_bounds(lower, upper)

    def _fold(self, value):
        return super()._fold(int(np.round(value)))

    def randomise(self, value):
        """
        Randomise the given value using the mechanism.

        :param value: Value to be randomised.
        :type value: `int`
        :return: Randomised value.
        :rtype: `int`
        """
        TruncationAndFoldingMachine.check_inputs(self, value)

        noisy_value = super().randomise(value)
        return int(np.round(self._fold(noisy_value)))
