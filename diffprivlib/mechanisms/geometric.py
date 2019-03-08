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
        super().check_inputs(value)

        if not isinstance(value, Integral):
            raise TypeError("Value to be randomised must be an integer")

        if self._sensitivity is None:
            raise ValueError("Sensitivity must be set")

    def set_epsilon_delta(self, epsilon, delta):
        if not delta == 0:
            raise ValueError("Delta must be zero")

        return super().set_epsilon_delta(epsilon, delta)

    def randomise(self, value):
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
        if not isinstance(lower, Integral) or not isinstance(upper, Integral):
            raise TypeError("Bounds must be integers")

        return super().set_bounds(lower, upper)

    def randomise(self, value):
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
        if not np.isclose(2 * lower, np.round(2 * lower)) or not np.isclose(2 * upper, np.round(2 * upper)):
            raise ValueError("Bounds must be integer or half-integer floats")

        return super().set_bounds(lower, upper)

    def _fold(self, value):
        return super()._fold(int(np.round(value)))

    def randomise(self, value):
        TruncationAndFoldingMachine.check_inputs(self, value)

        noisy_value = super().randomise(value)
        return int(np.round(self._fold(noisy_value)))
