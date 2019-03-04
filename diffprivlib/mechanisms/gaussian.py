from numbers import Real

from numpy import log, sqrt, cos, pi
from numpy.random import random

from . import DPMechanism


class Gaussian(DPMechanism):
    def __init__(self):
        super().__init__()
        self._sensitivity = None
        self._scale = None

    def set_epsilon_delta(self, epsilon, delta):
        if epsilon == 0 or delta == 0:
            raise ValueError("Neither Epsilon nor Delta can be zero")

        if isinstance(epsilon, Real) and epsilon > 1.0:
            raise ValueError("Epsilon cannot be greater than 1")

        self._scale = None
        return super().set_epsilon_delta(epsilon, delta)

    def set_sensitivity(self, sensitivity):
        if not isinstance(sensitivity, Real):
            raise TypeError("Sensitivity must be numeric")

        if sensitivity <= 0:
            raise ValueError("Sensitivity must be strictly positive")

        self._scale = None
        self._sensitivity = sensitivity
        return self

    def check_inputs(self, value):
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
        return 0.0

    def get_variance(self, value):
        self.check_inputs(0)

        return self._scale ** 2

    def randomise(self, value):
        self.check_inputs(value)

        u1 = random()
        u2 = random()

        return sqrt(- 2 * log(u1)) * cos(2 * pi * u2) * self._scale + value
