from numbers import Real

from numpy import exp
from numpy.random import geometric, random

from . import Laplace


class Staircase(Laplace):
    def __init__(self):
        super().__init__()
        self._gamma = None

    def set_gamma(self, gamma):
        if not isinstance(gamma, Real):
            raise TypeError("Gamma must be numeric")
        # noinspection PyTypeChecker
        if not (0.0 <= gamma <= 1.0):
            raise ValueError("Gamma must be in [0,1]")

        self._gamma = float(gamma)
        return self

    def check_inputs(self, value=None):
        super().check_inputs(value)

        if self._gamma is None:
            raise ValueError("Gamma must be set")

        return True

    def set_epsilon_delta(self, epsilon, delta):
        if not delta == 0:
            raise ValueError("Delta must be zero")

        return super().set_epsilon_delta(epsilon, delta)

    def get_bias(self, value):
        return 0.0

    def randomise(self, value):
        self.check_inputs(value)

        s = -1 if random() < 0.5 else 1
        g = geometric(1 - exp(- self._epsilon)) - 1
        u = random()
        b = 0 if random() < self._gamma / (self._gamma + (1 - self._gamma) * exp(- self._epsilon)) else 1

        return value + s * ((1 - b) * ((g + self._gamma * u) * self._sensitivity) +
                            b * ((g + self._gamma + (1 - self._gamma) * u) * self._sensitivity))
