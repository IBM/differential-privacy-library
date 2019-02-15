from numpy import exp
from numpy.random import geometric, random

from . import Laplace


class Staircase(Laplace):
    def __init__(self):
        super().__init__()
        self.gamma = None

    def set_gamma(self, gamma):
        if not isinstance(gamma, float):
            raise ValueError("Gamma must be a floating point number")

        if not (0 <= gamma <= 1):
            raise ValueError("Gamma must be in [0,1]")

        self.gamma = gamma
        return self

    def check_inputs(self, value):
        super().check_inputs(value)

        if self.gamma is None:
            raise ValueError("Gamma must be set")

        return True

    def get_variance(self, value):
        raise NotImplementedError

    def randomise(self, value):
        s = -1 if random() < 0.5 else 1
        g = geometric(1 - exp(- self.epsilon)) - 1
        u = random()
        b = 0 if random() < self.gamma / (self.gamma + (1 - self.gamma) * exp(- self.epsilon)) else 1

        return s * ((1 - b) * ((g + self.gamma * u) * self.sensitivity) +
                    b * ((g + self.gamma + (1 - self.gamma) * u) * self.sensitivity))
