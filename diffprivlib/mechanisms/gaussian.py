from numpy import log, sqrt, cos, pi
from numpy.random import random

from . import DPMechanism


class Gaussian(DPMechanism):
    def __init__(self):
        super().__init__()
        self.sensitivity = None
        self.scale = None

    def set_epsilon_delta(self, epsilon, delta):
        if epsilon * delta == 0:
            raise ValueError("Neither Epsilon nor Delta can be zero")

        if epsilon > 1.0:
            raise ValueError("Epsilon cannot be greater than 1")

        return super().set_epsilon_delta(epsilon, delta)

    def set_sensitivity(self, sensitivity):
        if sensitivity <= 0:
            raise ValueError("Sensitivity must be strictly positive")
        self.sensitivity = sensitivity
        return self

    def get_bias(self, value):
        return 0.0

    def get_variance(self, value):
        self.check_inputs(0)

        return self.scale ** 2

    def randomise(self, value):
        self.check_inputs(value)

        if self.scale is None:
            self.scale = sqrt(2 * log(1.25 / self.delta)) * self.sensitivity / self.epsilon

        u1 = random()
        u2 = random()

        return sqrt(- 2 * log(u1)) * cos(2 * pi * u2) * self.scale + value
