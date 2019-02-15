from numpy import log, sqrt, cos, pi
from numpy.random import random

from . import DPMechanism


class Gaussian(DPMechanism):
    def __init__(self):
        super().__init__()
        self.sensitivity = None
        self.shape = None

    def set_epsilon_delta(self, epsilon, delta):
        if epsilon * delta == 0:
            raise ValueError("Neither Epsilon not Delta can be zero")

        return super().set_epsilon_delta(epsilon, delta)

    def set_sensitivity(self, sensitivity):
        self.sensitivity = sensitivity
        return self

    def get_bias(self, value):
        return 0.0

    def get_variance(self, value):
        self.check_inputs(0)

        return self.shape ** 2

    def randomise(self, value):
        self.check_inputs(value)

        if self.shape is None:
            self.shape = sqrt(2 * log(1.25 / self.delta)) * self.sensitivity / self.epsilon

        u1 = random()
        u2 = random()

        return sqrt(- 2 * log(u1)) * cos(2 * pi * u2) * self.shape + value
