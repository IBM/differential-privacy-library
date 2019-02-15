from numpy.random import random

from . import DPMechanism


class Uniform(DPMechanism):
    def __init__(self):
        super().__init__()
        self.sensitivity = None

    def set_epsilon_delta(self, epsilon, delta):
        if epsilon != 0:
            raise ValueError("Epsilon must be strictly zero.")

        if not (0 < delta <= 0.5):
            raise ValueError("Delta must be strictly in (0,0.5]")

        return super().set_epsilon_delta(epsilon, delta)

    def set_sensitivity(self, sensitivity):
        """

        :param sensitivity: The sensitivity of the function being considered
        :type sensitivity: `float`
        :return:
        """

        if not isinstance(sensitivity, (int, float)):
            raise TypeError("Sensitivity must be numeric")

        if sensitivity <= 0:
            raise ValueError("Sensitivity must be strictly positive")

        self.sensitivity = sensitivity
        return self

    def get_bias(self, value):
        return 0.0

    def randomise(self, value):
        self.check_inputs(value)

        u = 2 * random() - 1
        u *= self.sensitivity / self.delta / 2

        return value + u
