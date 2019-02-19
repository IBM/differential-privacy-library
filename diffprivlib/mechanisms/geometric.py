from numpy.random import random
from numpy import exp, floor, log

from . import DPMechanism, TruncationAndFoldingMachine


class Geometric(DPMechanism):
    def __init__(self):
        super().__init__()
        self.sensitivity = None
        self.scale = None

    def __repr__(self):
        output = super().__repr__()
        output += ".setSensitivity(" + str(self.sensitivity) + ")" if self.sensitivity is not None else ""

        return output

    def set_sensitivity(self, sensitivity):
        """

        :param sensitivity:
        :type sensitivity `float`
        :return:
        """
        if not isinstance(sensitivity, int):
            raise TypeError("Sensitivity must be integer-valued")

        if sensitivity <= 0:
            raise ValueError("Sensitivity must be strictly positive")

        self.sensitivity = sensitivity
        return self

    def check_inputs(self, value):
        super().check_inputs(value)

        if not isinstance(value, int):
            raise TypeError("Value to be randomised must be an integer")

        if self.sensitivity is None:
            raise ValueError("Sensitivity must be set")

    def randomise(self, value):
        self.check_inputs(value)

        if self.scale is None:
            self.scale = - self.epsilon / self.sensitivity

        # Need to account for overlap of 0-value between distributions of different sign
        u = random() - 0.5
        u *= 1 + exp(self.scale)
        sgn = -1 if u < 0 else 1

        # Use formula for geometric distribution, with ratio of exp(-epsilon/sensitivity)
        return int(value + sgn * floor(log(sgn * u) / self.scale))


class GeometricTruncated(Geometric, TruncationAndFoldingMachine):
    def __init__(self):
        super().__init__()
        TruncationAndFoldingMachine.__init__(self)

    def __repr__(self):
        output = super().__repr__()
        output += TruncationAndFoldingMachine.__repr__(self)

        return output

    def randomise(self, value):
        TruncationAndFoldingMachine.check_inputs(self, value)

        noisy_value = super().randomise(value)
        return int(self.__truncate(noisy_value))


class GeometricFolded(Geometric, TruncationAndFoldingMachine):
    def __init__(self):
        super().__init__()
        TruncationAndFoldingMachine.__init__(self)

    def __repr__(self):
        output = super().__repr__()
        output += TruncationAndFoldingMachine.__repr__(self)

        return output

    def randomise(self, value):
        TruncationAndFoldingMachine.check_inputs(self, value)

        noisy_value = super().randomise(value)
        return int(self.__fold(noisy_value))
