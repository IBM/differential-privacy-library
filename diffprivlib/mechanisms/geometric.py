from numbers import Number
from numpy.random import random
from numpy import exp, floor, log

from diffprivlib.mechanisms import DPMechanism, TruncationMachine, FoldingMachine


class Geometric(DPMechanism):
    def __init__(self):
        super().__init__()
        self.sensitivity = None
        self.shape = None

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
        if not isinstance(sensitivity, Number):
            raise TypeError("Sensitivity must be numeric")

        if sensitivity <= 0:
            raise ValueError("Sensitivity must be strictly positive")

        self.sensitivity = sensitivity
        return self

    def check_inputs(self, value):
        super().check_inputs(value)

        if not isinstance(value, Number):
            raise TypeError("Value to be randomised must be a number")

        if self.sensitivity is None:
            raise ValueError("Sensitivity must be set")

    def randomise(self, value):
        self.check_inputs(value)

        if self.shape is None:
            self.shape = - self.epsilon / self.sensitivity

        # Need to account for overlap of 0-value between distributions of different sign
        u = random() - 0.5
        u *= 1 + exp(self.shape)
        sgn = -1 if u < 0 else 1

        # Use formula for geometric distribution, with ratio of exp(-epsilon/sensitivity)
        return int(value + sgn * floor(log(sgn * u) / self.shape))


class GeometricTruncated(Geometric, TruncationMachine):
    def __init__(self):
        super().__init__()
        TruncationMachine.__init__(self)

    def __repr__(self):
        output = super().__repr__()
        output += TruncationMachine.__repr__(self)

        return output

    def randomise(self, value):
        TruncationMachine.check_inputs(self, value)

        noisy_value = super().randomise(value)
        return int(super().truncate(noisy_value))


class GeometricFolded(Geometric, FoldingMachine):
    def __init__(self):
        super().__init__()
        FoldingMachine.__init__(self)

    def __repr__(self):
        output = super().__repr__()
        output += FoldingMachine.__repr__(self)

        return output

    def randomise(self, value):
        FoldingMachine.check_inputs(self, value)

        noisy_value = super().randomise(value)
        return super().fold(noisy_value)
