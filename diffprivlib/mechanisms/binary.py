from numpy.random import random
from numpy import exp

from . import DPMechanism


class Binary(DPMechanism):
    def __init__(self):
        super().__init__()
        self.value0 = None
        self.value1 = None

    def __repr__(self):
        output = super().__repr__()
        output += ".setLabels(" + str(self.value0) + ", " + str(self.value1) + ")" if self.value0 is not None else ""

        return output

    def set_labels(self, value0, value1):
        if (type(value0) is not str) or (type(value1) is not str):
            raise ValueError("Binary labels must be strings. Use a DPTransformer"
                             " (e.g. transformers.IntToString) for non-string labels")

        if len(value0) * len(value1) == 0:
            raise ValueError("Binary labels must be non-empty strings")

        if value0 == value1:
            raise ValueError("Binary labels must not match")

        self.value0 = value0
        self.value1 = value1
        return self

    def check_inputs(self, value):
        super().check_inputs(value)

        if (self.value0 is None) or (self.value1 is None):
            raise ValueError("Binary labels must be set")

        if type(value) is not str:
            raise ValueError("Value to be randomised must be a string")

        if value not in [self.value0, self.value1]:
            raise ValueError("Value to be randomised is not in the domain")

    def randomise(self, value):
        self.check_inputs(value)

        indicator = 0 if value == self.value0 else 1

        u = random() * (exp(self.epsilon) + 1)

        if u > exp(self.epsilon):
            indicator = 1 - indicator

        return self.value0 if indicator == 0 else self.value1
