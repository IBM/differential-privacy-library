from numpy.random import random
from numpy import exp

from . import DPMechanism


class Binary(DPMechanism):
    def __init__(self):
        super().__init__()
        self._value0 = None
        self._value1 = None

    def __repr__(self):
        output = super().__repr__()
        output += ".set_labels(" + str(self._value0) + ", " + str(self._value1) + ")" if self._value0 is not None else ""

        return output

    def set_labels(self, value0, value1):
        if (type(value0) is not str) or (type(value1) is not str):
            raise TypeError("Binary labels must be strings. Use a DPTransformer"
                             " (e.g. transformers.IntToString) for non-string labels")

        if len(value0) * len(value1) == 0:
            raise ValueError("Binary labels must be non-empty strings")

        if value0 == value1:
            raise ValueError("Binary labels must not match")

        self._value0 = value0
        self._value1 = value1
        return self

    def check_inputs(self, value):
        super().check_inputs(value)

        if (self._value0 is None) or (self._value1 is None):
            raise ValueError("Binary labels must be set")

        if type(value) is not str:
            raise TypeError("Value to be randomised must be a string")
        elif value not in [self._value0, self._value1]:
            raise ValueError("Value to be randomised is not in the domain {\"" +
                             self._value0 + "\", \"" + self._value1 + "\"}")

        return True

    def randomise(self, value):
        self.check_inputs(value)

        indicator = 0 if value == self._value0 else 1

        u = random() * (exp(self._epsilon) + 1)

        if u > exp(self._epsilon):
            indicator = 1 - indicator

        return self._value0 if indicator == 0 else self._value1
