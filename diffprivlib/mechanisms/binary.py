"""
The binary mechanism for differential privacy.
"""
import numpy as np
from numpy.random import random

from . import DPMechanism


class Binary(DPMechanism):
    """
    Classic binary mechanism for differential privacy.
    Paper link: https://arxiv.org/pdf/1612.05568.pdf
    """
    def __init__(self):
        super().__init__()
        self._value0 = None
        self._value1 = None

    def __repr__(self):
        output = super().__repr__()
        output += ".set_labels(" + str(self._value0) + ", " + str(self._value1) + ")" \
            if self._value0 is not None else ""

        return output

    def set_labels(self, value0, value1):
        """
        Set the binary labels of the mechanism. Labels must be unique, non-empty strings.

        If non-string labels are required, consider using a :class:`.DPTransformer`.

        :param value0: Binary label.
        :type value0: str
        :param value1: Binary label.
        :type value1: str
        :return: self
        """
        if not isinstance(value0, str) or not isinstance(value1, str):
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
        """
        Check that all parameters of the mechanism have been initialised correctly, and that the mechanism is ready
        to be used.

        :param value: Value to be checked.
        :type value: `str`
        :return: True if the mechanism is ready to be used.
        :rtype: `bool`
        """
        super().check_inputs(value)

        if (self._value0 is None) or (self._value1 is None):
            raise ValueError("Binary labels must be set")

        if not isinstance(value, str):
            raise TypeError("Value to be randomised must be a string")

        if value not in [self._value0, self._value1]:
            raise ValueError("Value to be randomised is not in the domain {\"" +
                             self._value0 + "\", \"" + self._value1 + "\"}")

        return True

    def randomise(self, value):
        """
        Randomise the given value. The value must be one of the pre-defined binary labels.

        :param value: Value to be randomised.
        :type value: `string`
        :return: Randomised value, one of the binary labels.
        :rtype: `string`
        """
        self.check_inputs(value)

        indicator = 0 if value == self._value0 else 1

        unif_rv = random() * (np.exp(self._epsilon) + 1)

        if unif_rv > np.exp(self._epsilon) + self._delta:
            indicator = 1 - indicator

        return self._value0 if indicator == 0 else self._value1
