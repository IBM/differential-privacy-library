# MIT License
#
# Copyright (C) IBM Corporation 2019
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
The binary mechanism for differential privacy.

"""
import numpy as np
from numpy.random import random

from diffprivlib.mechanisms.base import DPMechanism
from diffprivlib.utils import copy_docstring


class Binary(DPMechanism):
    """The classic binary mechanism in differential privacy.

    Given a binary input value, the mechanism randomly decides to flip to the other binary value or not, in order to
    satisfy differential privacy.

    Paper link: https://arxiv.org/pdf/1612.05568.pdf

    Notes
    -----
    * The binary attributes, known as `labels`, must be specified as strings.  If non-string labels are required (e.g.
      integer-valued labels), a :class:`.DPTransformer` can be used (e.g. :class:`.IntToString`).
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
        """Sets the binary labels of the mechanism.

        Labels must be unique, non-empty strings.  If non-string labels are required, consider using a
        :class:`.DPTransformer`.

        Parameters
        ----------
        value0 : str
            0th binary label.

        value1 : str
            1st binary label.

        Returns
        -------
        self : class

        """
        if not isinstance(value0, str) or not isinstance(value1, str):
            raise TypeError("Binary labels must be strings. Use a DPTransformer  (e.g. transformers.IntToString) for "
                            "non-string labels")

        if len(value0) * len(value1) == 0:
            raise ValueError("Binary labels must be non-empty strings")

        if value0 == value1:
            raise ValueError("Binary labels must not match")

        self._value0 = value0
        self._value1 = value1
        return self

    def check_inputs(self, value):
        """Checks that all parameters of the mechanism have been initialised correctly, and that the mechanism is ready
        to be used.

        Parameters
        ----------
        value : str
            The value to be checked.

        Returns
        -------
        True if the mechanism is ready to be used.

        Raises
        ------
        Exception
            If parameters have not been set correctly, or if `value` falls outside the domain of the mechanism.

        """
        super().check_inputs(value)

        if (self._value0 is None) or (self._value1 is None):
            raise ValueError("Binary labels must be set")

        if not isinstance(value, str):
            raise TypeError("Value to be randomised must be a string")

        if value not in [self._value0, self._value1]:
            raise ValueError("Value to be randomised is not in the domain {\"" + self._value0 + "\", \"" + self._value1
                             + "\"}")

        return True

    @copy_docstring(DPMechanism.get_bias)
    def get_bias(self, value):
        raise NotImplementedError

    @copy_docstring(DPMechanism.get_variance)
    def get_variance(self, value):
        raise NotImplementedError

    def randomise(self, value):
        """Randomise `value` with the mechanism.

        Parameters
        ----------
        value : str
            The value to be randomised.

        Returns
        -------
        str
            The randomised value.

        """
        self.check_inputs(value)

        indicator = 0 if value == self._value0 else 1

        unif_rv = random() * (np.exp(self._epsilon) + 1)

        if unif_rv > np.exp(self._epsilon) + self._delta:
            indicator = 1 - indicator

        return self._value0 if indicator == 0 else self._value1
