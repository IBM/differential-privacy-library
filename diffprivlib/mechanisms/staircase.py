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
The staircase mechanism in differential privacy.
"""
from numbers import Real

import numpy as np
from numpy.random import geometric, random

from diffprivlib.mechanisms.laplace import Laplace
from diffprivlib.utils import copy_docstring


class Staircase(Laplace):
    """
    The staircase mechanism in differential privacy.

    The staircase mechanism is an optimisation of the classical Laplace Mechanism (:class:`.Laplace`), described as a
    "geometric mixture of uniform random variables".
    Paper link: https://arxiv.org/pdf/1212.1186.pdf
    """
    def __init__(self, *, epsilon, sensitivity, gamma=None):
        super().__init__(epsilon=epsilon, delta=0, sensitivity=sensitivity)
        self.gamma = self._check_gamma(gamma, init=True)

    def __repr__(self):
        output = super().__repr__()
        output += ".set_gamma(" + str(self.gamma) + ")" if self.gamma is not None else ""

        return output

    def _check_gamma(self, gamma, init=False):
        r"""Sets the tuning parameter :math:`\gamma` for the mechanism.

        Must satisfy 0 <= `gamma` <= 1.  If not set, gamma defaults to minimise the expectation of the amplitude of
        noise,
        .. math:: \gamma = \frac{1}{1 + e^{\epsilon / 2}}

        Parameters
        ----------
        gamma : float
            Value of the tuning parameter gamma for the mechanism.

        Returns
        -------
        self : class

        Raises
        ------
        TypeError
            If `gamma` is not a float.

        ValueError
            If `gamma` is does not satisfy 0 <= `gamma` <= 1.

        """
        if init and gamma is None:
            gamma = 1 / (1 + np.exp(self.epsilon / 2))

        if not isinstance(gamma, Real):
            raise TypeError("Gamma must be numeric")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("Gamma must be in [0,1]")

        return float(gamma)

    @copy_docstring(Laplace._check_all)
    def _check_all(self, value):
        super()._check_all(value)
        self._check_gamma(self.gamma)

        return True

    def _check_epsilon_delta(self, epsilon, delta):
        r"""Sets the value of :math:`\epsilon` and :math:`\delta` to be used by the mechanism.

        For the staircase mechanism, `delta` must be zero and `epsilon` must be strictly positive.

        Parameters
        ----------
        epsilon : float
            The value of epsilon for achieving :math:`(\epsilon,\delta)`-differential privacy with the mechanism.  Must
            have `epsilon > 0`.

        delta : float
            For the staircase mechanism, `delta` must be zero.

        Returns
        -------
        self : class

        Raises
        ------
        ValueError
            If `epsilon` is zero or negative, or if `delta` is non-zero.

        """
        if not delta == 0:
            raise ValueError("Delta must be zero")

        return super()._check_epsilon_delta(epsilon, delta)

    @copy_docstring(Laplace.bias)
    def bias(self, value):
        return 0.0

    @copy_docstring(Laplace.variance)
    def variance(self, value):
        raise NotImplementedError

    @copy_docstring(Laplace.randomise)
    def randomise(self, value):
        self._check_all(value)

        sign = -1 if random() < 0.5 else 1
        geometric_rv = geometric(1 - np.exp(- self.epsilon)) - 1
        unif_rv = random()
        binary_rv = 0 if random() < self.gamma / (self.gamma + (1 - self.gamma) * np.exp(- self.epsilon)) else 1

        return value + sign * ((1 - binary_rv) * ((geometric_rv + self.gamma * unif_rv) * self.sensitivity) +
                               binary_rv * ((geometric_rv + self.gamma + (1 - self.gamma) * unif_rv) *
                                            self.sensitivity))
