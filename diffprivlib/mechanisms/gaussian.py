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
The classic Gaussian mechanism in differential privacy, and its derivatives.
"""
from math import erf
from numbers import Real

import numpy as np
from numpy.random import random

from diffprivlib.mechanisms.base import DPMechanism
from diffprivlib.mechanisms.laplace import Laplace
from diffprivlib.utils import copy_docstring


class Gaussian(DPMechanism):
    """The Gaussian mechanism in differential privacy.

    As first proposed by Dwork and Roth in "The algorithmic foundations of differential privacy".

    Paper link: https://www.nowpublishers.com/article/DownloadSummary/TCS-042

    """
    def __init__(self):
        super().__init__()
        self._sensitivity = None
        self._scale = None
        self._stored_gaussian = None

    def __repr__(self):
        output = super().__repr__()
        output += ".set_sensitivity(" + str(self._sensitivity) + ")" if self._sensitivity is not None else ""

        return output

    def set_epsilon_delta(self, epsilon, delta):
        r"""Sets the privacy parameters :math:`\epsilon` and :math:`\delta` for the mechanism.

        For the Gaussian mechanism, `epsilon` cannot be greater than 1, and `delta` must be non-zero.

        Parameters
        ----------
        epsilon : float
            Epsilon value of the mechanism. Must satisfy 0 < `epsilon` <= 1.

        delta : float
            Delta value of the mechanism. Must satisfy 0 < `delta` <= 1.

        Returns
        -------
        self : class

        """
        if epsilon == 0 or delta == 0:
            raise ValueError("Neither Epsilon nor Delta can be zero")

        if isinstance(epsilon, Real) and epsilon > 1.0:
            raise ValueError("Epsilon cannot be greater than 1")

        self._scale = None
        return super().set_epsilon_delta(epsilon, delta)

    @copy_docstring(Laplace.set_sensitivity)
    def set_sensitivity(self, sensitivity):
        if not isinstance(sensitivity, Real):
            raise TypeError("Sensitivity must be numeric")

        if sensitivity <= 0:
            raise ValueError("Sensitivity must be strictly positive")

        self._scale = None
        self._sensitivity = sensitivity
        return self

    @copy_docstring(Laplace.check_inputs)
    def check_inputs(self, value):
        super().check_inputs(value)

        if self._delta is None:
            raise ValueError("Delta must be set")

        if self._sensitivity is None:
            raise ValueError("Sensitivity must be set")

        if self._scale is None:
            self._scale = np.sqrt(2 * np.log(1.25 / self._delta)) * self._sensitivity / self._epsilon

        if not isinstance(value, Real):
            raise TypeError("Value to be randomised must be a number")

        return True

    @copy_docstring(Laplace.get_bias)
    def get_bias(self, value):
        return 0.0

    @copy_docstring(Laplace.get_variance)
    def get_variance(self, value):
        self.check_inputs(0)

        return self._scale ** 2

    @copy_docstring(Laplace.randomise)
    def randomise(self, value):
        self.check_inputs(value)

        if self._stored_gaussian is None:
            unif_rv1 = random()
            unif_rv2 = random()

            self._stored_gaussian = np.sqrt(- 2 * np.log(unif_rv1)) * np.sin(2 * np.pi * unif_rv2)
            standard_normal = np.sqrt(- 2 * np.log(unif_rv1)) * np.cos(2 * np.pi * unif_rv2)
        else:
            standard_normal = self._stored_gaussian
            self._stored_gaussian = None

        return standard_normal * self._scale + value


class GaussianAnalytic(Gaussian):
    """The analytic Gaussian mechanism in differential privacy.

    As first proposed by Balle and Wang in "Improving the Gaussian Mechanism for Differential Privacy: Analytical
    Calibration and Optimal Denoising".

    Paper link: https://arxiv.org/pdf/1805.06530.pdf

    """
    def set_epsilon_delta(self, epsilon, delta):
        r"""Sets the privacy parameters :math:`\epsilon` and :math:`\delta` for the mechanism.

        For the analytic Gaussian mechanism, `epsilon` and `delta` must be non-zero.

        Parameters
        ----------
        epsilon : float
            Epsilon value of the mechanism. Must satisfy 0 < `epsilon`.

        delta : float
            Delta value of the mechanism. Must satisfy 0 < `delta` < 1.

        Returns
        -------
        self : class

        """
        if epsilon == 0 or delta == 0:
            raise ValueError("Neither Epsilon nor Delta can be zero")

        self._scale = None
        return DPMechanism.set_epsilon_delta(self, epsilon, delta)

    @copy_docstring(Laplace.check_inputs)
    def check_inputs(self, value):
        if self._scale is None:
            self._scale = self._find_scale()

        super().check_inputs(value)

        return True

    def _find_scale(self):
        if self._epsilon is None or self._delta is None:
            raise ValueError("Epsilon and Delta must be set before calling _find_scale().")
        if self._sensitivity is None:
            raise ValueError("Sensitivity must be set before calling _find_scale().")

        epsilon = self._epsilon
        delta = self._delta

        def phi(val):
            return (1 + erf(val / np.sqrt(2))) / 2

        def b_plus(val):
            return phi(np.sqrt(epsilon * val)) - np.exp(epsilon) * phi(- np.sqrt(epsilon * (val + 2))) - delta

        def b_minus(val):
            return phi(- np.sqrt(epsilon * val)) - np.exp(epsilon) * phi(- np.sqrt(epsilon * (val + 2))) - delta

        delta_0 = b_plus(0)

        if delta_0 == 0:
            alpha = 1
        else:
            if delta_0 < 0:
                target_func = b_plus
            else:
                target_func = b_minus

            # Find the starting interval by doubling the initial size until the target_func sign changes, as suggested
            # in the paper
            left = 0
            right = 1

            while target_func(left) * target_func(right) > 0:
                left = right
                right *= 2

            # Binary search code copied from mechanisms.LaplaceBoundedDomain
            old_interval_size = (right - left) * 2

            while old_interval_size > right - left:
                old_interval_size = right - left
                middle = (right + left) / 2

                if target_func(middle) * target_func(left) <= 0:
                    right = middle
                if target_func(middle) * target_func(right) <= 0:
                    left = middle

            alpha = np.sqrt(1 + (left + right) / 4) + (-1 if delta_0 < 0 else 1) * np.sqrt((left + right) / 4)

        return alpha * self._sensitivity / np.sqrt(2 * self._epsilon)
