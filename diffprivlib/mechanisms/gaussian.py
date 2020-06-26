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
from numbers import Real, Integral

import numpy as np
from numpy.random import random

from diffprivlib.mechanisms.base import DPMechanism
from diffprivlib.mechanisms.geometric import Geometric
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
            Epsilon value of the mechanism.  Must satisfy 0 < `epsilon` <= 1.

        delta : float
            Delta value of the mechanism.  Must satisfy 0 < `delta` <= 1.

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

        if sensitivity < 0:
            raise ValueError("Sensitivity must be non-negative")

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

        return value + standard_normal * self._scale


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
            Epsilon value of the mechanism.  Must satisfy 0 < `epsilon`.

        delta : float
            Delta value of the mechanism.  Must satisfy 0 < `delta` < 1.

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
        if self._sensitivity / self._epsilon == 0:
            return 0.0

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


class GaussianDiscrete(DPMechanism):
    """The Discrete Gaussian mechanism in differential privacy.

    As proposed by Canonne, Kamath and Steinke, re-purposed for approximate differential privacy.

    Paper link: https://arxiv.org/pdf/2004.00010.pdf

    """
    def __init__(self):
        super().__init__()
        self._scale = None
        self._sensitivity = 1

    def set_epsilon_delta(self, epsilon, delta):
        r"""Sets the privacy parameters :math:`\epsilon` and :math:`\delta` for the mechanism.

        For the discrete Gaussian mechanism, `epsilon` and `delta` must be non-zero.

        Parameters
        ----------
        epsilon : float
            Epsilon value of the mechanism.  Must satisfy 0 < `epsilon`.

        delta : float
            Delta value of the mechanism.  Must satisfy 0 < `delta` < 1.

        Returns
        -------
        self : class

        """
        if epsilon == 0 or delta == 0:
            raise ValueError("Neither Epsilon nor Delta can be zero")

        self._scale = None
        return super().set_epsilon_delta(epsilon, delta)

    @copy_docstring(Geometric.set_sensitivity)
    def set_sensitivity(self, sensitivity):
        if not isinstance(sensitivity, Integral):
            raise TypeError("Sensitivity must be an integer")

        if sensitivity < 0:
            raise ValueError("Sensitivity must be non-negative")

        self._scale = None
        self._sensitivity = sensitivity
        return self

    @copy_docstring(Geometric.check_inputs)
    def check_inputs(self, value):
        super().check_inputs(value)

        if self._delta is None:
            raise ValueError("Delta must be set")

        if self._scale is None:
            self._scale = self._find_scale()

        if not isinstance(value, Integral):
            raise TypeError("Value to be randomised must be an integer")

        return True

    @copy_docstring(Laplace.get_bias)
    def get_bias(self, value):
        return 0.0

    @copy_docstring(Laplace.get_variance)
    def get_variance(self, value):
        raise NotImplementedError

    @copy_docstring(Geometric.randomise)
    def randomise(self, value):
        self.check_inputs(value)

        if self._scale == 0:
            return value

        tau = 1 / (1 + np.floor(self._scale))
        sigma2 = self._scale ** 2

        while True:
            geom_x = 0
            while self._bernoulli_exp(tau):
                geom_x += 1

            bern_b = np.random.binomial(1, 0.5)
            if bern_b and not geom_x:
                continue

            lap_y = int((1 - 2 * bern_b) * geom_x)
            bern_c = self._bernoulli_exp((abs(lap_y) - tau * sigma2) ** 2 / 2 / sigma2)
            if bern_c:
                return value + lap_y

    def _find_scale(self):
        """Determine the scale of the mechanism's distribution given epsilon and delta.
        """
        if self._epsilon is None or self._delta is None:
            raise ValueError("Epsilon and Delta must be set before calling _find_scale().")
        if self._sensitivity / self._epsilon == 0:
            return 0

        def objective(sigma, epsilon_, delta_, sensitivity_):
            """Function for which we are seeking its root. """
            idx_0 = int(np.floor(epsilon_ * sigma ** 2 / sensitivity_ - sensitivity_ / 2))
            idx_1 = int(np.floor(epsilon_ * sigma ** 2 / sensitivity_ + sensitivity_ / 2))
            idx = 1

            lhs, rhs, denom = float(idx_0 < 0), 0, 1
            _term, diff = 1, 1

            while _term > 0 and diff > 0:
                _term = np.exp(-idx ** 2 / 2 / sigma ** 2)

                if idx > idx_0:
                    lhs += _term

                    if idx_0 < -idx:
                        lhs += _term

                    if idx > idx_1:
                        diff = -rhs
                        rhs += _term
                        diff += rhs

                denom += 2 * _term
                idx += 1
                if idx > 1e6:
                    raise ValueError("Infinite sum not converging, aborting. Try changing the epsilon and/or delta.")

            return (lhs - np.exp(epsilon_) * rhs) / denom - delta_

        epsilon = self._epsilon
        delta = self._delta
        sensitivity = self._sensitivity

        # Begin by locating the root within an interval [2**i, 2**(i+1)]
        guess_0 = 1
        f_0 = objective(guess_0, epsilon, delta, sensitivity)
        pwr = 1 if f_0 > 0 else -1
        guess_1 = 2 ** pwr
        f_1 = objective(guess_1, epsilon, delta, sensitivity)

        while f_0 * f_1 > 0:
            guess_0 *= 2 ** pwr
            guess_1 *= 2 ** pwr

            f_0 = f_1
            f_1 = objective(guess_1, epsilon, delta, sensitivity)

        # Find the root (sigma) using the bisection method
        while not np.isclose(guess_0, guess_1, atol=1e-12, rtol=1e-6):
            guess_mid = (guess_0 + guess_1) / 2
            f_mid = objective(guess_mid, epsilon, delta, sensitivity)

            if f_mid * f_0 <= 0:
                f_1 = f_mid
                guess_1 = guess_mid
            if f_mid * f_1 <= 0:
                f_0 = f_mid
                guess_0 = guess_mid

        return (guess_0 + guess_1) / 2

    def _bernoulli_exp(self, gamma):
        """Sample from Bernoulli(exp(-gamma))

        Adapted from Appendix A of https://arxiv.org/pdf/2004.00010.pdf

        """
        if gamma > 1:
            gamma_ceil = np.ceil(gamma)
            for _ in np.arange(gamma_ceil):
                if not self._bernoulli_exp(gamma / gamma_ceil):
                    return 0

            return 1

        counter = 1

        while np.random.binomial(1, gamma / counter):
            counter += 1

        return counter % 2
