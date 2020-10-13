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
The classic Laplace mechanism in differential privacy, and its derivatives.
"""
from numbers import Real

import numpy as np

from diffprivlib.mechanisms.base import DPMechanism, TruncationAndFoldingMixin
from diffprivlib.utils import copy_docstring


class Laplace(DPMechanism):
    r"""
    The classic Laplace mechanism in differential privacy, as first proposed by Dwork, McSherry, Nissim and Smith.

    Paper link: https://link.springer.com/content/pdf/10.1007/11681878_14.pdf

    Includes extension to (relaxed) :math:`(\epsilon,\delta)`-differential privacy, as proposed by Holohan et al.

    Paper link: https://arxiv.org/pdf/1402.6124.pdf

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in [0, ∞].

    delta : float, default: 0.0
        Privacy parameter :math:`\delta` for the mechanism.  Must be in [0, 1].  Cannot be simultaneously zero with
        ``epsilon``.

    sensitivity : float
        The sensitivity of the mechanism.  Must be in [0, ∞).

    """
    def __init__(self, *, epsilon, delta=0.0, sensitivity):
        super().__init__(epsilon=epsilon, delta=delta)
        self.sensitivity = self._check_sensitivity(sensitivity)

    @classmethod
    def _check_sensitivity(cls, sensitivity):
        if not isinstance(sensitivity, Real):
            raise TypeError("Sensitivity must be numeric")

        if sensitivity < 0:
            raise ValueError("Sensitivity must be non-negative")

        return float(sensitivity)

    def _check_all(self, value):
        super()._check_all(value)
        self._check_sensitivity(self.sensitivity)

        if not isinstance(value, Real):
            raise TypeError("Value to be randomised must be a number")

        return True

    def bias(self, value):
        """Returns the bias of the mechanism at a given `value`.

        Parameters
        ----------
        value : int or float
            The value at which the bias of the mechanism is sought.

        Returns
        -------
        bias : float or None
            The bias of the mechanism at `value`.

        """
        return 0.0

    def variance(self, value):
        """Returns the variance of the mechanism at a given `value`.

        Parameters
        ----------
        value : float
            The value at which the variance of the mechanism is sought.

        Returns
        -------
        bias : float
            The variance of the mechanism at `value`.

        """
        self._check_all(0)

        return 2 * (self.sensitivity / (self.epsilon - np.log(1 - self.delta))) ** 2

    def randomise(self, value):
        """Randomise `value` with the mechanism.

        Parameters
        ----------
        value : float
            The value to be randomised.

        Returns
        -------
        float
            The randomised value.

        """
        self._check_all(value)

        scale = self.sensitivity / (self.epsilon - np.log(1 - self.delta))

        unif_rv = self._rng.random() - 0.5

        return value - scale * np.sign(unif_rv) * np.log(1 - 2 * np.abs(unif_rv))


class LaplaceTruncated(Laplace, TruncationAndFoldingMixin):
    r"""
    The truncated Laplace mechanism, where values outside a pre-described domain are mapped to the closest point
    within the domain.

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in [0, ∞].

    delta : float, default: 0.0
        Privacy parameter :math:`\delta` for the mechanism.  Must be in [0, 1].  Cannot be simultaneously zero with
        ``epsilon``.

    sensitivity : float
        The sensitivity of the mechanism.  Must be in [0, ∞).

    lower : float
        The lower bound of the mechanism.

    upper : float
        The upper bound of the mechanism.

    """
    def __init__(self, *, epsilon, delta=0.0, sensitivity, lower, upper):
        super().__init__(epsilon=epsilon, delta=delta, sensitivity=sensitivity)
        TruncationAndFoldingMixin.__init__(self, lower=lower, upper=upper)

    @copy_docstring(Laplace.bias)
    def bias(self, value):
        self._check_all(value)

        shape = self.sensitivity / self.epsilon

        return shape / 2 * (np.exp((self.lower - value) / shape) - np.exp((value - self.upper) / shape))

    @copy_docstring(Laplace.variance)
    def variance(self, value):
        self._check_all(value)

        shape = self.sensitivity / self.epsilon

        variance = value ** 2 + shape * (self.lower * np.exp((self.lower - value) / shape)
                                         - self.upper * np.exp((value - self.upper) / shape))
        variance += (shape ** 2) * (2 - np.exp((self.lower - value) / shape)
                                    - np.exp((value - self.upper) / shape))

        variance -= (self.bias(value) + value) ** 2

        return variance

    def _check_all(self, value):
        Laplace._check_all(self, value)
        TruncationAndFoldingMixin._check_all(self, value)

        return True

    @copy_docstring(Laplace.randomise)
    def randomise(self, value):
        self._check_all(value)

        noisy_value = super().randomise(value)
        return self._truncate(noisy_value)


class LaplaceFolded(Laplace, TruncationAndFoldingMixin):
    r"""
    The folded Laplace mechanism, where values outside a pre-described domain are folded around the domain until they
    fall within.

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in [0, ∞].

    delta : float, default: 0.0
        Privacy parameter :math:`\delta` for the mechanism.  Must be in [0, 1].  Cannot be simultaneously zero with
        ``epsilon``.

    sensitivity : float
        The sensitivity of the mechanism.  Must be in [0, ∞).

    lower : float
        The lower bound of the mechanism.

    upper : float
        The upper bound of the mechanism.

    """
    def __init__(self, *, epsilon, delta=0.0, sensitivity, lower, upper):
        super().__init__(epsilon=epsilon, delta=delta, sensitivity=sensitivity)
        TruncationAndFoldingMixin.__init__(self, lower=lower, upper=upper)

    @copy_docstring(Laplace.bias)
    def bias(self, value):
        self._check_all(value)

        shape = self.sensitivity / self.epsilon

        bias = shape * (np.exp((self.lower + self.upper - 2 * value) / shape) - 1)
        bias /= np.exp((self.lower - value) / shape) + np.exp((self.upper - value) / shape)

        return bias

    @copy_docstring(DPMechanism.variance)
    def variance(self, value):
        raise NotImplementedError

    def _check_all(self, value):
        super()._check_all(value)
        TruncationAndFoldingMixin._check_all(self, value)

        return True

    @copy_docstring(Laplace.randomise)
    def randomise(self, value):
        self._check_all(value)

        noisy_value = super().randomise(value)
        return self._fold(noisy_value)


class LaplaceBoundedDomain(LaplaceTruncated):
    r"""
    The bounded Laplace mechanism on a bounded domain.  The mechanism draws values directly from the domain, without any
    post-processing.

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in [0, ∞].

    delta : float, default: 0.0
        Privacy parameter :math:`\delta` for the mechanism.  Must be in [0, 1].  Cannot be simultaneously zero with
        ``epsilon``.

    sensitivity : float
        The sensitivity of the mechanism.  Must be in [0, ∞).

    lower : float
        The lower bound of the mechanism.

    upper : float
        The upper bound of the mechanism.

    """
    def __init__(self, *, epsilon, delta=0.0, sensitivity, lower, upper):
        super().__init__(epsilon=epsilon, delta=delta, sensitivity=sensitivity, lower=lower, upper=upper)
        self._scale = None

    def _find_scale(self):
        eps = self.epsilon
        delta = self.delta
        diam = self.upper - self.lower
        delta_q = self.sensitivity

        def _delta_c(shape):
            if shape == 0:
                return 2.0
            return (2 - np.exp(- delta_q / shape) - np.exp(- (diam - delta_q) / shape)) / (1 - np.exp(- diam / shape))

        def _f(shape):
            return delta_q / (eps - np.log(_delta_c(shape)) - np.log(1 - delta))

        left = delta_q / (eps - np.log(1 - delta))
        right = _f(left)
        old_interval_size = (right - left) * 2

        while old_interval_size > right - left:
            old_interval_size = right - left
            middle = (right + left) / 2

            if _f(middle) >= middle:
                left = middle
            if _f(middle) <= middle:
                right = middle

        return (right + left) / 2

    def _cdf(self, value):
        # Allow for infinite epsilon
        if self._scale == 0:
            return 0 if value < 0 else 1

        if value < 0:
            return 0.5 * np.exp(value / self._scale)

        return 1 - 0.5 * np.exp(-value / self._scale)

    def effective_epsilon(self):
        r"""Gets the effective epsilon of the mechanism, only for strict :math:`\epsilon`-differential privacy.  Returns
        ``None`` if :math:`\delta` is non-zero.

        Returns
        -------
        float
            The effective :math:`\epsilon` parameter of the mechanism.  Returns ``None`` if `delta` is non-zero.

        """
        if self._scale is None:
            self._scale = self._find_scale()

        if self.delta > 0.0:
            return None

        return self.sensitivity / self._scale

    @copy_docstring(Laplace.bias)
    def bias(self, value):
        self._check_all(value)

        if self._scale is None:
            self._scale = self._find_scale()

        bias = (self._scale - self.lower + value) / 2 * np.exp((self.lower - value) / self._scale) \
            - (self._scale + self.upper - value) / 2 * np.exp((value - self.upper) / self._scale)
        bias /= 1 - np.exp((self.lower - value) / self._scale) / 2 \
            - np.exp((value - self.upper) / self._scale) / 2

        return bias

    @copy_docstring(Laplace.variance)
    def variance(self, value):
        self._check_all(value)

        if self._scale is None:
            self._scale = self._find_scale()

        variance = value**2
        variance -= (np.exp((self.lower - value) / self._scale) * (self.lower ** 2)
                     + np.exp((value - self.upper) / self._scale) * (self.upper ** 2)) / 2
        variance += self._scale * (self.lower * np.exp((self.lower - value) / self._scale)
                                   - self.upper * np.exp((value - self.upper) / self._scale))
        variance += (self._scale ** 2) * (2 - np.exp((self.lower - value) / self._scale)
                                          - np.exp((value - self.upper) / self._scale))
        variance /= 1 - (np.exp(-(value - self.lower) / self._scale)
                         + np.exp(-(self.upper - value) / self._scale)) / 2

        variance -= (self.bias(value) + value) ** 2

        return variance

    @copy_docstring(Laplace.randomise)
    def randomise(self, value):
        self._check_all(value)

        if self._scale is None:
            self._scale = self._find_scale()

        value = min(value, self.upper)
        value = max(value, self.lower)

        unif_rv = self._rng.random()
        unif_rv *= self._cdf(self.upper - value) - self._cdf(self.lower - value)
        unif_rv += self._cdf(self.lower - value)
        unif_rv -= 0.5

        unif_rv = min(unif_rv, 0.5 - 1e-10)

        return value - self._scale * np.sign(unif_rv) * np.log(1 - 2 * np.abs(unif_rv))


class LaplaceBoundedNoise(Laplace):
    r"""
    The Laplace mechanism with bounded noise, only applicable for approximate differential privacy (delta > 0).

    Epsilon must be strictly positive, `epsilon` > 0. `delta` must be strictly in the interval (0, 0.5).
     - For zero `epsilon`, use :class:`.Uniform`.
     - For zero `delta`, use :class:`.Laplace`.

    Paper link: https://arxiv.org/pdf/1810.00877v1.pdf

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in (0, ∞].

    delta : float
        Privacy parameter :math:`\delta` for the mechanism.  Must be in (0, 0.5).

    sensitivity : float
        The sensitivity of the mechanism.  Must be in [0, ∞).

    """
    def __init__(self, *, epsilon, delta, sensitivity):
        super().__init__(epsilon=epsilon, delta=delta, sensitivity=sensitivity)
        self._scale = None
        self._noise_bound = None

    @classmethod
    def _check_epsilon_delta(cls, epsilon, delta):
        if epsilon == 0:
            raise ValueError("Epsilon must be strictly positive. For zero epsilon, use :class:`.Uniform`.")

        if isinstance(delta, Real) and not 0 < delta < 0.5:
            raise ValueError("Delta must be strictly in the interval (0,0.5). For zero delta, use :class:`.Laplace`.")

        return super(Laplace, cls)._check_epsilon_delta(epsilon, delta)

    def _cdf(self, value):
        if self._scale == 0:
            return 0 if value < 0 else 1

        if value < 0:
            return 0.5 * np.exp(value / self._scale)

        return 1 - 0.5 * np.exp(-value / self._scale)

    @copy_docstring(Laplace.bias)
    def bias(self, value):
        return 0.0

    @copy_docstring(DPMechanism.variance)
    def variance(self, value):
        raise NotImplementedError

    @copy_docstring(Laplace.randomise)
    def randomise(self, value):
        self._check_all(value)

        if self._scale is None or self._noise_bound is None:
            self._scale = self.sensitivity / self.epsilon
            self._noise_bound = -1 if self._scale == 0 else \
                self._scale * np.log(1 + (np.exp(self.epsilon) - 1) / 2 / self.delta)

        unif_rv = self._rng.random()
        unif_rv *= self._cdf(self._noise_bound) - self._cdf(- self._noise_bound)
        unif_rv += self._cdf(- self._noise_bound)
        unif_rv -= 0.5

        return value - self._scale * (np.sign(unif_rv) * np.log(1 - 2 * np.abs(unif_rv)))
