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
from numpy.random import random

from diffprivlib.mechanisms.base import DPMechanism, TruncationAndFoldingMixin
from diffprivlib.utils import copy_docstring


class Laplace(DPMechanism):
    r"""
    The classic Laplace mechanism in differential privacy, as first proposed by Dwork, McSherry, Nissim and Smith.

    Paper link: https://link.springer.com/content/pdf/10.1007/11681878_14.pdf

    Includes extension to (relaxed) :math:`(\epsilon,\delta)`-differential privacy, as proposed by Holohan et al.

    Paper link: https://arxiv.org/pdf/1402.6124.pdf

    """
    def __init__(self):
        super().__init__()
        self._sensitivity = None

    def __repr__(self):
        output = super().__repr__()
        output += ".set_sensitivity(" + str(self._sensitivity) + ")" if self._sensitivity is not None else ""

        return output

    def set_sensitivity(self, sensitivity):
        """Sets the sensitivity of the mechanism.

        Parameters
        ----------
        sensitivity : float
            The sensitivity of the mechanism.  Must satisfy `sensitivity` > 0.

        Returns
        -------
        self : class

        """
        if not isinstance(sensitivity, Real):
            raise TypeError("Sensitivity must be numeric")

        if sensitivity < 0:
            raise ValueError("Sensitivity must be non-negative")

        self._sensitivity = float(sensitivity)
        return self

    def check_inputs(self, value):
        """Checks that all parameters of the mechanism have been initialised correctly, and that the mechanism is ready
        to be used.

        Parameters
        ----------
        value : float
            The value to be checked

        Returns
        -------
        True if the mechanism is ready to be used.

        Raises
        ------
        Exception
            If parameters have not been set correctly, or if `value` falls outside the domain of the mechanism.

        """
        super().check_inputs(value)

        if not isinstance(value, Real):
            raise TypeError("Value to be randomised must be a number")

        if self._sensitivity is None:
            raise ValueError("Sensitivity must be set")

        return True

    def get_bias(self, value):
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

    def get_variance(self, value):
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
        self.check_inputs(0)

        return 2 * (self._sensitivity / (self._epsilon - np.log(1 - self._delta))) ** 2

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
        self.check_inputs(value)

        scale = self._sensitivity / (self._epsilon - np.log(1 - self._delta))

        unif_rv = random() - 0.5

        return value - scale * np.sign(unif_rv) * np.log(1 - 2 * np.abs(unif_rv))


class LaplaceTruncated(Laplace, TruncationAndFoldingMixin):
    """
    The truncated Laplace mechanism, where values outside a pre-described domain are mapped to the closest point
    within the domain.
    """
    def __init__(self):
        super().__init__()
        TruncationAndFoldingMixin.__init__(self)

    def __repr__(self):
        output = super().__repr__()
        output += TruncationAndFoldingMixin.__repr__(self)

        return output

    @copy_docstring(Laplace.get_bias)
    def get_bias(self, value):
        self.check_inputs(value)

        shape = self._sensitivity / self._epsilon

        return shape / 2 * (np.exp((self._lower_bound - value) / shape) - np.exp((value - self._upper_bound) / shape))

    @copy_docstring(Laplace.get_variance)
    def get_variance(self, value):
        self.check_inputs(value)

        shape = self._sensitivity / self._epsilon

        variance = value ** 2 + shape * (self._lower_bound * np.exp((self._lower_bound - value) / shape)
                                         - self._upper_bound * np.exp((value - self._upper_bound) / shape))
        variance += (shape ** 2) * (2 - np.exp((self._lower_bound - value) / shape)
                                    - np.exp((value - self._upper_bound) / shape))

        variance -= (self.get_bias(value) + value) ** 2

        return variance

    @copy_docstring(Laplace.check_inputs)
    def check_inputs(self, value):
        super().check_inputs(value)
        TruncationAndFoldingMixin.check_inputs(self, value)

        return True

    @copy_docstring(Laplace.randomise)
    def randomise(self, value):
        TruncationAndFoldingMixin.check_inputs(self, value)

        noisy_value = super().randomise(value)
        return self._truncate(noisy_value)


class LaplaceFolded(Laplace, TruncationAndFoldingMixin):
    """
    The folded Laplace mechanism, where values outside a pre-described domain are folded around the domain until they
    fall within.
    """
    def __init__(self):
        super().__init__()
        TruncationAndFoldingMixin.__init__(self)

    def __repr__(self):
        output = super().__repr__()
        output += TruncationAndFoldingMixin.__repr__(self)

        return output

    @copy_docstring(Laplace.get_bias)
    def get_bias(self, value):
        self.check_inputs(value)

        shape = self._sensitivity / self._epsilon

        bias = shape * (np.exp((self._lower_bound + self._upper_bound - 2 * value) / shape) - 1)
        bias /= np.exp((self._lower_bound - value) / shape) + np.exp((self._upper_bound - value) / shape)

        return bias

    @copy_docstring(DPMechanism.get_variance)
    def get_variance(self, value):
        raise NotImplementedError

    @copy_docstring(Laplace.check_inputs)
    def check_inputs(self, value):
        super().check_inputs(value)
        TruncationAndFoldingMixin.check_inputs(self, value)

        return True

    @copy_docstring(Laplace.randomise)
    def randomise(self, value):
        TruncationAndFoldingMixin.check_inputs(self, value)

        noisy_value = super().randomise(value)
        return self._fold(noisy_value)


class LaplaceBoundedDomain(LaplaceTruncated):
    """
    The bounded Laplace mechanism on a bounded domain.  The mechanism draws values directly from the domain, without any
    post-processing.
    """
    def __init__(self):
        super().__init__()
        self._scale = None

    def _find_scale(self):
        if self._epsilon is None or self._delta is None:
            raise ValueError("Epsilon and Delta must be set before calling _find_scale().")

        eps = self._epsilon
        delta = self._delta
        diam = self._upper_bound - self._lower_bound
        delta_q = self._sensitivity

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

    def get_effective_epsilon(self):
        r"""Gets the effective epsilon of the mechanism, only for strict :math:`\epsilon`-differential privacy.  Returns
        ``None`` if :math:`\delta` is non-zero.

        Returns
        -------
        float
            The effective :math:`\epsilon` parameter of the mechanism.  Returns ``None`` if `delta` is non-zero.

        """
        if self._scale is None:
            self._scale = self._find_scale()

        if self._delta > 0.0:
            return None

        return self._sensitivity / self._scale

    @copy_docstring(Laplace.get_bias)
    def get_bias(self, value):
        self.check_inputs(value)

        if self._scale is None:
            self._scale = self._find_scale()

        bias = (self._scale - self._lower_bound + value) / 2 * np.exp((self._lower_bound - value) / self._scale) \
            - (self._scale + self._upper_bound - value) / 2 * np.exp((value - self._upper_bound) / self._scale)
        bias /= 1 - np.exp((self._lower_bound - value) / self._scale) / 2 \
            - np.exp((value - self._upper_bound) / self._scale) / 2

        return bias

    @copy_docstring(Laplace.get_variance)
    def get_variance(self, value):
        self.check_inputs(value)

        if self._scale is None:
            self._scale = self._find_scale()

        variance = value**2
        variance -= (np.exp((self._lower_bound - value) / self._scale) * (self._lower_bound ** 2)
                     + np.exp((value - self._upper_bound) / self._scale) * (self._upper_bound ** 2)) / 2
        variance += self._scale * (self._lower_bound * np.exp((self._lower_bound - value) / self._scale)
                                   - self._upper_bound * np.exp((value - self._upper_bound) / self._scale))
        variance += (self._scale ** 2) * (2 - np.exp((self._lower_bound - value) / self._scale)
                                          - np.exp((value - self._upper_bound) / self._scale))
        variance /= 1 - (np.exp(-(value - self._lower_bound) / self._scale)
                         + np.exp(-(self._upper_bound - value) / self._scale)) / 2

        variance -= (self.get_bias(value) + value) ** 2

        return variance

    @copy_docstring(Laplace.randomise)
    def randomise(self, value):
        self.check_inputs(value)

        if self._scale is None:
            self._scale = self._find_scale()

        value = min(value, self._upper_bound)
        value = max(value, self._lower_bound)

        unif_rv = random()
        unif_rv *= self._cdf(self._upper_bound - value) - self._cdf(self._lower_bound - value)
        unif_rv += self._cdf(self._lower_bound - value)
        unif_rv -= 0.5

        unif_rv = min(unif_rv, 0.5 - 1e-10)

        return value - self._scale * np.sign(unif_rv) * np.log(1 - 2 * np.abs(unif_rv))


class LaplaceBoundedNoise(Laplace):
    """
    The Laplace mechanism with bounded noise, only applicable for approximate differential privacy (delta > 0).
    """
    def __init__(self):
        super().__init__()
        self._scale = None
        self._noise_bound = None

    def set_epsilon_delta(self, epsilon, delta):
        r"""Set the privacy parameters :math:`\epsilon` and :math:`\delta` for the mechanism.

        Epsilon must be strictly positive, `epsilon` > 0. `delta` must be strictly in the interval (0, 0.5).
         - For zero `epsilon`, use :class:`.Uniform`.
         - For zero `delta`, use :class:`.Laplace`.

        Parameters
        ----------
        epsilon : float
            The value of epsilon for achieving :math:`(\epsilon,\delta)`-differential privacy with the mechanism.  Must
            have `epsilon > 0`.

        delta : float
            The value of delta for achieving :math:`(\epsilon,\delta)`-differential privacy with the mechanism.  Must
            have `0 < delta < 0.5`.

        Returns
        -------
        self : class

        """
        if epsilon == 0:
            raise ValueError("Epsilon must be strictly positive. For zero epsilon, use :class:`.Uniform`.")

        if isinstance(delta, Real) and not 0 < delta < 0.5:
            raise ValueError("Delta must be strictly in (0,0.5). For zero delta, use :class:`.Laplace`.")

        return DPMechanism.set_epsilon_delta(self, epsilon, delta)

    def _cdf(self, value):
        if self._scale == 0:
            return 0 if value < 0 else 1

        if value < 0:
            return 0.5 * np.exp(value / self._scale)

        return 1 - 0.5 * np.exp(-value / self._scale)

    @copy_docstring(Laplace.get_bias)
    def get_bias(self, value):
        return 0.0

    @copy_docstring(DPMechanism.get_variance)
    def get_variance(self, value):
        raise NotImplementedError

    @copy_docstring(Laplace.randomise)
    def randomise(self, value):
        self.check_inputs(value)

        if self._scale is None or self._noise_bound is None:
            self._scale = self._sensitivity / self._epsilon
            self._noise_bound = -1 if self._scale == 0 else \
                self._scale * np.log(1 + (np.exp(self._epsilon) - 1) / 2 / self._delta)

        unif_rv = random()
        unif_rv *= self._cdf(self._noise_bound) - self._cdf(- self._noise_bound)
        unif_rv += self._cdf(- self._noise_bound)
        unif_rv -= 0.5

        return value - self._scale * (np.sign(unif_rv) * np.log(1 - 2 * np.abs(unif_rv)))
