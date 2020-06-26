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
The classic geometric mechanism for differential privacy, and its derivatives.
"""
from numbers import Integral

import numpy as np
from numpy.random import random

from diffprivlib.mechanisms.base import DPMechanism, TruncationAndFoldingMixin
from diffprivlib.utils import copy_docstring


class Geometric(DPMechanism):
    """
    The classic geometric mechanism for differential privacy, as first proposed by Ghosh, Roughgarden and Sundararajan.
    Extended to allow for non-unity sensitivity.

    Paper link: https://arxiv.org/pdf/0811.2841.pdf

    """
    def __init__(self):
        super().__init__()
        self._sensitivity = 1
        self._scale = None

    def __repr__(self):
        output = super().__repr__()
        output += ".set_sensitivity(" + str(self._sensitivity) + ")" if self._sensitivity is not None else ""

        return output

    def set_sensitivity(self, sensitivity):
        """Sets the sensitivity of the mechanism.

        Parameters
        ----------
        sensitivity : int
            The sensitivity of the mechanism.  Must satisfy `sensitivity` > 0.

        Returns
        -------
        self : class

        """
        if not isinstance(sensitivity, Integral):
            raise TypeError("Sensitivity must be an integer")

        if sensitivity < 0:
            raise ValueError("Sensitivity must be non-negative")

        self._sensitivity = sensitivity
        self._scale = None
        return self

    def check_inputs(self, value):
        """Checks that all parameters of the mechanism have been initialised correctly, and that the mechanism is ready
        to be used.

        Parameters
        ----------
        value : int
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

        if not isinstance(value, Integral):
            raise TypeError("Value to be randomised must be an integer")

        if self._scale is None:
            self._scale = - self._epsilon / self._sensitivity if self._sensitivity > 0 else - float("inf")

    def set_epsilon_delta(self, epsilon, delta):
        r"""Sets the value of :math:`\epsilon` and :math:`\delta` to be used by the mechanism.

        For the geometric mechanism, `delta` must be zero and `epsilon` must be strictly positive.

        Parameters
        ----------
        epsilon : float
            The value of epsilon for achieving :math:`(\epsilon,\delta)`-differential privacy with the mechanism.  Must
            have `epsilon > 0`.

        delta : float
            For the geometric mechanism, `delta` must be zero.

        Returns
        -------
        self : class

        Raises
        ------
        ValueError
            If `epsilon` is negative or zero, or if `delta` is non-zero.

        """
        if not delta == 0:
            raise ValueError("Delta must be zero")

        return super().set_epsilon_delta(epsilon, delta)

    @copy_docstring(DPMechanism.get_bias)
    def get_bias(self, value):
        return 0.0

    @copy_docstring(DPMechanism.get_variance)
    def get_variance(self, value):
        self.check_inputs(value)

        leading_factor = (1 - np.exp(self._scale)) / (1 + np.exp(self._scale))
        geom_series = np.exp(self._scale) / (1 - np.exp(self._scale))

        return 2 * leading_factor * (geom_series + 3 * (geom_series ** 2) + 2 * (geom_series ** 3))

    def randomise(self, value):
        """Randomise `value` with the mechanism.

        Parameters
        ----------
        value : int
            The value to be randomised.

        Returns
        -------
        int
            The randomised value.

        """
        self.check_inputs(value)

        # Need to account for overlap of 0-value between distributions of different sign
        unif_rv = random() - 0.5
        unif_rv *= 1 + np.exp(self._scale)
        sgn = -1 if unif_rv < 0 else 1

        # Use formula for geometric distribution, with ratio of exp(-epsilon/sensitivity)
        return int(np.round(value + sgn * np.floor(np.log(sgn * unif_rv) / self._scale)))


class GeometricTruncated(Geometric, TruncationAndFoldingMixin):
    """
    The truncated geometric mechanism, where values that fall outside a pre-described range are mapped back to the
    closest point within the range.
    """
    def __init__(self):
        super().__init__()
        TruncationAndFoldingMixin.__init__(self)

    def __repr__(self):
        output = super().__repr__()
        output += TruncationAndFoldingMixin.__repr__(self)

        return output

    def set_bounds(self, lower, upper):
        """Sets the lower and upper bounds of the mechanism.

        For the truncated geometric mechanism, `lower` and `upper` must be integer-valued.  Must have
        `lower` <= `upper`.

        Parameters
        ----------
        lower : int
            The lower bound of the mechanism.

        upper : int
            The upper bound of the mechanism.

        Returns
        -------
        self : class

        """
        if not isinstance(lower, Integral) or not isinstance(upper, Integral):
            raise TypeError("Bounds must be integers")

        return super().set_bounds(lower, upper)

    @copy_docstring(DPMechanism.get_bias)
    def get_bias(self, value):
        raise NotImplementedError

    @copy_docstring(DPMechanism.get_bias)
    def get_variance(self, value):
        raise NotImplementedError

    @copy_docstring(Geometric.randomise)
    def randomise(self, value):
        TruncationAndFoldingMixin.check_inputs(self, value)

        noisy_value = super().randomise(value)
        return int(np.round(self._truncate(noisy_value)))


class GeometricFolded(Geometric, TruncationAndFoldingMixin):
    """
    The folded geometric mechanism, where values outside a pre-described range are folded back toward the domain around
    the closest point within the domain.
    Half-integer bounds are permitted.
    """
    def __init__(self):
        super().__init__()
        TruncationAndFoldingMixin.__init__(self)

    def __repr__(self):
        output = super().__repr__()
        output += TruncationAndFoldingMixin.__repr__(self)

        return output

    def set_bounds(self, lower, upper):
        """Sets the lower and upper bounds of the mechanism.

        For the folded geometric mechanism, `lower` and `upper` must be integer or half-integer -valued.  Must have
        `lower` <= `upper`.

        Parameters
        ----------
        lower : int or float
            The lower bound of the mechanism.

        upper : int or float
            The upper bound of the mechanism.

        Returns
        -------
        self : class

        """
        if not np.isclose(2 * lower, np.round(2 * lower)) or not np.isclose(2 * upper, np.round(2 * upper)):
            raise ValueError("Bounds must be integer or half-integer floats")

        return super().set_bounds(lower, upper)

    def _fold(self, value):
        return super()._fold(int(np.round(value)))

    @copy_docstring(DPMechanism.get_bias)
    def get_bias(self, value):
        raise NotImplementedError

    @copy_docstring(DPMechanism.get_bias)
    def get_variance(self, value):
        raise NotImplementedError

    @copy_docstring(Geometric.randomise)
    def randomise(self, value):
        TruncationAndFoldingMixin.check_inputs(self, value)

        noisy_value = super().randomise(value)
        return int(np.round(self._fold(noisy_value)))
