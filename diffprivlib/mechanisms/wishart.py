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
The Wishart mechanism in differential privacy, for producing positive semi-definite perturbed second-moment matrices
"""
from numbers import Real

import numpy as np

from diffprivlib.mechanisms.base import DPMechanism
from diffprivlib.utils import copy_docstring


class Wishart(DPMechanism):
    """
    The Wishart mechanism in differential privacy.

    Used to achieve differential privacy on 2nd moment matrices.

    Paper link: https://ieeexplore.ieee.org/abstract/document/7472095/
    """
    def __init__(self):
        super().__init__()
        self._sensitivity = None

    def __repr__(self):
        output = super().__repr__()
        output += ".set_sensitivity(" + str(self._sensitivity) + ")" if self._sensitivity is not None else ""

        return output

    def set_epsilon_delta(self, epsilon, delta):
        r"""Sets the value of :math:`\epsilon` and :math:`\delta `to be used by the mechanism.

        For the Wishart mechanism, `delta` must be zero and `epsilon` must be strictly positive.

        Parameters
        ----------
        epsilon : float
            The value of epsilon for achieving :math:`(\epsilon,\delta)`-differential privacy with the mechanism.  Must
            have `epsilon > 0`.

        delta : float
            For the vector mechanism, `delta` must be zero.

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

        return super().set_epsilon_delta(epsilon, delta)

    def set_sensitivity(self, sensitivity):
        """Sets the l2-norm sensitivity of the data being processed by the mechanism.

        Parameters
        ----------
        sensitivity : float
            The maximum l2-norm of the data.

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
        value : method
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

        if not isinstance(value, np.ndarray):
            raise TypeError("Value to be randomised must be a numpy array, got %s" % type(value))
        if value.ndim != 2:
            raise ValueError("Array must be 2-dimensional, got %d dimensions" % value.ndim)
        if value.shape[0] != value.shape[1]:
            raise ValueError("Array must be square, got %d x %d" % (value.shape[0], value.shape[1]))

        if self._sensitivity is None:
            raise ValueError("Sensitivity must be set")

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
        value : numpy array
            The data to be randomised.

        Returns
        -------
        numpy array
            The randomised array.

        """
        self.check_inputs(value)

        scale = 1 / 2 / self._epsilon
        n_features = value.shape[0]

        noise_array = np.random.randn(n_features, n_features + 1) * scale * self._sensitivity
        noise_array = np.dot(noise_array, noise_array.T)

        return value + noise_array
