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
The Bingham mechanism in differential privacy, for estimating the first eigenvector of a covariance matrix.
"""
from numbers import Real

import numpy as np

from diffprivlib.mechanisms.base import DPMechanism
from diffprivlib.utils import copy_docstring


class Bingham(DPMechanism):
    """
    The Bingham mechanism in differential privacy.

    Used to estimate the first eigenvector (associated with the largest eigenvalue) of a covariance matrix.

    Paper link: http://eprints.whiterose.ac.uk/123206/7/simbingham8.pdf
    """
    def __init__(self):
        super().__init__()
        self._sensitivity = 1

    def __repr__(self):
        output = super().__repr__()
        output += ".set_sensitivity(" + str(self._sensitivity) + ")"

        return output

    def set_epsilon_delta(self, epsilon, delta):
        r"""Sets the value of :math:`\epsilon` and :math:`\delta `to be used by the mechanism.

        For the Bingham mechanism, `delta` must be zero and `epsilon` must be strictly positive.

        Parameters
        ----------
        epsilon : float
            The value of epsilon for achieving :math:`(\epsilon,\delta)`-differential privacy with the mechanism.  Must
            have `epsilon > 0`.

        delta : float
            For this mechanism, `delta` must be zero.

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
        """Sets the l2-norm sensitivity of the data which defines the input covariance matrix.

        Parameters
        ----------
        sensitivity : float
            The maximum l2-norm of the data which defines the input covariance matrix.

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
        if not np.allclose(value, value.T):
            raise ValueError("Array must be symmetric, supplied array is not.")

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
            The randomised eigenvector.

        """
        self.check_inputs(value)

        eigvals, eigvecs = np.linalg.eigh(value)
        d = value.shape[0]

        if d == 1:
            return np.ones((1, 1))
        if self._sensitivity / self._epsilon == 0:
            return eigvecs[:, eigvals.argmax()]

        value_translated = self._epsilon * (eigvals.max() * np.eye(d) - value) / 4 / self._sensitivity

        left, right, mid = 1, d, (1 + d) / 2
        old_interval_size = (right - left) * 2

        while right - left < old_interval_size:
            old_interval_size = right - left

            mid = (right + left) / 2
            f_mid = np.array([1 / (mid + 2 * e) for e in eigvals]).sum()

            if f_mid <= 1:
                right = mid

            if f_mid >= 1:
                left = mid

        b = mid
        omega = np.eye(d) + 2 * value_translated / b
        omega_inv = np.linalg.inv(omega)
        norm_const = np.exp(-(d - b) / 2) * ((d / b) ** (d / 2))

        while True:
            z = np.random.multivariate_normal(np.zeros(d), omega_inv)
            u = z / np.linalg.norm(z)
            prob = np.exp(-u.dot(value_translated).dot(u)) / norm_const / ((u.dot(omega).dot(u)) ** (d / 2))

            if np.random.random() >= prob:
                return u
