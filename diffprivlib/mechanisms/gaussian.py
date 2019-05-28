"""
The classic Gaussian mechanism in differential privacy, and its derivatives.
"""
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

        unif_rv1 = random()
        unif_rv2 = random()

        return np.sqrt(- 2 * np.log(unif_rv1)) * np.cos(2 * np.pi * unif_rv2) * self._scale + value
