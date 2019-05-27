"""
The staircase mechanism in differential privacy.
"""
import warnings
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
    Paper link: http://web.stanford.edu/~kairouzp/tstsp_2014.pdf
    """
    def __init__(self):
        super().__init__()
        self._gamma = None

    def set_gamma(self, gamma):
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
        if not isinstance(gamma, Real):
            raise TypeError("Gamma must be numeric")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("Gamma must be in [0,1]")

        self._gamma = float(gamma)
        return self

    @copy_docstring(Laplace.check_inputs)
    def check_inputs(self, value):
        super().check_inputs(value)

        if self._gamma is None:
            self._gamma = 1 / (1 + np.exp(self._epsilon / 2))
            raise warnings.warn("Gamma not set, falling back to default: 1 / (1 + exp(epsilon / 2)).", UserWarning)

        return True

    def set_epsilon_delta(self, epsilon, delta):
        r"""Sets the value of :math:`\epsilon` and :math:`\delta` to be used by the mechanism.

        For the staircase mechanism, `delta` must be zero and `epsilon` must be strictly positive.

        Parameters
        ----------
        epsilon : float
            The value of epsilon for achieving :math:`(\epsilon,\delta)`-differential privacy with the mechanism. Must
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

        return super().set_epsilon_delta(epsilon, delta)

    @copy_docstring(Laplace.get_bias)
    def get_bias(self, value):
        return 0.0

    @copy_docstring(Laplace.randomise)
    def randomise(self, value):
        self.check_inputs(value)

        sign = -1 if random() < 0.5 else 1
        geometric_rv = geometric(1 - np.exp(- self._epsilon)) - 1
        unif_rv = random()
        binary_rv = 0 if random() < self._gamma / (self._gamma + (1 - self._gamma) * np.exp(- self._epsilon)) else 1

        return value + sign * ((1 - binary_rv) * ((geometric_rv + self._gamma * unif_rv) * self._sensitivity) +
                               binary_rv * ((geometric_rv + self._gamma + (1 - self._gamma) * unif_rv) *
                                            self._sensitivity))
