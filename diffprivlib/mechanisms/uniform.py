"""
The uniform mechanism in differential privacy.
"""
from numbers import Real

from numpy.random import random

from diffprivlib.mechanisms.base import DPMechanism
from diffprivlib.mechanisms.laplace import Laplace
from diffprivlib.utils import copy_docstring


class Uniform(DPMechanism):
    """
    The Uniform mechanism in differential privacy.

    This emerges as a special case of the :class:`.LaplaceBoundedNoise` mechanism when epsilon = 0.
    Paper link: https://arxiv.org/pdf/1810.00877.pdf
    """
    def __init__(self):
        super().__init__()
        self._sensitivity = None

    def set_epsilon_delta(self, epsilon, delta):
        r"""Set privacy parameters :math:`\epsilon` and :math:`\delta` for the mechanism.

        For the uniform mechanism, `epsilon` must be strictly zero and `delta` must satisfy 0 < `delta` <= 0.5.

        Parameters
        ----------
        epsilon : float
            For the uniform mechanism, `epsilon` must be strictly zero.
        delta : float
            For the uniform mechanism, `delta` must satisfy 0 < `delta` <= 0.5.

        Returns
        -------
        self : class

        Raises
        ------
        ValueError
            If `epsilon` is non-zero or if `delta` does not satisfy 0 < `delta` <= 0.5.
        TypeError
            If `epsilon` or `delta` cannot be cast as floats.

        """
        if not epsilon == 0:
            raise ValueError("Epsilon must be strictly zero.")

        if not 0 < delta <= 0.5:
            raise ValueError("Delta must satisfy 0 < delta <= 0.5")

        return super().set_epsilon_delta(epsilon, delta)

    @copy_docstring(Laplace.set_sensitivity)
    def set_sensitivity(self, sensitivity):
        if not isinstance(sensitivity, Real):
            raise TypeError("Sensitivity must be numeric")

        if sensitivity <= 0:
            raise ValueError("Sensitivity must be strictly positive")

        self._sensitivity = float(sensitivity)
        return self

    @copy_docstring(Laplace.get_bias)
    def get_bias(self, value):
        return 0.0

    @copy_docstring(Laplace.check_inputs)
    def check_inputs(self, value):
        super().check_inputs(value)

        if not isinstance(value, Real):
            raise TypeError("Value to be randomised must be a number")

        if self._sensitivity is None:
            raise ValueError("Sensitivity must be set")

        return True

    @copy_docstring(Laplace.randomise)
    def randomise(self, value):
        self.check_inputs(value)

        unif_rv = 2 * random() - 1
        unif_rv *= self._sensitivity / self._delta / 2

        return value + unif_rv
