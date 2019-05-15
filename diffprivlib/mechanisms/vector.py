"""
The vector mechanism in differential privacy, for producing perturbed objectives
"""
from numbers import Real

import numpy as np

from diffprivlib.mechanisms import DPMechanism


class Vector(DPMechanism):
    """
    The vector mechanism in differential privacy.

    The vector mechanism is used when perturbing convex objective functions.
    Full paper: http://www.jmlr.org/papers/volume12/chaudhuri11a/chaudhuri11a.pdf
    """
    def __init__(self):
        super().__init__()
        self._function_sensitivity = None
        self._data_sensitivity = 1
        self._d = None
        self._n = None
        self._lambda = 0.01

    def set_epsilon_delta(self, epsilon, delta):
        """
        Set the privacy parameters epsilon and delta for the mechanism.

        For the vector mechanism, delta must be zero. Epsilon must be strictly positive, epsilon >= 0.

        :param epsilon: Epsilon value of the mechanism.
        :type epsilon: `float`
        :param delta: Delta value of the mechanism. For the geometric mechanism, this must be zero.
        :type delta: `float`
        :return: self
        """
        if not delta == 0:
            raise ValueError("Delta must be zero")

        return super().set_epsilon_delta(epsilon, delta)

    def set_sensitivity(self, function_sensitivity, data_sensitivity=1):
        """
        Sets the sensitivity of the function and data for use in the mechanism.

        :param function_sensitivity:
        :param data_sensitivity:
        :return:
        """
        if not isinstance(function_sensitivity, Real) or not isinstance(data_sensitivity, Real):
            raise TypeError("Sensitivities must be numeric")

        if function_sensitivity <= 0 or data_sensitivity <= 0:
            raise ValueError("Sensitivities must be strictly positive")

        self._function_sensitivity = function_sensitivity
        self._data_sensitivity = data_sensitivity
        return self

    def set_lambda(self, lam):
        """
        Set the regularisation parameter lambda for the mechanism.

        :param lam: Regularisation parameter, default is 0.01
        :type lam: `float`
        :return: self
        """
        if not isinstance(lam, Real):
            raise TypeError("Lambda must be numeric")

        if lam <= 0:
            raise ValueError("Lambda must be strictly positive")

        self._lambda = lam
        return self

    def check_inputs(self, value):
        """
        Checks that all parameters of the mechanism have been initialised correctly, and that the mechanism is ready
        to be used.

        :param value: Value to be checked.
        :type value: object
        :return: True if the mechanism is ready to be used.
        :rtype: `bool`
        """
        super().check_inputs(value)

        if not callable(value):
            raise TypeError("Value to be randomised must be a function")

        if self._data_sensitivity is None or self._function_sensitivity is None:
            raise ValueError("Sensitivities must be set")

        if self._n is None or self._d is None:
            raise ValueError("Dimensions n and d must be set")

        return True

    def set_dimensions(self, d, n):
        """
        Set the dimensions of the function output `d` and the number of datapoints `n`.

        :param d: Function output dimension.
        :type d: `int`
        :param n: Number of datapoints in the dataset.
        :type n: `int`
        :return: self
        """

        if not isinstance(d, Real) or not d >= 1 or not np.isclose(d, int(d)):
            raise ValueError("d must be a strictly positive integer")

        if not isinstance(n, Real) or not n >= 1 or not np.isclose(n, int(n)):
            raise ValueError("n must be a strictly positive integer")

        self._n = n
        self._d = d
        return self

    def randomise(self, value):
        """
        Randomise the given function using the mechanism.

        :param value: Function to be randomised.
        :type value: method
        :return: Randomised function.
        :rtype: method
        """
        self.check_inputs(value)

        c = self._function_sensitivity
        g = self._data_sensitivity
        epsilon_p = self._epsilon - 2 * np.log(1 + c * g / (self._lambda * self._n))
        delta = 0

        if epsilon_p <= 0:
            delta = c * g / (self._n * (np.exp(self._epsilon / 4) - 1)) - self._lambda
            epsilon_p = self._epsilon / 2

        scale = epsilon_p / 2 / g

        normed_noisy_vector = np.random.normal(0, 1, self._d)
        norm = np.linalg.norm(normed_noisy_vector, 2)
        noisy_norm = np.random.gamma(self._d, 1 / scale, 1)

        normed_noisy_vector = normed_noisy_vector / norm * noisy_norm

        def output_func(x):
            val = value(x)
            val += np.dot(normed_noisy_vector, x) / self._n
            val += delta / 2 * (np.linalg.norm(x) ** 2)
            return val

        return output_func
