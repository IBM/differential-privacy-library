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
The vector mechanism in differential privacy, for producing perturbed objectives
"""
from numbers import Real

import numpy as np

from diffprivlib.mechanisms.base import DPMechanism
from diffprivlib.utils import copy_docstring


class Vector(DPMechanism):
    """
    The vector mechanism in differential privacy.

    The vector mechanism is used when perturbing convex objective functions.
    Full paper: http://www.jmlr.org/papers/volume12/chaudhuri11a/chaudhuri11a.pdf
    """
    def __init__(self):
        super().__init__()
        self._function_sensitivity = None
        self._data_sensitivity = None
        self._vector_dim = None
        self._alpha = 0.01

    def __repr__(self):
        output = super().__repr__()
        output += ".set_alpha(" + str(self._alpha) + ")" if self._alpha != 0.01 else ""
        output += ".set_sensitivity(" + str(self._function_sensitivity) + ", " + str(self._data_sensitivity) + ")" \
            if self._function_sensitivity is not None or self._data_sensitivity != 1 else ""
        output += ".set_dimension(" + str(self._vector_dim) + ")" if self._vector_dim is not None else ""

        return output

    def set_epsilon_delta(self, epsilon, delta):
        r"""Sets the value of :math:`\epsilon` and :math:`\delta `to be used by the mechanism.

        For the vector mechanism, `delta` must be zero and `epsilon` must be strictly positive.

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

    def set_alpha(self, alpha):
        r"""Set the regularisation parameter :math:`\alpha` for the mechanism.

        `alpha` must be strictly positive.  Default is 0.01.

        Parameters
        ----------
        alpha : float
            Regularisation parameter.

        Returns
        -------
        self : class

        """
        if not isinstance(alpha, Real):
            raise TypeError("Alpha must be numeric")

        if alpha <= 0:
            raise ValueError("Alpha must be strictly positive")

        self._alpha = alpha
        return self

    def set_dimension(self, vector_dim):
        """Sets the dimension `vector_dim` of the domain of the mechanism.

        This dimension relates to the size of the input vector of the function being considered by the mechanism.  This
        corresponds to the size of the random vector produced by the mechanism.

        Parameters
        ----------
        vector_dim : int
            Function input dimension.

        Returns
        -------
        self : class

        """
        if not isinstance(vector_dim, Real) or not np.isclose(vector_dim, int(vector_dim)):
            raise TypeError("d must be integer-valued")
        if not vector_dim >= 1:
            raise ValueError("d must be strictly positive")

        self._vector_dim = int(vector_dim)
        return self

    def set_sensitivity(self, function_sensitivity, data_sensitivity=1):
        """Sets the sensitivity of the function and data being processed by the mechanism.

        - The sensitivity of the function relates to the max of its second derivative.  Must be strictly positive.
        - The sensitivity of the data relates to the max 2-norm of each row.  Must be strictly positive.

        Parameters
        ----------
        function_sensitivity : float
            The function sensitivity of the mechanism.

        data_sensitivity : float, default: 1.0
            The data sensitivity of the mechanism.

        Returns
        -------
        self : class

        """
        if not isinstance(function_sensitivity, Real) or not isinstance(data_sensitivity, Real):
            raise TypeError("Sensitivities must be numeric")

        if function_sensitivity < 0 or data_sensitivity < 0:
            raise ValueError("Sensitivities must be non-negative")

        self._function_sensitivity = function_sensitivity
        self._data_sensitivity = data_sensitivity
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

        if not callable(value):
            raise TypeError("Value to be randomised must be a function")

        if self._data_sensitivity is None or self._function_sensitivity is None:
            raise ValueError("Sensitivities must be set")

        if self._vector_dim is None:
            raise ValueError("Dimension d must be set")

        return True

    @copy_docstring(DPMechanism.get_bias)
    def get_bias(self, value):
        raise NotImplementedError

    @copy_docstring(DPMechanism.get_variance)
    def get_variance(self, value):
        raise NotImplementedError

    def randomise(self, value):
        """Randomise `value` with the mechanism.

        If `value` is a method of two outputs, they are taken as `f` and `fprime` (i.e., its gradient), and both are
        perturbed accordingly.

        Parameters
        ----------
        value : method
            The function to be randomised.

        Returns
        -------
        method
            The randomised method.

        """
        self.check_inputs(value)

        epsilon_p = self._epsilon - 2 * np.log(1 + self._function_sensitivity * self._data_sensitivity /
                                               (0.5 * self._alpha))
        delta = 0

        if epsilon_p <= 0:
            delta = (self._function_sensitivity * self._data_sensitivity / (np.exp(self._epsilon / 4) - 1)
                     - 0.5 * self._alpha)
            epsilon_p = self._epsilon / 2

        scale = (epsilon_p / 2 / self._data_sensitivity) if self._data_sensitivity > 0 else float("inf")

        normed_noisy_vector = np.random.normal(0, 1, self._vector_dim)
        norm = np.linalg.norm(normed_noisy_vector, 2)
        noisy_norm = np.random.gamma(self._vector_dim, 1 / scale, 1)

        normed_noisy_vector = normed_noisy_vector / norm * noisy_norm

        def output_func(*args):
            input_vec = args[0]

            func = value(*args)

            if isinstance(func, tuple):
                func, grad = func
            else:
                grad = None

            func += np.dot(normed_noisy_vector, input_vec)
            func += 0.5 * delta * np.dot(input_vec, input_vec)

            if grad is not None:
                grad += normed_noisy_vector + delta * input_vec

                return func, grad

            return func

        return output_func
