import warnings
from numbers import Real

from numpy import exp
from numpy.random import geometric, random

from . import Laplace


class Staircase(Laplace):
    """
    The Staircase mechanism in differential privacy.

    The staircase mechanism is an optimisation of the classical Laplace Mechanism (:class:`.Laplace`), described as a
    "geometric mixture of uniform random variables".
    Paper link: http://web.stanford.edu/~kairouzp/tstsp_2014.pdf
    """
    def __init__(self):
        super().__init__()
        self._gamma = None

    def set_gamma(self, gamma):
        """
        Set tuning parameter gamma for the mechanism.

        If not set, gamma defaults to minimise the expectation of the amplitude of noise,
        gamma = 1 / (1 + exp(epsilon/2)) .

        :param gamma: Gamma value of the mechanism.
        :type gamma: `float`
        :return: self
        :rtype: :class:`.Uniform`
        """
        if not isinstance(gamma, Real):
            raise TypeError("Gamma must be numeric")
        if not (0.0 <= gamma <= 1.0):
            raise ValueError("Gamma must be in [0,1]")

        self._gamma = float(gamma)
        return self

    def check_inputs(self, value):
        """
        Check that all parameters of the mechanism have been initialised correctly, and that the mechanism is ready
        to be used.

        :param value: Value to be checked.
        :type value: `float`
        :return: True if the mechanism is correctly initialised.
        :rtype: `bool`
        """
        super().check_inputs(value)

        if self._gamma is None:
            self._gamma = 1 / (1 + exp(self._epsilon / 2))
            raise warnings.warn("Gamma not set, falling back to default: 1 / (1 + exp(epsilon / 2)).", UserWarning)

        return True

    def set_epsilon_delta(self, epsilon, delta):
        """
        Set privacy parameters epsilon and delta for the mechanism.

        For the staircase mechanism, delta must be strictly zero.

        :param epsilon: Epsilon value of the mechanism.
        :type epsilon: `float`
        :param delta: Delta value of the mechanism.
        :type delta: `float`
        :return: self
        :rtype: :class:`.Uniform`
        """
        if not delta == 0:
            raise ValueError("Delta must be zero")

        return super().set_epsilon_delta(epsilon, delta)

    def get_bias(self, value):
        """
        Get the bias of the mechanism at a given `value`.

        :param value: The value at which the bias of the mechanism is sought.
        :type value: `float`
        :return: The bias of the mechanism at `value`.
        :rtype: `float`
        """
        return 0.0

    def randomise(self, value):
        """
        Randomise the given value using the mechanism.

        :param value: Value to be randomised.
        :type value: `float`
        :return: Randomised value.
        :rtype: `float`
        """
        self.check_inputs(value)

        s = -1 if random() < 0.5 else 1
        g = geometric(1 - exp(- self._epsilon)) - 1
        u = random()
        b = 0 if random() < self._gamma / (self._gamma + (1 - self._gamma) * exp(- self._epsilon)) else 1

        return value + s * ((1 - b) * ((g + self._gamma * u) * self._sensitivity) +
                            b * ((g + self._gamma + (1 - self._gamma) * u) * self._sensitivity))
