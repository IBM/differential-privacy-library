from numpy import sign, log, abs, exp
from numpy.random import random

from . import DPMechanism, TruncationAndFoldingMachine


class Laplace(DPMechanism):
    def __init__(self):
        super().__init__()
        self._sensitivity = None

    def __repr__(self):
        output = super().__repr__()
        output += ".setSensitivity(" + str(self._sensitivity) + ")" if self._sensitivity is not None else ""

        return output

    def set_sensitivity(self, sensitivity):
        """

        :param sensitivity: The sensitivity of the function being considered
        :type sensitivity: `float`
        :return:
        """

        if not isinstance(sensitivity, (int, float)):
            raise TypeError("Sensitivity must be numeric")

        if sensitivity <= 0:
            raise ValueError("Sensitivity must be strictly positive")

        self._sensitivity = sensitivity
        return self

    def check_inputs(self, value):
        super().check_inputs(value)

        if not isinstance(value, (int, float)):
            raise TypeError("Value to be randomised must be a number")

        if self._sensitivity is None:
            raise ValueError("Sensitivity must be set")

        return True

    def get_bias(self, value):
        return 0.0

    def get_variance(self, value):
        self.check_inputs(0)

        return 2 * (self._sensitivity / self._epsilon) ** 2

    def randomise(self, value):
        self.check_inputs(value)

        scale = self._sensitivity / self._epsilon

        u = random() - 0.5

        return value - scale * sign(u) * log(1 - 2 * abs(u))


class LaplaceTruncated(Laplace, TruncationAndFoldingMachine):
    def __init__(self):
        super().__init__()
        TruncationAndFoldingMachine.__init__(self)

    def __repr__(self):
        output = super().__repr__()
        output += TruncationAndFoldingMachine.__repr__(self)

        return output

    def get_bias(self, value):
        self.check_inputs(value)

        shape = self._sensitivity / self._epsilon

        return shape / 2 * (exp((self._lower_bound - value) / shape) - exp((value - self._upper_bound) / shape))

    def get_variance(self, value):
        self.check_inputs(value)

        shape = self._sensitivity / self._epsilon

        variance = value ** 2 + shape * (self._lower_bound * exp((self._lower_bound - value) / shape)
                                         - self._upper_bound * exp((value - self._upper_bound) / shape))
        variance += (shape ** 2) * (2 - exp((self._lower_bound - value) / shape)
                                    - exp((value - self._upper_bound) / shape))

        variance -= (self.get_bias(value) + value) ** 2

        return variance

    def check_inputs(self, value):
        super().check_inputs(value)
        TruncationAndFoldingMachine.check_inputs(self, value)

        return True

    def randomise(self, value):
        TruncationAndFoldingMachine.check_inputs(self, value)

        noisy_value = super().randomise(value)
        return self._truncate(noisy_value)


class LaplaceFolded(Laplace, TruncationAndFoldingMachine):
    def __init__(self):
        super().__init__()
        TruncationAndFoldingMachine.__init__(self)

    def __repr__(self):
        output = super().__repr__()
        output += TruncationAndFoldingMachine.__repr__(self)

        return output

    def get_bias(self, value):
        self.check_inputs(value)

        shape = self._sensitivity / self._epsilon

        bias = shape * (exp((self._lower_bound + self._upper_bound - 2 * value) / shape) - 1)
        bias /= exp((self._lower_bound - value) / shape) + exp((self._upper_bound - value) / shape)

        return bias

    def check_inputs(self, value):
        super().check_inputs(value)
        TruncationAndFoldingMachine.check_inputs(self, value)

        return True

    def randomise(self, value):
        TruncationAndFoldingMachine.check_inputs(self, value)

        noisy_value = super().randomise(value)
        return self._fold(noisy_value)


class LaplaceBoundedDomain(LaplaceTruncated):
    def __init__(self):
        super().__init__()
        self._scale = None

    def _find_scale(self):
        eps = self._epsilon
        delta = 0.0
        diam = self._upper_bound - self._lower_bound
        dq = self._sensitivity

        def delta_c(shape):
            return (2 - exp(- dq / shape) - exp(- (diam - dq) / shape)) / (1 - exp(- diam / shape))

        def f(shape):
            return dq / (eps - log(delta_c(shape)) - log(1 - delta))

        left = dq / (eps - log(1 - delta))
        right = f(left)
        old_interval_size = (right - left) * 2

        while old_interval_size > right - left:
            old_interval_size = right - left
            middle = (right + left) / 2

            if f(middle) >= middle:
                left = middle
            if f(middle) <= middle:
                right = middle

        return (right + left) / 2

    def _cdf(self, x):
        if x < 0:
            return 0.5 * exp(x / self._scale)
        else:
            return 1 - 0.5 * exp(-x / self._scale)

    def get_effective_epsilon(self):
        if self._scale is None:
            self._scale = self._find_scale()

        return self._sensitivity / self._scale

    def get_bias(self, value):
        self.check_inputs(value)

        if self._scale is None:
            self._scale = self._find_scale()

        bias = (self._scale - self._lower_bound + value) / 2 * exp((self._lower_bound - value) / self._scale) \
            - (self._scale + self._upper_bound - value) / 2 * exp((value - self._upper_bound) / self._scale)
        bias /= 1 - exp((self._lower_bound - value) / self._scale) / 2 \
            - exp((value - self._upper_bound) / self._scale) / 2

        return bias

    def get_variance(self, value):
        self.check_inputs(value)

        if self._scale is None:
            self._scale = self._find_scale()

        variance = value**2
        variance -= (exp((self._lower_bound - value) / self._scale) * (self._lower_bound ** 2)
                     + exp((value - self._upper_bound) / self._scale) * (self._upper_bound ** 2)) / 2
        variance += self._scale * (self._lower_bound * exp((self._lower_bound - value) / self._scale)
                                   - self._upper_bound * exp((value - self._upper_bound) / self._scale))
        variance += (self._scale ** 2) * (2 - exp((self._lower_bound - value) / self._scale)
                                          - exp((value - self._upper_bound) / self._scale))
        variance /= 1 - (exp(-(value - self._lower_bound) / self._scale)
                         + exp(-(self._upper_bound - value) / self._scale)) / 2

        variance -= (self.get_bias(value) + value) ** 2

        return variance

    def randomise(self, value):
        self.check_inputs(value)

        if self._scale is None:
            self._scale = self._find_scale()

        value = min(value, self._upper_bound)
        value = max(value, self._lower_bound)

        u = random()
        u *= self._cdf(self._upper_bound - value) - self._cdf(self._lower_bound - value)
        u += self._cdf(self._lower_bound - value)
        u -= 0.5

        return value - self._scale * sign(u) * log(1 - 2 * abs(u))


class LaplaceBoundedNoise(Laplace):
    def __init__(self):
        super().__init__()
        self._shape = None
        self._noise_bound = None

    def set_epsilon_delta(self, epsilon, delta):
        if epsilon == 0:
            raise ValueError("Epsilon must be strictly positive. For zero epsilon, use `mechanisms.Uniform`.")

        if not (0 < delta < 0.5):
            raise ValueError("Delta must be strictly in (0,0.5). For zero delta, use `mechanisms.Laplace`.")

        return DPMechanism.set_epsilon_delta(self, epsilon, delta)

    def _cdf(self, x):
        if x < 0:
            return 0.5 * exp(x / self._shape)
        else:
            return 1 - 0.5 * exp(-x / self._shape)

    def get_bias(self, value):
        return 0.0

    def randomise(self, value):
        self.check_inputs(value)

        if self._shape is None or self._noise_bound is None:
            self._shape = self._sensitivity / self._epsilon
            self._noise_bound = self._shape * log(1 + (exp(self._epsilon) - 1) / 2 / self._delta)

        u = random()
        u *= self._cdf(self._noise_bound) - self._cdf(- self._noise_bound)
        u += self._cdf(- self._noise_bound)
        u -= 0.5

        return value - self._shape * sign(u) * log(1 - 2 * abs(u))
