from numpy import sign, log, abs, exp
from numpy.random import random

from . import DPMechanism, TruncationAndFoldingMachine


class Laplace(DPMechanism):
    def __init__(self):
        super().__init__()
        self.sensitivity = None

    def __repr__(self):
        output = super().__repr__()
        output += ".setSensitivity(" + str(self.sensitivity) + ")" if self.sensitivity is not None else ""

        return output

    def set_sensitivity(self, sensitivity):
        """

        :param sensitivity: The sensitivity of the function being considered
        :type sensitivity: `float`
        :return:
        """

        if not isinstance(sensitivity, int) and not isinstance(sensitivity, float):
            raise TypeError("Sensitivity must be numeric")

        if sensitivity <= 0:
            raise ValueError("Sensitivity must be strictly positive")

        self.sensitivity = sensitivity
        return self

    def check_inputs(self, value):
        super().check_inputs(value)

        if not isinstance(value, int) and not isinstance(value, float):
            raise TypeError("Value to be randomised must be a number")

        if self.sensitivity is None:
            raise ValueError("Sensitivity must be set")

        return True

    def get_bias(self, value):
        return 0.0

    def get_variance(self, value):
        self.check_inputs(0)

        return 2 * (self.sensitivity / self.epsilon) ** 2

    def randomise(self, value):
        self.check_inputs(value)

        shape = self.sensitivity / self.epsilon

        u = random() - 0.5

        return value - shape * sign(u) * log(1 - 2 * abs(u))


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

        shape = self.sensitivity / self.epsilon

        return shape / 2 * (exp((self.lower_bound - value) / shape) - exp((value - self.upper_bound) / shape))

    def get_variance(self, value):
        self.check_inputs(value)

        shape = self.sensitivity / self.epsilon

        variance = value ** 2 + shape * (self.lower_bound * exp((self.lower_bound - value) / shape)
                                         - self.upper_bound * exp((value - self.upper_bound) / shape))
        variance += (shape ** 2) * (2 - exp((self.lower_bound - value) / shape)
                                    - exp((value - self.upper_bound) / shape))

        variance -= (self.get_bias(value) + value) ** 2

        return variance

    def check_inputs(self, value):
        super().check_inputs(value)
        TruncationAndFoldingMachine.check_inputs(self, value)

        return True

    def randomise(self, value):
        TruncationAndFoldingMachine.check_inputs(self, value)

        noisy_value = super().randomise(value)
        return self.__truncate(noisy_value)


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

        shape = self.sensitivity / self.epsilon

        bias = shape * (exp((self.lower_bound + self.upper_bound - 2 * value) / shape) - 1)
        bias /= exp((self.lower_bound - value) / shape) + exp((self.upper_bound - value) / shape)

        return bias

    def check_inputs(self, value):
        super().check_inputs(value)
        TruncationAndFoldingMachine.check_inputs(self, value)

        return True

    def randomise(self, value):
        TruncationAndFoldingMachine.check_inputs(self, value)

        noisy_value = super().randomise(value)
        return self.__fold(noisy_value)


class LaplaceBoundedDomain(LaplaceTruncated):
    def __init__(self):
        super().__init__()
        self.shape = None

    def __find_shape(self):
        eps = self.epsilon
        delta = 0.0
        diam = self.upper_bound - self.lower_bound
        dq = self.sensitivity

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

    def __cdf(self, x):
        if x < 0:
            return 0.5 * exp(x / self.shape)
        else:
            return 1 - 0.5 * exp(-x / self.shape)

    def get_effective_epsilon(self):
        if self.shape is None:
            self.shape = self.__find_shape()

        return self.sensitivity / self.shape

    def get_bias(self, value):
        self.check_inputs(value)

        if self.shape is None:
            self.shape = self.__find_shape()

        bias = (self.shape - self.lower_bound + value) / 2 * exp((self.lower_bound - value) / self.shape) \
            - (self.shape + self.upper_bound - value) / 2 * exp((value - self.upper_bound) / self.shape)
        bias /= 1 - exp((self.lower_bound - value) / self.shape) / 2 \
            - exp((value - self.upper_bound) / self.shape) / 2

        return bias

    def get_variance(self, value):
        self.check_inputs(value)

        if self.shape is None:
            self.shape = self.__find_shape()

        variance = value**2
        variance -= (exp((self.lower_bound - value) / self.shape) * (self.lower_bound ** 2)
                     + exp((value - self.upper_bound) / self.shape) * (self.upper_bound ** 2)) / 2
        variance += self.shape * (self.lower_bound * exp((self.lower_bound - value) / self.shape)
                                  - self.upper_bound * exp((value - self.upper_bound) / self.shape))
        variance += (self.shape ** 2) * (2 - exp((self.lower_bound - value) / self.shape)
                                         - exp((value - self.upper_bound) / self.shape))
        variance /= 1 - (exp(-(value - self.lower_bound) / self.shape)
                         + exp(-(self.upper_bound - value) / self.shape)) / 2

        variance -= (self.get_bias(value) + value) ** 2

        return variance

    def randomise(self, value):
        self.check_inputs(value)

        if self.shape is None:
            self.shape = self.__find_shape()

        value = min(value, self.upper_bound)
        value = max(value, self.lower_bound)

        u = random()
        u *= self.__cdf(self.upper_bound - value) - self.__cdf(self.lower_bound - value)
        u += self.__cdf(self.lower_bound - value)
        u -= 0.5

        return value - self.shape * sign(u) * log(1 - 2 * abs(u))


class LaplaceBoundedNoise(Laplace):
    def __init__(self):
        super().__init__()
        self.shape = None
        self.noise_bound = None

    def set_epsilon_delta(self, epsilon, delta):
        if epsilon == 0:
            raise ValueError("Epsilon must be strictly positive. For zero epsilon, use `mechanisms.Uniform`.")

        if not (0 < delta < 0.5):
            raise ValueError("Delta must be strictly in (0,0.5). For zero delta, use `mechanisms.Laplace`.")

        return DPMechanism.set_epsilon_delta(self, epsilon, delta)

    def __cdf(self, x):
        if x < 0:
            return 0.5 * exp(x / self.shape)
        else:
            return 1 - 0.5 * exp(-x / self.shape)

    def get_bias(self, value):
        return 0.0

    def randomise(self, value):
        self.check_inputs(value)

        if self.shape is None or self.noise_bound is None:
            self.shape = self.sensitivity / self.epsilon
            self.noise_bound = self.shape * log(1 + (exp(self.epsilon) - 1) / 2 / self.delta)

        u = random()
        u *= self.__cdf(self.noise_bound) - self.__cdf(- self.noise_bound)
        u += self.__cdf(- self.noise_bound)
        u -= 0.5

        return value - self.shape * sign(u) * log(1 - 2 * abs(u))
