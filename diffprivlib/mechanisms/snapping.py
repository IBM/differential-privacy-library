
"""
The Snapping mechanism in differential privacy, which eliminates a weakness to floating point errors in the classic
Laplace mechanism with standard Laplace sampling.
"""
import math
import struct

import crlibm
import numpy as np

from diffprivlib.mechanisms import LaplaceTruncated


class Snapping(LaplaceTruncated):
    r"""
    The Snapping mechanism for differential privacy.

    First proposed by Ilya Mironov [M12]_.

    It eliminates a vulnerability stemming from the representation of reals as floating-point numbers in implementations
    of the classic Laplace mechanism and its variants which use the inverse CDF of the Laplace distribution to sample
    it. It causes a high degree of reduction in the granularity of the output.

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in [:math:`2*\eta`, ∞], where math:`\eta` is the
        machine epsilon of the floating point type.

    sensitivity : float
        The sensitivity of the mechanism.  Must be in [0, ∞).

    lower : float
        The lower bound of the mechanism.

    upper : float
        The upper bound of the mechanism.
    References
    ----------
    .. [Mir12] Mironov, Ilya. "On significance of the least significant bits for differential privacy." Proceedings of
     the 2012 ACM conference on Computer and communications security (2012).
    """
    def __init__(self, *, epsilon, sensitivity, lower, upper):
        super().__init__(epsilon=epsilon, sensitivity=sensitivity, delta=0.0, lower=lower, upper=upper)
        self.epsilon = self._check_epsilon_machine_epsilon(epsilon)
        self._bound = self._scale_bound()
        self._epsilon_0 = self.effective_epsilon()
        self.scale = 1.0 / self._epsilon_0  # everything is scaled to sensitivity 1
        self._lambda = self._get_nearest_power_of_2(self.scale)

    @staticmethod
    def _check_epsilon_machine_epsilon(epsilon):
        machine_epsilon = np.finfo(float).epsneg
        if epsilon < 2 * machine_epsilon:
            raise ValueError("Epsilon must be at least as large as twice the machine epsilon for the floating point "
                             "type, as the effective epsilon must be non-negative")
        return epsilon

    def _check_all(self, value):
        super()._check_all(value)
        self._check_epsilon_machine_epsilon(self.epsilon)
        return True

    def _scale_bound(self):
        """
        Scales the lower and upper bounds to be proportionate to sensitivity 1, and symmetrical about 0.
        For sensitivity 0, only centres the bound, as scaling up and down is not defined.

        Returns
        -------
        float
            A symmetric bound around 0 scaled to sensitivity 1
        """
        if self.sensitivity == 0:
            return (self.upper - self.lower) / 2.0
        return (self.upper - self.lower) / 2.0 / self.sensitivity

    def _truncate(self, value):
        if value > self._bound:
            return self._bound
        if value < -self._bound:
            return -self._bound

        return value

    def bias(self, value):
        raise NotImplementedError

    def variance(self, value):
        raise NotImplementedError

    def effective_epsilon(self):
        r"""
        Returns the effective value used in the Snapping mechanism to give the required :math:`(\epsilon, 0)`-DP, based
        on the bounds and the machine epsilon.
        Based on section 5.2 of [Mir12]_.

        Returns
        -------
        float
            The effective value of :math:`\epsilon`
        """
        try:
            return self._epsilon_0
        except AttributeError:
            machine_epsilon = np.finfo(float).epsneg
            return (self.epsilon - 2*machine_epsilon) / (1 + 12*self._bound*machine_epsilon)

    def _scale_and_offset_value(self, value):
        """
        Centre value around 0 with symmetric bound and scale to sensitivity 1

        Parameters
        ----------
        value : float
            value to be scaled
        Returns
        -------
        float
            value offset to be centered on 0 and scaled to sensitivity 1
        """
        value_scaled = value / self.sensitivity
        return value_scaled - self._bound - (self.lower / self.sensitivity)

    def _reverse_scale_and_offset_value(self, value):
        return (value + self._bound) * self.sensitivity + self.lower

    @staticmethod
    def _get_nearest_power_of_2(x):
        def float_to_bits(d):
            s = struct.pack('>d', d)
            return struct.unpack('>q', s)[0]

        def bits_to_float(b):
            s = struct.pack('>q', b)
            return struct.unpack('>d', s)[0]

        bits = float_to_bits(x)
        if bits % (1 << 52) == 0:
            return x
        return bits_to_float(((bits >> 52) + 1) << 52)

    def _round_to_nearest_power_of_2(self, value):
        """ Performs the rounding step from [Mir12]_ with ties resolved towards +∞

        Parameters
        ----------
        value : float
            Value to be rounded

        Returns
        -------
        float
            Rounded value

        """
        if self._epsilon_0 == float('inf'):  # infinitely small rounding
            return value
        remainder = value % self._lambda
        if remainder > self._lambda / 2:
            return value - remainder + self._lambda
        if remainder == self._lambda / 2:
            return value + remainder
        return value - remainder

    def _uniform_sampler(self):
        """
        Uniformly sample the full domain of floating-point numbers between (0, 1), rather than only multiples of 2^-53.
        A uniform distribution over D ∩ (0, 1) can be generated by independently sampling an exponent
        from the geometric distribution with parameter .5 and a significand by drawing a uniform string from
        {0, 1}^52 [Mir12]_

        Based on code recipe in Python standard library documentation [Py21]_.

        Returns
        -------
        float
            A value sampled from float in (0, 1) with probability proportional to the size of the infinite-precision
            real interval each float represents

        References
        ----------
        .. [Py21]  The Python Standard Library. "random — Generate pseudo-random numbers", 2021
        https://docs.python.org/3/library/random.html#recipes
        """
        mantissa = 1 << 52 | self._rng.getrandbits(52)
        exponent = -53
        x = 0
        while not x:
            x = self._rng.getrandbits(32)
            exponent += x.bit_length() - 32
        return math.ldexp(mantissa, exponent)

    @staticmethod
    def _laplace_sampler(unif_bit, unif):
        r"""
        Laplace inverse CDF random sampling implementation which uses full domain uniform sampling and exact log
        implementation from crlibm, as mentioned in [Mir12]_.
        Outputs a random value scaled according to privacy budget and sensitivity 1, as bounds and input are scaled to
        sensitivity 1 before Laplacian noise is added.

        Returns
        -------
        float
            Random value from Laplace distribution scaled according to :math:`\epsilon`
        """
        laplace = (-1)**unif_bit * crlibm.log_rn(unif)
        return laplace

    def randomise(self, value):
        """Randomise `value` with the mechanism.

        Parameters
        ----------
        value : float
            The value to be randomised.

        Returns
        -------
        float
            The randomised value.

        """
        self._check_all(value)
        if self.sensitivity == 0:
            return self._truncate(value)
        value_scaled_offset = self._scale_and_offset_value(value)
        value_clamped = self._truncate(value_scaled_offset)
        laplace = self.scale * self._laplace_sampler(self._rng.getrandbits(1), self._uniform_sampler())
        value_rounded = self._round_to_nearest_power_of_2(value_clamped + laplace)
        return self._reverse_scale_and_offset_value(self._truncate(value_rounded))