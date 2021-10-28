
"""
The Snapping mechanism in differential privacy, which eliminates a weakness to floating point errors in the classic
Laplace mechanism with standard Laplace sampling.
"""
import math
import secrets
import struct
from numbers import Real

import crlibm
import numpy as np

from diffprivlib.mechanisms import DPMechanism
from diffprivlib.mechanisms.base import TruncationAndFoldingMixin


class Snapping(DPMechanism, TruncationAndFoldingMixin):
    r"""
    The Snapping mechanism for differential privacy.

    First proposed by Ilya Mironov [M12]_.

    It eliminates a vulnerability stemming from the representation of reals as floating-point numbers in implementations
    of the classic Laplace mechanism and its variants which use the inverse CDF of the Laplace distribution to sample
    it. It causes a high degree of reduction in the granularity of the output.

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in [0, ∞].

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
        super().__init__(epsilon=epsilon, delta=0.0)
        self._check_sensitivity(sensitivity)
        self.sensitivity = sensitivity
        TruncationAndFoldingMixin.__init__(self, lower=lower, upper=upper)
        self._bound = self._scale_bound()

    @classmethod
    def _check_sensitivity(cls, sensitivity):
        if not isinstance(sensitivity, Real):
            raise TypeError("Sensitivity must be numeric")

        if sensitivity < 0:
            raise ValueError("Sensitivity must be non-negative")
        return float(sensitivity)

    def _check_all(self, value):
        super()._check_all(value)
        TruncationAndFoldingMixin._check_all(self, value)
        self._check_sensitivity(sensitivity=self.sensitivity)

        if not isinstance(value, Real):
            raise TypeError("Value to be randomised must be a number")

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
        if self.sensitivity > 0:
            return (self.upper - self.lower) / 2.0 / self.sensitivity
        return (self.upper - self.lower) / 2.0

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

    def mse(self, value):
        raise NotImplementedError

    def effective_epsilon(self):
        r"""
        Computes the effective value of :math:`\epsilon` that the Snapping mechanism guarantees compared to an
        equivalent Laplace mechanisms based on the bounds and the machine epsilon.
        Defined in section 5.2 of [Mir12]_.

        Returns
        -------
        float
            The effective value of :math:`\epsilon`
        """
        if self.sensitivity > 0:
            machine_epsilon = np.finfo(float).epsneg
            return (self.epsilon + 12.0 * self._bound * self.epsilon + 2.0 * machine_epsilon) / self.sensitivity
        else:
            return float('inf')

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

    def _get_nearest_power_of_2(self, x):
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
        if self.epsilon == float('inf'):  # infinitely small rounding
            return value
        base = self._get_nearest_power_of_2(1.0 / self.epsilon)
        remainder = value % base
        if remainder > base / 2:
            return value - remainder + base
        if remainder == base / 2:
            return value + remainder
        return value - remainder

    def _sample_uniform(self):
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
        mantissa = 1 << 52 | secrets.randbits(52)
        exponent = -53
        x = 0
        while not x:
            x = secrets.randbits(32)
            exponent += x.bit_length() - 32
        return math.ldexp(mantissa, exponent)

    def _sample_laplace(self):
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
        sign = secrets.randbits(1)
        uniform = self._sample_uniform()
        laplace = (-1)**sign * 1.0 / self.epsilon * crlibm.log_rn(uniform)
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
        if self.sensitivity > 0:
            value_scaled_offset = self._scale_and_offset_value(value)
            value_clamped = self._truncate(value_scaled_offset)
            laplace = self._sample_laplace()
            value_rounded = self._round_to_nearest_power_of_2(value_clamped + laplace)
            return self._reverse_scale_and_offset_value(self._truncate(value_rounded))
        else:
            return self._truncate(value)
