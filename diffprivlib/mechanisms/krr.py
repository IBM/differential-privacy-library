# MIT License
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
The k-RR mechanism in differential privacy, and its derivatives.
"""
from numbers import Real
from math import e, log
import numpy as np

from diffprivlib.mechanisms.base import DPMechanism, TruncationAndFoldingMixin
from diffprivlib.utils import copy_docstring


class KRR(DPMechanism):
    r"""
    The k-RR mechanism - also known as flat mechanism or randomized response
    mechanism - is one of the simplest LPD mechanisms. It can be visualized as
    a generalization of the randomised response proposed by [Warner, 1965].
  
    The k-RR mechahism returns the original value of an attribute domain with
    probability :math:`\frac{e^\varepsilon}{|X|-1+e^\varepsilon}` and returns
    any other value different from the original with probability
    :math:`\frac{1}{|X|-1+e^\varepsilon}`, where :math:`|X|` is the domain
    size. P. Kairouz, S. Oh, and P. Viswanath has showed that the mechanism is
    optimal in the low privacy regime for a large class of information
    theoretic utility functions.

    Paper link: https://proceedings.neurips.cc/paper/2014/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in [0, ∞]
    
    domain_values : list
        Domain of values of the mechanism.
    
    """

    def __init__(self, *, epsilon, domain_values):
        super().__init__(epsilon=epsilon, delta=0.0)
        self._domain_values, self._data_size = self._check_domain(domain_values)

    def _check_domain(self, domain_values):
        if not isinstance(domain_values, list):
            raise TypeError("Domain must be a list")
        
        if len(domain_values) <= 1:
            raise ValueError("The domain must have at least 2 elements")

        return domain_values, len(domain_values)

    @copy_docstring(DPMechanism.bias)
    def bias(self, value):
        raise NotImplementedError

    @copy_docstring(DPMechanism.variance)
    def variance(self, value):
        raise NotImplementedError

    def randomise(self, value):
        """Randomise `value` with the mechanism.

        Parameters
        ----------
        value : int or other
            The value to be randomised.

        Returns
        -------
        int or other
            The randomised value, same type as `value`.

        """

        self._check_all(value)
        unif_prob = self._rng.random()

        if (e**self.epsilon == float("inf") or 
            unif_prob <= e**self.epsilon / 
            (self._data_size - 1 + e**self.epsilon)):
            return value
        
        # Select a random index from domain that is different from the index of the value that is being randomised
        idx = self._rng.randrange(self._data_size)
        while self._domain_values[idx] == value:
            idx = self._rng.randrange(self._data_size)

        return self._domain_values[idx]
