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
Basic mechanisms for achieving differential privacy, the basic building blocks of the library.
"""
from diffprivlib.mechanisms.base import DPMachine, DPMechanism, TruncationAndFoldingMixin

from diffprivlib.mechanisms.binary import Binary
from diffprivlib.mechanisms.bingham import Bingham
from diffprivlib.mechanisms.exponential import Exponential, ExponentialCategorical, ExponentialHierarchical, \
    PermuteAndFlip
from diffprivlib.mechanisms.gaussian import Gaussian, GaussianAnalytic, GaussianDiscrete
from diffprivlib.mechanisms.geometric import Geometric, GeometricFolded, GeometricTruncated
from diffprivlib.mechanisms.laplace import Laplace, LaplaceBoundedDomain, LaplaceBoundedNoise, LaplaceFolded,\
    LaplaceTruncated
from diffprivlib.mechanisms.staircase import Staircase
from diffprivlib.mechanisms.uniform import Uniform
from diffprivlib.mechanisms.vector import Vector
