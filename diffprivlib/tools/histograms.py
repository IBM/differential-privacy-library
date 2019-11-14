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
#
#
# Copyright (c) 2005-2019, NumPy Developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice, this list of conditions and the following
#       disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
#       following disclaimer in the documentation and/or other materials provided with the distribution.
#
#     * Neither the name of the NumPy Developers nor the names of any contributors may be used to endorse or promote
#       products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
Differentially private histogram-related functions
Builds upon the histogram functionality of Numpy
"""
import warnings
from sys import maxsize

import numpy as np

from diffprivlib.mechanisms import GeometricTruncated
from diffprivlib.utils import PrivacyLeakWarning


# noinspection PyShadowingBuiltins
def histogram(sample, epsilon=1, bins=10, range=None, normed=None, weights=None, density=None):
    r"""
    Compute the differentially private histogram of a set of data.

    The histogram is computed using :obj:`numpy.histogram`, and noise added using :class:`.GeometricTruncated` to
    satisfy differential privacy.  If the `range` parameter is not specified correctly, a :class:`.PrivacyLeakWarning`
    is thrown.  Users are referred to :obj:`numpy.histogram` for more usage notes.

    Parameters
    ----------
    sample : array_like
        Input data. The histogram is computed over the flattened array.

    epsilon : float
        Privacy parameter :math:`\epsilon` to be applied.

    bins : int or sequence of scalars or str, optional
        If `bins` is an int, it defines the number of equal-width bins in the given range (10, by default). If `bins` is
        a sequence, it defines a monotonically increasing array of bin edges, including the rightmost edge, allowing for
        non-uniform bin widths.

        If `bins` is a string, it defines the method used to calculate the optimal bin width, as defined by
        `histogram_bin_edges`.

    range : (float, float), optional
        The lower and upper range of the bins.  If not provided, range is simply ``(a.min(), a.max())``.  Values outside
        the range are ignored. The first element of the range must be less than or equal to the second. `range` affects
        the automatic bin computation as well. While bin width is computed to be optimal based on the actual data within
        `range`, the bin count will fill the entire range including portions containing no data.

    normed : bool, optional, deprecated
        This is equivalent to the `density` argument, but produces incorrect results for unequal bin widths. It should
        not be used.  In diffprivlib, this option is ignored.

    weights : array_like, optional
        An array of weights, of the same shape as `a`.  Each value in `a` only contributes its associated weight
        towards the bin count (instead of 1). If `density` is True, the weights are normalized, so that the integral
        of the density over the range remains 1.

    density : bool, optional
        If ``False``, the result will contain the number of samples in each bin. If ``True``, the result is the value
        of the probability *density* function at the bin, normalized such that the *integral* over the range is 1.
        Note that the sum of the histogram values will not be equal to 1 unless bins of unity width are chosen; it is
        not a probability *mass* function.

    Returns
    -------
    hist : array
        The values of the histogram. See `density` and `weights` for a
        description of the possible semantics.
    bin_edges : array of dtype float
        Return the bin edges ``(length(hist)+1)``.


    See Also
    --------
    histogramdd, histogram2d

    Notes
    -----
    All but the last (righthand-most) bin is half-open.  In other words, if `bins` is::

      [1, 2, 3, 4]

    then the first bin is ``[1, 2)`` (including 1, but excluding 2) and the second ``[2, 3)``.  The last bin, however,
    is ``[3, 4]``, which *includes* 4.

    """
    if range is None:
        warnings.warn("Range parameter has not been specified. Falling back to taking range from the data.\n"
                      "To ensure differential privacy, and no additional privacy leakage, the range must be "
                      "specified independently of the data (i.e., using domain knowledge).", PrivacyLeakWarning)

    hist, bin_edges = np.histogram(sample, bins=bins, range=range, normed=None, weights=weights, density=None)

    dp_mech = GeometricTruncated().set_epsilon(epsilon).set_sensitivity(1).set_bounds(0, maxsize)

    dp_hist = np.zeros_like(hist)

    for i in np.arange(dp_hist.shape[0]):
        dp_hist[i] = dp_mech.randomise(int(hist[i]))

    # dp_hist = dp_hist.astype(float, casting='safe')

    if normed or density:
        bin_sizes = np.array(np.diff(bin_edges), float)
        return dp_hist / bin_sizes / dp_hist.sum(), bin_edges

    return dp_hist, bin_edges


# noinspection PyShadowingBuiltins
def histogramdd(sample, epsilon=1.0, bins=10, range=None, normed=None, weights=None, density=None):
    r"""
    Compute the differentially private multidimensional histogram of some data.

    The histogram is computed using :obj:`numpy.histogramdd`, and noise added using :class:`.GeometricTruncated` to
    satisfy differential privacy.  If the `range` parameter is not specified correctly, a :class:`.PrivacyLeakWarning`
    is thrown.  Users are referred to :obj:`numpy.histogramdd` for more usage notes.

    Parameters
    ----------
    sample : (N, D) array, or (D, N) array_like
        The data to be histogrammed.

        Note the unusual interpretation of sample when an array_like:

        * When an array, each row is a coordinate in a D-dimensional space - such as
          ``histogramgramdd(np.array([p1, p2, p3]))``.
        * When an array_like, each element is the list of values for single coordinate - such as
          ``histogramgramdd((X, Y, Z))``.

        The first form should be preferred.

    epsilon : float
        Privacy parameter :math:`\epsilon` to be applied.

    bins : sequence or int, optional
        The bin specification:

        * A sequence of arrays describing the monotonically increasing bin edges along each dimension.
        * The number of bins for each dimension (nx, ny, ... =bins)
        * The number of bins for all dimensions (nx=ny=...=bins).

    range : sequence, optional
        A sequence of length D, each an optional (lower, upper) tuple giving the outer bin edges to be used if the edges
        are not given explicitly in `bins`.
        An entry of None in the sequence results in the minimum and maximum values being used for the corresponding
        dimension.
        The default, None, is equivalent to passing a tuple of D None values.

    density : bool, optional
        If False, the default, returns the number of samples in each bin. If True, returns the probability *density*
        function at the bin, ``bin_count / sample_count / bin_volume``.

    normed : bool, optional
        An alias for the density argument that behaves identically. To avoid confusion with the broken normed argument
        to `histogram`, `density` should be preferred.

    weights : (N,) array_like, optional
        An array of values `w_i` weighing each sample `(x_i, y_i, z_i, ...)`. Weights are normalized to 1 if normed is
        True. If normed is False, the values of the returned histogram are equal to the sum of the weights belonging to
        the samples falling into each bin.

    Returns
    -------
    H : ndarray
        The multidimensional histogram of sample x. See normed and weights for the different possible semantics.
    edges : list
        A list of D arrays describing the bin edges for each dimension.

    See Also
    --------
    histogram: 1-D differentially private histogram
    histogram2d: 2-D differentially private histogram

    """
    if range is None or (isinstance(range, list) and None in range):
        warnings.warn("Range parameter has not been specified (or has missing elements). Falling back to taking range "
                      "from the data.\n "
                      "To ensure differential privacy, and no additional privacy leakage, the range must be "
                      "specified for each dimension independently of the data (i.e., using domain knowledge).",
                      PrivacyLeakWarning)

    hist, bin_edges = np.histogramdd(sample, bins=bins, range=range, normed=None, weights=weights, density=None)

    dp_mech = GeometricTruncated().set_epsilon(epsilon).set_sensitivity(1).set_bounds(0, maxsize)

    dp_hist = np.zeros_like(hist)
    iterator = np.nditer(hist, flags=['multi_index'])

    while not iterator.finished:
        dp_hist[iterator.multi_index] = dp_mech.randomise(int(iterator[0]))
        iterator.iternext()

    dp_hist = dp_hist.astype(float, casting='safe')

    if density or normed:
        # calculate the probability density function
        dims = len(dp_hist.shape)
        dp_hist_sum = dp_hist.sum()
        for i in np.arange(dims):
            shape = np.ones(dims, int)
            shape[i] = dp_hist.shape[i]
            # noinspection PyUnresolvedReferences
            dp_hist = dp_hist / np.diff(bin_edges[i]).reshape(shape)
        dp_hist /= dp_hist_sum

    return dp_hist, bin_edges


# noinspection PyShadowingBuiltins
def histogram2d(x, y, epsilon=1.0, bins=10, range=None, normed=None, weights=None, density=None):
    r"""
    Compute the differentially private bi-dimensional histogram of two data samples.

    Parameters
    ----------
    x : array_like, shape (N,)
        An array containing the x coordinates of the points to be histogrammed.

    y : array_like, shape (N,)
        An array containing the y coordinates of the points to be histogrammed.

    epsilon : float
        Privacy parameter :math:`\epsilon` to be applied.

    bins : int or array_like or [int, int] or [array, array], optional
        The bin specification:

          * If int, the number of bins for the two dimensions (nx=ny=bins).
          * If array_like, the bin edges for the two dimensions (x_edges=y_edges=bins).
          * If [int, int], the number of bins in each dimension (nx, ny = bins).
          * If [array, array], the bin edges in each dimension (x_edges, y_edges = bins).
          * A combination [int, array] or [array, int], where int is the number of bins and array is the bin edges.

    range : array_like, shape(2,2), optional
        The leftmost and rightmost edges of the bins along each dimension (if not specified explicitly in the `bins`
        parameters): ``[[xmin, xmax], [ymin, ymax]]``. All values outside of this range will be considered outliers and
        not tallied in the histogram.

    density : bool, optional
        If False, the default, returns the number of samples in each bin.  If True, returns the probability *density*
        function at the bin, ``bin_count / sample_count / bin_area``.

    normed : bool, optional
        An alias for the density argument that behaves identically. To avoid confusion with the broken normed argument
        to `histogram`, `density` should be preferred.

    weights : array_like, shape(N,), optional
        An array of values ``w_i`` weighing each sample ``(x_i, y_i)``.  Weights are normalized to 1 if `normed` is
        True. If `normed` is False, the values of the returned histogram are equal to the sum of the weights belonging
        to the samples falling into each bin.

    Returns
    -------
    H : ndarray, shape(nx, ny)
        The bi-dimensional histogram of samples `x` and `y`. Values in `x` are histogrammed along the first dimension
        and values in `y` are histogrammed along the second dimension.

    xedges : ndarray, shape(nx+1,)
        The bin edges along the first dimension.

    yedges : ndarray, shape(ny+1,)
        The bin edges along the second dimension.

    See Also
    --------
    histogram : 1D differentially private histogram
    histogramdd : Differentially private Multidimensional histogram

    Notes
    -----
    When `normed` is True, then the returned histogram is the sample density, defined such that the sum over bins of the
    product ``bin_value * bin_area`` is 1.

    Please note that the histogram does not follow the Cartesian convention where `x` values are on the abscissa and `y`
    values on the ordinate axis.  Rather, `x` is histogrammed along the first dimension of the array (vertical), and `y`
    along the second dimension of the array (horizontal).  This ensures compatibility with `histogramdd`.

    """
    try:
        num_bins = len(bins)
    except TypeError:
        num_bins = 1

    if num_bins not in (1, 2):
        xedges = yedges = np.asarray(bins)
        bins = [xedges, yedges]

    hist, edges = histogramdd([x, y], epsilon=epsilon, bins=bins, range=range, normed=normed, weights=weights,
                              density=density)
    return hist, edges[0], edges[1]
