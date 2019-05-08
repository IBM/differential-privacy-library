"""
Differentially private histogram-related functions
Builds upon the histogram functionality of Numpy
"""
import warnings
from sys import maxsize

import numpy as np

from diffprivlib import mechanisms


# noinspection PyShadowingBuiltins
def histogram(sample, epsilon=1, bins=10, range=None, normed=None, weights=None, density=None):
    """
    Differentially private histogram. Identical functionality to Numpy's `histogram`, but with required `range`
    argument. See numpy.histogram for full help.

    :param sample: Input data. The histogram is computed over the flattened array.
    :type sample: array-like
    :param epsilon: Privacy parameter epsilon to be applied.
    :type epsilon: `float`
    :param bins: If `bins` is an int, it defines the number of equal-width bins in the given range (10, by default). If
        `bins` is a sequence, it defines a monotonically increasing array of bin edges,  including the rightmost edge,
        allowing for non-uniform bin widths.
    :type bins: `int` or sequence of scalars or `str`, optional
    :param range: The lower and upper range of the bins. Values outside the
        range are ignored. The first element of the range must be less than or equal to the second.
    :type range: (float, float)
    :param normed: Deprecated. use `density` instead.
    :type normed: `bool`, optional
    :param weights: An array of weights, of the same shape as `a`.  Each value in `a` only contributes its associated
        weight towards the bin count (instead of 1). If `density` is True, the weights are normalized, so that the
        integral of the density over the range remains 1.
    :type weights: array_like, optional
    :param density: If ``False``, the result will contain the number of samples in each bin. If ``True``, the result is
        the value of the probability *density* function at the bin, normalized such that the *integral* over the range
        is 1. Note that the sum of the histogram values will not be equal to 1 unless bins of unity width are chosen; it
        is not a probability *mass* function.
    :type density: `bool`, optional
    :return: (hist, bin_edges)
        hist: The values of the histogram. See `density` and `weights` for a description of the possible semantics.
        bin_edges: Return the bin edges ``(length(hist)+1)``.
    :rtype: (array, array of dtype float)

    Some extra test with an example:

    """
    if range is None:
        warnings.warn("Range parameter has not been specified. Falling back to taking range from the data.\n"
                      "To ensure differential privacy, and no additional privacy leakage, the range must be "
                      "specified independently of the data (i.e., using domain knowledge).", RuntimeWarning)

    hist, bin_edges = np.histogram(sample, bins=bins, range=range, normed=None, weights=weights, density=None)

    dp_mech = mechanisms.GeometricTruncated().set_epsilon(epsilon).set_sensitivity(1).set_bounds(0, maxsize)

    dp_hist = np.zeros_like(hist)

    for i in np.arange(dp_hist.shape[0]):
        dp_hist[i] = dp_mech.randomise(int(hist[i]))

    # dp_hist = dp_hist.astype(float, casting='safe')

    if normed or density:
        bin_sizes = np.array(np.diff(bin_edges), float)
        return dp_hist / bin_sizes / dp_hist.sum(), bin_edges

    return dp_hist, bin_edges


# noinspection PyShadowingBuiltins
# todo Throw warning if range not specified, and use sample.min() and sample.max() as substitute
def histogramdd(sample, epsilon, bins=10, range=None, normed=None, weights=None, density=None):
    """
    Compute the differentially private multidimensional histogram of some data. Behaves the same as Numpy's
    `histogramdd`, but parameter `range` is required.

    :param sample: The data to be histogrammed.

        Note the unusual interpretation of sample when an array_like:

        * When an array, each row is a coordinate in a D-dimensional space - such as
          ``histogramgramdd(np.array([p1, p2, p3]))``.
        * When an array_like, each element is the list of values for single coordinate - such as
          ``histogramgramdd((X, Y, Z))``.

        The first form should be preferred.
    :type sample: (N, D) array, or (D, N) array_like
    :param epsilon: Privacy parameter epsilon for differential privacy.
    :type epsilon: `float`
    :param bins: The bin specification:

        * A sequence of arrays describing the monotonically increasing bin edges along each dimension.
        * The number of bins for each dimension (nx, ny, ... =bins)
        * The number of bins for all dimensions (nx=ny=...=bins).
    :type bins: sequence or `int`, optional
    :param range: A sequence of length D, each a (lower, upper) tuple giving the outer bin edges to be used if the edges
        are not given explicitly in `bins`.
        For `dp_histogramdd`, a range for each dimension is required.
    :type range: sequence
    :param normed: An alias for the density argument that behaves identically. To avoid confusion with the broken normed
        argument to `histogram`, `density` should be preferred.
    :type normed: `bool`, optional
    :param weights: An array of values `w_i` weighing each sample `(x_i, y_i, z_i, ...)`. Weights are normalized to 1 if
        normed is True. If normed is False, the values of the returned histogram are equal to the sum of the weights
        belonging to the samples falling into each bin.
    :type weights: (N,) array_like, optional
    :param density: If False, the default, returns the number of samples in each bin. If True, returns the probability
        *density* function at the bin, ``bin_count / sample_count / bin_volume``.
    :type density: `bool`, optional
    :return: (H, edges)
        H: The multidimensional histogram of sample x. See normed and weights for the different possible semantics.
        edges: A list of D arrays describing the bin edges for each dimension.
    :rtype: (ndarray, list)
    """
    if range is None or (isinstance(range, list) and None in range):
        raise ValueError("Range must be specified for each dimension, as tuple of (lower, upper)")

    hist, bin_edges = np.histogramdd(sample, bins=bins, range=range, normed=None, weights=weights, density=None)

    dp_mech = mechanisms.GeometricTruncated().set_epsilon(epsilon).set_sensitivity(1).set_bounds(0, maxsize)

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
def histogram2d(x, y, epsilon, bins=10, range=None, normed=None, weights=None,
                density=None):
    """
    Compute the bi-dimensional histogram of two data samples.

    Parameters
    ----------
    x : array_like, shape (N,)
        An array containing the x coordinates of the points to be
        histogrammed.
    y : array_like, shape (N,)
        An array containing the y coordinates of the points to be
        histogrammed.
    epsilon : float
        Privacy parameter epsilon for differential privacy.
    bins : int or array_like or [int, int] or [array, array], optional
        The bin specification:

          * If int, the number of bins for the two dimensions (nx=ny=bins).
          * If array_like, the bin edges for the two dimensions
            (x_edges=y_edges=bins).
          * If [int, int], the number of bins in each dimension
            (nx, ny = bins).
          * If [array, array], the bin edges in each dimension
            (x_edges, y_edges = bins).
          * A combination [int, array] or [array, int], where int
            is the number of bins and array is the bin edges.

    range : array_like, shape(2,2), optional
        The leftmost and rightmost edges of the bins along each dimension
        (if not specified explicitly in the `bins` parameters):
        ``[[xmin, xmax], [ymin, ymax]]``. All values outside of this range
        will be considered outliers and not tallied in the histogram.
    density : bool, optional
        If False, the default, returns the number of samples in each bin.
        If True, returns the probability *density* function at the bin,
        ``bin_count / sample_count / bin_area``.
    normed : bool, optional
        An alias for the density argument that behaves identically. To avoid
        confusion with the broken normed argument to `histogram`, `density`
        should be preferred.
    weights : array_like, shape(N,), optional
        An array of values ``w_i`` weighing each sample ``(x_i, y_i)``.
        Weights are normalized to 1 if `normed` is True. If `normed` is
        False, the values of the returned histogram are equal to the sum of
        the weights belonging to the samples falling into each bin.

    Returns
    -------
    H : ndarray, shape(nx, ny)
        The bi-dimensional histogram of samples `x` and `y`. Values in `x`
        are histogrammed along the first dimension and values in `y` are
        histogrammed along the second dimension.
    xedges : ndarray, shape(nx+1,)
        The bin edges along the first dimension.
    yedges : ndarray, shape(ny+1,)
        The bin edges along the second dimension.

    See Also
    --------
    histogram : 1D histogram
    histogramdd : Multidimensional histogram

    Notes
    -----
    When `normed` is True, then the returned histogram is the sample
    density, defined such that the sum over bins of the product
    ``bin_value * bin_area`` is 1.

    Please note that the histogram does not follow the Cartesian convention
    where `x` values are on the abscissa and `y` values on the ordinate
    axis.  Rather, `x` is histogrammed along the first dimension of the
    array (vertical), and `y` along the second dimension of the array
    (horizontal).  This ensures compatibility with `histogramdd`.

    Examples
    --------
    >>> from matplotlib.image import NonUniformImage
    >>> import matplotlib.pyplot as plt

    Construct a 2-D histogram with variable bin width. First define the bin
    edges:

    >>> xedges = [0, 1, 3, 5]
    >>> yedges = [0, 2, 3, 4, 6]

    Next we create a histogram H with random bin content:

    >>> x = np.random.normal(2, 1, 100)
    >>> y = np.random.normal(1, 1, 100)
    >>> H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
    >>> H = H.T  # Let each row list bins with common y range.

    :func:`imshow <matplotlib.pyplot.imshow>` can only display square bins:

    >>> fig = plt.figure(figsize=(7, 3))
    >>> ax = fig.add_subplot(131, title='imshow: square bins')
    >>> plt.imshow(H, interpolation='nearest', origin='low',
    ...         extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

    :func:`pcolormesh <matplotlib.pyplot.pcolormesh>` can display actual edges:

    >>> ax = fig.add_subplot(132, title='pcolormesh: actual edges',
    ...         aspect='equal')
    >>> X, Y = np.meshgrid(xedges, yedges)
    >>> ax.pcolormesh(X, Y, H)

    :class:`NonUniformImage <matplotlib.image.NonUniformImage>` can be used to
    display actual bin edges with interpolation:

    >>> ax = fig.add_subplot(133, title='NonUniformImage: interpolated',
    ...         aspect='equal', xlim=xedges[[0, -1]], ylim=yedges[[0, -1]])
    >>> im = NonUniformImage(ax, interpolation='bilinear')
    >>> xcenters = (xedges[:-1] + xedges[1:]) / 2
    >>> ycenters = (yedges[:-1] + yedges[1:]) / 2
    >>> im.set_data(xcenters, ycenters, H)
    >>> ax.images.append(im)
    >>> plt.show()

    """
    try:
        n = len(bins)
    except TypeError:
        n = 1

    if n != 1 and n != 2:
        xedges = yedges = np.asarray(bins)
        bins = [xedges, yedges]
    hist, edges = histogramdd([x, y], epsilon, bins, range, normed, weights, density)
    return hist, edges[0], edges[1]
