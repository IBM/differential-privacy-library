"""
K-means clustering algorithm satisfying differential privacy.
"""
import warnings

import numpy as np
from sklearn import cluster as skcluster

from diffprivlib.mechanisms import LaplaceBoundedDomain, GeometricFolded
from diffprivlib.models.utils import _check_bounds
from diffprivlib.utils import PrivacyLeakWarning, warn_unused_args


class KMeans(skcluster.KMeans):
    r"""K-Means clustering with differential privacy

    Parameters
    ----------
    epsilon : float, optional, default: 1.0
        Privacy parameter :math:`\epsilon`.

    bounds : list or None, optional, default: None
        Bounds of the data, provided as a list of tuples, with one tuple per dimension.  If not provided, the bounds
        are computed on the data when ``.fit()`` is first called, resulting in a :class:`.PrivacyLeakWarning`.

    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of centroids to generate.

    unused_args :
        Placeholder for arguments used by :obj:`sklearn.cluster.KMeans`, but not used by `diffprivlib`. Specifying any of
        these parameters will result in a :class:`.DiffprivlibCompatibilityWarning`.

    Attributes
    ----------
    cluster_centers_ : array, [n_clusters, n_features]

        Coordinates of cluster centers. If the algorithm stops before fully converging, these will not be consistent
        with ``labels_``.

    labels_ :
        Labels of each point

    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.

    n_iter_ : int
        Number of iterations run.

    Examples
    --------

    >>> from diffprivlib.models import KMeans
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [0, 3],
    ...               [10, 2], [10, 4], [9, 3]])
    >>> kmeans = KMeans(bounds=[(0,10), (2, 4)], n_clusters=2).fit(X)
    >>> kmeans.labels_
    array([1, 1, 1, 0, 0, 0], dtype=int32)
    >>> kmeans.predict([[0, 0], [12, 3]])
    array([1, 0], dtype=int32)
    >>> kmeans.cluster_centers_
    array([[10.,  2.],
           [ 1.,  2.]])

    """

    def __init__(self, epsilon=1.0, bounds=None, n_clusters=8, **unused_args):
        super().__init__(n_clusters=n_clusters)

        self.epsilon = epsilon
        self.bounds = bounds

        warn_unused_args(unused_args)

        self.cluster_centers_ = None
        self.bounds_processed = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None

    def fit(self, X, y=None, **unused_args):
        """Computes k-means clustering with differential privacy.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Training instances to cluster.

        y : Ignored
            not used, present here for API consistency by convention.

        unused_args :
            Placeholder for arguments present in :obj:`sklearn.cluster.KMeans`, but not used in diffprivlib.  Specifying
            any of these parameters will result in a :class:`.DiffprivlibCompatibilityWarning`.

        Returns
        -------
        self : class

        """
        warn_unused_args(unused_args)
        del y

        # Todo: Determine iters on-the-fly as a function of epsilon
        iters = 7

        if X.ndim != 2:
            raise ValueError(
                "Expected 2D array, got array with %d dimensions instead. Reshape your data using array.reshape(-1, 1),"
                "or array.reshape(1, -1) if your data contains only one sample." % X.ndim)

        dims = X.shape[1]

        if self.bounds is None:
            warnings.warn("Bounds have not been specified and will be calculated on the data provided.  This will "
                          "result in additional privacy leakage. To ensure differential privacy and no additional "
                          "privacy leakage, specify `bounds` for each dimension.", PrivacyLeakWarning)
            self.bounds = list(zip(np.min(X, axis=0), np.max(X, axis=0)))

        self.bounds = _check_bounds(self.bounds, dims)

        centers = self._init_centers(dims)
        labels = None
        distances = None

        # Run _update_centers first to ensure consistency of `labels` and `centers`, since convergence unlikely
        for _ in range(-1, iters):
            if labels is not None:
                centers = self._update_centers(X, centers=centers, labels=labels, dims=dims, total_iters=iters)

            distances, labels = self._distances_labels(X, centers)

        self.cluster_centers_ = centers
        self.labels_ = labels
        self.inertia_ = distances[np.arange(len(labels)), labels].sum()
        self.n_iter_ = iters

        return self

    def _init_centers(self, dims):
        # Todo: Fix to ensure initialised centers are at least a distance d from the domain boundaries, and 2d from
        #  other custer centres
        if self.bounds_processed is None:
            bounds_processed = np.zeros(shape=(dims, 2))

            for dim in range(dims):
                lower = self.bounds[dim][0]
                upper = self.bounds[dim][1]

                bounds_processed[dim, :] = [upper - lower, lower]

            self.bounds_processed = bounds_processed

        cluster_proximity = np.min(self.bounds_processed[:, 0]) / 2.0

        while cluster_proximity > 0:
            centers = np.zeros(shape=(self.n_clusters, dims))
            cluster, retry = 0, 0

            while retry < 100:
                if cluster >= self.n_clusters:
                    break

                temp_center = np.random.random(dims) * self.bounds_processed[:, 0] + self.bounds_processed[:, 1]

                if cluster == 0:
                    centers[0, :] = temp_center
                    cluster += 1
                    continue

                min_distance = ((centers[:cluster, :] - temp_center) ** 2).sum(axis=1).min()

                if np.sqrt(min_distance) >= cluster_proximity:
                    centers[cluster, :] = temp_center
                    cluster += 1
                    retry = 0
                else:
                    retry += 1

            if cluster >= self.n_clusters:
                return centers

            cluster_proximity /= 2.0

        return None

    def _distances_labels(self, X, centers):
        distances = np.zeros((X.shape[0], self.n_clusters))

        for cluster in range(self.n_clusters):
            distances[:, cluster] = ((X - centers[cluster, :])**2).sum(axis=1)

        labels = np.argmin(distances, axis=1)
        return distances, labels

    def _update_centers(self, X, centers, labels, dims, total_iters):
        """Updates the centers of the KMeans algorithm for the current iteration, while satisfying differential
        privacy.

        Differential privacy is satisfied by adding (integer-valued, using :class:`.GeometricFolded`) random noise to
        the count of nearest neighbours to the previous cluster centers, and adding (real-valued, using
        :class:`.LaplaceBoundedDomain`) random noise to the sum of values per dimension.

        """
        epsilon_0, epsilon_i = self._split_epsilon(dims, total_iters)
        geometric_mech = GeometricFolded().set_sensitivity(1).set_bounds(0.5, float("inf")).set_epsilon(epsilon_0)
        laplace_mech = LaplaceBoundedDomain().set_epsilon(epsilon_i)

        for cluster in range(self.n_clusters):
            if cluster not in labels:
                continue

            cluster_count = sum(labels == cluster)
            noisy_count = geometric_mech.randomise(cluster_count)

            cluster_sum = np.sum(X[labels == cluster], axis=0)
            # Extra np.array() a temporary fix for PyLint bug: https://github.com/PyCQA/pylint/issues/2747
            noisy_sum = np.array(np.zeros_like(cluster_sum))

            for i in range(dims):
                laplace_mech.set_sensitivity(self.bounds[i][1] - self.bounds[i][0])\
                    .set_bounds(noisy_count * self.bounds[i][0], noisy_count * self.bounds[i][1])
                noisy_sum[i] = laplace_mech.randomise(cluster_sum[i])

            centers[cluster, :] = noisy_sum / noisy_count

        return centers

    def _split_epsilon(self, dims, total_iters, rho=0.225):
        """Split epsilon between sum perturbation and count perturbation, as proposed by Su et al.

        Parameters
        ----------
        dims : int
            Number of dimensions to split `epsilon` across.

        total_iters : int
            Total number of iterations to split `epsilon` across.

        rho : float, default 0.225
            Coordinate normalisation factor.

        Returns
        -------
        epsilon_0 : float
            The epsilon value for satisfying differential privacy on the count of a cluster.

        epsilon_i : float
            The epsilon value for satisfying differential privacy on each dimension of the center of a cluster.

        """
        epsilon_i = 1
        epsilon_0 = np.cbrt(4 * dims * rho ** 2)

        normaliser = self.epsilon / total_iters / (epsilon_i * dims + epsilon_0)

        return epsilon_i * normaliser, epsilon_0 * normaliser
