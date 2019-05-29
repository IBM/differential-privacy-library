import warnings

import numpy as np
from sklearn import cluster as skcluster

from diffprivlib.mechanisms import LaplaceBoundedDomain, GeometricFolded
from diffprivlib.models.utils import _check_bounds
from diffprivlib.utils import DiffprivlibCompatibilityWarning, PrivacyLeakWarning, warn_unused_args


class KMeans(skcluster.KMeans):
    def __init__(self, epsilon=1.0, bounds=None, n_clusters=8, **unused_args):
        self.epsilon = epsilon
        self.bounds = bounds
        self.n_clusters = n_clusters

        warn_unused_args(unused_args)

        self.cluster_centers_ = None
        self.bounds_processed = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None

    def fit(self, X, y=None, sample_weight=None):
        """

        Parameters
        ----------
        X : array-like
        y
        sample_weight

        Returns
        -------

        """
        if sample_weight is not None:
            warnings.warn("For diffprivlib, sample_weight is not used. Remove or set to None to suppress this warning.",
                          DiffprivlibCompatibilityWarning)
            del sample_weight

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
            # Add slack to guard against features with
            self.bounds = list(zip(np.min(X, axis=0), np.max(X, axis=0)))

        self.bounds = _check_bounds(self.bounds, dims)

        centers = self._init_centers(dims)
        labels = None
        distances = None

        for _ in range(iters):
            distances, labels = self._distances_labels(X, centers)

            centers = self._update_centers(X, centers, labels, dims)

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

    def _update_centers(self, X, centers, labels, dims):
        epsilon_0, epsilon_i = self._split_epsilon(dims)
        geometric_mech = GeometricFolded().set_sensitivity(1).set_bounds(0.5, float("inf")).set_epsilon(epsilon_0)
        laplace_mech = LaplaceBoundedDomain().set_epsilon(epsilon_i)

        for cluster in range(self.n_clusters):
            if cluster not in labels:
                continue

            cluster_count = sum(labels == cluster)
            noisy_count = geometric_mech.randomise(cluster_count)

            cluster_sum = np.sum(X[labels == cluster], axis=0)
            noisy_sum = np.zeros_like(cluster_sum)

            for i in range(dims):
                laplace_mech.set_sensitivity(self.bounds[i][1] - self.bounds[i][0])\
                    .set_bounds(noisy_count * self.bounds[i][0], noisy_count * self.bounds[i][1])
                noisy_sum[i] = laplace_mech.randomise(cluster_sum[i])

            centers[cluster, :] = noisy_sum / noisy_count

        return centers

    def _split_epsilon(self, dims, rho=0.225):
        """Split epsilon between sum perturbation and count perturbation, as proposed by Su et al.

        Parameters
        ----------
        dims : int
            Number of dimensions to split epsilon across.
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

        normaliser = self.epsilon / (epsilon_i * dims + epsilon_0)

        return epsilon_i * normaliser, epsilon_0 * normaliser
