"""
Gaussian Naive Bayes classifier satisfying differential privacy
"""
import numpy as np
from numbers import Real

import sklearn.naive_bayes as sk_nb

from diffprivlib.mechanisms import Laplace, LaplaceBoundedDomain


class GaussianNB(sk_nb.GaussianNB):
    def __init__(self, epsilon, bounds, priors=None, var_smoothing=1e-9):
        super().__init__(priors, var_smoothing)

        if not isinstance(epsilon, Real) or epsilon <= 0.0:
            raise ValueError("Epsilon must be specified as a positive float.")

        if bounds is None or not isinstance(bounds, list):
            raise ValueError("Bounds must be specified as a list of tuples.")

        for bound in bounds:
            if not isinstance(bound, tuple):
                raise TypeError("Bounds must be specified as a list of tuples")
            if not isinstance(bound[0], Real) or not isinstance(bound[1], Real) or bound[0] >= bound[1]:
                raise ValueError("For each feature bound, lower bound must be strictly lower than upper bound"
                                 "(error found in bound %s" % str(bound))

        self.dp_epsilon = epsilon
        self.bounds = bounds

    def _partial_fit(self, X, y, classes=None, _refit=False, sample_weight=None):
        # Store size of current X to apply differential privacy later on
        self.new_X_n = X.shape[0]

        super()._partial_fit(X, y, classes, _refit, sample_weight)

        del self.new_X_n
        return self

    def _update_mean_variance(self, n_past, mu, var, X, sample_weight=None):
        """Compute online update of Gaussian mean and variance.

        Given starting sample count, mean, and variance, a new set of
        points X, and optionally sample weights, return the updated mean and
        variance. (NB - each dimension (column) in X is treated as independent
        -- you get variance, not covariance).

        Can take scalar mean and variance, or vector mean and variance to
        simultaneously update a number of independent Gaussians.

        See Stanford CS tech report STAN-CS-79-773 by Chan, Golub, and LeVeque:

        http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf

        Parameters
        ----------
        n_past : int
            Number of samples represented in old mean and variance. If sample
            weights were given, this should contain the sum of sample
            weights represented in old mean and variance.

        mu : array-like, shape (number of Gaussians,)
            Means for Gaussians in original set.

        var : array-like, shape (number of Gaussians,)
            Variances for Gaussians in original set.

        sample_weight : array-like, shape (n_samples,), optional (default=None)
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        total_mu : array-like, shape (number of Gaussians,)
            Updated mean for each Gaussian over the combined set.

        total_var : array-like, shape (number of Gaussians,)
            Updated variance for each Gaussian over the combined set.
        """
        if X.shape[0] == 0:
            return mu, var

        # Compute (potentially weighted) mean and variance of new datapoints
        if sample_weight is not None:
            n_new = float(sample_weight.sum())
            new_mu = np.average(X, axis=0, weights=sample_weight / n_new)
            new_var = np.average((X - new_mu) ** 2, axis=0,
                                 weights=sample_weight / n_new)
        else:
            n_new = X.shape[0]
            new_var = np.var(X, axis=0)
            new_mu = np.mean(X, axis=0)

        # Apply differential privacy to the new means and variances
        new_mu, new_var = self._randomise(new_mu, new_var, self.new_X_n)

        if n_past == 0:
            return new_mu, new_var

        n_total = float(n_past + n_new)

        # Combine mean of old and new data, taking into consideration
        # (weighted) number of observations
        total_mu = (n_new * new_mu + n_past * mu) / n_total

        # Combine variance of old and new data, taking into consideration
        # (weighted) number of observations. This is achieved by combining
        # the sum-of-squared-differences (ssd)
        old_ssd = n_past * var
        new_ssd = n_new * new_var
        total_ssd = (old_ssd + new_ssd +
                     (n_past / float(n_new * n_total)) *
                     (n_new * mu - n_new * new_mu) ** 2)
        total_var = total_ssd / n_total

        return total_mu, total_var

    def _randomise(self, mu, var, n):
        features = var.shape[0]

        local_epsilon = self.dp_epsilon / 2
        local_epsilon /= features

        if len(self.bounds) != features:
            raise ValueError("Bounds must be specified for each feature dimension")

        new_mu = np.zeros_like(mu)
        new_var = np.zeros_like(var)

        for feature in range(features):
            local_diameter = self.bounds[feature][1] - self.bounds[feature][0]
            mech_mu = Laplace().set_sensitivity(local_diameter / (n + 1)).set_epsilon(local_epsilon)
            mech_var = LaplaceBoundedDomain().set_sensitivity(np.sqrt(n) * local_diameter / (n + 1))\
                .set_epsilon(local_epsilon).set_bounds(0, float("inf"))

            new_mu[feature] = mech_mu.randomise(mu[feature])
            new_var[feature] = mech_var.randomise(var[feature])

        return new_mu, new_var
