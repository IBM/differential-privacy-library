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
Gaussian Naive Bayes classifier satisfying differential privacy
"""
import warnings

import numpy as np
import sklearn.naive_bayes as sk_nb

from diffprivlib.mechanisms import Laplace, LaplaceBoundedDomain
from diffprivlib.models.utils import _check_bounds
from diffprivlib.utils import PrivacyLeakWarning, warn_unused_args


class GaussianNB(sk_nb.GaussianNB):
    r"""Gaussian Naive Bayes (GaussianNB) with differential privacy

    Inherits the :class:`sklearn.naive_bayes.GaussianNB` class from Scikit Learn and adds noise to satisfy differential
    privacy to the learned means and variances.  Adapted from the work presented in [VSB13]_.

    Parameters
    ----------
    epsilon : float, default: 1.0
        Privacy parameter :math:`\epsilon` for the model.

    bounds : list or None, default: None
        Bounds of the data, provided as a list of tuples, with one tuple per dimension.  If not provided, the bounds
        are computed on the data when ``.fit()`` is first called, resulting in a :class:`.PrivacyLeakWarning`.

    priors : array-like, shape (n_classes,)
        Prior probabilities of the classes. If specified the priors are not adjusted according to the data.

    var_smoothing : float, optional (default=1e-9)
        Portion of the largest variance of all features that is added to variances for calculation stability.

    Attributes
    ----------
    class_prior_ : array, shape (n_classes,)
        probability of each class.

    class_count_ : array, shape (n_classes,)
        number of training samples observed in each class.

    theta_ : array, shape (n_classes, n_features)
        mean of each feature per class

    sigma_ : array, shape (n_classes, n_features)
        variance of each feature per class

    epsilon_ : float
        absolute additive value to variances (unrelated to ``epsilon`` parameter for differential privacy)

    References
    ----------
    .. [VSB13] Vaidya, Jaideep, Basit Shafiq, Anirban Basu, and Yuan Hong. "Differentially private naive bayes
        classification." In 2013 IEEE/WIC/ACM International Joint Conferences on Web Intelligence (WI) and Intelligent
        Agent Technologies (IAT), vol. 1, pp. 571-576. IEEE, 2013.

    """

    def __init__(self, epsilon=1, bounds=None, priors=None, var_smoothing=1e-9):
        super().__init__(priors, var_smoothing)

        self.epsilon = epsilon
        self.bounds = bounds

    def _partial_fit(self, X, y, classes=None, _refit=False, sample_weight=None):
        if sample_weight is not None:
            warn_unused_args("sample_weight")

        # Store size of current X to apply differential privacy later on
        self.new_n_samples = X.shape[0]

        if self.bounds is None:
            warnings.warn("Bounds have not been specified and will be calculated on the data provided. This will "
                          "result in additional privacy leakage. To ensure differential privacy and no additional "
                          "privacy leakage, specify bounds for each dimension.", PrivacyLeakWarning)
            self.bounds = list(zip(np.min(X, axis=0), np.max(X, axis=0)))

        self.bounds = _check_bounds(self.bounds, X.shape[1])

        super()._partial_fit(X, y, classes, _refit, sample_weight=None)

        del self.new_n_samples
        return self

    def _update_mean_variance(self, n_past, mu, var, X, sample_weight=None):
        """Compute online update of Gaussian mean and variance.

        Given starting sample count, mean, and variance, a new set of points X return the updated mean and variance.
        (NB - each dimension (column) in X is treated as independent -- you get variance, not covariance).

        Can take scalar mean and variance, or vector mean and variance to simultaneously update a number of
        independent Gaussians.

        See Stanford CS tech report STAN-CS-79-773 by Chan, Golub, and LeVeque:

        http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf

        Parameters
        ----------
        n_past : int
            Number of samples represented in old mean and variance. If sample weights were given, this should contain
            the sum of sample weights represented in old mean and variance.

        mu : array-like, shape (number of Gaussians,)
            Means for Gaussians in original set.

        var : array-like, shape (number of Gaussians,)
            Variances for Gaussians in original set.

        sample_weight : ignored
            Ignored in diffprivlib.

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
            warn_unused_args("sample_weight")

        n_new = X.shape[0]
        new_var = np.var(X, axis=0)
        new_mu = np.mean(X, axis=0)

        # Apply differential privacy to the new means and variances
        new_mu, new_var = self._randomise(new_mu, new_var, self.new_n_samples)

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
        total_ssd = old_ssd + new_ssd + (n_past / float(n_new * n_total)) * (n_new * mu - n_new * new_mu) ** 2
        total_var = total_ssd / n_total

        return total_mu, total_var

    def _randomise(self, mean, var, n_samples):
        """Randomises the learned means and variances subject to differential privacy."""
        features = var.shape[0]

        local_epsilon = self.epsilon / 2
        local_epsilon /= features

        if len(self.bounds) != features:
            raise ValueError("Bounds must be specified for each feature dimension")

        new_mu = np.zeros_like(mean)
        new_var = np.zeros_like(var)

        for feature in range(features):
            local_diameter = self.bounds[feature][1] - self.bounds[feature][0]
            mech_mu = Laplace().set_sensitivity(local_diameter / n_samples).set_epsilon(local_epsilon)
            mech_var = LaplaceBoundedDomain().set_sensitivity((n_samples - 1) * local_diameter ** 2 / n_samples ** 2)\
                .set_epsilon(local_epsilon).set_bounds(0, float("inf"))

            new_mu[feature] = mech_mu.randomise(mean[feature])
            new_var[feature] = mech_var.randomise(var[feature])

        return new_mu, new_var
