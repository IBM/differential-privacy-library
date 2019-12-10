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
# New BSD License
#
# Copyright (c) 2007–2019 The scikit-learn developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
#   a. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
#      disclaimer.
#   b. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
#      following disclaimer in the documentation and/or other materials provided with the distribution.
#   c. Neither the name of the Scikit-learn Developers  nor the names of its contributors may be used to endorse or
#      promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
"""
Principal Component Analysis with differential privacy
"""
import warnings

import numpy as np
try:
    import sklearn.decomposition._pca as sk_pca
except ImportError:
    import sklearn.decomposition.pca as sk_pca
from sklearn.utils.extmath import stable_cumsum, svd_flip

from diffprivlib import tools
from diffprivlib.mechanisms import Wishart
from diffprivlib.utils import warn_unused_args, copy_docstring, PrivacyLeakWarning


# noinspection PyPep8Naming
class PCA(sk_pca.PCA):
    """Principal component analysis (PCA) with differential privacy.

    This class is a child of :obj:`sklearn.decomposition.PCA`, with amendments to allow for the implementation of
    differential privacy as given in [IS16b]_.  Some parameters of `Scikit Learn`'s model have therefore had to be
    fixed, including:

        - The only permitted `svd_solver` is 'full'.  Specifying the ``svd_solver`` option will result in a warning;
        - The parameters ``tol`` and ``iterated_power`` are not applicable (as a consequence of fixing ``svd_solver =
          'full'``).

    Parameters
    ----------
    n_components : int, float, None or str
        Number of components to keep.
        If n_components is not set all components are kept::

            n_components == min(n_samples, n_features)

        If ``n_components == 'mle'``, Minka's MLE is used to guess the dimension.

        If ``0 < n_components < 1``, select the number of components such that the amount of variance that needs to be
        explained is greater than the percentage specified by n_components.

        Hence, the None case results in::

            n_components == min(n_samples, n_features) - 1

    copy : bool, default=True
        If False, data passed to fit are overwritten and running fit(X).transform(X) will not yield the expected
        results, use fit_transform(X) instead.

    whiten : bool, optional (default False)
        When True (False by default) the `components_` vectors are multiplied by the square root of n_samples and
        then divided by the singular values to ensure uncorrelated outputs with unit component-wise variances.

        Whitening will remove some information from the transformed signal (the relative variance scales of the
        components) but can sometime improve the predictive accuracy of the downstream estimators by making their
        data respect some hard-wired assumptions.

    random_state : int, RandomState instance or None, optional (default None)
        If int, random_state is the seed used by the random number generator; If RandomState instance, random_state
        is the random number generator.

    Attributes
    ----------
    components_ : array, shape (n_components, n_features)
        Principal axes in feature space, representing the directions of maximum variance in the data. The components
        are sorted by ``explained_variance_``.

    explained_variance_ : array, shape (n_components,)
        The amount of variance explained by each of the selected components.

        Equal to n_components largest eigenvalues of the covariance matrix of X.

    explained_variance_ratio_ : array, shape (n_components,)
        Percentage of variance explained by each of the selected components.

        If ``n_components`` is not set then all components are stored and the sum of the ratios is equal to 1.0.

    singular_values_ : array, shape (n_components,)
        The singular values corresponding to each of the selected components. The singular values are equal to the
        2-norms of the ``n_components`` variables in the lower-dimensional space.

    mean_ : array, shape (n_features,)
        Per-feature empirical mean, estimated from the training set.

        Equal to `X.mean(axis=0)`.

    n_components_ : int
        The estimated number of components. When n_components is set to 'mle' or a number between 0 and 1 (with
        svd_solver == 'full') this number is estimated from input data. Otherwise it equals the parameter
        n_components, or the lesser value of n_features and n_samples if n_components is None.

    n_features_ : int
        Number of features in the training data.

    n_samples_ : int
        Number of samples in the training data.

    noise_variance_ : float
        The estimated noise covariance following the Probabilistic PCA model from Tipping and Bishop 1999. See
        "Pattern Recognition and Machine Learning" by C. Bishop, 12.2.1 p. 574 or
        http://www.miketipping.com/papers/met-mppca.pdf. It is required to compute the estimated data covariance and
        score samples.

        Equal to the average of (min(n_features, n_samples) - n_components) smallest eigenvalues of the covariance
        matrix of X.

    See Also
    --------
    :obj:`sklearn.decomposition.PCA` : Scikit-learn implementation Principal Component Analysis.

    References
    ----------
    .. [IS16b] Imtiaz, Hafiz, and Anand D. Sarwate. "Symmetric matrix perturbation for differentially-private principal
        component analysis." In 2016 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP),
        pp. 2339-2343. IEEE, 2016.
    """
    def __init__(self, n_components=None, centered=False, epsilon=1, data_norm=None, range=None, copy=True,
                 whiten=False, random_state=None, **unused_args):
        super().__init__(n_components=n_components, copy=copy, whiten=whiten, svd_solver='full', tol=0.0,
                         iterated_power='auto', random_state=random_state)
        self.centered = centered
        self.epsilon = epsilon
        self.data_norm = data_norm
        self.range = range

        warn_unused_args(unused_args)

    def _fit_full(self, X, n_components):
        n_samples, n_features = X.shape

        if self.centered:
            self.mean_ = np.zeros_like(np.mean(X, axis=0))
        else:
            self.mean_ = tools.mean(X, epsilon=self.epsilon / 2, range=self.range, axis=0)

        X -= self.mean_

        max_norm = np.linalg.norm(X, axis=1).max()

        if self.data_norm is None:
            warnings.warn("Data norm has not been specified and will be calculated on the data provided.  This will "
                          "result in additional privacy leakage. To ensure differential privacy and no additional "
                          "privacy leakage, specify `data_norm` at initialisation.", PrivacyLeakWarning)
            self.data_norm = max_norm

        if max_norm > self.data_norm:
            warnings.warn("Differential privacy is only guaranteed for data whose rows have a 2-norm of at most %g. "
                          "Got %f\n"
                          "Translate and/or scale the data accordingly to ensure differential privacy is achieved."
                          % (self.data_norm, max_norm), PrivacyLeakWarning)

        XtX = np.dot(X.T, X)

        mech = Wishart().set_epsilon(self.epsilon if self.centered else self.epsilon / 2).\
            set_sensitivity(self.data_norm)
        noisy_input = mech.randomise(XtX)

        u, s, v = np.linalg.svd(noisy_input)
        u, v = svd_flip(u, v)
        s = np.sqrt(s)

        components_ = v

        # Get variance explained by singular values
        explained_variance_ = (s ** 2) / (n_samples - 1)
        total_var = explained_variance_.sum()
        explained_variance_ratio_ = explained_variance_ / total_var
        singular_values_ = s.copy()  # Store the singular values.

        # Post-process the number of components required
        if n_components == 'mle':
            n_components = sk_pca._infer_dimension_(explained_variance_, n_samples, n_features)
        elif 0 < n_components < 1.0:
            # number of components for which the cumulated explained
            # variance percentage is superior to the desired threshold
            ratio_cumsum = stable_cumsum(explained_variance_ratio_)
            n_components = np.searchsorted(ratio_cumsum, n_components) + 1

        # Compute noise covariance using Probabilistic PCA model
        # The sigma2 maximum likelihood (cf. eq. 12.46)
        if n_components < min(n_features, n_samples):
            self.noise_variance_ = explained_variance_[n_components:].mean()
        else:
            self.noise_variance_ = 0.

        self.n_samples_, self.n_features_ = n_samples, n_features
        self.components_ = components_[:n_components]
        self.n_components_ = n_components
        self.explained_variance_ = explained_variance_[:n_components]
        self.explained_variance_ratio_ = explained_variance_ratio_[:n_components]
        self.singular_values_ = singular_values_[:n_components]

        return u, s, v

    @copy_docstring(sk_pca.PCA.fit_transform)
    def fit_transform(self, X, y=None):
        del y

        self._fit(X)

        return self.transform(X)
