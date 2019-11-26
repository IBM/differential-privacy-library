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
import sklearn.decomposition.pca as sk_pca
from sklearn.utils.extmath import stable_cumsum, svd_flip

from diffprivlib import tools
from diffprivlib.mechanisms import Wishart
from diffprivlib.utils import warn_unused_args, copy_docstring, PrivacyLeakWarning


# noinspection PyPep8Naming
class PCA(sk_pca.PCA):
    """PCA model, documentation to follow.
    """
    def __init__(self, n_components=None, centered=False, epsilon=1, data_norm=None, range=None, copy=True,
                 whiten=False, tol=0.0, iterated_power='auto', random_state=None, **unused_args):
        super().__init__(n_components=n_components, copy=copy, whiten=whiten, svd_solver='full', tol=tol,
                         iterated_power=iterated_power, random_state=random_state)
        self.centered = centered
        self.epsilon = epsilon if centered else epsilon / 2  # Must use some privacy budget for calculating the mean
        self.data_norm = data_norm
        self.range = range

        warn_unused_args(unused_args)

    def _fit_full(self, X, n_components):
        n_samples, n_features = X.shape

        if self.centered:
            self.mean_ = np.zeros_like(np.mean(X, axis=0))
        else:
            self.mean_ = tools.mean(X, epsilon=self.epsilon, range=self.range, axis=0)

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

        mech = Wishart().set_epsilon(self.epsilon).set_sensitivity(self.data_norm)
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
