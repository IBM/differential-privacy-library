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
Linear Regression with differential privacy
"""
import warnings

import numpy as np
from sklearn import linear_model as sk_lr
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import FLOAT_DTYPES

from diffprivlib.mechanisms import Wishart
from diffprivlib.tools import mean
from diffprivlib.utils import warn_unused_args, PrivacyLeakWarning

_range = range


def _preprocess_data(X, y, fit_intercept, epsilon=1.0, range_X=None, range_y=None, copy=True, check_input=True,
                     **unused_args):
    warn_unused_args(unused_args)

    if check_input:
        X = check_array(X, copy=copy, accept_sparse=False, dtype=FLOAT_DTYPES)
    elif copy:
        X = X.copy(order='K')

    y = np.asarray(y, dtype=X.dtype)
    X_scale = np.ones(X.shape[1], dtype=X.dtype)

    if fit_intercept:
        X_offset = mean(X, axis=0, range=range_X, epsilon=epsilon)
        X -= X_offset
        y_offset = mean(y, axis=0, range=range_y, epsilon=epsilon)
        y = y - y_offset
    else:
        X_offset = np.zeros(X.shape[1], dtype=X.dtype)
        if y.ndim == 1:
            y_offset = X.dtype.type(0)
        else:
            y_offset = np.zeros(y.shape[1], dtype=X.dtype)

    return X, y, X_offset, y_offset, X_scale


class LinearRegression(sk_lr.LinearRegression):
    def __init__(self, epsilon=1, data_norm=None, range_X=None, range_y=None, fit_intercept=True, copy_X=True,
                 **unused_args):
        super().__init__(fit_intercept=fit_intercept, normalize=False, copy_X=copy_X, n_jobs=None)

        self.epsilon = epsilon
        self.data_norm = data_norm
        self.range_X = range_X
        self.range_y = range_y

        warn_unused_args(unused_args)

    def fit(self, X, y, **unused_args):
        """
        Fit linear model.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Training data

        y : array_like, shape (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary

        Returns
        -------
        self : returns an instance of self.
        """

        warn_unused_args(unused_args)

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

        if self.fit_intercept and (self.range_X is None or self.range_y is None):
            warnings.warn("Range parameters haven't been specified, so falling back to determining range from the "
                          "data.\n"
                          "This will result in additional privacy leakage. To ensure differential privacy with no "
                          "additional privacy loss, specify `range_X` and `range_y`.",
                          PrivacyLeakWarning)

            if self.range_X is None:
                self.range_X = np.maximum(np.max(X, axis=0) - np.min(X, axis=0), 1e-5)
            if self.range_y is None:
                self.range_y = np.maximum(np.max(y, axis=0) - np.min(y, axis=0), 1e-5)

        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True, multi_output=True)

        n_features = X.shape[1]
        epsilon_intercept_scale = 1 / (n_features + 1) if self.fit_intercept else 0

        X, y, X_offset, y_offset, X_scale = self._preprocess_data(X, y, fit_intercept=self.fit_intercept,
                                                                  range_X=self.range_X, range_y=self.range_y,
                                                                  epsilon=self.epsilon * epsilon_intercept_scale,
                                                                  copy=self.copy_X)

        A = np.hstack((X, y[:, np.newaxis] if y.ndim == 1 else y))
        AtA = np.dot(A.T, A)

        mech = Wishart().set_epsilon(self.epsilon * (1 - epsilon_intercept_scale)).set_sensitivity(self.data_norm)
        noisy_AtA = mech.randomise(AtA)

        noisy_AtA = noisy_AtA[:n_features, :]
        XtX = noisy_AtA[:, :n_features]
        Xty = noisy_AtA[:, n_features:]

        self.coef_, self._residues, self.rank_, self.singular_ = np.linalg.lstsq(XtX, Xty, rcond=-1)
        self.coef_ = self.coef_.T

        if y.ndim == 1:
            self.coef_ = np.ravel(self.coef_)
        self._set_intercept(X_offset, y_offset, X_scale)
        return self

    _preprocess_data = staticmethod(_preprocess_data)
