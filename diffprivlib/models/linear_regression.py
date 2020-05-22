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
import sklearn.linear_model as sk_lr
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import FLOAT_DTYPES

from diffprivlib.accountant import BudgetAccountant
from diffprivlib.mechanisms import Wishart
from diffprivlib.tools import mean
from diffprivlib.utils import warn_unused_args, PrivacyLeakWarning
from diffprivlib.validation import clip_to_norm, check_bounds, clip_to_bounds

_range = range


# noinspection PyPep8Naming
def _preprocess_data(X, y, fit_intercept, epsilon=1.0, bounds_X=None, bounds_y=None, copy=True, check_input=True,
                     **unused_args):
    warn_unused_args(unused_args)

    if check_input:
        X = check_array(X, copy=copy, accept_sparse=False, dtype=FLOAT_DTYPES)
    elif copy:
        X = X.copy(order='K')

    y = np.asarray(y, dtype=X.dtype)
    X_scale = np.ones(X.shape[1], dtype=X.dtype)

    if fit_intercept:
        bounds_X = check_bounds(bounds_X, X.shape[1])
        bounds_y = check_bounds(bounds_y, y.shape[1] if y.ndim > 1 else 1)

        X = clip_to_bounds(X, bounds_X)
        y = clip_to_bounds(y, bounds_y)

        X_offset = mean(X, axis=0, bounds=bounds_X, epsilon=epsilon, accountant=BudgetAccountant())
        X -= X_offset
        y_offset = mean(y, axis=0, bounds=bounds_y, epsilon=epsilon, accountant=BudgetAccountant())
        y = y - y_offset
    else:
        X_offset = np.zeros(X.shape[1], dtype=X.dtype)
        if y.ndim == 1:
            y_offset = X.dtype.type(0)
        else:
            y_offset = np.zeros(y.shape[1], dtype=X.dtype)

    return X, y, X_offset, y_offset, X_scale


# noinspection PyPep8Naming,PyAttributeOutsideInit
class LinearRegression(sk_lr.LinearRegression):
    r"""
    Ordinary least squares Linear Regression with differential privacy.

    LinearRegression fits a linear model with coefficients w = (w1, ..., wp) to minimize the residual sum of squares
    between the observed targets in the dataset, and the targets predicted by the linear approximation.  Differential
    privacy is guaranteed with respect to the training sample.

    Differential privacy is achieved by adding noise to the second moment matrix using the :class:`.Wishart` mechanism.
    This method is demonstrated in  [She15]_, but our implementation takes inspiration from the use of the Wishart
    distribution in  [IS16]_ to achieve a strict differential privacy guarantee.

    Parameters
    ----------
    epsilon : float, default: 1.0
        Privacy parameter :math:`\epsilon`.

    data_norm : float, optional
        The max l2 norm of any row of the concatenated dataset A = [X; y].  This defines the spread of data that will be
        protected by differential privacy.

        If not specified, the max norm is taken from the data when ``.fit()`` is first called, but will result in a
        :class:`.PrivacyLeakWarning`, as it reveals information about the data.  To preserve differential privacy fully,
        `data_norm` should be selected independently of the data, i.e. with domain knowledge.

    bounds_X:  tuple, optional
        Bounds of the data, provided as a tuple of the form (min, max).  `min` and `max` can either be scalars, covering
        the min/max of the entire data, or vectors with one entry per feature.  If not provided, the bounds are computed
        on the data when ``.fit()`` is first called, resulting in a :class:`.PrivacyLeakWarning`.

    bounds_y : tuple
        Same as `bounds_X`, but for the training label set `y`.

    fit_intercept : bool, default: True
        Whether to calculate the intercept for this model.  If set to False, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    copy_X : bool, default: True
        If True, X will be copied; else, it may be overwritten.

    accountant : BudgetAccountant, optional
        Accountant to keep track of privacy budget.

    Attributes
    ----------
    coef_ : array of shape (n_features, ) or (n_targets, n_features)
        Estimated coefficients for the linear regression problem.  If multiple targets are passed during the fit (y 2D),
        this is a 2D array of shape (n_targets, n_features), while if only one target is passed, this is a 1D array of
        length n_features.

    rank_ : int
        Rank of matrix `X`.

    singular_ : array of shape (min(X, y),)
        Singular values of `X`.

    intercept_ : float or array of shape of (n_targets,)
        Independent term in the linear model.  Set to 0.0 if `fit_intercept = False`.

    References
    ----------
    .. [She15] Sheffet, Or. "Private approximations of the 2nd-moment matrix using existing techniques in linear
        regression." arXiv preprint arXiv:1507.00056 (2015).

    .. [IS16] Imtiaz, Hafiz, and Anand D. Sarwate. "Symmetric matrix perturbation for differentially-private principal
        component analysis." In 2016 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP),
        pp. 2339-2343. IEEE, 2016.
    """
    def __init__(self, epsilon=1.0, data_norm=None, bounds_X=None, bounds_y=None, fit_intercept=True, copy_X=True,
                 accountant=None, **unused_args):
        super().__init__(fit_intercept=fit_intercept, normalize=False, copy_X=copy_X, n_jobs=None)

        self.epsilon = epsilon
        self.data_norm = data_norm
        self.bounds_X = bounds_X
        self.bounds_y = bounds_y
        self.accountant = BudgetAccountant.load_default(accountant)

        warn_unused_args(unused_args)

    def fit(self, X, y, sample_weight=None):
        """
        Fit linear model.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Training data

        y : array_like, shape (n_samples, n_targets)
            Target values.  Will be cast to X's dtype if necessary

        sample_weight : ignored
            Ignored by diffprivlib.  Present for consistency with sklearn API.

        Returns
        -------
        self : returns an instance of self.
        """
        self.accountant.check(self.epsilon, 0)

        if sample_weight is not None:
            warn_unused_args("sample_weight")

        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True, multi_output=True)

        if self.fit_intercept:
            if self.bounds_X is None or self.bounds_y is None:
                warnings.warn(
                    "Bounds parameters haven't been specified, so falling back to determining bounds from the "
                    "data.\n"
                    "This will result in additional privacy leakage. To ensure differential privacy with no "
                    "additional privacy loss, specify `bounds_X` and `bounds_y`.",
                    PrivacyLeakWarning)

                if self.bounds_X is None:
                    self.bounds_X = (np.min(X, axis=0), np.max(X, axis=0))
                if self.bounds_y is None:
                    self.bounds_y = (np.min(y, axis=0), np.max(y, axis=0))

            self.bounds_X = check_bounds(self.bounds_X, X.shape[1])
            self.bounds_y = check_bounds(self.bounds_y, y.shape[1] if y.ndim > 1 else 1)

        n_features = X.shape[1]
        epsilon_intercept_scale = 1 / (n_features + 1) if self.fit_intercept else 0

        X, y, X_offset, y_offset, X_scale = self._preprocess_data(X, y, fit_intercept=self.fit_intercept,
                                                                  bounds_X=self.bounds_X, bounds_y=self.bounds_y,
                                                                  epsilon=self.epsilon * epsilon_intercept_scale,
                                                                  copy=self.copy_X)

        if self.data_norm is None:
            warnings.warn("Data norm has not been specified and will be calculated on the data provided.  This will "
                          "result in additional privacy leakage. To ensure differential privacy and no additional "
                          "privacy leakage, specify `data_norm` at initialisation.", PrivacyLeakWarning)
            self.data_norm = np.linalg.norm(X, axis=1).max()

        X = clip_to_norm(X, self.data_norm)

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

        self.accountant.spend(self.epsilon, 0)

        return self

    _preprocess_data = staticmethod(_preprocess_data)
