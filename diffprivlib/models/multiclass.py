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
Multiclass classification strategies satisfying differential privacy.
"""

import warnings

from sklearn import multiclass
from sklearn.base import _fit_context, clone
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.metadata_routing import _raise_for_params, process_routing

from diffprivlib.utils import PrivacyLeakWarning
from diffprivlib.validation import DiffprivlibMixin


def _fit_binary(estimator, X, y, fit_params, classes):
    """Fit a single binary estimator."""
    estimator = clone(estimator)
    try:
        estimator.classes_ = classes
    except AttributeError:
        pass
    estimator.fit(X, y, **fit_params)

    return estimator


class OneVsRestClassifier(multiclass.OneVsRestClassifier, DiffprivlibMixin):
    """One-vs-the-rest (OvR) multiclass strategy.

    Also known as one-vs-all, this strategy consists in fitting one classifier
    per class. For each classifier, the class is fitted against all the other
    classes. In addition to its computational efficiency (only `n_classes`
    classifiers are needed), one advantage of this approach is its
    interpretability. Since each class is represented by one and one classifier
    only, it is possible to gain knowledge about the class by inspecting its
    corresponding classifier. This is the most commonly used strategy for
    multiclass classification and is a fair default choice.

    OneVsRestClassifier can also be used for multilabel classification. To use
    this feature, provide an indicator matrix for the target `y` when calling
    `.fit`. In other words, the target labels should be formatted as a 2D
    binary (0/1) matrix, where [i, j] == 1 indicates the presence of label j
    in sample i. This estimator uses the binary relevance method to perform
    multilabel classification, which involves training one binary classifier
    independently for each label.

    Read more in the :ref:`User Guide <ovr_classification>`.

    Parameters
    ----------
    estimator : estimator object
        A regressor or a classifier that implements :term:`fit`.
        When a classifier is passed, :term:`decision_function` will be used
        in priority and it will fallback to :term:`predict_proba` if it is not
        available.
        When a regressor is passed, :term:`predict` is used.

    n_jobs : int, default=None
        The number of jobs to use for the computation: the `n_classes`
        one-vs-rest problems are computed in parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

        .. versionchanged:: 0.20
           `n_jobs` default changed from 1 to None

    verbose : int, default=0
        The verbosity level, if non zero, progress messages are printed.
        Below 50, the output is sent to stderr. Otherwise, the output is sent
        to stdout. The frequency of the messages increases with the verbosity
        level, reporting all iterations at 10. See :class:`joblib.Parallel` for
        more details.

        .. versionadded:: 1.1

    Attributes
    ----------
    estimators_ : list of `n_classes` estimators
        Estimators used for predictions.

    classes_ : array, shape = [`n_classes`]
        Class labels.

    n_classes_ : int
        Number of classes.

    label_binarizer_ : LabelBinarizer object
        Object used to transform multiclass labels to binary labels and
        vice-versa.

    multilabel_ : boolean
        Whether a OneVsRestClassifier is a multilabel classifier.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

        .. versionadded:: 1.0

    See Also
    --------
    OneVsOneClassifier : One-vs-one multiclass strategy.
    OutputCodeClassifier : (Error-Correcting) Output-Code multiclass strategy.
    sklearn.multioutput.MultiOutputClassifier : Alternate way of extending an
        estimator for multilabel classification.
    sklearn.preprocessing.MultiLabelBinarizer : Transform iterable of iterables
        to binary indicator matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from diffprivlib.models import LogisticRegression
    >>> from diffprivlib.models.multiclass import OneVsRestClassifier
    >>> X = np.array([
    ...     [10, 10],
    ...     [8, 10],
    ...     [-5, 5.5],
    ...     [-5.4, 5.5],
    ...     [-20, -20],
    ...     [-15, -20]
    ... ])
    >>> y = np.array([0, 0, 1, 1, 2, 2])
    >>> clf = OneVsRestClassifier(LogisticRegression(data_norm=30, random_state=0), epsilon=100,
    ...                           classes=[0, 1, 2]).fit(X, y)
    >>> clf.predict([[-19, -20], [9, 9], [-5, 5]])
    array([2, 0, 1])
    """

    _parameter_constraints = DiffprivlibMixin._copy_parameter_constraints(
        multiclass.OneVsRestClassifier, "estimator", "n_jobs", "verbose")

    def __init__(self, estimator, *, n_jobs=None, verbose=0, classes=None, epsilon=None, delta=None):
        super().__init__(estimator, n_jobs=n_jobs, verbose=verbose)

        if epsilon is not None:
            self.estimator.epsilon = epsilon

        if delta is not None:
            self.estimator.delta = delta

        self.classes = classes

    @property
    def epsilon(self):
        r"""Privacy parameter :math:`\epsilon` of the underlying estimator.

        Returns
        -------
        float or None
            The ``epsilon`` attribute of ``self.estimator``, or ``None`` if the
            estimator does not expose that attribute.
        """
        try:
            return self.estimator.epsilon
        except AttributeError:
            return None

    @property
    def delta(self):
        r"""Privacy parameter :math:`\delta` of the underlying estimator.

        :math:`\delta` is the failure probability in :math:`(\epsilon, \delta)`-differential
        privacy.  A value of ``0`` corresponds to pure :math:`\epsilon`-differential privacy.

        Returns
        -------
        float or None
            The ``delta`` attribute of ``self.estimator``, or ``None`` if the
            estimator does not expose that attribute.
        """
        try:
            return self.estimator.delta
        except AttributeError:
            return None

    @_fit_context(
        # OneVsRestClassifier.estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y, **fit_params):
        """Fit underlying estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data.

        y : {array-like, sparse matrix} of shape (n_samples,) or (n_samples, n_classes)
            Multi-class targets. An indicator matrix turns on multilabel
            classification.

        **fit_params : dict
            Parameters passed to the ``estimator.fit`` method of each
            sub-estimator.

        Returns
        -------
        self : object
            Instance of fitted estimator.
        """
        _raise_for_params(fit_params, self, "fit")

        routed_params = process_routing(
            self,
            "fit",
            **fit_params,
        )

        # A sparse LabelBinarizer, with sparse_output=True, has been shown to
        # outperform or match a dense label binarizer in all cases and has also
        # resulted in less or equal memory consumption in the fit_ovr function
        # overall.
        self.label_binarizer_ = LabelBinarizer(sparse_output=True)
        if self.classes is not None:
            self.label_binarizer_.fit(self.classes)
            Y = self.label_binarizer_.transform(y)
        else:
            warnings.warn("Classes have not been specified and will be calculated on the data provided. This will "
                          "result in additional privacy leakage. To ensure differential privacy and no additional "
                          "privacy leakage, specify the prediction classes for model.", PrivacyLeakWarning)
            Y = self.label_binarizer_.fit_transform(y)

        # classes_ needs to be set separately, as sklearn regonises this as a fitted parameter
        self.classes_ = self.label_binarizer_.classes_
        self.classes = self.classes_.tolist()
        Y = Y.tocsc()
        columns = (col.toarray().ravel() for col in Y.T)

        # In cases where individual estimators are very fast to train setting
        # n_jobs > 1 in can results in slower performance due to the overhead
        # of spawning threads.  See joblib issue #112.
        self.estimators_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_fit_binary)(
                self.estimator,
                X,
                column,
                fit_params=routed_params.estimator.fit,
                classes=[0, 1]
            )
            for column in columns
        )

        if hasattr(self.estimators_[0], "n_features_in_"):
            self.n_features_in_ = self.estimators_[0].n_features_in_
        if hasattr(self.estimators_[0], "feature_names_in_"):
            self.feature_names_in_ = self.estimators_[0].feature_names_in_

        return self
