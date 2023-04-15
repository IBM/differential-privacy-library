# pip install diffprivlib

# Commented out IPython magic to ensure Python compatibility.
"""
Isolation Forest with Differential Privacy
"""
import warnings
import numpy as np
import numbers
import numpy as np
from scipy.sparse import issparse
from warnings import warn
from numbers import Integral, Real
from joblib import Parallel, delayed
from sklearn.tree import ExtraTreeRegressor
from sklearn.tree._tree import DTYPE as tree_dtype
from sklearn.utils import (
    check_random_state,
    check_array,
    gen_batches,
    get_chunk_n_rows,
)
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, _num_samples
from sklearn.base import OutlierMixin
from sklearn.ensemble._bagging import BaseBagging
from sklearn.datasets import make_classification

from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import PrivacyLeakWarning, check_random_state
from diffprivlib.mechanisms import PermuteAndFlip
from diffprivlib.validation import DiffprivlibMixin

from sklearn.exceptions import DataConversionWarning
from sklearn.tree._tree import Tree, DOUBLE, DTYPE, NODE_DTYPE
from sklearn.tree import DecisionTreeClassifier as skDecisionTreeClassifier
from sklearn.ensemble import IsolationForest as skIsolationForest
from sklearn.ensemble._forest import _parallel_build_trees
from collections import namedtuple
MAX_INT = np.iinfo(np.int32).max

__all__ = ["IsolationForest"]


class IsolationForest(skIsolationForest, DiffprivlibMixin):
    r"""
    Isolation Forest Algorithm with differential privacy.
    This class implements Differentially Private Isolation Forests

    Return the anomaly score of each sample using the IsolationForest algorithm
    The IsolationForest 'isolates' observations by randomly selecting a feature
    and then randomly selecting a split value between the maximum and minimum
    values of the selected feature.

    Since recursive partitioning can be represented by a tree structure, the
    number of splittings required to isolate a sample is equivalent to the path
    length from the root node to the terminating node.
    This path length, averaged over a forest of such random trees, is a
    measure of normality and our decision function.

    Random partitioning produces noticeably shorter paths for anomalies.
    Hence, when a forest of random trees collectively produce shorter path
    lengths for particular samples, they are highly likely to be anomalies.
    Read more in the :ref:`User Guide <isolation_forest>`.
    .. versionadded:: 0.18


    Parameters
    ----------
    n_estimators : int, default=100
        The number of base estimators in the ensemble.
    max_samples : "auto", int or float, default="auto"
        The number of samples to draw from X to train each base estimator.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.
            - If "auto", then `max_samples=min(256, n_samples)`.
        If max_samples is larger than the number of samples provided,
        all samples will be used for all trees (no sampling).
    contamination : 'auto' or float, default='auto'
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. Used when fitting to define the threshold
        on the scores of the samples.
            - If 'auto', the threshold is determined as in the
              original paper.
            - If float, the contamination should be in the range (0, 0.5].
        .. versionchanged:: 0.22
           The default value of ``contamination`` changed from 0.1
           to ``'auto'``.
    max_features : int or float, default=1.0
        The number of features to draw from X to train each base estimator.
            - If int, then draw `max_features` features.
            - If float, then draw `max(1, int(max_features * n_features_in_))` features.
        Note: using a float number less than 1.0 or integer less than number of
        features will enable feature subsampling and leads to a longerr runtime.
    bootstrap : bool, default=False
        If True, individual trees are fit on random subsets of the training
        data sampled with replacement. If False, sampling without replacement
        is performed.
    n_jobs : int, default=None
        The number of jobs to run in parallel for both :meth:`fit` and
        :meth:`predict`. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors. See :term:`Glossary <n_jobs>` for more details.
    random_state : int, RandomState instance or None, default=None
        Controls the pseudo-randomness of the selection of the feature
        and split values for each branching step and each tree in the forest.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.
    verbose : int, default=0
        Controls the verbosity of the tree building process.
    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.
        .. versionadded:: 0.21
    Attributes
    ----------
    estimator_ : :class:`~sklearn.tree.ExtraTreeRegressor` instance
        The child estimator template used to create the collection of
        fitted sub-estimators.
        .. versionadded:: 1.2
           `base_estimator_` was renamed to `estimator_`.
    base_estimator_ : ExtraTreeRegressor instance
        The child estimator template used to create the collection of
        fitted sub-estimators.
        .. deprecated:: 1.2
            `base_estimator_` is deprecated and will be removed in 1.4.
            Use `estimator_` instead.
    estimators_ : list of ExtraTreeRegressor instances
        The collection of fitted sub-estimators.
    estimators_features_ : list of ndarray
        The subset of drawn features for each base estimator.
    estimators_samples_ : list of ndarray
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator.
    max_samples_ : int
        The actual number of samples.
    offset_ : float
        Offset used to define the decision function from the raw scores. We
        have the relation: ``decision_function = score_samples - offset_``.
        ``offset_`` is defined as follows. When the contamination parameter is
        set to "auto", the offset is equal to -0.5 as the scores of inliers are
        close to 0 and the scores of outliers are close to -1. When a
        contamination parameter different than "auto" is provided, the offset
        is defined in such a way we obtain the expected number of outliers
        (samples with decision function < 0) in training.
        .. versionadded:: 0.20
    n_features_in_ : int
        Number of features seen during :term:`fit`.
        .. versionadded:: 0.24
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
        .. versionadded:: 1.0
    _parameter_constraints = DiffprivlibMixin._copy_parameter_constraints(
        skIsolationForest, "n_estimators", "n_jobs", "verbose", "random_state", "warm_start")
    Example
    ----------
    >> from sklearn.datasets import make_classification
    >> obj1= IsolationForest()
    >> X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0,
                            random_state=0, shuffle=False)
    >> clf = obj1.fit(X, y)
    >> print(clf.predict([[0, 0, 0, 0]]))

    [1]


    References
    ----------
    .. [1] Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua. "Isolation forest."
           Data Mining, 2008. ICDM'08. Eighth IEEE International Conference on. """

    _parameter_constraints = DiffprivlibMixin._copy_parameter_constraints(
        skIsolationForest, "n_estimators", "n_jobs", "verbose", "random_state", "warm_start")
    """parameter_constraints: dict = {
        "n_estimators": [Interval(Integral, 1, None, closed="left")],
        "max_samples": [
            StrOptions({"auto"}),
            Interval(Integral, 1, None, closed="left"),
            Interval(Real, 0, 1, closed="right"),
        ],
        "contamination": [
            StrOptions({"auto"}),
            Interval(Real, 0, 0.5, closed="right"),
        ],
        "max_features": [
            Integral,
            Interval(Real, 0, 1, closed="right"),
        ],
        "bootstrap": ["boolean"],
        "n_jobs": [Integral, None],
        "random_state": ["random_state"],
        "verbose": ["verbose"],
        "warm_start": ["boolean"],
    }"""

    def __init__(
        self,
        *,
        n_estimators=100,
        # bootstrap_features=False,
        max_samples="auto",
        contamination="auto",
        max_features=1.0,
        bootstrap=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
    ):
        super().__init__(

            bootstrap=bootstrap,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

        self.contamination = contamination

    def _set_oob_score(self, X, y):
        raise NotImplementedError("OOB score not supported by iforest")

    def _parallel_args(self):
        # ExtraTreeRegressor releases the GIL, so it's more efficient to use
        # a thread-based backend rather than a process-based backend so as
        # to avoid suffering from communication overhead and extra memory
        # copies.
        return {"prefer": "threads"}

    def fit(self, X, y=None, sample_weight=None):

        """
        Fit estimator:
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Use ``dtype=np.float32`` for maximum
            efficiency. Sparse matrices are also supported, use sparse
            ``csc_matrix`` for maximum efficiency.
        y : Ignored
            Not used, present for API consistency by convention.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
        Returns
        -------
        self : object
            Fitted estimator.
        """
        self._validate_params()
        # self.accountant.check(self.epsilon, 0)

        X = self._validate_data(X, accept_sparse=["csc"], dtype=tree_dtype)
        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        rnd = check_random_state(self.random_state)
        y = rnd.uniform(size=X.shape[0])

        # ensure that max_sample is in [1, n_samples]:
        n_samples = X.shape[0]

        if isinstance(self.max_samples, str) and self.max_samples == "auto":
            max_samples = min(256, n_samples)

        elif isinstance(self.max_samples, numbers.Integral):
            if self.max_samples > n_samples:
                warn(
                    "max_samples (%s) is greater than the "
                    "total number of samples (%s). max_samples "
                    "will be set to n_samples for estimation."
#                     % (self.max_samples, n_samples)
                )
                max_samples = n_samples
            else:
                max_samples = self.max_samples
        else:  # max_samples is float
            max_samples = int(self.max_samples * X.shape[0])

        self.max_samples_ = max_samples
        max_depth = int(np.ceil(np.log2(max(max_samples, 2))))
        super()._fit(
            X,
            y,
            max_samples,
            max_depth=max_depth,
            sample_weight=sample_weight,
            check_input=False,
        )

        if self.contamination == "auto":
            # 0.5 plays a special role as described in the original paper.
            # we take the opposite as we consider the opposite of their score.
            self.offset_ = -0.5
            return self

        # else, define offset_ wrt contamination parameter
        self.offset_ = np.percentile(self.score_samples(X), 100.0 * self.contamination)

        return self

    def predict(self, X):
        """
        Predict if a particular sample is an outlier or not.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        Returns
        -------
        is_inlier : ndarray of shape (n_samples,)
            For each observation, tells whether or not (+1 or -1) it should
            be considered as an inlier according to the fitted model.
        """
        check_is_fitted(self)
        decision_func = self.decision_function(X)
        is_inlier = np.ones_like(decision_func, dtype=int)
        is_inlier[decision_func < 0] = -1
        return is_inlier

    def decision_function(self, X):
        """
        Average anomaly score of X of the base classifiers.
        The anomaly score of an input sample is computed as
        the mean anomaly score of the trees in the forest.
        The measure of normality of an observation given a tree is the depth
        of the leaf containing this observation, which is equivalent to
        the number of splittings required to isolate this point. In case of
        several observations n_left in the leaf, the average path length of
        a n_left samples isolation tree is added.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        Returns
        -------
        scores : ndarray of shape (n_samples,)
            The anomaly score of the input samples.
            The lower, the more abnormal. Negative scores represent outliers,
            positive scores represent inliers.
        """
        # We subtract self.offset_ to make 0 be the threshold value for being
        # an outlier:

        return self.score_samples(X) - self.offset_

    def score_samples(self, X):
        """
        Opposite of the anomaly score defined in the original paper.
        The anomaly score of an input sample is computed as
        the mean anomaly score of the trees in the forest.
        The measure of normality of an observation given a tree is the depth
        of the leaf containing this observation, which is equivalent to
        the number of splittings required to isolate this point. In case of
        several observations n_left in the leaf, the average path length of
        a n_left samples isolation tree is added.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        scores : ndarray of shape (n_samples,)
            The anomaly score of the input samples.
            The lower, the more abnormal.
        """
        # code structure from ForestClassifier/predict_proba

        check_is_fitted(self)

        # Check data
        X = self._validate_data(X, accept_sparse="csr", reset=False)

        # Take the opposite of the scores as bigger is better (here less
        # abnormal)
        return -self._compute_chunked_score_samples(X)

    def _compute_chunked_score_samples(self, X):

        n_samples = _num_samples(X)

        if self._max_features == X.shape[1]:
            subsample_features = False
        else:
            subsample_features = True

        # We get as many rows as possible within our working_memory budget
        # (defined by sklearn.get_config()['working_memory']) to store
        # self._max_features in each row during computation.
        #
        # Note:
        #  - this will get at least 1 row, even if 1 row of score will
        #    exceed working_memory.
        #  - this does only account for temporary memory usage while loading
        #    the data needed to compute the scores -- the returned scores
        #    themselves are 1D.

        chunk_n_rows = get_chunk_n_rows(
            row_bytes=16 * self._max_features, max_n_rows=n_samples
        )
        slices = gen_batches(n_samples, chunk_n_rows)

        scores = np.zeros(n_samples, order="f")

        for sl in slices:
            # compute score on the slices of test samples:
            scores[sl] = self._compute_score_samples(X[sl], subsample_features)

        return scores

    def _compute_score_samples(self, X, subsample_features):
        """
        Compute the score of each samples in X going through the extra trees.
        Parameters
        ----------
        X : array-like or sparse matrix
            Data matrix.
        subsample_features : bool
            Whether features should be subsampled.
        """
        n_samples = X.shape[0]

        depths = np.zeros(n_samples, order="f")

        for tree, features in zip(self.estimators_, self.estimators_features_):
            X_subset = X[:, features] if subsample_features else X

            leaves_index = tree.apply(X_subset)
            node_indicator = tree.decision_path(X_subset)
            n_samples_leaf = tree.tree_.n_node_samples[leaves_index]

            depths += (
                np.ravel(node_indicator.sum(axis=1))
                + _average_path_length(n_samples_leaf)
                - 1.0
            )
        denominator = len(self.estimators_) * _average_path_length([self.max_samples_])
        scores = 2 ** (
            # For a single training sample, denominator and depth are 0.
            # Therefore, we set the score manually to 1.
            -np.divide(
                depths, denominator, out=np.ones_like(depths), where=denominator != 0
            )
        )
        return scores

    def _more_tags(self):
        return {
            "_xfail_checks": {
                "check_sample_weights_invariance": (
                    "zero sample_weight is not equivalent to removing samples"
                ),
            }
        }


def _average_path_length(n_samples_leaf):
    """
    The average path length in a n_samples iTree, which is equal to
    the average path length of an unsuccessful BST search since the
    latter has the same structure as an isolation tree.
    Parameters
    ----------
    n_samples_leaf : array-like of shape (n_samples,)
        The number of training samples in each test sample leaf, for
        each estimators.
    Returns
    -------
    average_path_length : ndarray of shape (n_samples,)
    """

    n_samples_leaf = check_array(n_samples_leaf, ensure_2d=False)

    n_samples_leaf_shape = n_samples_leaf.shape
    n_samples_leaf = n_samples_leaf.reshape((1, -1))
    average_path_length = np.zeros(n_samples_leaf.shape)

    mask_1 = n_samples_leaf <= 1
    mask_2 = n_samples_leaf == 2
    not_mask = ~np.logical_or(mask_1, mask_2)

    average_path_length[mask_1] = 0.0
    average_path_length[mask_2] = 1.0
    average_path_length[not_mask] = (
        2.0 * (np.log(n_samples_leaf[not_mask] - 1.0) + np.euler_gamma)
        - 2.0 * (n_samples_leaf[not_mask] - 1.0) / n_samples_leaf[not_mask]
    )

    return average_path_length.reshape(n_samples_leaf_shape)
class DecisionTreeClassifier(skDecisionTreeClassifier, DiffprivlibMixin):
    r"""Decision Tree Classifier with differential privacy.
    This class implements the base differentially private decision tree classifier
    for the Isolation Forest classifier algorithm. Not meant to be used separately.
    Parameters
    ----------
    max_depth : int, default: 5
        The maximum depth of the tree.
    epsilon : float, default: 1.0
        Privacy parameter :math:`\epsilon`.
    bounds : tuple, optional
        Bounds of the data, provided as a tuple of the form (min, max).  `min` and `max` can either be scalars, covering
        the min/max of the entire data, or vectors with one entry per feature.  If not provided, the bounds are computed
        on the data when ``.fit()`` is first called, resulting in a :class:`.PrivacyLeakWarning`.
    classes : array-like of shape (n_classes,), optional
        Array of class labels. If not provided, the classes will be read from the data when ``.fit()`` is first called,
        resulting in a :class:`.PrivacyLeakWarning`.
    random_state : int or RandomState, optional
        Controls the randomness of the estimator.  At each split, the feature to split on is chosen randomly, as is the
        threshold at which to split.  The classification label at each leaf is then randomised, subject to differential
        privacy constraints. To obtain a deterministic behaviour during randomisation, ``random_state`` has to be fixed
        to an integer.
    accountant : BudgetAccountant, optional
        Accountant to keep track of privacy budget.
    Attributes
    ----------
    n_features_in_: int
        The number of features when fit is performed.
    n_classes_: int
        The number of classes.
    classes_: array of shape (n_classes, )
        The class labels.
    """

    _parameter_constraints = DiffprivlibMixin._copy_parameter_constraints(
        skDecisionTreeClassifier, "max_depth", "random_state")

    def __init__(self, max_depth=5, *, epsilon=1, bounds=None, classes=None, random_state=None, accountant=None,
                 **unused_args):
        try:
            super().__init__(
                splitter=None,
                max_depth=max_depth,
                min_samples_split=None,
                min_samples_leaf=None,
                min_weight_fraction_leaf=None,
                max_features=None,
                random_state=random_state,
                max_leaf_nodes=None,
                min_impurity_decrease=None,
                min_impurity_split=None
            )
        except TypeError:
            super().__init__(
                splitter=None,
                max_depth=max_depth,
                min_samples_split=None,
                min_samples_leaf=None,
                min_weight_fraction_leaf=None,
                max_features=None,
                random_state=random_state,
                max_leaf_nodes=None,
                min_impurity_decrease=None
            )
        self.epsilon = epsilon
        self.bounds = bounds
        self.classes = classes
        self.accountant = BudgetAccountant.load_default(accountant)

        self._warn_unused_args(unused_args)

    def fit(self, X, y, sample_weight=None, check_input=True):
        """Build a differentially-private decision tree classifier from the training set (X, y).
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples. Internally, it will be converted to ``dtype=np.float32``.
        y : array-like of shape (n_samples,)
            The target values (class labels) as integers or strings.
        sample_weight : ignored
            Ignored by diffprivlib.  Present for consistency with sklearn API.
        check_input : bool, default=True
            Allow to bypass several input checking. Don't use this parameter unless you know what you do.
        Returns
        -------
        self : DecisionTreeClassifier
            Fitted estimator.
        """
        self._validate_params()
        random_state = check_random_state(self.random_state)

        self.accountant.check(self.epsilon, 0)

        if sample_weight is not None:
            self._warn_unused_args("sample_weight")

        if check_input:
            X, y = self._validate_data(X, y, multi_output=False)
        self.n_outputs_ = 1

        if self.bounds is None:
            warnings.warn("Bounds have not been specified and will be calculated on the data provided. This will "
                          "result in additional privacy leakage. To ensure differential privacy and no additional "
                          "privacy leakage, specify bounds for each dimension.", PrivacyLeakWarning)
            self.bounds = (np.min(X, axis=0), np.max(X, axis=0))
        self.bounds = self._check_bounds(self.bounds, shape=X.shape[1])
        X = self._clip_to_bounds(X, self.bounds)

        if self.classes is None:
            warnings.warn("Classes have not been specified and will be calculated on the data provided. This will "
                          "result in additional privacy leakage. To ensure differential privacy and no additional "
                          "privacy leakage, specify the prediction classes for model.", PrivacyLeakWarning)
            self.classes = np.unique(y)
        self.classes_ = np.ravel(self.classes)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]

        # Build and fit the _FittingTree
        fitting_tree = _FittingTree(self.max_depth, self.n_features_in_, self.classes_, self.epsilon, self.bounds,
                                    random_state)
        fitting_tree.build()
        fitting_tree.fit(X, y)

        # Load params from _FittingTree into sklearn.Tree
        d = fitting_tree.__getstate__()
        tree = Tree(self.n_features_in_, np.array([self.n_classes_]), self.n_outputs_)
        tree.__setstate__(d)
        self.tree_ = tree

        self.accountant.spend(self.epsilon, 0)

        return self

    @property
    def n_features_(self):
        return self.n_features_in_

    def _more_tags(self):
        return {}


class _FittingTree(DiffprivlibMixin):
    r"""Array-based representation of a binary decision tree, trained with differential privacy.
    This tree mimics the architecture of the corresponding Tree from sklearn.tree.tree_, but without many methods given
    in Tree. The purpose of _FittingTree is to fit the parameters of the model, and have those parameters passed to
    Tree (using _FittingTree.__getstate__() and Tree.__setstate__()), to be used for prediction.
    Parameters
    ----------
    max_depth : int
        The maximum depth of the tree.
    n_features : int
        The number of features of the training dataset.
    classes : array-like of shape (n_classes,)
        The classes of the training dataset.
    epsilon : float
        Privacy parameter :math:`\epsilon`.
    bounds : tuple
        Bounds of the data, provided as a tuple of the form (min, max).  `min` and `max` can either be scalars, covering
        the min/max of the entire data.
    random_state : RandomState
        Controls the randomness of the building and training process: the feature to split at each node, the threshold
        to split at and the randomisation of the label at each leaf.
    """
    _TREE_LEAF = -1
    _TREE_UNDEFINED = -2
    StackNode = namedtuple("StackNode", ["parent", "is_left", "depth", "bounds"])

    def __init__(self, max_depth, n_features, classes, epsilon, bounds, random_state):
        self.node_count = 0
        self.nodes = []
        self.max_depth = max_depth
        self.n_features = n_features
        self.classes = classes
        self.epsilon = epsilon
        self.bounds = bounds
        self.random_state = random_state

    def __getstate__(self):
        """Get state of _FittingTree to feed into __setstate__ of sklearn.Tree"""
        d = {"max_depth": self.max_depth,
             "node_count": self.node_count,
             "nodes": np.array([tuple(node) for node in self.nodes], dtype=NODE_DTYPE),
             "values": self.values_}
        return d

    def build(self):
        """Build the decision tree using random feature selection and random thresholding."""
        stack = [self.StackNode(parent=self._TREE_UNDEFINED, is_left=False, depth=0, bounds=self.bounds)]

        while stack:
            parent, is_left, depth, bounds = stack.pop()
            node_id = self.node_count
            bounds_lower, bounds_upper = self._check_bounds(bounds, shape=self.n_features)

            # Update parent node with its child
            if parent != self._TREE_UNDEFINED:
                if is_left:
                    self.nodes[parent].left_child = node_id
                else:
                    self.nodes[parent].right_child = node_id

            # Check if we have a leaf node, then add it
            if depth >= self.max_depth:
                node = _Node(node_id, self._TREE_UNDEFINED, self._TREE_UNDEFINED)
                node.left_child = self._TREE_LEAF
                node.right_child = self._TREE_LEAF

                self.nodes.append(node)
                self.node_count += 1
                continue

            # We have a decision node, so pick feature and threshold
            feature = self.random_state.randint(self.n_features)
            threshold = self.random_state.uniform(bounds_lower[feature], bounds_upper[feature])

            left_bounds_upper = bounds_upper.copy()
            left_bounds_upper[feature] = threshold
            right_bounds_lower = bounds_lower.copy()
            right_bounds_lower[feature] = threshold

            self.nodes.append(_Node(node_id, feature, threshold))
            self.node_count += 1

            stack.append(self.StackNode(parent=node_id, is_left=True, depth=depth+1,
                                        bounds=(bounds_lower, left_bounds_upper)))
            stack.append(self.StackNode(parent=node_id, is_left=False, depth=depth+1,
                                        bounds=(right_bounds_lower, bounds_upper)))

        return self

    def fit(self, X, y):
        """Fit the tree to the given training data.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target vector relative to X.
        """
        if not self.nodes:
            raise ValueError("Fitting Tree must be built before calling fit().")

        leaves = self.apply(X)
        unique_leaves = np.unique(leaves)
        values = np.zeros(shape=(self.node_count, 1, len(self.classes)))

        # Populate value of real leaves
        for leaf in unique_leaves:
            idxs = (leaves == leaf)
            leaf_y = y[idxs]

            counts = [np.sum(leaf_y == cls) for cls in self.classes]
            mech = PermuteAndFlip(epsilon=self.epsilon, sensitivity=1, monotonic=True, utility=counts,
                                  random_state=self.random_state)
            values[leaf, 0, mech.randomise()] = 1

        # Populate value of empty leaves
        for node in self.nodes:
            if values[node.node_id].sum() or node.left_child != self._TREE_LEAF:
                continue

            values[node.node_id, 0, self.random_state.randint(len(self.classes))] = 1

        self.values_ = values

        return self

    def apply(self, X):
        """Finds the terminal region (=leaf node) for each sample in X."""
        n_samples = X.shape[0]
        out = np.zeros((n_samples,), dtype=int)
        out_ptr = out.data

        for i in range(n_samples):
            node = self.nodes[0]

            while node.left_child != self._TREE_LEAF:
                if X[i, node.feature] <= node.threshold:
                    node = self.nodes[node.left_child]
                else:
                    node = self.nodes[node.right_child]

            out_ptr[i] = node.node_id

        return out


class _Node:
    """Base storage structure for the nodes in a _FittingTree object."""
    def __init__(self, node_id, feature, threshold):
        self.feature = feature
        self.threshold = threshold
        self.left_child = -1
        self.right_child = -1
        self.node_id = node_id

    def __iter__(self):
        """Defines parameters needed to populate NODE_DTYPE for Tree.__setstate__ using tuple(_Node)."""
        yield self.left_child
        yield self.right_child
        yield self.feature
        yield self.threshold
        yield 0.0  # Impurity
        yield 0  # n_node_samples
        yield 0.0  # weighted_n_node_samples

# Example

obj1= IsolationForest()
X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0,
                            random_state=0, shuffle=False)
clf = obj1.fit(X, y)

print(clf.predict([[0, 0, 0, 0]]))

# Example

obj2= IsolationForest()
X = [[-1.1], [0.3], [0.5], [100]]
clf = obj2.fit(X)
print(clf.predict([[0.1], [0], [90]]))