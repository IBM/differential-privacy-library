# MIT License
#
# Copyright (C) IBM Corporation 2021
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
Random Forest Classifier with Differential Privacy
"""
from collections import namedtuple
import warnings

from joblib import Parallel, delayed
import numpy as np
from sklearn.exceptions import DataConversionWarning
from sklearn.tree._tree import Tree, DOUBLE, DTYPE, NODE_DTYPE  # pylint: disable=no-name-in-module
from sklearn.ensemble._forest import RandomForestClassifier as skRandomForestClassifier, _parallel_build_trees
from sklearn.tree import DecisionTreeClassifier as skDecisionTreeClassifier

from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import PrivacyLeakWarning, check_random_state
from diffprivlib.mechanisms import PermuteAndFlip
from diffprivlib.validation import DiffprivlibMixin

MAX_INT = np.iinfo(np.int32).max


class RandomForestClassifier(skRandomForestClassifier, DiffprivlibMixin):  # pylint: disable=too-many-ancestors
    r"""Random Forest Classifier with differential privacy.

    This class implements Differentially Private Random Decision Forests using [1].
    :math:`\epsilon`-Differential privacy is achieved by constructing decision trees via random splitting criterion and
    applying the :class:`.PermuteAndFlip` Mechanism to determine a noisy label.

    Parameters
    ----------
    n_estimators : int, default: 10
        The number of trees in the forest.

    epsilon : float, default: 1.0
        Privacy parameter :math:`\epsilon`.

    bounds :  tuple, optional
        Bounds of the data, provided as a tuple of the form (min, max).  `min` and `max` can either be scalars, covering
        the min/max of the entire data, or vectors with one entry per feature.  If not provided, the bounds are computed
        on the data when ``.fit()`` is first called, resulting in a :class:`.PrivacyLeakWarning`.

    classes : array-like of shape (n_classes,)
        Array of classes to be trained on.  If not provided, the classes will be read from the data when ``.fit()`` is
        first called, resulting in a :class:`.PrivacyLeakWarning`.

    n_jobs : int, default: 1
        Number of CPU cores used when parallelising over classes. ``-1`` means using all processors.

    verbose : int, default: 0
        Set to any positive number for verbosity.

    random_state : int or RandomState, optional
        Controls both the randomness of the shuffling of the samples used when building trees (if ``shuffle=True``) and
        training of the differentially-private :class:`.DecisionTreeClassifier` to construct the forest.  To obtain a
        deterministic behaviour during randomisation, ``random_state`` has to be fixed to an integer.

    accountant : BudgetAccountant, optional
        Accountant to keep track of privacy budget.

    max_depth : int, default: 5
        The maximum depth of the tree.  The depth translates to an exponential increase in memory usage.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit and add more estimators to the ensemble,
        otherwise, just fit a whole new forest.

    shuffle : bool, default=False
        When set to ``True``, shuffles the datapoints to be trained on trees at random.  In diffprivlib, each datapoint
        is used to train exactly one tree. When set to ``False``, datapoints are chosen in-order to their tree in
        sequence.

    Attributes
    ----------
    base_estimator_ : DecisionTreeClassifier
        The child estimator template used to create the collection of fitted sub-estimators.

    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    classes_ : ndarray of shape (n_classes,) or a list of such arrays
        The classes labels.

    n_classes_ : int or list
        The number of classes.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X` has feature names that are all strings.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from diffprivlib.models import RandomForestClassifier
    >>> X, y = make_classification(n_samples=1000, n_features=4,
    ...                            n_informative=2, n_redundant=0,
    ...                            random_state=0, shuffle=False)
    >>> clf = RandomForestClassifier(n_estimators=100, random_state=0)
    >>> clf.fit(X, y)
    >>> print(clf.predict([[0, 0, 0, 0]]))
    [1]

    References
    ----------
    [1] Sam Fletcher, Md Zahidul Islam. "Differentially Private Random Decision Forests using Smooth Sensitivity"
    https://arxiv.org/abs/1606.03572

    """

    def __init__(self, n_estimators=10, *, epsilon=1.0, bounds=None, classes=None, n_jobs=1, verbose=0, accountant=None,
                 random_state=None, max_depth=5, warm_start=False, shuffle=False, **unused_args):
        super().__init__(
            n_estimators=n_estimators,
            criterion=None,
            bootstrap=False,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)
        self.epsilon = epsilon
        self.bounds = bounds
        self.classes = classes
        self.max_depth = max_depth
        self.shuffle = shuffle
        self.accountant = BudgetAccountant.load_default(accountant)

        self.base_estimator = DecisionTreeClassifier()
        self.estimator_params = ("max_depth", "epsilon", "bounds", "classes")

        self._warn_unused_args(unused_args)

    def fit(self, X, y, sample_weight=None):
        """
        Build a forest of trees from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted to ``dtype=np.float32``.

        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in regression).

        sample_weight : ignored
            Ignored by diffprivlib.  Present for consistency with sklearn API.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.accountant.check(self.epsilon, 0)

        if sample_weight is not None:
            self._warn_unused_args("sample_weight")

        # Validate or convert input data
        X, y = self._validate_data(X, y, multi_output=False, dtype=DTYPE)

        if self.bounds is None:
            warnings.warn("Bounds have not been specified and will be calculated on the data provided. This will "
                          "result in additional privacy leakage. To ensure differential privacy and no additional "
                          "privacy leakage, specify bounds for each dimension.", PrivacyLeakWarning)
            self.bounds = (np.min(X, axis=0), np.max(X, axis=0))
        self.bounds = self._check_bounds(self.bounds, shape=X.shape[1])
        X = self._clip_to_bounds(X, self.bounds)

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warnings.warn("A column-vector y was passed when a 1d array was expected. Please change the shape of y to "
                          "(n_samples,), for example using ravel().", DataConversionWarning, stacklevel=2)

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        if self.classes is None:
            warnings.warn("Classes have not been specified and will be calculated on the data provided. This will "
                          "result in additional privacy leakage. To ensure differential privacy and no additional "
                          "privacy leakage, specify the prediction classes for model.", PrivacyLeakWarning)
            self.classes = np.unique(y)
        self.classes_ = np.ravel(self.classes)
        self.n_classes_ = len(self.classes_)

        # y, expanded_class_weight = self._validate_y_class_weight(y)
        y = np.searchsorted(self.classes_, y)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        # Check parameters
        self._validate_estimator()

        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError(f"n_estimators={self.n_estimators} must be larger or equal to len(estimators_)="
                             f"{len(self.estimators_)} when warm_start==True")
        if n_more_estimators == 0:
            warnings.warn("Warm-start fitting without increasing n_estimators does not fit new trees.")
            return self

        if self.warm_start and len(self.estimators_) > 0:
            # We draw from the random state to get the random state we
            # would have got if we hadn't used a warm_start.
            random_state.randint(MAX_INT, size=len(self.estimators_))

        trees = [
            self._make_estimator(append=False, random_state=random_state)
            for _ in range(n_more_estimators)
        ]

        # Split samples between trees as evenly as possible (randomly if shuffle==True)
        n_samples = X.shape[0]
        tree_idxs = random_state.permutation(n_samples) if self.shuffle else np.arange(n_samples)
        tree_idxs = (tree_idxs // (n_samples / n_more_estimators)).astype(int)

        # Parallel loop: we prefer the threading backend as the Cython code
        # for fitting the trees is internally releasing the Python GIL
        # making threading more efficient than multiprocessing in
        # that case. However, for joblib 0.12+ we respect any
        # parallel_backend contexts set at a higher level,
        # since correctness does not rely on using threads.
        try:
            trees = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, prefer="threads")(
                delayed(_parallel_build_trees)(
                    tree=t,
                    bootstrap=False,
                    X=X[tree_idxs == i],
                    y=y[tree_idxs == i],
                    sample_weight=None,
                    tree_idx=i,
                    n_trees=len(trees),
                    verbose=self.verbose,
                )
                for i, t in enumerate(trees)
            )
        except TypeError:
            trees = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, prefer="threads")(
                delayed(_parallel_build_trees)(
                    tree=t,
                    forest=self,
                    X=X[tree_idxs == i],
                    y=y[tree_idxs == i],
                    sample_weight=None,
                    tree_idx=i,
                    n_trees=len(trees),
                    verbose=self.verbose,
                )
                for i, t in enumerate(trees)
            )

        # Collect newly grown trees
        self.estimators_.extend(trees)

        self.accountant.spend(self.epsilon, 0)

        return self


class DecisionTreeClassifier(skDecisionTreeClassifier, DiffprivlibMixin):
    r"""Decision Tree Classifier with differential privacy.

    This class implements the base differentially private decision tree classifier
    for the Random Forest classifier algorithm. Not meant to be used separately.

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

    def __init__(self, max_depth=5, *, epsilon=1, bounds=None, classes=None, random_state=None, accountant=None,
                 **unused_args):
        # TODO: Remove try...except when sklearn v1.0 is min-requirement
        try:
            super().__init__(  # pylint: disable=unexpected-keyword-arg
                criterion=None,
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
                criterion=None,
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
