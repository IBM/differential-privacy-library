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
from collections import defaultdict, namedtuple
import numbers
import warnings
from joblib import Parallel, delayed
import numpy as np

from sklearn.utils import check_array
from sklearn.ensemble._forest import ForestClassifier
from sklearn.tree import DecisionTreeClassifier as BaseDecisionTreeClassifier

from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import PrivacyLeakWarning
from diffprivlib.mechanisms import PermuteAndFlip
from diffprivlib.validation import DiffprivlibMixin

Dataset = namedtuple('Dataset', ['X', 'y'])


class RandomForestClassifier(ForestClassifier, DiffprivlibMixin):
    r"""Random Forest Classifier with differential privacy.

    This class implements Differentially Private Random Decision Forests using Smooth Sensitivity [1].
    :math:`\epsilon`-Differential privacy is achieved by constructing decision trees via random splitting criterion and
    applying Exponential Mechanism to produce a noisy label.

    Parameters
    ----------
    n_estimators: int, default: 10
        The number of trees in the forest.

    epsilon: float, default: 1.0
        Privacy parameter :math:`\epsilon`.

    cat_feature_threshold: int, default: 10
        Threshold value used to determine categorical features. For example, value of ``10`` means
        any feature that has less than or equal to 10 unique values will be treated as a categorical feature.

    n_jobs : int, default: 1
        Number of CPU cores used when parallelising over classes. ``-1`` means
        using all processors.

    verbose : int, default: 0
        Set to any positive number for verbosity.

    accountant : BudgetAccountant, optional
        Accountant to keep track of privacy budget.

    max_depth: int, default: 15
        The maximum depth of the tree. Final depth of the tree will be calculated based on the number of continuous
        and categorical features, but it wont be more than this number.
        Note: The depth translates to an exponential increase in memory usage.

    random_state: float, optional
        Sets the numpy random seed.

    feature_domains: dict, optional
        A dictionary of domain values for all features where keys are the feature indexes in the training data and
        the values are an array of domain values for categorical features and an array of min and max values for
        continuous features. For example, if the training data is [[2, 'dog'], [5, 'cat'], [7, 'dog']], then
        the feature_domains would be {'0': [2, 7], '1': ['dog', 'cat']}. If not provided, feature domains will
        be constructed from the data, but this will result in :class:`.PrivacyLeakWarning`.

    Attributes
    ----------
    n_features_in_: int
        The number of features when fit is performed.

    n_classes_: int
        The number of classes.

    classes_: array of shape (n_classes, )
        The classes labels.

    cat_features_: array of categorical feature indexes
        Categorical feature indexes.

    max_depth_: int
        Final max depth used for constructing decision trees.

    estimators_: list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    feature_domains_: dictionary of domain values mapped to feature
        indexes in the training data

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

    def __init__(self, n_estimators=10, *, epsilon=1.0, bounds=None, n_jobs=1, verbose=0, accountant=None,
                 max_depth=15, random_state=None, **unused_args):
        super().__init__(
            base_estimator=DecisionTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=("cat_feature_threshold", "max_depth", "epsilon", "random_state"),
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)
        self.epsilon = epsilon
        self.bounds = bounds
        self.max_depth = max_depth
        self.accountant = BudgetAccountant.load_default(accountant)

        if random_state is not None:
            np.random.seed(random_state)

        self._warn_unused_args(unused_args)

    def fit(self, X, y, sample_weight=None):
        """Fit the model to the given training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        sample_weight : ignored
            Ignored by diffprivlib.  Present for consistency with sklearn API.

        Returns
        -------
        self: class

        """
        self.accountant.check(self.epsilon, 0)

        if sample_weight is not None:
            self._warn_unused_args("sample_weight")

        X, y = self._validate_data(X, y)

        if not float(self.n_estimators).is_integer() or self.n_estimators < 0:
            raise ValueError(f'Number of estimators should be a positive integer; got {self.n_estimators}')

        if self.bounds is None:
            warnings.warn("Bounds have not been specified and will be calculated on the data provided. This will "
                          "result in additional privacy leakage. To ensure differential privacy and no additional "
                          "privacy leakage, specify bounds for each dimension.", PrivacyLeakWarning)
            self.bounds = (np.min(X, axis=0), np.max(X, axis=0))
        self.bounds = self._check_bounds(self.bounds, shape=X.shape[1])
        X = self._clip_to_bounds(X, self.bounds)

        self.n_outputs_ = 1
        self.n_features_in_ = X.shape[1]
        self.max_depth_ = calc_tree_depth(n_features=self.n_features_in_, max_depth=self.max_depth)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        if self.n_estimators > len(X):
            raise ValueError('Number of estimators is more than the available samples')

        subset_size = int(len(X) / self.n_estimators)
        datasets = []
        estimators = []

        for i in range(self.n_estimators):
            estimator = DecisionTreeClassifier(max_depth=self.max_depth_,
                                               epsilon=self.epsilon,
                                               bounds=self.bounds,
                                               classes=self.classes_)
            estimators.append(estimator)
            datasets.append(Dataset(X=X[i*subset_size:(i+1)*subset_size], y=y[i*subset_size:(i+1)*subset_size]))

        estimators = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, prefer='processes')(
            delayed(lambda estimator, X, y: estimator.fit(X, y))(estimator, dataset.X, dataset.y)
            for estimator, dataset in zip(estimators, datasets)
        )

        self.estimators_ = estimators
        self.accountant.spend(self.epsilon, 0)
        self.fitted_ = True

        return self


class DecisionTreeClassifier(BaseDecisionTreeClassifier, DiffprivlibMixin):
    r"""Decision Tree Classifier with differential privacy.

    This class implements the base differentially private decision tree classifier
    for the Random Forest classifier algorithm. Not meant to be used separately.

    Parameters
    ----------
    max_depth: int, default: 15
        The maximum depth of the tree.

    epsilon: float, default: 1.0
        Privacy parameter :math:`\epsilon`.

    bounds:  tuple, optional
        Bounds of the data, provided as a tuple of the form (min, max).  `min` and `max` can either be scalars, covering
        the min/max of the entire data, or vectors with one entry per feature.  If not provided, the bounds are computed
        on the data when ``.fit()`` is first called, resulting in a :class:`.PrivacyLeakWarning`.

    classes: array of shape (n_classes_, ), optional
        Array of class labels. If not provided, will be determined from the data.

    random_state: float, optional
        Sets the numpy random seed.

    Attributes
    ----------
    n_features_in_: int
        The number of features when fit is performed.

    n_classes_: int
        The number of classes.

    classes_: array of shape (n_classes, )
        The class labels.

    """
    def __init__(self, max_depth=5, *, epsilon=1, bounds=None, classes=None, random_state=None):
        # TODO: Remove try...except when sklearn v1.0 is min-requirement
        try:
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

        if random_state is not None:
            np.random.seed(random_state)

    def _build(self, features, bounds, current_depth=1):
        if not features or current_depth >= self.max_depth+1:
            return DecisionNode(level=current_depth, classes=self.classes_)

        bounds_lower, bounds_upper = self._check_bounds(bounds, shape=len(features))

        split_feature = np.random.choice(features)
        node = DecisionNode(level=current_depth, classes=self.classes_, split_feature=split_feature)

        split_value = np.random.uniform(bounds_lower[split_feature], bounds_upper[split_feature])
        node.set_split_value(split_value)

        left_bounds_upper = bounds_upper.copy()
        left_bounds_upper[split_feature] = split_value
        right_bounds_lower = bounds_lower.copy()
        right_bounds_lower[split_feature] = split_value

        left_child = self._build(features, (bounds_lower, left_bounds_upper), current_depth + 1)
        right_child = self._build(features, (right_bounds_lower, bounds_upper), current_depth + 1)
        node.set_left_child(left_child)
        node.set_right_child(right_child)

        return node

    def fit(self, X, y, sample_weight=None, check_input=True, X_idx_sorted="deprecated"):
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

        self.classes_ = self.classes

        if self.classes_ is None:
            self.classes_ = np.unique(y)

        self.n_classes_ = len(self.classes_)

        self.n_features_in_ = X.shape[1]
        features = list(range(self.n_features_in_))

        self.tree_ = self._build(features, self.bounds)

        for i, _ in enumerate(X):
            node = self.tree_.classify(X[i])
            node.update_class_count(y[i].item())

        self.tree_.set_noisy_label(self.epsilon, self.classes_)

        return self

    @property
    def n_features_(self):
        return self.n_features_in_

    def _more_tags(self):
        return {}


class DecisionNode:
    """Base Decision Node
    """

    def __init__(self, level, classes, split_feature=None, split_value=None):
        """
        Initialize DecisionNode

        Parameters
        ----------
        level: int
            Node level in the tree

        classes: list
            List of class labels

        split_feature: int
            Split feature index

        split_value: Any
            Feature value to split at

        """
        self._level = level
        self._classes = classes
        self._split_feature = split_feature
        self._split_value = split_value
        self._left_child = None
        self._right_child = None
        self._class_counts = defaultdict(int)
        self._noisy_label = None

    @property
    def noisy_label(self):
        """Get noisy label"""
        return self._noisy_label

    def set_split_value(self, split_value):
        """Set split value"""
        self._split_value = split_value

    def set_left_child(self, node):
        """Set left child of the node"""
        self._left_child = node

    def set_right_child(self, node):
        """Set right child of the node"""
        self._right_child = node

    def is_leaf(self):
        """Check whether the node is leaf node"""
        return not self._left_child and not self._right_child

    def update_class_count(self, class_value):
        """Update the class count for the given class"""
        self._class_counts[class_value] += 1

    def classify(self, x):
        """Classify the given data"""
        if self.is_leaf():
            return self

        x_val = x[self._split_feature]
        if x_val < self._split_value:
            child = self._left_child
        else:
            child = self._right_child

        return child.classify(x)

    def set_noisy_label(self, epsilon, class_values):
        """Set the noisy label for this node"""
        if self.is_leaf():
            if self._noisy_label is None:
                for val in class_values:
                    if val not in self._class_counts:
                        self._class_counts[val] = 0

                utility = list(self._class_counts.values())
                candidates = list(self._class_counts.keys())
                mech = PermuteAndFlip(epsilon=epsilon, sensitivity=1, monotonic=True, utility=utility,
                                      candidates=candidates)
                self._noisy_label = mech.randomise()
        else:
            if self._left_child:
                self._left_child.set_noisy_label(epsilon, class_values)
            if self._right_child:
                self._right_child.set_noisy_label(epsilon, class_values)

    def predict(self, X):
        """Predict using this node"""
        y = []
        X = np.array(X)
        check_array(X)

        for x in X:
            node = self.classify(x)
            proba = np.zeros(len(self._classes))
            proba[np.where(self._classes == node.noisy_label)[0].item()] = 1
            y.append(proba)

        return np.array(y)


def calc_tree_depth(n_features, max_depth=5):
    """Calculate tree depth

    Args:
        n_features (int): Number of features
        max_depth (int, optional): Max depth tree. Defaults to 15.

    Returns:
        [int]: Final depth tree
    """
    # Designed using balls-in-bins probability. See the paper for details.
    m = float(n_features)
    depth = 0
    expected_empty = m   # the number of unique attributes not selected so far
    while expected_empty > m / 2:   # repeat until we have less than half the attributes being empty
        expected_empty = m * ((m - 1) / m) ** depth
        depth += 1
    # the above was only for half the numerical attributes. now add half the categorical attributes
    return min(max_depth, depth)
