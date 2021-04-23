import numbers
import numpy as np
from joblib import Parallel, delayed
from collections import defaultdict, Counter, namedtuple

from scipy import stats
from sklearn.utils import check_X_y, check_array
from sklearn.utils.fixes import _joblib_parallel_args

from diffprivlib.accountant import BudgetAccountant
from diffprivlib.utils import warn_unused_args

Dataset = namedtuple('Dataset', ['X', 'y'])


class RandomForestClassifier:
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

    Attributes
    ----------
    n_features: int
        The number of features when fit is performed.

    n_classes: int
        The number of classes.

    classes_: array of shape (n_classes, )
        The classes labels.

    cat_features: array of categorical feature indexes
        Categorical feature indexes.

    max_depth: int
        Final max depth used for constructing decision trees.

    estimators: list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from diffprivlib.models.tree import RandomForestClassifier
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

    def __init__(self, n_estimators=10, epsilon=1.0, cat_feature_threshold=10,
                 n_jobs=1, verbose=0, accountant=None, max_depth=15, random_state=None, **unused_args):
        self._n_estimators = n_estimators
        self._n_jobs = n_jobs
        self._epsilon = epsilon
        self._cat_feature_threshold = cat_feature_threshold
        self._random_state = random_state
        self._estimators = []
        self._verbose = verbose
        self._max_depth = max_depth
        self._accountant = BudgetAccountant.load_default(accountant)
        self._n_features = None
        self._classes = None
        self._cat_features = None

        if random_state is not None:
            np.random.seed(random_state)

        warn_unused_args(unused_args)

    @property
    def n_features(self):
        return self._n_features

    @property
    def n_classes(self):
        return self._n_classes

    @property
    def classes_(self):
        return self._classes

    @property
    def cat_features(self):
        return self._cat_features

    @property
    def max_depth(self):
        return self._max_depth

    @property
    def estimators(self):
        return self._estimators

    def fit(self, X, y):
        """Fit the model to the given training data.

        Parameters:
            X : array-like, shape (n_samples, n_features)
                Training vector, where n_samples is the number of samples and n_features is the number of features.

            y : array-like, shape (n_samples,)
                Target vector relative to X.

        Returns:
            self: class object
        """
        if not isinstance(self._n_estimators, numbers.Integral) or self._n_estimators < 0:
            raise ValueError(f'Number of estimators should be a positive integer; got {self._n_estimators}')

        if not isinstance(self._cat_feature_threshold, numbers.Integral) or self._cat_feature_threshold < 0:
            raise ValueError('Categorical feature threshold should be a positive integer;'
                             f'got {self._cat_feature_threshold}')

        self._accountant.check(self._epsilon, 0)

        X, y = np.array(X), np.array(y)
        check_X_y(X, y)

        self._n_features = X.shape[1]
        self._cat_features = get_cat_features(X, self._cat_feature_threshold)
        self._max_depth = calc_tree_depth(n_cont_features=self._n_features-len(self._cat_features),
                                          n_cat_features=len(self._cat_features), max_depth=self._max_depth)
        self._feature_domains = get_feature_domains(X, self._cat_features)
        features = list(range(self._n_features))
        self._classes = np.unique(y)
        self._n_classes = len(self._classes)

        if self._n_estimators > len(X):
            raise ValueError('Number of estimators is more than the available samples')

        subset_size = int(len(X) / self._n_estimators)
        datasets = []
        estimators = []

        for i in range(self._n_estimators):
            estimator = DecisionTreeClassifier(max_depth=self._max_depth, epsilon=self._epsilon)
            estimator.set_cat_features(self._cat_features)
            estimator.set_feature_domains(self._feature_domains)
            estimator.set_classes(self._classes)
            estimators.append(estimator)
            datasets.append(Dataset(X=X[i*subset_size:(i+1)*subset_size], y=y[i*subset_size:(i+1)*subset_size]))

        estimators = Parallel(n_jobs=self._n_jobs, verbose=self._verbose, **_joblib_parallel_args(prefer='processes'))(
            delayed(lambda estimator, X, y: estimator.fit(X, y))(estimator, dataset.X, dataset.y)
            for estimator, dataset in zip(estimators, datasets)
        )

        self._estimators = estimators
        self._accountant.spend(self._epsilon, 0)

        return self

    def check_is_fitted(self):
        """Check if the model is fitted

        Returns:
            [bool]: True if the model is fitted, False otherwise.
        """
        if self._estimators:
            for estimator in self._estimators:
                if not estimator.check_is_fitted():
                    return False
            return True
        return False

    def predict(self, X):
        """Predict on a given data.

        Parameters:
            X : array-like, shape (n_samples, n_features)
                Training vector, where n_samples is the number of samples and n_features is the number of features.

        Raises:
            Exception: If the model is not fitted before prediction.

        Returns:
            [nd.array]: Numpy array of predictions.
        """
        if not self.check_is_fitted():
            raise Exception('Model is not fitted yet')

        preds = []

        for estimator in self._estimators:
            preds.append(np.array(estimator.predict(X)).reshape(-1, 1))

        y = np.hstack(preds)

        return np.apply_along_axis(lambda x: Counter(list(x)).most_common(1)[0][0], arr=y, axis=1)


class DecisionTreeClassifier:
    r"""Decision Tree Classifier with differential privacy.

    This class implements the base differentially private decision tree classifier
    for the Random Forest classifier algorithm. Not meant to be used separately.

    Parameters
    ----------
    epsilon: float, default: 1.0
        Privacy parameter :math:`\epsilon`.

    cat_feature_threshold: int, default: 10
        Threshold value used to determine categorical features. For example, value of ``10`` means
        any feature that has less than or equal to 10 unique values will be treated as a categorical feature.

    max_depth: int, default: 15
        The maximum depth of the tree.

    random_state: float, optional
        Sets the numpy random seed.

    """
    def __init__(self, cat_feature_threshold=10, max_depth=15, epsilon=1, random_state=None):
        self._feature_domains = None
        self._cat_features = None
        self._classes = None
        self._cat_feature_threshold = cat_feature_threshold
        self._max_depth = max_depth
        self._epsilon = epsilon
        self._root = None
        self._fitted = False

        if random_state is not None:
            np.random.seed(random_state)

    def set_feature_domains(self, feature_domains):
        self._feature_domains = feature_domains

    def set_cat_features(self, cat_features):
        self._cat_features = cat_features

    def set_classes(self, classes):
        self._classes = classes

    def _build(self, features, feature_domains, current_depth=1):
        if not features or current_depth >= self._max_depth+1:
            return DecisionNode(level=current_depth)

        split_feature = np.random.choice(features)
        node = DecisionNode(level=current_depth, split_feature=split_feature)

        if split_feature in self._cat_features:
            node.set_split_type(DecisionNode.CAT_SPLIT)
            for value in feature_domains[str(split_feature)]:
                child_node = self._build([f for f in features if f != split_feature], feature_domains, current_depth+1)
                node.add_cat_child(value, child_node)
        else:
            node.set_split_type(DecisionNode.CONT_SPLIT)
            split_value = np.random.uniform(feature_domains[str(split_feature)][0],
                                            feature_domains[str(split_feature)][1])
            node.set_split_value(split_value)
            left_domain = {k: v if k != str(split_feature) else [v[0], split_value]
                           for k, v in feature_domains.items()}
            right_domain = {k: v if k != str(split_feature) else [split_value, v[1]]
                            for k, v in feature_domains.items()}
            left_child = self._build(features, left_domain, current_depth+1)
            right_child = self._build(features, right_domain, current_depth+1)
            node.set_left_child(left_child)
            node.set_right_child(right_child)

        return node

    def fit(self, X, y):
        if not isinstance(self._cat_feature_threshold, numbers.Integral) or self._cat_feature_threshold < 0:
            raise ValueError('Categorical feature threshold should be a positive integer;'
                             f'got {self._cat_feature_threshold}')

        X, y = np.array(X), np.array(y)
        check_X_y(X, y)

        if self._cat_features is None:
            self._cat_features = get_cat_features(X, self._cat_feature_threshold)

        if self._feature_domains is None:
            self._feature_domains = get_feature_domains(X, self._cat_features)

        if self._classes is None:
            self._classes = np.unique(y)

        n_features = X.shape[1]
        features = list(range(n_features))

        self._root = self._build(features, self._feature_domains)

        for i in range(len(X)):
            node = self._root.classify(X[i])
            node.update_class_count(y[i])

        self._root.set_noisy_label(self._epsilon, self._classes)
        self._fitted = True

        return self

    def predict(self, X):
        if not self.check_is_fitted():
            raise Exception('Model is not fitted yet')

        y = []
        X = np.array(X)
        check_array(X)

        for x in X:
            node = self._root.classify(x)
            y.append(node.noisy_label)

        return np.array(y)

    def check_is_fitted(self):
        return self._fitted


class DecisionNode:
    """Base Decision Node
    """
    CONT_SPLIT = 0
    CAT_SPLIT = 1

    def __init__(self, level, split_feature=None, split_value=None, split_type=None):
        self._level = level
        self._split_type = split_type
        self._split_feature = split_feature
        self._split_value = split_value
        self._left_child = None
        self._right_child = None
        self._cat_children = {}
        self._class_counts = defaultdict(int)
        self._noisy_label = None
        self._sensitivity = -1.0

    @property
    def noisy_label(self):
        return self._noisy_label

    def set_split_value(self, split_value):
        self._split_value = split_value

    def set_split_type(self, split_type):
        self._split_type = split_type

    def set_left_child(self, node):
        self._left_child = node

    def set_right_child(self, node):
        self._right_child = node

    def add_cat_child(self, cat_value, node):
        self._cat_children[str(cat_value)] = node

    def is_leaf(self):
        return not self._left_child and not self._right_child and not self._cat_children

    def update_class_count(self, class_value):
        self._class_counts[class_value] += 1

    def classify(self, x):
        if self.is_leaf():
            return self

        child = None

        if self._split_type == self.CAT_SPLIT:
            x_val = str(x[self._split_feature])
            child = self._cat_children.get(x_val)
        else:
            x_val = x[self._split_feature]
            if x_val < self._split_value:
                child = self._left_child
            else:
                child = self._right_child

        if child is None:
            return self

        return child.classify(x)

    def set_noisy_label(self, epsilon, class_values):
        if self.is_leaf():
            if not self._noisy_label:
                for val in class_values:
                    if val not in self._class_counts:
                        self._class_counts[val] = 0

                if max([v for k, v in self._class_counts.items()]) < 1:
                    self._noisy_label = np.random.choice([k for k, v in self._class_counts.items()])
                else:
                    all_counts = sorted([v for k, v in self._class_counts.items()], reverse=True)
                    count_difference = all_counts[0] - all_counts[1]
                    self._sensitivity = np.exp(-1 * count_difference * epsilon)
                    self._noisy_label = self._calc_label_by_expo_mech(epsilon, self._sensitivity, self._class_counts)
        else:
            if self._left_child:
                self._left_child.set_noisy_label(epsilon, class_values)
            if self._right_child:
                self._right_child.set_noisy_label(epsilon, class_values)
            for child_node in self._cat_children.values():
                child_node.set_noisy_label(epsilon, class_values)

    def _calc_label_by_expo_mech(self, epsilon, sensitivity, counts):
        ''' For this implementation of the Exponetial Mechanism, we use a piecewise linear scoring function,
        where the element with the maximum count has a score of 1, and all other elements have a score of 0. '''
        weighted = []
        max_count = max([v for k, v in counts.items()])

        for label, count in counts.items():
            ''' if the score is non-monotonic, sensitivity needs to be multiplied by 2 '''
            if count == max_count:
                if sensitivity < 1.0e-10:
                    power = 50   # e^50 is already astronomical. sizes beyond that dont matter
                else:
                    power = min(50, (epsilon*1)/(2*sensitivity))   # score = 1
            else:
                power = 0   # score = 0
            weighted.append([label, np.exp(power)])

        count_sum = 0.
        for label, count in weighted:
            count_sum += count
        for i in range(len(weighted)):
            weighted[i][1] /= count_sum

        custom_dist = stats.rv_discrete(name='customDist',
                                        values=([label for label, count in weighted],
                                                [count for label, count in weighted]))
        best = custom_dist.rvs()

        return int(best)


def get_feature_domains(X, cat_features):
    """Calculate feature domains from the data.

    Parameters:
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and n_features is the number of features.

        cat_features : array of integers
            List of categorical feature indexes

    Returns:
        [dict]: Dictionary with keys as feature indexes and values as feature domains.
    """
    feature_domains = {}
    X_t = np.transpose(X)
    cont_features = list(set(range(X.shape[1])) - set(cat_features))

    for i in cat_features:
        feature_domains[str(i)] = [str(x) for x in set(X_t[i])]

    for i in cont_features:
        vals = [float(x) for x in X_t[i]]
        feature_domains[str(i)] = [min(vals), max(vals)]

    return feature_domains


def get_cat_features(X, feature_threshold=2):
    """Determine categorical features

    Parameters:
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and n_features is the number of features.

        feature_threshold: int, defaults to 2.
            Threshold value used to determine categorical features. For example, value of ``10`` means
            any feature that has less than or equal to 10 unique values will be treated as a categorical feature.

    Returns:
        [list]: List of categorical feature indexes
    """
    n_features = X.shape[1]
    cat_features = []

    for i in range(n_features):
        values = set(X[:, i])
        if len(values) <= feature_threshold:
            cat_features.append(i)

    return cat_features


def calc_tree_depth(n_cont_features, n_cat_features, max_depth=15):
    """Calculate tree depth

    Args:
        n_cont_features (int): Number of continuous features
        n_cat_features ([type]): Number of categorical features
        max_depth (int, optional): Max depth tree. Defaults to 15.

    Returns:
        [int]: Final depth tree
    """
    if n_cont_features < 1:
        return min(max_depth, np.floor(n_cat_features / 2.))
    else:
        ''' Designed using balls-in-bins probability. See the paper for details. '''
        m = float(n_cont_features)
        depth = 0
        expected_empty = m   # the number of unique attributes not selected so far
        while expected_empty > m / 2.:   # repeat until we have less than half the attributes being empty
            expected_empty = m * ((m - 1.) / m) ** depth
            depth += 1
        # the above was only for half the numerical attributes. now add half the categorical attributes
        final_depth = np.floor(depth + (n_cat_features / 2.))
        return min(max_depth, final_depth)
