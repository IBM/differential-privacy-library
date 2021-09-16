import numpy as np
from unittest import TestCase

from diffprivlib.models.tree import RandomForestClassifier, DecisionTreeClassifier, get_cat_features, get_feature_domains, calc_tree_depth
from diffprivlib.utils import PrivacyLeakWarning, global_seed, BudgetError


class TestRandomForestClassifier(TestCase):
    def setUp(self):
        global_seed(2718281828)
    
    def test_not_none(self):
        self.assertIsNotNone(RandomForestClassifier)

    def test_bad_params(self):
        X = [[1]]
        y = [0]

        with self.assertRaises(ValueError):
            RandomForestClassifier(n_estimators="10").fit(X, y)
        
        with self.assertRaises(ValueError):
            RandomForestClassifier(cat_feature_threshold="5").fit(X, y)

    def test_bad_data(self):
        with self.assertRaises(ValueError):
            RandomForestClassifier().fit([[1]], None)

        with self.assertRaises(ValueError):
            RandomForestClassifier().fit([[1], [2]], [1])

        with self.assertRaises(ValueError):
            RandomForestClassifier().fit([[1], [2]], [[1, 2], [2, 4]])

        with self.assertRaises(ValueError):
            RandomForestClassifier().fit([['foo', 'bar']], [1])

        with self.assertRaises(ValueError):
            RandomForestClassifier().fit([[1, 2], [3, 4]], [1, 0])

    def test_simple(self):
        X = np.array([[12, 3, 14], [12, 3, 4], [12, 3, 4], [2, 13, 4], [2, 13, 14], [2, 3, 14], [3, 5, 15]])
        y = np.array([1, 1, 1, 0, 0, 0, 1])
        model = RandomForestClassifier(epsilon=2, n_estimators=5, random_state=2021, cat_feature_threshold=2)
        self.assertFalse(model.check_is_fitted())
        # when `feature_domains` is not provided, we should get a privacy leakage warning
        with self.assertWarns(PrivacyLeakWarning):
            model.fit(X, y)
        self.assertTrue(model.check_is_fitted())
        self.assertEqual(model.n_features, 3)
        self.assertEqual(model.n_classes, 2)
        self.assertEqual(set(model.classes_), set([0, 1]))
        self.assertEqual(model.cat_features, [])
        self.assertEqual(model.max_depth, 3)
        self.assertTrue(model.estimators)
        self.assertEqual(len(model._estimators), 5)
        self.assertTrue(model.predict(np.array([[12, 3, 14]])))
        self.assertEqual(model.feature_domains, {'0': [2.0, 12.0], '1': [3.0, 13.0], '2': [4.0, 15.0]})

    def test_with_feature_domains(self):
        X = np.array([[12, 3, 14], [12, 3, 4], [12, 3, 4], [2, 13, 4], [2, 13, 14], [2, 3, 14], [3, 5, 15]])
        y = np.array([1, 1, 1, 0, 0, 0, 1])
        model = RandomForestClassifier(epsilon=2, n_estimators=5, random_state=2021, cat_feature_threshold=2,
                                       feature_domains={'0': [2.0, 12.0], '1': [3.0, 13.0], '2': [4.0, 15.0]})
        self.assertFalse(model.check_is_fitted())
        model.fit(X, y)
        self.assertTrue(model.check_is_fitted())
        self.assertEqual(model.n_features, 3)
        self.assertEqual(model.n_classes, 2)
        self.assertEqual(set(model.classes_), set([0, 1]))
        self.assertEqual(model.cat_features, [])
        self.assertEqual(model.max_depth, 3)
        self.assertTrue(model.estimators)
        self.assertEqual(len(model._estimators), 5)
        self.assertTrue(model.predict(np.array([[12, 3, 14]])))

    def test_with_not_enough_feature_domains(self):
        X = np.array([[12, 3, 14], [12, 3, 4], [12, 3, 4], [2, 13, 4], [2, 13, 14], [2, 3, 14], [3, 5, 15]])
        y = np.array([1, 1, 1, 0, 0, 0, 1])
        model = RandomForestClassifier(epsilon=2, n_estimators=5, random_state=2021, feature_domains={'0': [2.0, 12.0], '1': [3.0, 13.0]})
        self.assertFalse(model.check_is_fitted())
        with self.assertRaises(ValueError):
            model.fit(X, y)

    def test_accountant(self):
        from diffprivlib.accountant import BudgetAccountant
        acc = BudgetAccountant()

        X = np.array([[12, 3, 14], [12, 3, 4], [12, 3, 4], [2, 13, 4], [2, 13, 14], [2, 3, 14], [3, 5, 15]])
        y = np.array([1, 1, 1, 0, 0, 0, 1])
        model = RandomForestClassifier(epsilon=2, n_estimators=5, accountant=acc)
        model.fit(X, y)
        self.assertEqual((2, 0), acc.total())
        
        with BudgetAccountant(3, 0) as acc2:
            model = RandomForestClassifier(epsilon=2, n_estimators=5)
            model.fit(X, y)
            self.assertEqual((2, 0), acc2.total())

            with self.assertRaises(BudgetError):
                model.fit(X, y)


class TestDecisionTreeClassifier(TestCase):
    def setUp(self):
        global_seed(2718281828)

    def test_not_none(self):
        self.assertIsNotNone(DecisionTreeClassifier)

    def test_bad_params(self):
        X = [[1]]
        y = [0]
        
        with self.assertRaises(ValueError):
            DecisionTreeClassifier(cat_feature_threshold="5").fit(X, y)

    def test_bad_data(self):
        with self.assertRaises(ValueError):
            DecisionTreeClassifier().fit([[1]], None)

        with self.assertRaises(ValueError):
            DecisionTreeClassifier().fit([[1], [2]], [1])

        with self.assertRaises(ValueError):
            DecisionTreeClassifier().fit([[1], [2]], [[1, 2], [2, 4]])

        with self.assertRaises(ValueError):
            DecisionTreeClassifier().fit([['foo', 'bar']], [1])

    def test_simple(self):
        X = np.array([[12, 3, 14], [12, 3, 4], [12, 3, 4], [2, 13, 4], [2, 13, 14], [2, 3, 14], [3, 5, 15]])
        y = np.array([1, 1, 1, 0, 0, 0, 1])
        model = DecisionTreeClassifier(epsilon=2, cat_feature_threshold=2)
        self.assertFalse(model.check_is_fitted())
        # when `feature_domains` is not provided, we should get a privacy leakage warning
        with self.assertWarns(PrivacyLeakWarning):
            model.fit(X, y)
        self.assertTrue(model.check_is_fitted())
        self.assertTrue(model.predict(np.array([[12, 3, 14]])))

    def test_with_feature_domains(self):
        X = np.array([[12, 3, 14], [12, 3, 4], [12, 3, 4], [2, 13, 4], [2, 13, 14], [2, 3, 14], [3, 5, 15]])
        y = np.array([1, 1, 1, 0, 0, 0, 1])
        model = DecisionTreeClassifier(epsilon=2, cat_feature_threshold=2, feature_domains={'0': [2.0, 12.0], '1': [3.0, 13.0], '2': [4.0, 15.0]})
        self.assertFalse(model.check_is_fitted())
        model.fit(X, y)
        self.assertTrue(model.check_is_fitted())
        self.assertTrue(model.predict(np.array([[12, 3, 14]])))

class TestUtils(TestCase):
    def test_calc_tree_depth(self):
        self.assertEqual(calc_tree_depth(0, 4), 2)
        self.assertEqual(calc_tree_depth(0, 100, max_depth=20), 20)
        self.assertEqual(calc_tree_depth(4, 5), 6)
        self.assertEqual(calc_tree_depth(4, 5, max_depth=3), 3)
        self.assertEqual(calc_tree_depth(40, 50), 15)

    def test_get_feature_domains(self):
        X = np.array([[12, 3, 14, 21], [0.1, 0.5, 0.7, 1], ['cat', 'dog', 'mouse', 'cat'], [0, 1, 0, 1]]).T
        cat_features = [2, 3]
        feature_domains = get_feature_domains(X, cat_features)
        self.assertEqual(feature_domains['0'], [3, 21])
        self.assertEqual(feature_domains['1'], [0.1, 1])
        self.assertEqual(set(feature_domains['2']), set(['dog', 'cat', 'mouse']))
        self.assertEqual(set(feature_domains['3']), set(['0', '1']))
    
    def test_get_cat_features(self):
        X = np.array([[12, 3, 14, 21], [0.1, 0.5, 0.7, 1], ['cat', 'dog', 'mouse', 'cat'], [0, 1, 0, 1]]).T
        cat_features = get_cat_features(X)
        self.assertEqual(cat_features, [3])
        cat_features = get_cat_features(X, feature_threshold=3)
        self.assertEqual(cat_features, [2, 3])