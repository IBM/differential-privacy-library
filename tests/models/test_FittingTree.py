from unittest import TestCase

import numpy as np
from sklearn.tree._tree import Tree

from diffprivlib.models.forest import _FittingTree
from diffprivlib.utils import check_random_state


class TestFittingTree(TestCase):
    def test_not_none(self):
        self.assertIsNotNone(_FittingTree)

    def test_no_data(self):
        classes = np.array([0, 1, 2])
        tree = _FittingTree(5, 3, classes, 1, (0, 1), check_random_state(None))
        tree.build()
        tree.values_ = np.zeros(shape=(tree.node_count, 1, len(tree.classes)))
        d = tree.__getstate__()

        sktree = Tree(3, np.array([len(classes)]), 1)
        sktree.__setstate__(d)

    def test_fit_before_build(self):
        tree = _FittingTree(5, 3, [0, 1], 1, (0, 1), check_random_state(None))

        with self.assertRaises(ValueError):
            tree.fit([[1, 1, 1]], [0])
