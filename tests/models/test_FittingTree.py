from unittest import TestCase

import numpy as np
from sklearn.tree._tree import Tree

from diffprivlib.models.forest import _FittingTree


class TestFittingTree(TestCase):
    def test_not_none(self):
        self.assertIsNotNone(_FittingTree)

    def test_no_data(self):
        classes = np.array([0, 1, 2])
        tree = _FittingTree(5, 3, classes, 1, (0, 1))
        tree.build()
        tree.values_ = np.zeros(shape=(tree.node_count, 1, len(tree.classes)))
        d = tree.__getstate__()

        sktree = Tree(3, np.array([len(classes)]), 1)
        sktree.__setstate__(d)
