from unittest import TestCase
from sklearn.dummy import DummyClassifier

from diffprivlib.models import OneVsRestClassifier

class TestOneVsRestClassifier(TestCase):
    def test_not_none(self):
        self.assertIsNotNone(OneVsRestClassifier)

    def test_no_params(self):
        X = [[1]]
        y = [0]
        clf = OneVsRestClassifier(DummyClassifier(), classes=[0])
        clf.fit(X, y)
