from unittest import TestCase

from diffprivlib.models.utils import _check_bounds


class TestGaussianNB(TestCase):
    def test_none(self):
        self.assertIsNone(_check_bounds(None))

    def test_non_tuple(self):
        with self.assertRaises(TypeError):
            _check_bounds([1, 2, 3])

    def test_incorrect_entries(self):
        with self.assertRaises(ValueError):
            _check_bounds(([1, 2], 1))

        with self.assertRaises(ValueError):
            _check_bounds(([1, 2], [1, 2, 3]))

        with self.assertRaises(ValueError):
            _check_bounds(([1, 2], [1, 2], [1, 2]))

    def test_consistency(self):
        bounds = _check_bounds(([1, 1], [2, 2]), shape=2)
        bounds2 = _check_bounds(bounds, shape=2)
        self.assertTrue((bounds[0] == bounds2[0]).all())
        self.assertTrue((bounds[1] == bounds2[1]).all())

    def test_array_output(self):
        import numpy as np
        bounds = _check_bounds((1, 2))
        self.assertIsInstance(bounds[0], np.ndarray)
        self.assertIsInstance(bounds[1], np.ndarray)

    def test_wrong_dims(self):
        with self.assertRaises(ValueError):
            _check_bounds(([1, 1], [2, 2]), shape=3)

    def test_wrong_order(self):
        with self.assertRaises(ValueError):
            _check_bounds((2, 1))

    def test_non_numeric(self):
        with self.assertRaises(Exception):
            _check_bounds(("1", "2"))

    def test_min_separation(self):
        bounds = _check_bounds((1, 1), min_separation=2)
        self.assertEqual(0, bounds[0])
        self.assertEqual(2, bounds[1])

        bounds = _check_bounds((1., 1.), min_separation=1)
        self.assertEqual(0.5, bounds[0])
        self.assertEqual(1.5, bounds[1])

        bounds = _check_bounds((0.9, 1.1), min_separation=1)
        self.assertEqual(0.5, bounds[0])
        self.assertEqual(1.5, bounds[1])
