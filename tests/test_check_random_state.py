import secrets
from unittest import TestCase

import numpy as np

from diffprivlib.utils import check_random_state


class TestCheckRandomState(TestCase):
    def test_secure(self):
        rng = secrets.SystemRandom()
        self.assertIs(rng, check_random_state(rng, True))
        self.assertIsNot(rng, check_random_state(None, True))
        self.assertIsInstance(check_random_state(None, True), secrets.SystemRandom)

        self.assertRaises(ValueError, check_random_state, rng, False)

    def test_not_secure(self):
        rng = np.random.mtrand._rand
        self.assertIs(check_random_state(None), rng)
        self.assertIsInstance(check_random_state(1), np.random.RandomState)

        self.assertIsNot(check_random_state(1), rng)

        rng = np.random.RandomState(1)
        self.assertIs(check_random_state(rng), rng)
        self.assertIsNot(check_random_state(1), rng)
        self.assertEqual(rng.random(), check_random_state(1).random())
        self.assertNotEqual(rng.random(), check_random_state(1).random())
