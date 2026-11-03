"""Validate linear-kernel MMD against distances and empirical kernel means."""
import math
import unittest

import numpy as np
import pandas as pd

from popalign.evaluation.mmd import mmd_linear
from popalign.evaluation import bigfive_metrics, ood_metrics


class MMDTests(unittest.TestCase):
    def test_known_mean_shift_and_symmetry(self):
        x = np.ones((3, 5))
        y = np.tile([3., 2., 1., 1., 1.], (7, 1))
        self.assertAlmostEqual(mmd_linear(x, y), math.sqrt(5))
        self.assertAlmostEqual(mmd_linear(y, x), math.sqrt(5))

    def test_matches_empirical_linear_kernel_expression(self):
        x = np.array([[1., 2.], [3., 1.], [2., 4.]])
        y = np.array([[2., 3.], [0., 1.]])
        # Include all diagonal entries: empirical kernel-mean distance.
        squared = (x @ x.T).mean() + (y @ y.T).mean() - 2 * (x @ y.T).mean()
        self.assertAlmostEqual(mmd_linear(x, y), math.sqrt(squared))

    def test_equal_means_with_different_spread_have_zero_distance(self):
        x = np.array([[1., 1.], [5., 5.]])
        y = np.array([[3., 3.], [3., 3.], [3., 3.]])
        self.assertEqual(mmd_linear(x, y), 0.)
        self.assertEqual(mmd_linear(x, x), 0.)

    def test_uses_all_rows_beyond_previous_sampling_limits(self):
        x = np.zeros((10001, 2))
        x[-1] = [3 * len(x), 4 * len(x)]
        self.assertAlmostEqual(mmd_linear(x, [[0., 0.]]), 5.)

    def test_singletons_are_valid(self):
        self.assertAlmostEqual(mmd_linear([[1., 2.]], [[4., 6.]]), 5.)

    def test_invalid_inputs_are_rejected(self):
        for x, y in [
            ([], [[1.]]), ([[1.]], []), ([1., 2.], [[1., 2.]]),
            ([[1.]], [[1., 2.]]), ([[]], [[]]),
            ([[np.nan]], [[1.]]), ([[1.]], [[np.inf]]),
        ]:
            with self.subTest(x=x, y=y), self.assertRaises(ValueError):
                mmd_linear(x, y)

    def test_both_evaluators_report_linear_mmd(self):
        rng = np.random.default_rng(7)
        x = rng.uniform(1.5, 3.5, (16, 5))
        y = x + [0.3, 0.4, 0., 0., 0.]
        a = pd.DataFrame(x, columns=bigfive_metrics.TRAITS)
        b = pd.DataFrame(y, columns=bigfive_metrics.TRAITS)
        self.assertAlmostEqual(bigfive_metrics.compute_metrics(a, b)['MMD'], 0.5)
        ood, _ = ood_metrics.compute_metrics(a, b, use_joint=False)
        self.assertAlmostEqual(ood['MMD'], 0.5)


if __name__ == '__main__':
    unittest.main()
