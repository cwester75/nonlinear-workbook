import unittest
import numpy as np
from nonlinear_workbook import correlation, logistic_map, lyapunov_map, kantz_curve, hurst_rs, lz_complexity

class ChapterTwoTests(unittest.TestCase):
    def test_correlation_affine_and_constant(self):
        x = np.arange(20.)
        self.assertAlmostEqual(correlation(x, 3*x+7), 1)
        self.assertAlmostEqual(correlation(x, -2*x), -1)
        with self.assertRaises(ValueError): correlation(x, np.ones(20))

    def test_logistic_known_exponents(self):
        self.assertAlmostEqual(lyapunov_map(logistic_map(100000)), np.log(2), delta=0.005)
        self.assertAlmostEqual(lyapunov_map(logistic_map(1000, r=2.5), r=2.5), np.log(0.5), places=10)

    def test_kantz_logistic_growth(self):
        lags, curve, count = kantz_curve(logistic_map(2500), epsilon=0.005, horizon=5)
        self.assertGreater(count, 100)
        self.assertAlmostEqual(np.polyfit(lags[:3], curve[:3], 1)[0], np.log(2), delta=0.15)
        with self.assertRaises(ValueError): kantz_curve(np.ones(50))

    def test_kantz_embedding_and_bounds(self):
        lags, curve, count = kantz_curve(logistic_map(1000), dimension=2, delay=2,
                                         epsilon=0.05, horizon=8, theiler=20)
        self.assertEqual(len(curve), 8)
        self.assertTrue(np.all(np.isfinite(curve)))
        self.assertGreater(count, 0)
        with self.assertRaises(ValueError): kantz_curve([1, 2], dimension=3)
        with self.assertRaises(ValueError): correlation([1, 2], [1, 2, 3])

    def test_hurst_iid_and_constant(self):
        noise = np.random.default_rng(20261010).normal(size=65536)
        h, scales, rs = hurst_rs(noise)
        self.assertAlmostEqual(h, 0.5, delta=0.08)
        self.assertTrue(np.all(rs > 0))
        with self.assertRaises(ValueError): hurst_rs(np.ones(100))

    def test_lz_reference_parses(self):
        self.assertEqual(lz_complexity("0"*100)[0], 2)
        self.assertEqual(lz_complexity("01"*50)[0], 3)
        self.assertEqual(lz_complexity("00011010010001")[0], 6)
        random = np.random.default_rng(42).integers(0, 2, 4000)
        self.assertGreater(lz_complexity(random)[1], lz_complexity("01"*2000)[1])
        with self.assertRaises(ValueError): lz_complexity("")

if __name__ == "__main__": unittest.main()
