import unittest
import numpy as np
from nonlinear_workbook import (delay_embedding, wolf_lyapunov,
    local_lyapunov_spectrum, direct_lyapunov, logistic_map)


def wolf(x, **overrides):
    params = dict(dimension=1, delay=1, theiler=10, min_distance=1e-6,
                  max_distance=0.01, max_angle=0.3, max_steps=4)
    params.update(overrides)
    return wolf_lyapunov(x, **params)


class ResearchTests(unittest.TestCase):
    def test_backward_embedding_and_horizon(self):
        vectors, indices = delay_embedding(np.arange(12.), dimension=3, delay=2, horizon=2)
        np.testing.assert_array_equal(indices, np.arange(4, 10))
        np.testing.assert_array_equal(vectors[0], [4, 2, 0])
        with self.assertRaises(ValueError):
            delay_embedding([1, 2], dimension=3, delay=1)

    def test_wolf_chaotic_reference_and_accounting(self):
        result = wolf(logistic_map(4000))
        # Finite neighborhoods, folds, and replacements: absolute error budget .15.
        self.assertAlmostEqual(result.exponent, np.log(2), delta=0.15)
        self.assertGreater(result.replacements, 10)
        self.assertLessEqual(result.replacements, result.replacement_attempts)
        elapsed = sum(s.steps for s in result.segments)
        growth = sum(np.log(s.final_distance / s.initial_distance) for s in result.segments)
        self.assertAlmostEqual(result.elapsed_time, elapsed)
        self.assertAlmostEqual(result.exponent, growth / elapsed)
        scaled = wolf(logistic_map(4000), sample_interval=2)
        self.assertAlmostEqual(scaled.exponent, result.exponent / 2)

    def test_wolf_controlled_growth_and_no_replacement(self):
        # Two disjoint exact doubling segments; temporal exclusion leaves one pair.
        result = wolf([1., 2., 4., 1.1, 2.2, 4.4], theiler=2,
                      min_distance=.09, max_distance=.15, max_steps=1, max_angle=0)
        self.assertAlmostEqual(result.segments[0].log_growth, np.log(2))
        self.assertEqual(result.segments[0].steps, 1)
        self.assertEqual(result.termination_reason, 'no_replacement_at_boundary')
        with self.assertRaises(ValueError): wolf(np.ones(100))
        with self.assertRaises(ValueError): wolf(logistic_map(50), theiler=100)

    def test_local_linear_orientation_qr_and_units(self):
        # Nonsymmetric Jacobian: transpose mistakes change finite-time QR growth.
        matrix = np.array([[1.01, .03], [0, .99]])
        states = [np.array([.7, 1.3])]
        for _ in range(100): states.append(matrix @ states[-1])
        states = np.asarray(states)
        result = local_lyapunov_spectrum(states, dimension=2, delay=1, theiler=2,
                                         neighbors=12, qr_burn_in=5)
        np.testing.assert_allclose(result.exponents, np.log([1.01, .99]), atol=1e-10)
        self.assertEqual(result.usable_steps, 100)
        self.assertEqual(result.accumulated_steps, 95)
        self.assertEqual(result.running_estimates.shape, (95, 2))
        self.assertTrue(np.all(result.fit_ranks == 2))
        self.assertTrue(np.all(np.isfinite(result.condition_numbers)))
        scaled = local_lyapunov_spectrum(states, dimension=2, delay=1, theiler=2,
                                         neighbors=12, sample_interval=.5)
        np.testing.assert_allclose(scaled.exponents, 2 * np.log([1.01, .99]), atol=1e-10)

    def test_local_rank_failure_and_burn_in(self):
        line = np.column_stack([np.arange(20.), 2*np.arange(20.)])
        with self.assertRaisesRegex(ValueError, 'rank_deficient'):
            local_lyapunov_spectrum(line, dimension=2, delay=1, theiler=0, neighbors=4)
        with self.assertRaises(ValueError):
            local_lyapunov_spectrum(np.ones(30), dimension=1, delay=1, theiler=0, neighbors=4)
        with self.assertRaises(ValueError):
            local_lyapunov_spectrum(logistic_map(30), dimension=1, delay=1, theiler=0,
                                     neighbors=4, qr_burn_in=100)

    def test_direct_estimators(self):
        x = logistic_map(2500)
        for epsilon in (None, .005):
            value, count = direct_lyapunov(x, epsilon=epsilon, theiler=10)
            self.assertAlmostEqual(value, np.log(2), delta=.08)
            self.assertGreater(count, 100)
        with self.assertRaises(ValueError): direct_lyapunov(np.ones(20))

    def test_partial_result_stops_at_rank_failure(self):
        matrix = np.array([[1.01, .03], [0., .99]])
        states = [np.array([.7, 1.3])]
        for _ in range(20): states.append(matrix @ states[-1])
        states.extend([np.array([20. + i, 0.]) for i in range(30)])
        result = local_lyapunov_spectrum(states, dimension=2, delay=1,
                                         theiler=0, neighbors=2)
        self.assertEqual(result.termination_reason, 'rank_deficient_fit')
        self.assertEqual(result.fit_ranks[-1], 1)
        self.assertEqual(len(result.fit_ranks), result.usable_steps + 1)
        self.assertLess(result.usable_steps, len(states) - 1)

    def test_zero_evolved_separation_and_successor_exclusion(self):
        with self.assertRaisesRegex(ValueError, 'No usable evolution'):
            wolf([0., 1., 0.1, 1.], theiler=1, min_distance=.05,
                 max_distance=.2, max_steps=1)
        with self.assertRaises(ValueError):
            local_lyapunov_spectrum(logistic_map(20), dimension=1, delay=1,
                                     theiler=100, neighbors=2)

    def test_invalid_controls(self):
        x = logistic_map(50)
        for override in (dict(max_steps=0), dict(max_angle=-1), dict(min_distance=0),
                         dict(sample_interval=0), dict(dimension=True), dict(theiler=-1)):
            with self.subTest(override=override), self.assertRaises(ValueError): wolf(x, **override)
        for override in (dict(neighbors=0), dict(neighbors=100), dict(delay=0),
                         dict(sample_interval=np.nan), dict(qr_burn_in=-1)):
            params = dict(dimension=1, delay=1, theiler=0, neighbors=4)
            params.update(override)
            with self.subTest(override=override), self.assertRaises(ValueError):
                local_lyapunov_spectrum(x, **params)

if __name__ == '__main__': unittest.main()
