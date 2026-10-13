"""Independent regression checks for numerical boundaries and actual work."""
import unittest
from unittest.mock import patch
import numpy as np
from nonlinear_workbook import chapter12 as c
from nonlinear_workbook.chapter01.diagnostics import monte_carlo


class NumericalRegressionTests(unittest.TestCase):
    def test_decode_extreme_finite_bounds(self):
        with np.errstate(all='raise'):
            self.assertEqual(c.decode_binary([0,0],-1e308,1e308),-1e308)
            self.assertEqual(c.decode_binary([1,1],-1e308,1e308),1e308)
            self.assertAlmostEqual(c.decode_binary([0,1],-1e308,1e308)/1e308,-1/3)
            self.assertAlmostEqual(c.decode_binary([1,0],-1e308,1e308)/1e308,1/3)
            for bounds in ((1e308,1.7e308),(-1.7e308,-1e308)):
                result=c.decode_binary([1,0],*bounds)
                self.assertTrue(np.isfinite(result))
                self.assertTrue(bounds[0] <= result <= bounds[1])

    def test_tsp_counts_every_length_call(self):
        original=np.roll
        for size in (4,5):
            for generations in (0,3):
                for crossover in ('pmx','composition'):
                    # Every tour calculation calls roll exactly once.
                    with patch.object(c.np,'roll',wraps=original) as calls:
                        result=c.tsp_ga(np.ones((4,4))-np.eye(4),population_size=size,
                                      generations=generations,seed=3,crossover=crossover)
                    expected=size+generations*(size+2*(size//2))
                    self.assertEqual(result.evaluations,expected)
                    self.assertEqual(calls.call_count,expected)

    def test_monte_carlo_requires_scalar_outputs(self):
        for method in ('mean','hit_or_miss'):
            for value in ([0,1],[.5],np.array([[.5]]),.5+0j):
                with self.assertRaisesRegex(ValueError,'one real scalar'):
                    monte_carlo(lambda x: value,n=10,seed=1,method=method)
            result=monte_carlo(lambda x: np.array(.5),n=10,seed=1,method=method)
            self.assertTrue(np.isfinite(result.estimate))
        result=monte_carlo(lambda x: .5,n=10,seed=1)
        self.assertEqual(result.estimate,.5)
        self.assertEqual(result.standard_error,0)
