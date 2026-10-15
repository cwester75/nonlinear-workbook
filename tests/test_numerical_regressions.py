"""Independent regression checks for numerical boundaries and actual work."""
import unittest
from unittest.mock import patch
import numpy as np
from nonlinear_workbook import chapter12 as c
from nonlinear_workbook.chapter01.diagnostics import monte_carlo


class NumericalRegressionTests(unittest.TestCase):
    def test_integration_numerical_trials_and_parameter_errors(self):
        from nonlinear_workbook.chapter10 import integrate
        from nonlinear_workbook.chapter04 import toda_rhs
        from nonlinear_workbook.chapter03 import van_der_pol
        for method in ('euler','rkf45','rkf45_adaptive'):
            kw=dict(t_span=(0,1),method=method,step=.1)
            for rhs,state in ((lambda t,y:toda_rhs(t,y,periodic=True),[1000,-1000,0,0,0,0]),
                              (lambda t,y:van_der_pol(t,y,mu=1),[1e200,1e200])):
                result=integrate(rhs,state,**kw)
                self.assertEqual(result.status,'nonfinite')
                self.assertEqual(result.accepted_steps,0)
                self.assertEqual(result.evaluations,1)
                self.assertTrue(np.isfinite(result.states).all())
            def invalid_parameter(t,y): raise ValueError('invalid parameter')
            with self.assertRaisesRegex(ValueError,'invalid parameter'):
                integrate(invalid_parameter,[1],**kw)
            with self.assertRaisesRegex(ValueError,'shape mismatch'):
                integrate(lambda t,y:[1,2],[1],**kw)
            with self.assertRaisesRegex(ValueError,'invalid parameters'):
                integrate(lambda t,y:van_der_pol(t,y,mu=np.nan),[1,1],**kw)
        calls=[]
        def overflow_later(t,y):
            calls.append(t)
            if len(calls)==3: raise OverflowError('numerical trial overflow')
            return y
        result=integrate(overflow_later,[1],t_span=(0,1),method='rkf45',step=.1)
        self.assertEqual(result.status,'nonfinite'); self.assertEqual(result.evaluations,3)
        result=integrate(lambda t,y:van_der_pol(t,y,mu=1),[1e100,1e100],
                         t_span=(0,1),method='rkf45',step=.1)
        self.assertEqual(result.status,'nonfinite'); self.assertEqual(result.evaluations,2)
        calls=[]
        def finite_large_derivative(t,y):
            if not np.isfinite(y).all(): raise ValueError('model rejects nonfinite state')
            calls.append(t)
            return np.full_like(y,1e308)
        result=integrate(finite_large_derivative,[1],t_span=(0,2),method='rkf45',step=2)
        self.assertEqual(result.status,'nonfinite'); self.assertEqual(result.evaluations,len(calls))

    def test_central_difference_initial_escape_without_evaluation(self):
        from nonlinear_workbook.chapter10 import central_difference
        for n in (2,10):
            for initial in ((100,100),(100,0),(0,-100)):
                def forbidden(y): self.fail('already escaped input was evaluated')
                times,states,status=central_difference(forbidden,*initial,step=.1,n=n,escape_radius=1)
                self.assertEqual(status,'escaped'); self.assertEqual(len(times),2)
                np.testing.assert_array_equal(states,np.array(initial).reshape(2,1))
        _,_,status=central_difference(lambda y:0*y,1,1,step=.1,n=2,escape_radius=1)
        self.assertEqual(status,'completed')

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
