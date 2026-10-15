"""Independent matrix/accepted-history oracles for the release audit."""
import unittest
import numpy as np
from nonlinear_workbook.chapter09 import elementary_step,evolve
from nonlinear_workbook.chapter10 import integrate

class ReleaseAuditTests(unittest.TestCase):
    def test_rule90_against_gf2_matrix(self):
        rng=np.random.default_rng(102015)
        for width in (1,2,3,5,8,16):
            matrix=np.zeros((width,width),dtype=int)
            for i in range(width):
                matrix[i,(i-1)%width]^=1; matrix[i,(i+1)%width]^=1
            initial=rng.integers(0,2,width)
            history=evolve(lambda a:elementary_step(a,rule=90,boundary='periodic'),initial,updates=20)
            reference=initial.copy()
            for row in history:
                np.testing.assert_array_equal(row,reference)
                reference=(matrix@reference)%2

    def test_failed_trial_retains_only_accepted_history(self):
        for method in ('euler','rkf45','rkf45_adaptive'):
            calls=[]
            def rhs(t,y):
                self.assertTrue(np.isfinite(y).all()); calls.append(t)
                if t>.15: raise FloatingPointError('failed later numerical trial')
                return np.ones_like(y)
            result=integrate(rhs,[0],t_span=(0,1),method=method,step=.1,max_step=.1)
            self.assertEqual(result.status,'nonfinite'); self.assertEqual(result.evaluations,len(calls))
            self.assertEqual(len(result.states),result.accepted_steps+1)
            self.assertEqual(len(result.step_sizes),result.accepted_steps)
            self.assertGreaterEqual(result.accepted_steps,1)
            np.testing.assert_allclose(result.states[:,0],result.times,atol=1e-14)

if __name__=='__main__': unittest.main()
