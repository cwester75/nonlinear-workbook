import unittest
from fractions import Fraction
import numpy as np
from nonlinear_workbook.chapter01 import *

class StageATests(unittest.TestCase):
    def test_exact_book_trajectories_and_iteration_contract(self):
        result=iterate(collatz_step,56,n=20,exact=True)
        self.assertEqual(result.values[:6].tolist(),[56,28,14,7,22,11])
        self.assertEqual(result.values[-3:].tolist(),[4,2,1])
        exact=iterate(logistic_step,Fraction(1,3),n=5,exact=True)
        self.assertEqual(exact.values.tolist(),[Fraction(1,3),Fraction(8,9),Fraction(32,81),Fraction(6272,6561),Fraction(7250432,43046721)])
        self.assertEqual(iterate(bernoulli_step,Fraction(1,17),n=4,exact=True).values.tolist(),[Fraction(1,17),Fraction(2,17),Fraction(4,17),Fraction(8,17)])
        self.assertEqual(iterate(gauss_step,Fraction(23,101),n=6,exact=True).values.tolist(),[Fraction(23,101),Fraction(9,23),Fraction(5,9),Fraction(4,5),Fraction(1,4),Fraction(0)])
        self.assertEqual(iterate(lambda x:x+1,0,n=3,burn_in=2).values.tolist(),[2,3,4])
        self.assertEqual(iterate(lambda x:2*x,1,n=20,escape_radius=10).status,'escaped')
        self.assertEqual(iterate(lambda x:float('nan'),1,n=3).status,'nonfinite')

    def test_piecewise_boundaries_and_precision_collapse(self):
        for x in [Fraction(0),Fraction(1,3),Fraction(1,2),Fraction(2,3),Fraction(1)]:
            self.assertEqual(bungalow_step(x,a=Fraction(1,3)),tent_step(x))
        self.assertEqual(bernoulli_step(1),0)
        self.assertEqual(tent_step(.5),1)
        self.assertEqual(gauss_step(0),0)
        self.assertTrue(np.all(iterate(bernoulli_step,.618,n=80).values[60:]==0))
        self.assertTrue(np.all(iterate(tent_step,.618,n=80).values[60:]==0))
        self.assertEqual(quadratic_step(Fraction(1,2),r=Fraction(2)),Fraction(1,2))
        with self.assertRaises(ValueError): logistic_step(2)
        with self.assertRaises(ValueError): bungalow_step(.3,a=.5)
        with self.assertRaises(ValueError): collatz_step(True)

    def test_stability_lyapunov_moments_and_autocovariance(self):
        self.assertEqual(stability(lambda x:4*(1-2*x),.75)[1],'unstable')
        self.assertEqual(stability(lambda x:2.5*(1-2*x),.6)[1],'stable')
        self.assertEqual(stability(lambda x:1,0)[1],'nonhyperbolic')
        x=logistic_map(100000)
        self.assertAlmostEqual(scalar_lyapunov(x,logistic_derivative),np.log(2),delta=.005)
        self.assertEqual(scalar_lyapunov([.5],logistic_derivative),float('-inf'))
        np.testing.assert_allclose(moments([1,2,3]),[2,14/3])
        lags,cov,pairs=autocovariance([1,2,3],max_lag=2)
        np.testing.assert_allclose(cov,[2/3,0,-1]); np.testing.assert_array_equal(pairs,[3,2,1])
        _,curve,_=autocovariance(x,max_lag=5)
        self.assertAlmostEqual(curve[0],1/8,delta=.002)
        self.assertTrue(np.max(np.abs(curve[1:]))<.002)

    def test_invariant_densities(self):
        edges=np.linspace(0,1,21); x=logistic_map(100000)
        result=invariant_density(x,edges=edges)
        target=np.diff(2/np.pi*np.arcsin(np.sqrt(edges)))
        self.assertEqual(result.retained,100000)
        self.assertLess(np.max(np.abs(result.counts/result.retained-target)),.005)
        self.assertAlmostEqual(np.sum(result.density*np.diff(edges)),1)
        # Invariance under a single tent step on an IID uniform ensemble avoids
        # finite-precision collapse of a long float64 tent trajectory.
        uniform=np.random.default_rng(20261011).random(100000)
        pushed=np.array([tent_step(v) for v in uniform])
        density=invariant_density(pushed,edges=edges)
        self.assertLess(np.max(np.abs(density.counts/100000-.05)),.005)
        endpoint=invariant_density([0,1],edges=[0,.5,1])
        np.testing.assert_array_equal(endpoint.counts,[1,1])
        with self.assertRaises(ValueError): invariant_density([2],edges=[0,1])

    def test_dft_and_independent_fft(self):
        for n in (1,2,8,32):
            rng=np.random.default_rng(n); x=rng.normal(size=n)+1j*rng.normal(size=n)
            np.testing.assert_allclose(dft(x),np.fft.fft(x)/n,atol=1e-13)
            np.testing.assert_allclose(radix2_fft(x),dft(x),atol=1e-13)
            np.testing.assert_allclose(radix2_fft(radix2_fft(x),inverse=True),x,atol=1e-13)
            np.testing.assert_allclose(dft(dft(x),inverse=True),x,atol=1e-12)
        cosine=np.cos(2*np.pi*np.arange(8)/8); coefficients=dft(cosine)
        target=np.zeros(8); target[[1,7]]=.5
        np.testing.assert_allclose(coefficients,target,atol=1e-14)
        with self.assertRaises(ValueError): radix2_fft([1,2,3])

    def test_random_maps_and_monte_carlo(self):
        self.assertEqual(iterate(lambda x:lcg_step(x,multiplier=7,increment=7,modulus=10),7,n=5,exact=True).values.tolist(),[7,6,9,0,7])
        self.assertTrue(0<=polynomial_random_step(.5)<1)
        target=1-np.cos(1)
        for method in ('mean','hit_or_miss'):
            r=monte_carlo(np.sin,n=100000,seed=20261011,method=method)
            self.assertAlmostEqual(r.estimate,target,delta=5*r.standard_error)
        with self.assertRaises(ValueError): monte_carlo(lambda x:2,n=10,seed=1,method='hit_or_miss')

    def test_rotation_newton_and_symbolic_inversion(self):
        self.assertAlmostEqual(rotation_number(omega=.37,strength=0,n=1000),.37,places=11)
        self.assertAlmostEqual(rotation_number(omega=0,strength=.8),0,places=10)
        result=newton(lambda x:x-np.sin(np.pi*x),lambda x:1-np.pi*np.cos(np.pi*x),.7)
        self.assertEqual(result.status,'converged'); self.assertLess(result.residuals[-1],1e-12)
        self.assertEqual(newton(lambda x:x*x+1,lambda x:2*x,0).status,'singular_derivative')
        self.assertEqual(newton(lambda x:x*x+1,lambda x:2*x,1,max_steps=1).status,'max_steps')
        np.testing.assert_array_equal(symbolic_partition([-.2,0,.2],boundaries=[0]),[0,1,1])
        trajectory=iterate(lambda x:quadratic_step(x,r=1.85),.2,n=8).values
        symbols=['L' if x<0 else 'R' for x in trajectory[:-1]]
        self.assertAlmostEqual(inverse_symbolic(symbols,trajectory[-1],r=1.85),trajectory[0],places=10)
        with self.assertRaises(ValueError): inverse_symbolic(['X'],0,r=2)

    def test_period_doubling_multiplier_crossings(self):
        roots=[]
        for p,r in zip([1,2,4,8,16],[3.,3.45,3.544,3.564,3.5687]):
            result=period_doubling(p,r_guess=r)
            self.assertEqual(result.status,'converged',result)
            roots.append(result.root[1])
        self.assertAlmostEqual(roots[0],3,delta=1e-9)
        self.assertAlmostEqual(roots[1],1+np.sqrt(6),delta=1e-9)
        self.assertAlmostEqual(feigenbaum_ratios(roots)[-1],4.669201609,delta=.05)
        with self.assertRaises(ValueError): period_doubling(3,r_guess=3.5)

if __name__=='__main__': unittest.main()
