import unittest
import numpy as np
from nonlinear_workbook import chapter03 as c
from nonlinear_workbook.chapter10 import integrate
from nonlinear_workbook.validation import sampled_observations,kantz_sampling_sweep


SOLVER=dict(method='rkf45_adaptive',step=.05,atol=1e-10,rtol=1e-10)


class Chapter03Tests(unittest.TestCase):
    def test_jacobians(self):
        for rhs,jac,params in ((c.van_der_pol,c.van_der_pol_jacobian,dict(mu=5)),(c.radial_equilibrium,c.radial_equilibrium_jacobian,{}),(c.homoclinic,c.homoclinic_jacobian,{}),(c.pendulum,c.pendulum_jacobian,{}),(c.sine_damping,c.sine_damping_jacobian,dict(mu=5)),(c.lotka_volterra,c.lotka_volterra_jacobian,dict(a=2,b=.8,c=3))):
            y=np.array([.7,.4]); h=1e-6
            fd=np.column_stack([(rhs(0,y+e*h,**params)-rhs(0,y-e*h,**params))/(2*h) for e in np.eye(2)])
            np.testing.assert_allclose(jac(0,y,**params),fd,atol=1e-6,rtol=0)

    def test_equilibria(self):
        for name,rhs,params in (('van_der_pol',c.van_der_pol,dict(mu=1)),('sine_damping',c.sine_damping,dict(mu=5)),('homoclinic',c.homoclinic,{}),('pendulum',c.pendulum,{}),('lotka_volterra',c.lotka_volterra,dict(a=2,b=3,c=4))):
            for state in c.equilibria(name,**params): np.testing.assert_allclose(rhs(0,state,**params),0,atol=1e-14)
        for state in c.equilibria('radial',circle_angles=[0,.5,2]): np.testing.assert_allclose(c.radial_equilibrium(0,state),0,atol=1e-14)
        self.assertEqual(c.classify_planar(c.pendulum_jacobian(0,[np.pi,0])).geometric_type,'saddle')

    def test_classification(self):
        for matrix,kind,stability in (([[-1,0],[0,-2]],'node','stable'),([[1,0],[0,2]],'node','unstable'),([[1,0],[0,-1]],'saddle','unstable'),([[-1,-1],[1,-1]],'spiral','stable'),([[0,-1],[1,0]],'linear_center','nonhyperbolic'),([[0,1],[0,0]],'degenerate','nonhyperbolic'),([[-1,1],[0,-1]],'defective_node','stable'),([[-1,0],[0,-1]],'star_node','stable')):
            r=c.classify_planar(matrix)
            self.assertEqual((r.geometric_type,r.stability),(kind,stability))

    def test_homoclinic_reference(self):
        # For d=2+sqrt(13)cosh(s), q''=-3*d''/d^2+6*d'^2/d^3.
        t=np.linspace(-10,10,101); d=2+np.sqrt(13)*np.cosh(t/np.sqrt(2))
        dp=np.sqrt(13/2)*np.sinh(t/np.sqrt(2)); dd=np.sqrt(13)/2*np.cosh(t/np.sqrt(2))
        exact=c.homoclinic_reference(t); acceleration=-3*dd/d**2+6*dp**2/d**3
        np.testing.assert_allclose(acceleration,exact[:,0]/2-exact[:,0]**2-exact[:,0]**3,atol=1e-14)
        np.testing.assert_allclose([c.homoclinic_energy(y) for y in exact],0,atol=1e-14)
        r=integrate(c.homoclinic,exact[0],t_span=(0,20),**SOLVER)
        self.assertLess(np.max(abs(r.states-c.homoclinic_reference(r.times-10))),1e-6)
        self.assertTrue(np.isfinite(c.homoclinic_reference([-1e4,1e4])).all())

    def test_invariants_and_refinement(self):
        for rhs,energy,state in ((c.pendulum,c.pendulum_energy,[1,0]),(lambda t,y:c.lotka_volterra(t,y,a=1,b=1,c=1),lambda y:c.lotka_volterra_invariant(y,a=1,b=1,c=1),[2,1.5])):
            r=integrate(rhs,state,t_span=(0,20),**SOLVER)
            self.assertEqual(r.status,'completed')
            self.assertLess(max(abs(energy(y)-energy(state)) for y in r.states),1e-6)
            errors=[]
            for h in (.2,.1):
                fixed=integrate(rhs,state,t_span=(0,20),method='rkf45',step=h)
                errors.append(max(abs(energy(y)-energy(state)) for y in fixed.states))
            self.assertLess(errors[1],errors[0]/8)

    def test_small_pendulum_period(self):
        r=integrate(c.pendulum,[.001,0],t_span=(0,30),max_step=.02,**SOLVER)
        times,states,periods=c.section_crossings(r.times,r.states,coordinate=0,level=0,direction=1)
        np.testing.assert_allclose(periods,2*np.pi,atol=1e-5,rtol=0)
        self.assertEqual(states.shape[1],2)

    def test_listing_is_dissipative_and_distinct(self):
        state=np.array([.05,.01]); h=.005; mu=5
        x=state[0]+h*state[1]
        np.testing.assert_allclose(c.van_der_pol_listing_step(state,step=h,mu=mu),[x,state[1]*np.exp(-h*mu*(x*x-1))-h*x])
        self.assertGreater(c.van_der_pol(0,state,mu=5)[1],c.sine_damping(0,state,mu=5)[1])

    def test_harness_failures_and_repeatability(self):
        params=dict(samples=20,sample_interval=.1,burn_in=1,noise_std=.02,seed=3,observation=lambda y:y[0],solver_parameters=SOLVER)
        a=sampled_observations(c.pendulum,[1,0],**params); b=sampled_observations(c.pendulum,[1,0],**params)
        np.testing.assert_array_equal(a['observed'],b['observed'])
        with self.assertRaisesRegex(ValueError,'integration failed'):
            sampled_observations(c.pendulum,[1,0],**dict(params,solver_parameters=dict(SOLVER,max_steps=1)))

    def test_invalid(self):
        for state in ([1],[1,np.nan]):
            with self.assertRaises(ValueError): c.pendulum(0,state)
        with self.assertRaises(ValueError): c.van_der_pol(0,[1,0],mu=np.inf)
        with self.assertRaises(ValueError): c.lotka_volterra(0,[1,1],a=0,b=1,c=1)
        with self.assertRaises(ValueError): c.lotka_volterra_invariant([0,1],a=1,b=1,c=1)
        with self.assertRaises(ValueError): c.classify_planar(np.ones((3,3)))
        with self.assertRaises(ValueError): c.section_crossings([0,0],[[1,0],[2,0]],coordinate=0,level=0,direction=1)

    def test_shared_sweep(self):
        controls=dict(sample_counts=[80],sample_intervals=[.2],noise_levels=[0],dimensions=[2],
            delay=2,theiler=3,epsilon=.5,horizon=5,fit_slices=[(0,3)],burn_in=0,seed=1,
            observation=lambda y:y[0],solver_parameters=SOLVER)
        record=kantz_sampling_sweep(c.pendulum,[1,0],**controls)[0]
        self.assertEqual(record['status'],'completed')
        self.assertGreater(record['reference_count'],0)
        expected=np.polyfit(record['lags'][:3]*.2,record['curve'][:3],1)[0]
        self.assertAlmostEqual(record['fits'][0]['slope_per_time'],expected)
        empty=kantz_sampling_sweep(c.pendulum,[1,0],**dict(controls,epsilon=1e-30))[0]
        self.assertEqual(empty['status'],'no_valid_neighbors')
        self.assertEqual(empty['fits'],[])
        with self.assertRaises(ValueError): kantz_sampling_sweep(c.pendulum,[1,0],**dict(controls,fit_slices=[(0,1)]))
