import unittest
import numpy as np
from nonlinear_workbook import chapter10 as c
from nonlinear_workbook.validation import sampled_observations


class Chapter10Tests(unittest.TestCase):
    def test_orders_and_time_stages(self):
        f=lambda t,y:y
        def error(method,h): return abs(c.integrate(f,[1],t_span=(0,1),method=method,step=h).states[-1,0]-np.e)
        self.assertTrue(1.8<error('euler',.05)/error('euler',.025)<2.2)
        self.assertTrue(25<error('rkf45',.1)/error('rkf45',.05)<40)
        r=c.integrate(lambda t,y:np.array([t**4]),[0],t_span=(0,1),method='rkf45',step=.3)
        self.assertAlmostEqual(r.states[-1,0],.2,places=12)
        self.assertEqual(r.times[-1],1)
        self.assertEqual(r.evaluations,6*r.accepted_steps)

    def test_adaptive_and_stopping(self):
        r=c.integrate(lambda t,y:y,[1],t_span=(0,2),method='rkf45_adaptive',step=2,atol=1e-10,rtol=1e-10)
        self.assertEqual(r.status,'completed'); self.assertGreater(r.rejected_steps,0)
        self.assertLess(abs(r.states[-1,0]-np.exp(2)),1e-8)
        self.assertEqual(r.evaluations,6*(r.accepted_steps+r.rejected_steps))
        for f,kw,status in ((lambda t,y:y,dict(max_steps=1),'max_steps'),(lambda t,y:y,dict(escape_radius=1.1),'escaped'),(lambda t,y:np.full_like(y,np.nan),{},'nonfinite')):
            self.assertEqual(c.integrate(f,[1],t_span=(0,10),method='euler',step=.2,**kw).status,status)
        r=c.integrate(lambda t,y:100*y,[1],t_span=(0,1),method='rkf45_adaptive',step=.5,min_step=.4,atol=1e-15,rtol=0)
        self.assertEqual(r.status,'step_underflow')

    def test_taylor_and_ghost(self):
        self.assertAlmostEqual(c.taylor_step([.5],.1,[[.25],[0]])[0],.525)
        times,y,status=c.central_difference(lambda u:u*(1-u),.5,.5+.05*.25,step=.05,n=1000,escape_radius=10)
        self.assertEqual(y[2,0],.5+.1*(.5125*(1-.5125)))
        exact=1/(1+np.exp(-times))
        self.assertGreater(np.max(abs(y[:,0]-exact)),.05)

    def test_symplectic_geometry_energy(self):
        def mapping(y,method):
            q,p=c.symplectic_step([y[0]],[y[1]],.1,grad_potential=lambda q:q,grad_kinetic=lambda p:p,method=method)
            return np.array([q[0],p[0]])
        for method in ('euler','verlet'):
            columns=[(mapping(np.eye(2)[i]*1e-5,method)-mapping(-np.eye(2)[i]*1e-5,method))/2e-5 for i in range(2)]
            self.assertAlmostEqual(np.linalg.det(np.array(columns).T),1,places=12)
        q,p=np.array([1.]),np.array([0.]); errors=[]
        for _ in range(5000):
            q,p=c.symplectic_step(q,p,.1,grad_potential=lambda q:q,grad_kinetic=lambda p:p,method='verlet')
            errors.append(abs(float((q@q+p@p)/2)-.5))
        self.assertLess(max(errors),.0013)
        np.testing.assert_allclose(c.quartic_discrete_step([.9,.75]),[.321,.9])

    def test_sampling_harness(self):
        parameters=dict(samples=20,sample_interval=.1,burn_in=.2,noise_std=.01,seed=8,
            observation=lambda y:y[0],solver_parameters=dict(method='rkf45_adaptive',step=.1,atol=1e-11,rtol=1e-11))
        a=sampled_observations(lambda t,y:-y,[1],**parameters)
        b=sampled_observations(lambda t,y:-y,[1],**parameters)
        np.testing.assert_array_equal(a['observed'],b['observed'])
        np.testing.assert_allclose(a['clean'],np.exp(-a['times']),atol=1e-10,rtol=0)

    def test_invalid_controls(self):
        for kw in (dict(step=0),dict(atol=0),dict(method='other'),dict(max_steps=0)):
            options=dict(t_span=(0,1),method='euler',step=.1); options.update(kw)
            with self.assertRaises(ValueError): c.integrate(lambda t,y:y,[1],**options)
        with self.assertRaises(ValueError): c.rkf45_step(lambda t,y:[1,2],0,[1],.1)
