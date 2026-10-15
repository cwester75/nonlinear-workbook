import unittest
import numpy as np
from nonlinear_workbook import chapter04 as c
from nonlinear_workbook.chapter10 import integrate


SOLVER=dict(method='rkf45_adaptive',step=.01,max_step=.02,atol=1e-10,rtol=1e-10)


class Chapter04Tests(unittest.TestCase):
    def test_gradients_jacobians_and_hamiltonian_identity(self):
        fixtures=[(c.henon_heiles_rhs,c.henon_heiles_jacobian,c.henon_heiles_energy,[.2,-.3,.4,.1]),
            (lambda t,y:c.quartic_rhs(t,y,r=.15),lambda t,y:c.quartic_jacobian(t,y,r=.15),lambda y:c.quartic_energy(y,r=.15),[.4,-.2,.1,.3])]
        for periodic in (False,True):
            fixtures.append((lambda t,y,p=periodic:c.toda_rhs(t,y,periodic=p),lambda t,y,p=periodic:c.toda_jacobian(t,y,periodic=p),lambda y,p=periodic:c.toda_energy(y,periodic=p),[-.3,.1,.2,.2,-.1,-.1]))
        for rhs,jac,energy,state in fixtures:
            y=np.array(state); n=len(y)//2; eye=np.eye(len(y)); h=1e-6
            fd=np.column_stack([(rhs(0,y+h*e)-rhs(0,y-h*e))/(2*h) for e in eye])
            np.testing.assert_allclose(jac(0,y),fd,atol=1e-6,rtol=0)
            omega=np.block([[np.zeros((n,n)),np.eye(n)],[-np.eye(n),np.zeros((n,n))]])
            np.testing.assert_allclose(jac(0,y).T@omega+omega@jac(0,y),0,atol=1e-12)
            gradient=np.array([(energy(y+h*e)-energy(y-h*e))/(2*h) for e in eye])
            np.testing.assert_allclose(rhs(0,y),omega@gradient,atol=1e-6,rtol=0)
        np.testing.assert_allclose(c.henon_heiles_rhs(0,[0,0,0,0]),0)
        np.testing.assert_allclose(c.henon_heiles_rhs(0,[0,1,0,0]),0)
        np.testing.assert_allclose(c.toda_rhs(0,[1,1,1,0,0,0],periodic=True),0)

    def test_toda_lax_and_canonical_derivative(self):
        y=np.array([-.3,.1,.2,.2,-.1,-.1]); d=c.open_toda_lax(y)
        np.testing.assert_allclose(d['residual'],0,atol=1e-12)
        self.assertAlmostEqual(np.trace(d['L']@d['L']),c.toda_energy(y,periodic=False)/2)
        h=1e-5; f=c.toda_rhs(0,y,periodic=False)
        fd=(c.open_toda_lax(y+h*f)['L']-c.open_toda_lax(y-h*f)['L'])/(2*h)
        np.testing.assert_allclose(fd,d['derivative'],atol=1e-12,rtol=0)
        p=y[3:]; _,e=c._toda(y[:3],True)
        # Independent analytic gradient of the periodic cubic invariant.
        gq=np.array([-p[2]*e[0]+p[1]*e[2],p[2]*e[0]-p[0]*e[1],p[0]*e[1]-p[1]*e[2]])
        gp=np.array([p[1]*p[2]-e[1],p[0]*p[2]-e[2],p[0]*p[1]-e[0]])
        self.assertAlmostEqual(np.r_[gq,gp]@c.toda_rhs(0,y,periodic=True),0,places=12)

    def test_invariant_drift(self):
        y=np.array([-.3,.1,.2,.2,-.1,-.1])
        for periodic in (True,False):
            result=integrate(lambda t,y:c.toda_rhs(t,y,periodic=periodic),y,t_span=(0,20),**SOLVER)
            self.assertEqual(result.status,'completed')
            vals=np.array([[c.toda_energy(z,periodic=periodic),*(c.periodic_toda_integrals(z) if periodic else c.open_toda_lax(z)['invariants'])] for z in result.states])
            self.assertLess(np.max(abs(vals-vals[0])/np.maximum(1,abs(vals[0]))),1e-6)
        result=integrate(lambda t,y:c.quartic_rhs(t,y,r=0),[1,1,1.5,1],t_span=(0,20),**SOLVER)
        v=np.array([[c.quartic_energy(z,r=0),c.quartic_integral(z)] for z in result.states])
        self.assertLess(np.max(abs(v-v[0])/np.maximum(1,abs(v[0]))),1e-6)

    def test_quartic_corrected_integral_and_rotation(self):
        x,y,p,v=.4,.7,.3,-.2; state=np.array([x,y,p,v])
        grad=np.array([y*(3*x*x+y*y),x*(x*x+3*y*y),3*v,3*p])
        self.assertAlmostEqual(grad@c.quartic_rhs(0,state,r=0),0,places=14)
        printed_grad=np.array([y*(2*x+y),x*(x+2*y),3*v,3*p])
        self.assertGreater(abs(printed_grad@c.quartic_rhs(0,state,r=0)),.01)
        z=c.quartic_rotation(state)
        self.assertAlmostEqual(c.quartic_energy(state,r=0),sum(z[2:]**2/2+z[:2]**4/6))
        self.assertAlmostEqual(c.quartic_integral(state),3*((z[2]**2-z[3]**2)/2+(z[0]**4-z[1]**4)/6))
        with self.assertRaises(ValueError): c.quartic_integral(state,r=.1)

    def test_mathieu_analytical_monodromy(self):
        for delta in (0.,.4,-.4):
            if delta==0: expected=np.array([[1,np.pi],[0,1]])
            elif delta>0:
                w=np.sqrt(delta); expected=np.array([[np.cos(w*np.pi),np.sin(w*np.pi)/w],[-w*np.sin(w*np.pi),np.cos(w*np.pi)]])
            else:
                w=np.sqrt(-delta); expected=np.array([[np.cosh(w*np.pi),np.sinh(w*np.pi)/w],[w*np.sinh(w*np.pi),np.cosh(w*np.pi)]])
            result=c.mathieu_floquet(delta=delta,epsilon=0,solver_parameters=SOLVER)
            np.testing.assert_allclose(result['monodromy'],expected,atol=1e-7,rtol=0)
            self.assertLess(result['determinant_error'],1e-7)
            self.assertEqual(result['band'],{0.:'boundary_unresolved',.4:'bounded_oscillatory',-.4:'exponentially_unstable'}[delta])
            if delta==0: self.assertFalse(result['scalar_boundary'])
        result=c.mathieu_floquet(delta=1,epsilon=0,solver_parameters=SOLVER)
        self.assertTrue(result['scalar_boundary'])
        fd=np.column_stack([(c.mathieu_rhs(.2,np.eye(2)[j]*1e-6,delta=1,epsilon=.5)-c.mathieu_rhs(.2,-np.eye(2)[j]*1e-6,delta=1,epsilon=.5))/2e-6 for j in range(2)])
        np.testing.assert_allclose(fd,c.mathieu_jacobian(.2,[0,0],delta=1,epsilon=.5),atol=1e-6)

    def test_symplectic_refinement_and_accounting(self):
        for method,ratio in [('euler',1.8),('verlet',3.8)]:
            errors=[]
            for step in (.1,.05):
                result=c.integrate_separable([1,0],grad_potential=lambda q:q,t_span=(0,1),step=step,method=method)
                errors.append(np.linalg.norm(result.states[-1]-[np.cos(1),-np.sin(1)]))
                self.assertEqual(result.times[-1],1)
                self.assertEqual(result.evaluations,result.accepted_steps*(1 if method=='euler' else 2))
            self.assertGreater(errors[0]/errors[1],ratio)
        r=c.integrate_separable([1,0],grad_potential=lambda q:q,t_span=(0,100),step=.05,method='verlet')
        self.assertLess(np.max(abs(np.sum(r.states**2,axis=1)/2-.5)),.001)
        r=c.integrate_separable([1,0],grad_potential=lambda q:q,t_span=(0,.25),step=.1,method='verlet')
        np.testing.assert_allclose(r.step_sizes,[.1,.1,.05])

    def test_refined_sections_and_endpoint_policy(self):
        rhs=lambda t,y:np.array([y[1],-y[0],0.])
        t=np.linspace(0,4*np.pi,53); y=np.column_stack((np.cos(t),-np.sin(t),np.ones(len(t))))
        r=c.refined_sections(rhs,t,y,coordinate=0,level=0,direction=-1,max_crossings=10,solver_parameters=SOLVER)
        np.testing.assert_allclose(r.times,[np.pi/2,5*np.pi/2],atol=1e-7,rtol=0)
        self.assertEqual(r.status,'completed'); self.assertLess(max(r.residuals),1e-10)
        np.testing.assert_allclose(r.states[:,2],1)
        r=c.refined_sections(lambda t,y:np.ones(1),[0,1,2],[[-1],[0],[1]],coordinate=0,level=0,direction=1,max_crossings=2,solver_parameters=SOLVER)
        np.testing.assert_allclose(r.times,[1])
        r=c.refined_sections(lambda t,y:np.ones(1),[0,1,2],[[1],[0],[1]],coordinate=0,level=0,direction=1,max_crossings=2,solver_parameters=SOLVER)
        self.assertEqual(len(r.times),0)  # tangent contact

    def test_section_failures(self):
        kw=dict(coordinate=0,level=0,direction=1,max_crossings=1,solver_parameters=SOLVER)
        r=c.refined_sections(lambda t,y:-np.ones(1),[0,1],[[-1],[1]],**kw)
        self.assertEqual(r.failures[0][1],'lost_bracket')
        kw['solver_parameters']=dict(SOLVER,max_steps=1)
        r=c.refined_sections(lambda t,y:np.ones(1),[0,2],[[-1],[1]],**kw)
        self.assertEqual(r.failures[0][1],'max_steps')
        r=c.mathieu_floquet(delta=1,epsilon=.5,solver_parameters=dict(SOLVER,max_steps=1))
        self.assertEqual(r['band'],'integration_failure')

    def test_shell_and_invalid_inputs(self):
        y=c.henon_heiles_shell(energy=1/12,q2=0,p2=0,momentum_sign=-1)
        self.assertAlmostEqual(c.henon_heiles_energy(y),1/12)
        for f in [lambda:c.henon_heiles_shell(energy=-1,q2=0,p2=0,momentum_sign=1),lambda:c.quartic_rhs(0,[0]*4,r=2),lambda:c.toda_rhs(0,[0]*4,periodic=True),lambda:c.henon_heiles_rhs(0,[np.nan]*4),lambda:c.mathieu_rhs(0,[0,0],delta=np.inf,epsilon=0),lambda:c.integrate_separable([1,0],grad_potential=lambda q:q,t_span=(0,1),step=0,method='verlet')]:
            with self.assertRaises(ValueError): f()
        with self.assertRaises(FloatingPointError): c.toda_potential([1000,-1000,0],periodic=True)
        with self.assertRaises(ValueError): c.refined_sections(lambda t,y:y,[1,0],[[1],[2]],coordinate=0,level=0,direction=1,max_crossings=2,solver_parameters=SOLVER)

    def test_driver_failure_statuses(self):
        kw=dict(t_span=(0,1),step=.1,method='verlet')
        self.assertEqual(c.integrate_separable([1,0],grad_potential=lambda q:q,max_steps=1,**kw).status,'max_steps')
        self.assertEqual(c.integrate_separable([1,0],grad_potential=lambda q:q,escape_radius=.5,**kw).status,'escaped')
        self.assertEqual(c.integrate_separable([1,0],grad_potential=lambda q:np.full_like(q,np.inf),**kw).status,'nonfinite')
        with self.assertRaises(ValueError): c.integrate_separable([1,0],grad_potential=lambda q:np.zeros(2),**kw)
        self.assertEqual(c.integrate_separable([0,1],grad_potential=lambda q:0*q,escape_radius=1.01,t_span=(0,2),step=.2,method='verlet').status,'escaped')
        self.assertEqual(c.integrate_separable([1e200,1e200],grad_potential=lambda q:q**3,**kw).status,'nonfinite')

    def test_section_limits_empty_and_exhaustion(self):
        kw=dict(coordinate=0,level=0,direction=1,max_crossings=1,solver_parameters=SOLVER)
        r=c.refined_sections(lambda t,y:np.ones(1),[0,1],[[-1],[1]],**kw)
        self.assertEqual(r.status,'crossing_limit')
        r=c.refined_sections(lambda t,y:np.ones(1),[0,1],[[1],[2]],**kw)
        self.assertEqual(r.states.shape,(0,1)); self.assertEqual(r.brackets.shape,(0,2))
        r=c.refined_sections(lambda t,y:np.ones(1),[0,1],[[-.4],[.6]],max_iterations=1,**kw)
        self.assertEqual(r.failures[0][1],'max_iterations')
        # A negative-side grazing contact is not retained as a transverse crossing.
        r=c.refined_sections(lambda t,y:np.array([-2*(t-1)]),[0,1,2],[[-1],[0],[-1]],**kw)
        self.assertEqual(len(r.times),0)


if __name__=='__main__': unittest.main()
