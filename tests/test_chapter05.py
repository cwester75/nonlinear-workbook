import unittest
import numpy as np
from nonlinear_workbook import chapter05 as c
from nonlinear_workbook.chapter10 import integrate
from nonlinear_workbook.validation import flow_lyapunov_spectrum,exterior_area_growth,second_compound,wedge


SOLVER=dict(method='rkf45_adaptive',step=.02,max_step=.02,atol=1e-10,rtol=1e-10)


class Chapter05Tests(unittest.TestCase):
    def test_jacobians(self):
        for rhs,jac,y,p in ((c.lorenz,c.lorenz_jacobian,[.7,1.2,2.3],dict(sigma=16,r=40,b=4)),(c.generalized_lotka_volterra,c.generalized_lotka_volterra_jacobian,[.7,1.2,2.3],dict(a=2.9851,b=3,c=7)),(c.hyperchaotic,c.hyperchaotic_jacobian,[-1,2,3,4],dict(a=.25,b=3,c=.5,d=.05)),(c.belousov_zhabotinsky,c.belousov_zhabotinsky_jacobian,[.7,1.2,2.3],dict(s=10,q=.01,f=1,w=.2))):
            y=np.array(y,float); h=1e-6
            finite=np.column_stack([(rhs(0,y+h*e,**p)-rhs(0,y-h*e,**p))/(2*h) for e in np.eye(len(y))])
            np.testing.assert_allclose(jac(0,y,**p),finite,atol=1e-6,rtol=0)

    def test_equilibria_classification(self):
        for r in (0,1,40):
            for y in c.lorenz_equilibria(sigma=16,r=r,b=4): np.testing.assert_allclose(c.lorenz(0,y,sigma=16,r=r,b=4),0,atol=1e-10)
        states,positive=c.chemical_equilibria(s=10,q=.01,f=1,w=.2)
        self.assertEqual(positive.sum(),1)
        for y in states: np.testing.assert_allclose(c.belousov_zhabotinsky(0,y,s=10,q=.01,f=1,w=.2),0,atol=1e-10)
        r=c.classify_equilibrium(np.diag([-2,0,1]))
        self.assertEqual((r.stable_dimension,r.center_dimension,r.unstable_dimension),(1,1,1))
        self.assertFalse(r.hyperbolic); self.assertEqual(r.stability,'unstable')
        self.assertEqual(c.classify_equilibrium(np.diag([-1,-2])).stability,'stable')

    def test_linear_spectrum_and_area(self):
        matrix=np.diag([.3,-.2,-1.]); rhs=lambda t,y:matrix@y; jac=lambda t,y:matrix
        r=flow_lyapunov_spectrum(rhs,jac,[1,1,1],burn_in=.2,duration=2,qr_interval=.13,initial_basis=np.eye(3),solver_parameters=SOLVER)
        np.testing.assert_allclose(r.exponents,np.diag(matrix),atol=1e-7,rtol=0)
        self.assertAlmostEqual(r.elapsed,2); self.assertEqual(r.status,'completed')
        self.assertAlmostEqual(r.exponents.sum(),r.trace_average,places=7)
        area=exterior_area_growth(rhs,jac,[1,1,1],burn_in=.2,duration=2,normalization_interval=.13,initial_vectors=np.eye(3)[:2],solver_parameters=SOLVER)
        self.assertAlmostEqual(area.rate,.1,places=7)
        contract=exterior_area_growth(rhs,jac,[1,1,1],burn_in=0,duration=2,normalization_interval=.1,initial_vectors=np.eye(3)[1:],solver_parameters=SOLVER)
        self.assertAlmostEqual(contract.rate,-1.2,places=7)

    def test_compound_identity(self):
        rng=np.random.default_rng(9)
        for n in (2,3,4):
            j=rng.normal(size=(n,n)); u,v=rng.normal(size=(2,n))
            np.testing.assert_allclose(second_compound(j)@wedge(u,v),wedge(j@u,v)+wedge(u,j@v),atol=1e-14)

    def test_variable_divergence(self):
        rhs=lambda t,y:np.array([-y[0]**2,-y[1]])
        jac=lambda t,y:np.diag([-2*y[0],-1])
        result=flow_lyapunov_spectrum(rhs,jac,[1,1],burn_in=0,duration=2,qr_interval=.1,initial_basis=np.eye(2),solver_parameters=SOLVER)
        np.testing.assert_allclose(result.exponents,[-np.log(3),-1],atol=1e-7,rtol=0)
        self.assertLess(abs(result.exponents.sum()-result.trace_average),1e-7)

    def test_lorenz_sum(self):
        p=dict(sigma=16,r=40,b=4)
        r=flow_lyapunov_spectrum(lambda t,y:c.lorenz(t,y,**p),lambda t,y:c.lorenz_jacobian(t,y,**p),[.8]*3,burn_in=1,duration=3,qr_interval=.1,initial_basis=np.eye(3),solver_parameters=SOLVER)
        self.assertLess(abs(r.exponents.sum()+21),1e-5)

    def test_time_integral(self):
        p=dict(sigma=10,r=0,b=1)
        for state in ([1,1,1],[-2,.2,1.3]):
            rhs=c.lorenz(0,state,**p); y,z=state[1:]
            self.assertLess(abs(2*y*rhs[1]+2*z*rhs[2]+2*(y*y+z*z)),1e-12)
        result=integrate(lambda t,y:c.lorenz(t,y,**p),[1,1,1],t_span=(0,5),**SOLVER)
        values=[c.lorenz_time_integral(t,y,**p) for t,y in zip(result.times,result.states)]
        self.assertLess(max(abs(np.array(values)/values[0]-1)),1e-6)

    def test_hopf_candidates(self):
        threshold=c.lorenz_hopf_threshold(sigma=16,b=4)
        result=c.hopf_candidate(lambda r,y:c.lorenz_jacobian(0,y,sigma=16,r=r,b=4),lambda r:c.lorenz_equilibria(sigma=16,r=r,b=4)[1],bracket=(threshold-1,threshold+1),tolerance=1e-8)
        self.assertEqual(result['status'],'hopf_candidate'); self.assertLess(abs(result['parameter']-threshold),1e-6)
        branch=lambda p:np.zeros(2)
        self.assertEqual(c.hopf_candidate(lambda p,y:[[p,-1],[1,p]],branch,bracket=(1,2))['status'],'not_bracketed')
        self.assertEqual(c.hopf_candidate(lambda p,y:np.diag([p,-1]),branch,bracket=(-.5,.5))['status'],'no_complex_pair')
        with self.assertRaises(ValueError): c.hopf_candidate(lambda p,y:np.kron(np.eye(2),[[p,-1],[1,p]]),lambda p:np.zeros(4),bracket=(-1,1))

    def test_failures(self):
        rhs=lambda t,y:y; jac=lambda t,y:np.eye(len(y))
        base=dict(burn_in=0,duration=1,qr_interval=.1,initial_basis=np.eye(2),solver_parameters=SOLVER)
        with self.assertRaises(ValueError): flow_lyapunov_spectrum(rhs,jac,[1,1],**dict(base,initial_basis=np.zeros((2,2))))
        with self.assertRaises(ValueError): exterior_area_growth(rhs,jac,[1,1],burn_in=0,duration=1,normalization_interval=.1,initial_vectors=[[1,1],[2,2]],solver_parameters=SOLVER)
        with self.assertRaises(ValueError): flow_lyapunov_spectrum(rhs,lambda t,y:np.eye(3),[1,1],**base)
        for parameters,status in ((dict(SOLVER,max_steps=1),'max_steps'),(dict(SOLVER,escape_radius=.5),'escaped')):
            result=flow_lyapunov_spectrum(rhs,jac,[1,1],**dict(base,solver_parameters=parameters))
            self.assertEqual(result.status,status); self.assertEqual(result.elapsed,0)
        result=flow_lyapunov_spectrum(lambda t,y:np.full_like(y,np.nan),jac,[1,1],**base)
        self.assertEqual(result.status,'nonfinite')
        for call in (lambda:c.lorenz(0,[1,2],sigma=16,r=40,b=4),lambda:c.lorenz(0,[1,2,3],sigma=0,r=1,b=1),lambda:c.generalized_lotka_volterra(0,[1,2,3],a=-1,b=1,c=1),lambda:c.lorenz_time_integral(0,[1,1,1],sigma=10,r=0,b=2),lambda:c.lorenz_time_integral(1000,[1,1,1],sigma=10,r=0,b=1)):
            with self.assertRaises(ValueError): call()
