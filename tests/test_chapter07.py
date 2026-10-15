import unittest
import numpy as np
from nonlinear_workbook import chapter07 as c
from nonlinear_workbook.chapter05 import lorenz,lorenz_jacobian
from nonlinear_workbook.chapter06 import stroboscopic_sample,anharmonic
from nonlinear_workbook.chapter10 import integrate
from nonlinear_workbook.flow_validation import conditional_flow_spectrum,flow_lyapunov_spectrum
from nonlinear_workbook.chapter01.maps import coupled_logistic_step,coupled_logistic_jacobian

SOLVER=dict(method='rkf45_adaptive',step=.01,max_step=.02,atol=1e-10,rtol=1e-10)


class Chapter07Tests(unittest.TestCase):
    def test_analytic_jacobians(self):
        fixtures=[(lambda t,y:c.modulated_lorenz(t,y,sigma=10,r=80,b=.4,k=2,omega=6.28318),lambda t,y:c.modulated_lorenz_jacobian(t,y,sigma=10,r=80,b=.4,k=2,omega=6.28318),[.8,.4,.3]),
            (lambda t,y:c.rossler(t,y,a=.2,b=.2,c=9),lambda t,y:c.rossler_jacobian(t,y,a=.2,b=.2,c=9),[.8,.4,.3]),
            (lambda t,y:c.rossler_drive_response(t,y,a=.2,b=.2,c=9),lambda t,y:c.rossler_drive_response_jacobian(t,y,a=.2,b=.2,c=9),[.8,.4,.8,1.1,2.3])]
        base=lambda t,y:lorenz(t,y,sigma=16,r=40,b=4)
        jac=lambda t,y:lorenz_jacobian(t,y,sigma=16,r=40,b=4)
        fixtures.append((lambda t,y:c.diffusive_rhs(base,t,y,coupling=.7),lambda t,y:c.diffusive_jacobian(jac,t,y,coupling=.7),[.8,.8,.8,1,.4,.3]))
        for rhs,J,state in fixtures:
            y=np.array(state); h=1e-6
            fd=np.column_stack([(rhs(.3,y+h*e)-rhs(.3,y-h*e))/(2*h) for e in np.eye(len(y))])
            np.testing.assert_allclose(J(.3,y),fd,atol=1e-6,rtol=0)
        y=np.array([.3,.4]); h=1e-6
        fd=np.column_stack([(c.henon_additive_step(y+h*e,a=1.4,b=.3)-c.henon_additive_step(y-h*e,a=1.4,b=.3))/(2*h) for e in np.eye(2)])
        np.testing.assert_allclose(fd,c.henon_additive_jacobian(y,a=1.4,b=.3),atol=1e-6)

    def test_targets_and_pole_placement(self):
        for period in (1,2):
            targets=c.logistic_targets(r=3.8,period=period)
            for i,x in enumerate(targets): self.assertAlmostEqual(3.8*x*(1-x),targets[(i+1)%period],places=12)
        points=c.henon_additive_fixed_points(a=1.4,b=.3)
        for p in points: np.testing.assert_allclose(c.henon_additive_step(p,a=1.4,b=.3),p,atol=1e-12)
        A=c.henon_additive_jacobian(points[-1],a=1.4,b=.3); B=np.array([1.,0.])
        for poles in ([.2,.3],[.2+.3j,.2-.3j]):
            K=c.place_poles_2d(A,B,poles=poles)
            np.testing.assert_allclose(np.sort_complex(np.linalg.eigvals(A-np.outer(B,K))),np.sort_complex(poles),atol=1e-12,rtol=0)
        np.testing.assert_allclose(abs(np.linalg.eigvals(A-np.outer(B,[-.1,1.3]))),1,atol=1e-12)
        with self.assertRaises(ValueError): c.place_poles_2d(np.eye(2),B,poles=[.2,.3])

    def test_local_ogy_cancellation(self):
        target=c.logistic_targets(r=3.8,period=1)[0]; errors=[]
        for eps in (1e-4,5e-5):
            r=c.logistic_ogy(target+eps,r=3.8,period=1,updates=1,neighborhood=.001,max_perturbation=.01)
            self.assertTrue(r.active[0]); errors.append(abs(r.states[-1,0]-target))
        self.assertGreater(errors[0]/errors[1],3.9)
        p=c.henon_additive_fixed_points(a=1.4,b=.3)[-1]; A=c.henon_additive_jacobian(p,a=1.4,b=.3)
        K=c.place_poles_2d(A,[1,0],poles=[.2,.3]); e=np.array([1e-4,-2e-4])
        r=c.henon_ogy(p+e,a=1.4,b=.3,target=p,gain=K,updates=20,neighborhood=.005,max_perturbation=.01,escape_radius=100)
        np.testing.assert_allclose(r.states[1]-p,(A-np.outer([1,0],K))@e+[-e[0]**2,0],atol=1e-14)
        self.assertLess(np.max(abs(r.states[-1]-p)),1e-10)

    def test_control_gates_resets_and_noise(self):
        kw=dict(r=3.8,period=1,updates=1000,neighborhood=.001,max_perturbation=.01)
        a=c.logistic_ogy(.28,noise_std=1e-6,seed=20261015,**kw)
        b=c.logistic_ogy(.28,noise_std=1e-6,seed=20261015,**kw)
        np.testing.assert_array_equal(a.states,b.states); np.testing.assert_array_equal(a.measured_states,b.measured_states)
        self.assertLessEqual(np.max(abs(a.controls)),.01)
        np.testing.assert_array_equal(a.parameters[~a.active],np.full(np.sum(~a.active),3.8))
        p=c.logistic_targets(r=3.8,period=1)[0]
        boundary=p+.001
        r=c.logistic_ogy(boundary,r=3.8,period=1,updates=1,neighborhood=abs(boundary-p),max_perturbation=.01)
        self.assertEqual(r.reasons,['outside_neighborhood'])
        r=c.logistic_ogy(p+1e-4,r=3.8,period=1,updates=1,neighborhood=.001,max_perturbation=0)
        self.assertEqual(r.reasons,['perturbation_bound']); self.assertEqual(r.controls[0],0)
        p=c.henon_additive_fixed_points(a=1.4,b=.3)[-1]
        r=c.henon_ogy(p+[.01,0],a=1.4,b=.3,target=p,gain=[1,1],updates=1,neighborhood=.005,max_perturbation=.01,escape_radius=100)
        self.assertEqual(r.reasons,['outside_neighborhood'])

    def test_projection_and_simultaneous_updates(self):
        J=np.array([[2,.3],[1,0.]])
        delta,status=c.unstable_projection_control(J,[1,0],[.001,.002],[1,2],denominator_tolerance=1e-12)
        self.assertEqual(status,'eligible'); self.assertAlmostEqual(np.array([1,2])@(J@np.array([.001,.002])+[delta,0]),0)
        self.assertEqual(c.unstable_projection_control(J,[0,1],[.1,.2],[1,0],denominator_tolerance=1e-12)[1],'singular_direction')
        kw=dict(a=1.4,b=.3,updates=1,neighborhood=.01,max_perturbation=.01,escape_radius=100)
        master=np.array([.5,-.8]); slave=np.array([.001,.001]); r=c.henon_synchronize(master,slave,**kw)
        np.testing.assert_allclose(r.states[1],np.r_[c.henon_additive_step(master,a=1.4,b=.3),c.henon_additive_step(slave,a=1.4,b=.3)])
        r=c.henon_synchronize(master,master,**kw)
        np.testing.assert_allclose(r.states[:,0:2],r.states[:,2:])
        r=c.henon_synchronize([0,-1.4/.3],[1e-4,-1.4/.3],**kw)
        self.assertEqual(r.reasons,['unavailable_direction'])

    def test_synchronization_manifolds_and_blocks(self):
        y=np.array([.8,.4,.8]); r=np.r_[y,y[0],y[2]]
        f=c.rossler_drive_response(0,r,a=.2,b=.2,c=9)
        np.testing.assert_allclose(f[[0,2]],f[3:])
        np.testing.assert_allclose(c.rossler_response_jacobian(0,r,a=.2,b=.2,c=9),c.rossler_drive_response_jacobian(0,r,a=.2,b=.2,c=9)[3:,3:])
        A=np.array([[0,-2],[2,.3]]); gamma=.7
        J=c.diffusive_jacobian(lambda t,y:A,0,[1,2,1,2],coupling=gamma)
        S=np.block([[np.eye(2),np.eye(2)],[np.eye(2),-np.eye(2)]])/np.sqrt(2)
        np.testing.assert_allclose(S.T@J@S,np.block([[A,np.zeros((2,2))],[np.zeros((2,2)),A-2*gamma*np.eye(2)]]),atol=1e-12)
        f=c.diffusive_rhs(lambda t,y:A@y,0,[1,2,1,2],coupling=gamma)
        np.testing.assert_allclose(f[:2],f[2:])
        np.testing.assert_allclose(c.diffusive_rhs(lambda t,y:A@y,0,[1,2,3,4],coupling=0),np.r_[A@[1,2],A@[3,4]])

    def test_conditional_linear_spectrum(self):
        A=np.diag([.2,-.7]); rhs=lambda t,y:np.r_[1.,A@y[1:]]
        kw=dict(burn_in=.2,duration=2.,qr_interval=.1,initial_basis=np.eye(2),solver_parameters=SOLVER,response_dimension=2)
        result=conditional_flow_spectrum(rhs,lambda t,y:A,[0,1,1],**kw)
        np.testing.assert_allclose(result.exponents,[.2,-.7],atol=1e-7,rtol=0)
        self.assertAlmostEqual(result.trace_average,-.5,places=9); self.assertEqual(result.status,'completed')
        np.testing.assert_allclose(result.final_state[0],2.2,atol=1e-12)
        shifted=conditional_flow_spectrum(rhs,lambda t,y:A-.6*np.eye(2),[0,1,1],**kw)
        np.testing.assert_allclose(shifted.exponents,result.exponents-.6,atol=1e-7)
        full=flow_lyapunov_spectrum(rhs,lambda t,y:np.diag([0,.2,-.7]),[0,1,1],burn_in=.2,duration=2.,qr_interval=.1,initial_basis=np.eye(3),solver_parameters=SOLVER)
        np.testing.assert_allclose(full.exponents,[0,.2,-.7],atol=1e-7)

    def test_conditional_failure_validation(self):
        kw=dict(response_dimension=1,burn_in=0,duration=1,qr_interval=.1,initial_basis=np.eye(1),solver_parameters=SOLVER)
        result=conditional_flow_spectrum(lambda t,y:y,lambda t,y:np.ones((1,1)),[1,1],**dict(kw,solver_parameters=dict(SOLVER,max_steps=1)))
        self.assertEqual(result.status,'max_steps'); self.assertEqual(result.elapsed,0); self.assertTrue(np.isnan(result.exponents).all())
        for overrides in [dict(initial_basis=np.zeros((1,1))),dict(response_dimension=3),dict(response_dimension=0)]:
            with self.assertRaises(ValueError): conditional_flow_spectrum(lambda t,y:y,lambda t,y:np.eye(1),[1,1],**dict(kw,**overrides))
        with self.assertRaises(ValueError): conditional_flow_spectrum(lambda t,y:y,lambda t,y:np.eye(2),[1,1],**kw)
        with self.assertRaises(ValueError): flow_lyapunov_spectrum(lambda t,y:y,lambda t,y:np.eye(1),1.,burn_in=0,duration=1,qr_interval=.1,initial_basis=np.eye(1),solver_parameters=SOLVER)
        result=conditional_flow_spectrum(lambda t,y:np.full(2,np.nan),lambda t,y:np.eye(1),[1,1],**kw)
        self.assertEqual(result.status,'nonfinite')
        result=conditional_flow_spectrum(lambda t,y:y,lambda t,y:np.eye(1),[1,1],**dict(kw,solver_parameters=dict(SOLVER,escape_radius=.5)))
        self.assertEqual(result.status,'escaped')

    def test_logistic_transverse_fixed_point(self):
        r=2.8; x=1-1/r
        for gamma in (0,.05,.15):
            result=c.logistic_transverse_curve(np.full(100,x),r=r,coupling=gamma)
            np.testing.assert_allclose(result['exponent'],np.log(abs(2-r-2*gamma)),atol=1e-12)
            J=coupled_logistic_jacobian([x,x],r=r,coupling=gamma)
            np.testing.assert_allclose(np.sort(np.linalg.eigvals(J)),np.sort([2-r,2-r-2*gamma]),atol=1e-12)
            np.testing.assert_allclose(coupled_logistic_step([x,x],r=r,coupling=gamma),[x,x])
        self.assertGreater(c.logistic_transverse_curve([x],r=r,coupling=.15)['exponent'],0)
        self.assertTrue(np.isneginf(c.logistic_transverse_curve([.5],r=4,coupling=0)['exponent']))

    def test_feedforward_exact_and_linear_solution(self):
        target=lambda t:np.array([.5*np.cos(t),-.5*np.sin(t)])
        derivative=lambda t:np.array([-.5*np.sin(t),-.5*np.cos(t)])
        nonlinear=lambda t,y:anharmonic(t,y,a=1,b=1,c=.1,k1=0,k2=0,omega=1)
        nonlinear_controlled=lambda t,y:nonlinear(t,y)+c.feedforward_force(nonlinear,target,derivative,t)
        for t in (0,.3,2):
            np.testing.assert_allclose(nonlinear_controlled(t,target(t)),derivative(t),atol=1e-12)
        result=integrate(nonlinear_controlled,target(0),t_span=(0,20),**SOLVER)
        np.testing.assert_allclose(result.states,np.array([target(t) for t in result.times]),atol=1e-7,rtol=0)
        A=np.array([[0,1],[-1,-1.]])
        rhs=lambda t,y:A@y
        controlled=lambda t,y:rhs(t,y)+c.feedforward_force(rhs,target,derivative,t)
        for t in (0,.3,2): np.testing.assert_allclose(controlled(t,target(t)),derivative(t),atol=1e-12)
        for offset in (0.,.1):
            result=integrate(controlled,target(0)+[offset,0],t_span=(0,20),**SOLVER)
            t=result.times; w=np.sqrt(3)/2
            q=offset*np.exp(-t/2)*(np.cos(w*t)+np.sin(w*t)/(2*w))
            p=-offset*np.exp(-t/2)*np.sin(w*t)/w
            expected=np.array([target(s) for s in t])+np.column_stack((q,p))
            np.testing.assert_allclose(result.states,expected,atol=1e-7,rtol=0)

    def test_modulation_reductions_and_stroboscopic_3d(self):
        y=[.8,.8,.8]; kw=dict(sigma=10,r=80,b=.4,k=0,omega=6.28318)
        np.testing.assert_allclose(c.modulated_lorenz(.3,y,**kw),lorenz(.3,y,sigma=10,r=80,b=.4))
        kw['k']=2
        np.testing.assert_allclose(c.modulated_lorenz(.3,y,**kw),c.modulated_lorenz(.3+2*np.pi/kw['omega'],y,**kw),atol=1e-12)
        result=stroboscopic_sample(lambda t,y:np.ones(3),[0,0,0],initial_time=.3,period=.2,burn_in_periods=2,samples=3,drive_frequencies=[2*np.pi/.2],drive_phases=[.1],solver_parameters=SOLVER)
        np.testing.assert_allclose(result.times,[.7,.9,1.1]); np.testing.assert_allclose(result.states,np.repeat([[.4],[.6],[.8]],3,axis=1),atol=1e-12)
        np.testing.assert_allclose(np.exp(1j*result.phases[:,0]),np.exp(1j*(result.times*2*np.pi/.2+.1)))
        failed=stroboscopic_sample(lambda t,y:y,[1,1,1],initial_time=0,period=1,burn_in_periods=1,samples=3,drive_frequencies=[2*np.pi],drive_phases=[0],solver_parameters=dict(SOLVER,max_steps=1))
        self.assertEqual(failed.status,'max_steps'); self.assertEqual(failed.states.shape,(0,3))

    def test_invalid_and_escape(self):
        for call in [lambda:c.logistic_ogy(.28,r=3.8,period=2,updates=0,neighborhood=.001,max_perturbation=.01),lambda:c.logistic_targets(r=2.8,period=2),lambda:c.henon_additive_step([1],a=1.4,b=.3),lambda:c.rossler(0,[1,np.nan,1],a=.2,b=.2,c=9),lambda:c.diffusive_rhs(lambda t,y:y,0,[1,2,3],coupling=.1),lambda:c.modulated_lorenz(0,[1,1,1],sigma=10,r=80,b=.4,k=2,omega=0),lambda:c.logistic_ogy(.28,r=3.8,period=1,updates=1,neighborhood=.001,max_perturbation=.01,noise_std=1e-6)]:
            with self.assertRaises(ValueError): call()
        p=c.henon_additive_fixed_points(a=1.4,b=.3)[-1]
        result=c.henon_ogy([10,10],a=1.4,b=.3,target=p,gain=[1,1],updates=10,neighborhood=.005,max_perturbation=.01,escape_radius=100)
        self.assertEqual(result.status,'escaped')
        result=c.henon_ogy([1e200,0],a=1.4,b=.3,target=p,gain=[1,1],updates=1,neighborhood=.005,max_perturbation=.01,escape_radius=1e300)
        self.assertEqual(result.status,'nonfinite')
        with self.assertRaises(ValueError): c.logistic_targets(r=.5,period=1)


if __name__=='__main__': unittest.main()
