import unittest
import numpy as np
from nonlinear_workbook import chapter06 as c
from nonlinear_workbook import chapter03
from nonlinear_workbook.validation import flow_lyapunov_spectrum


SOLVER=dict(method='rkf45_adaptive',step=.02,max_step=.02,atol=1e-11,rtol=1e-11)
FIXTURES={
    'anharmonic':dict(a=1,b=-10,c=100,k1=0,k2=1.2,omega=3.5),
    'driven_pendulum':dict(a=.2,b=1,k=1.1,omega=.8),
    'parametric_pendulum':dict(a=.15,k=.94,omega=1.56),
    'driven_van_der_pol':dict(a=5,k=5,omega=2.466),
    'two_drive_pendulum':dict(a=.1,k1=.3,k2=.3,omega1=.618034,omega2=1.618034)}


class Chapter06Tests(unittest.TestCase):
    def test_state_and_lifted_jacobians(self):
        for name,p in FIXTURES.items():
            rhs=getattr(c,name); jac=getattr(c,name+'_jacobian'); y=np.array([.2,.4]); h=1e-6; t=.7
            fd=np.column_stack([(rhs(t,y+h*e,**p)-rhs(t,y-h*e,**p))/(2*h) for e in np.eye(2)])
            np.testing.assert_allclose(jac(t,y,**p),fd,atol=1e-6,rtol=0)
            lift=c.phase_lift(name,**p); z=lift.initial_state(y,time=t)
            fd=np.column_stack([(lift.rhs(0,z+h*e)-lift.rhs(0,z-h*e))/(2*h) for e in np.eye(len(z))])
            np.testing.assert_allclose(lift.jacobian(0,z),fd,atol=1e-6,rtol=0)
            np.testing.assert_allclose(lift.rhs(0,z)[:2],rhs(t,y,**p),atol=1e-14)

    def test_reductions_and_periodicity(self):
        y=np.array([.7,.2])
        np.testing.assert_allclose(c.driven_van_der_pol(.3,y,a=5,k=0,omega=2),chapter03.van_der_pol(.3,y,mu=5))
        np.testing.assert_allclose(c.driven_pendulum(.3,y,a=0,b=1,k=0,omega=2),chapter03.pendulum(.3,y))
        np.testing.assert_allclose(c.parametric_pendulum(.3,y,a=0,k=0,omega=2),chapter03.pendulum(.3,y))
        np.testing.assert_allclose(c.two_drive_pendulum(.3,y,a=0,k1=0,k2=0,omega1=2,omega2=3),chapter03.pendulum(.3,y))
        for name,p in FIXTURES.items():
            if name=='two_drive_pendulum': continue
            rhs=getattr(c,name)
            np.testing.assert_allclose(rhs(.3,y,**p),rhs(.3+2*np.pi/p['omega'],y,**p),atol=1e-13)
        np.testing.assert_allclose(c.wrap_angle([np.pi,-np.pi,3*np.pi]),[-np.pi]*3)

    def test_stroboscopic_time_phase_failure(self):
        kwargs=dict(initial_time=.3,period=.7,burn_in_periods=2,samples=4,drive_frequencies=[2,3],drive_phases=[.1,.2],solver_parameters=SOLVER)
        result=c.stroboscopic_sample(lambda t,y:np.array([1.,0.]),[0,0],**kwargs)
        expected=.3+np.arange(2,6)*.7
        np.testing.assert_array_equal(result.times,expected)
        np.testing.assert_allclose(result.states[:,0],expected-.3,atol=1e-13)
        np.testing.assert_allclose(result.phases,(expected[:,None]*[2,3]+[.1,.2])%(2*np.pi))
        self.assertEqual(result.evaluations,6*(result.accepted_steps+result.rejected_steps))
        failed=c.stroboscopic_sample(lambda t,y:np.full(2,np.nan),[0,0],**kwargs)
        self.assertEqual(failed.status,'nonfinite'); self.assertEqual(len(failed.times),0)

    def test_linear_endpoint_monodromy_shooting(self):
        # q''+2q'+2q=cos(3t). Steady particular Re(exp(3it)/(-7+6i)).
        omega=3.; period=2*np.pi/omega; t0=.4
        params=dict(a=2,b=2,c=0,k1=0,k2=1,omega=omega)
        rhs=lambda t,y:c.anharmonic(t,y,**params); jac=lambda t,y:c.anharmonic_jacobian(t,y,**params)
        exact=lambda t:np.array([(np.exp(3j*t)/(-7+6j)).real,(3j*np.exp(3j*t)/(-7+6j)).real])
        result=c.variational_map(rhs,jac,exact(t0),initial_time=t0,period=period,solver_parameters=SOLVER)
        np.testing.assert_allclose(result.endpoint,exact(t0+period),atol=1e-7,rtol=0)
        m=np.array([[0,1],[-2,-2]])
        analytic=np.exp(-period)*(np.eye(2)*np.cos(period)+(m+np.eye(2))*np.sin(period))
        np.testing.assert_allclose(result.monodromy,analytic,atol=1e-7,rtol=0)
        shot=c.shoot_periodic(rhs,jac,[1,1],initial_time=t0,period=period,period_multiple=1,tolerance=1e-9,max_iterations=5,escape_radius=100,solver_parameters=SOLVER)
        self.assertEqual(shot.status,'converged'); self.assertLess(shot.residual_history[-1],1e-9)
        np.testing.assert_allclose(shot.state,exact(t0),atol=1e-7)
        twice=c.shoot_periodic(rhs,jac,[1,1],initial_time=t0,period=period,period_multiple=2,tolerance=1e-9,max_iterations=5,escape_radius=100,solver_parameters=SOLVER)
        self.assertEqual(twice.status,'converged')
        np.testing.assert_allclose(twice.state,exact(t0),atol=1e-7)

    def test_shooting_failures(self):
        rhs=lambda t,y:np.array([1.,0.]); jac=lambda t,y:np.zeros((2,2))
        controls=dict(initial_time=0,period=1,period_multiple=1,tolerance=1e-9,max_iterations=3,escape_radius=100,solver_parameters=SOLVER)
        self.assertEqual(c.shoot_periodic(rhs,jac,[0,0],**controls).status,'singular_solve')
        self.assertEqual(c.shoot_periodic(rhs,jac,[0,0],**dict(controls,max_iterations=0)).status,'max_iterations')
        self.assertEqual(c.shoot_periodic(rhs,jac,[0,0],**dict(controls,solver_parameters=dict(SOLVER,max_steps=1))).status,'integration_max_steps')

    def test_multipliers(self):
        for matrix,dims,label in ((np.diag([.5,.2]),(2,0,0),'PD'),(np.diag([2,.5]),(1,1,0),'ND'),(np.diag([-2,.5]),(1,1,0),'NI'),(np.diag([-2,3]),(0,2,0),'PI'),(np.diag([1,.5]),(1,0,1),None)):
            r=c.classify_multipliers(matrix)
            self.assertEqual((r['stable_dimension'],r['unstable_dimension'],r['neutral_dimension']),dims); self.assertEqual(r['label'],label)

    def test_spectrum_sum(self):
        for name,p in (('anharmonic',FIXTURES['anharmonic']),('driven_van_der_pol',FIXTURES['driven_van_der_pol'])):
            result=flow_lyapunov_spectrum(lambda t,y:getattr(c,name)(t,y,**p),lambda t,y:getattr(c,name+'_jacobian')(t,y,**p),[.1,.2],burn_in=.5,duration=2,qr_interval=.1,initial_basis=np.eye(2),solver_parameters=SOLVER)
            self.assertLess(abs(result.exponents.sum()-result.trace_average),1e-5)
            if name=='anharmonic': self.assertLess(abs(result.exponents.sum()+1),1e-5)

    def test_signal_normalization(self):
        for n in (63,64):
            x=np.cos(2*np.pi*5*np.arange(n)/n); dt=.2
            for window in ('rectangular','hann'):
                f,p=c.periodogram(x,sample_interval=dt,demean=False,window=window)
                w=np.ones(n) if window=='rectangular' else np.hanning(n)
                self.assertAlmostEqual(p.sum()/(n*dt),np.sum((x*w)**2)/np.sum(w*w),places=12)
                self.assertAlmostEqual(f[p.argmax()],5/(n*dt),places=12)
        f,p=c.periodogram(np.ones(64),sample_interval=1,demean=False,window='rectangular'); self.assertEqual(p[0],64)
        f,p=c.periodogram((-1.)**np.arange(64),sample_interval=1,demean=False,window='rectangular'); self.assertEqual(p[-1],64)
        lags,curve,counts=c.normalized_autocorrelation([1,2,3],sample_interval=.2,max_lag=2)
        np.testing.assert_allclose(curve,[1,0,-1.5]); np.testing.assert_array_equal(counts,[3,2,1]); np.testing.assert_allclose(lags,[0,.2,.4])

    def test_invalid(self):
        for call in (lambda:c.anharmonic(0,[1,2],a=-1,b=1,c=1,k1=0,k2=0,omega=1),lambda:c.driven_pendulum(0,[1,np.nan],a=1,b=1,k=1,omega=1),lambda:c.phase_lift('wrong'),lambda:c.normalized_autocorrelation([1,1],sample_interval=1,max_lag=1),lambda:c.periodogram([1,2],sample_interval=1,demean=False,window='hann'),lambda:c.periodogram([1+1j,2],sample_interval=1,demean=False,window='rectangular')):
            with self.assertRaises(ValueError): call()
