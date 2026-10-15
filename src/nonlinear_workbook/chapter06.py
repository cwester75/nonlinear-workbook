"""Chapter 6 driven equations; phases are radians, time remains absolute."""
from dataclasses import dataclass
from typing import Callable
import numpy as np
from . import chapter03
from .chapter10 import integrate
from .chapter01.diagnostics import autocovariance


def _state(y,n=2):
    y=np.asarray(y,float)
    if y.shape!=(n,) or not np.isfinite(y).all(): raise ValueError('invalid finite state vector')
    return y


def _controls(t,a,omega,*parameters):
    if not np.isfinite([t,a,omega,*parameters]).all() or a<0 or omega<=0: raise ValueError('finite parameters, nonnegative damping and positive frequency required')


def anharmonic(t,state,*,a,b,c,k1,k2,omega,phi=0.):
    x,v=_state(state); _controls(t,a,omega,b,c,k1,k2,phi)
    return np.array([v,-a*v-b*x-c*x**3+k1+k2*np.cos(omega*t+phi)])


def anharmonic_jacobian(t,state,*,a,b,c,k1,k2,omega,phi=0.):
    x,v=_state(state); _controls(t,a,omega,b,c,k1,k2,phi)
    return np.array([[0.,1.],[-b-3*c*x*x,-a]])


def driven_pendulum(t,state,*,a,b,k,omega,phi=0.):
    q,p=_state(state); _controls(t,a,omega,b,k,phi)
    return np.array([p,-a*p-b*np.sin(q)+k*np.cos(omega*t+phi)])


def driven_pendulum_jacobian(t,state,*,a,b,k,omega,phi=0.):
    q,p=_state(state); _controls(t,a,omega,b,k,phi)
    return np.array([[0.,1.],[-b*np.cos(q),-a]])


def parametric_pendulum(t,state,*,a,k,omega,phi=0.):
    q,p=_state(state); _controls(t,a,omega,k,phi)
    return np.array([p,-a*p-(1+k*np.cos(omega*t+phi))*np.sin(q)])


def parametric_pendulum_jacobian(t,state,*,a,k,omega,phi=0.):
    q,p=_state(state); _controls(t,a,omega,k,phi)
    return np.array([[0.,1.],[-(1+k*np.cos(omega*t+phi))*np.cos(q),-a]])


def driven_van_der_pol(t,state,*,a,k,omega,phi=0.):
    y=_state(state); _controls(t,a,omega,k,phi)
    result=chapter03.van_der_pol(t,y,mu=a)
    result[1]+=k*np.cos(omega*t+phi)
    return result


def driven_van_der_pol_jacobian(t,state,*,a,k,omega,phi=0.):
    y=_state(state); _controls(t,a,omega,k,phi)
    return chapter03.van_der_pol_jacobian(t,y,mu=a)


def two_drive_pendulum(t,state,*,a,k1,k2,omega1,omega2,phi1=0.,phi2=0.):
    q,p=_state(state); _controls(t,a,omega1,k1,k2,omega2,phi1,phi2)
    if omega2<=0: raise ValueError('positive second frequency required')
    return np.array([p,-a*p-(1+k2*np.cos(omega2*t+phi2))*np.sin(q)+k1*np.cos(omega1*t+phi1)])


def two_drive_pendulum_jacobian(t,state,*,a,k1,k2,omega1,omega2,phi1=0.,phi2=0.):
    q,p=_state(state); _controls(t,a,omega1,k1,k2,omega2,phi1,phi2)
    if omega2<=0: raise ValueError('positive second frequency required')
    return np.array([[0.,1.],[-(1+k2*np.cos(omega2*t+phi2))*np.cos(q),-a]])


def wrap_angle(angle):
    angle=np.asarray(angle,float)
    if not np.isfinite(angle).all(): raise ValueError('finite angles required')
    return (angle+np.pi)%(2*np.pi)-np.pi


@dataclass(frozen=True)
class PhaseLift:
    rhs: Callable
    jacobian: Callable
    frequencies: np.ndarray
    offsets: np.ndarray

    def initial_state(self,state,*,time):
        y=_state(state)
        if not np.isfinite(time): raise ValueError('finite time required')
        return np.concatenate((y,self.frequencies*time+self.offsets))


def phase_lift(system,**parameters):
    """Autonomous adapter; appended coordinates ARE phases, not clock times.

    Names: anharmonic, driven_pendulum, parametric_pendulum,
    driven_van_der_pol, two_drive_pendulum. Parameters match the two-state RHS.
    """
    models=dict(anharmonic=(anharmonic,anharmonic_jacobian),driven_pendulum=(driven_pendulum,driven_pendulum_jacobian),parametric_pendulum=(parametric_pendulum,parametric_pendulum_jacobian),driven_van_der_pol=(driven_van_der_pol,driven_van_der_pol_jacobian),two_drive_pendulum=(two_drive_pendulum,two_drive_pendulum_jacobian))
    if system not in models: raise ValueError('unknown driven system')
    function,jacobian=models[system]; p=dict(parameters)
    function(0.,np.zeros(2),**p)  # validate before constructing closures
    double=system=='two_drive_pendulum'
    frequencies=np.array([p['omega1'],p['omega2']] if double else [p['omega']],float)
    offsets=np.array([p.get('phi1',0.),p.get('phi2',0.)] if double else [p.get('phi',0.)],float)
    def args(y):
        new=dict(p)
        if double: new.update(phi1=y[2],phi2=y[3])
        else: new['phi']=y[2]
        return new
    def rhs(t,state):
        y=_state(state,2+len(frequencies))
        return np.concatenate((function(0.,y[:2],**args(y)),frequencies))
    def jac(t,state):
        y=_state(state,2+len(frequencies)); matrix=np.zeros((len(y),len(y)))
        matrix[:2,:2]=jacobian(0.,y[:2],**args(y))
        if double: matrix[1,2]=-p['k1']*np.sin(y[2]); matrix[1,3]=p['k2']*np.sin(y[3])*np.sin(y[0])
        elif system=='anharmonic': matrix[1,2]=-p['k2']*np.sin(y[2])
        elif system=='parametric_pendulum': matrix[1,2]=p['k']*np.sin(y[2])*np.sin(y[0])
        else: matrix[1,2]=-p['k']*np.sin(y[2])
        return matrix
    return PhaseLift(rhs,jac,frequencies,offsets)


def _integer(n,minimum):
    if not isinstance(n,(int,np.integer)) or isinstance(n,bool) or n<minimum: raise ValueError('invalid integer count')


@dataclass
class StroboscopicResult:
    times: np.ndarray
    states: np.ndarray
    phases: np.ndarray
    evaluations: int
    accepted_steps: int
    rejected_steps: int
    status: str


def stroboscopic_sample(rhs,y0,*,initial_time,period,burn_in_periods,samples,
                        drive_frequencies,drive_phases,solver_parameters):
    """First retained state is at initial_time+burn_in_periods*period.

    Phases=(omega*absolute_time+offset) modulo2pi. For two drives these
    metadata do not make the projected two-state samples a closed return map.
    """
    raw=np.asarray(y0,float)
    if raw.ndim!=1 or not len(raw): raise ValueError('nonempty vector required')
    y=_state(y0,n=len(raw)); dimension=len(y); _integer(burn_in_periods,0); _integer(samples,1)
    omega=np.asarray(drive_frequencies,float); offsets=np.asarray(drive_phases,float)
    if omega.ndim!=1 or not len(omega) or offsets.shape!=omega.shape or not np.isfinite(omega).all() or not np.isfinite(offsets).all() or (omega<=0).any() or not np.isfinite([initial_time,period]).all() or period<=0: raise ValueError('invalid sampling controls')
    if 't_span' in solver_parameters: raise ValueError('sampler owns t_span')
    times=[]; states=[]; calls=accepted=rejected=0; status='completed'; t=float(initial_time)
    for i in range(burn_in_periods+samples):
        endpoint=initial_time+i*period
        if i:
            result=integrate(rhs,y,t_span=(t,endpoint),**solver_parameters)
            calls+=result.evaluations; accepted+=result.accepted_steps; rejected+=result.rejected_steps
            if result.status!='completed': status=result.status; break
            y=result.states[-1]; t=endpoint
        if i>=burn_in_periods: times.append(t); states.append(y.copy())
    times=np.asarray(times)
    return StroboscopicResult(times,np.array(states).reshape(-1,dimension),(times[:,None]*omega+offsets)%(2*np.pi),calls,accepted,rejected,status)


@dataclass
class VariationalMapResult:
    endpoint: np.ndarray
    monodromy: np.ndarray
    evaluations: int
    status: str


def variational_map(rhs,jacobian,state,*,initial_time,period,solver_parameters):
    y=_state(state)
    if not np.isfinite([initial_time,period]).all() or period<=0 or 't_span' in solver_parameters: raise ValueError('invalid map controls')
    def augmented(t,z):
        f=np.asarray(rhs(t,z[:2].copy()),float); j=np.asarray(jacobian(t,z[:2].copy()),float)
        if f.shape!=(2,) or j.shape!=(2,2): raise ValueError('RHS/Jacobian shape mismatch')
        return np.concatenate((f,(j@z[2:].reshape(2,2)).ravel()))
    result=integrate(augmented,np.concatenate((y,np.eye(2).ravel())),t_span=(initial_time,initial_time+period),**solver_parameters)
    return VariationalMapResult(result.states[-1,:2],result.states[-1,2:].reshape(2,2),result.evaluations,result.status)


@dataclass
class ShootingResult:
    state: np.ndarray
    endpoint: np.ndarray
    monodromy: np.ndarray
    residual_history: np.ndarray
    evaluations: int
    status: str


def shoot_periodic(rhs,jacobian,initial,*,initial_time,period,period_multiple,
                   tolerance,max_iterations,escape_radius,solver_parameters):
    """Newton on P^m(y)-y. Unwrapped state equality, no quotient-angle residual."""
    y=_state(initial); _integer(period_multiple,1); _integer(max_iterations,0)
    if not np.isfinite([tolerance,escape_radius]).all() or tolerance<=0 or escape_radius<=0: raise ValueError('invalid shooting controls')
    history=[]; calls=0; status='max_iterations'
    endpoint=y.copy(); monodromy=np.eye(2)
    for iteration in range(max_iterations+1):
        if np.max(abs(y))>escape_radius: status='escaped'; break
        result=variational_map(rhs,jacobian,y,initial_time=initial_time,period=period*period_multiple,solver_parameters=solver_parameters)
        calls+=result.evaluations; endpoint=result.endpoint; monodromy=result.monodromy
        if result.status!='completed': status='integration_'+result.status; break
        residual=endpoint-y; history.append(float(np.linalg.norm(residual,np.inf)))
        if history[-1]<=tolerance: status='converged'; break
        if iteration==max_iterations: break
        try: delta=np.linalg.solve(monodromy-np.eye(2),-residual)
        except np.linalg.LinAlgError: status='singular_solve'; break
        y=y+delta
        if not np.isfinite(y).all(): status='nonfinite'; break
    return ShootingResult(y,endpoint,monodromy,np.array(history),calls,status)


def classify_multipliers(matrix,*,tolerance=1e-10):
    m=np.asarray(matrix,float)
    if m.ndim!=2 or m.shape[0]!=m.shape[1] or not len(m) or not np.isfinite(m).all() or not np.isfinite(tolerance) or tolerance<0: raise ValueError('invalid multiplier matrix')
    values=np.linalg.eigvals(m); mod=np.abs(values)
    stable=int(np.sum(mod<1-tolerance)); unstable=int(np.sum(mod>1+tolerance)); neutral=len(values)-stable-unstable
    label=None
    if neutral==0 and np.all(mod>0):
        orientation=np.prod(values[mod>1+tolerance]).real
        label=('P' if unstable%2==0 else 'N')+('D' if orientation>0 else 'I')
    return dict(multipliers=values,stable_dimension=stable,unstable_dimension=unstable,neutral_dimension=neutral,
                hyperbolic=neutral==0,stability='unstable' if unstable else ('nonhyperbolic' if neutral else 'stable'),label=label)


def normalized_autocorrelation(x,*,sample_interval,max_lag):
    if not np.isfinite(sample_interval) or sample_interval<=0: raise ValueError('positive interval required')
    lags,covariance,counts=autocovariance(x,max_lag=max_lag)
    if covariance[0]<=0: raise ValueError('nonzero variance required')
    return lags*sample_interval,covariance/covariance[0],counts


def periodogram(x,*,sample_interval,demean,window):
    """One-sided density; integral equals window-weighted mean square.

    Frequencies cycles/time, density signal^2*time; symmetric np.hanning window.
    """
    y=np.asarray(x)
    if y.ndim!=1 or len(y)<2 or not np.isrealobj(y) or not np.isfinite(y).all() or not np.isfinite(sample_interval) or sample_interval<=0 or not isinstance(demean,bool): raise ValueError('invalid real signal/controls')
    if window not in ('rectangular','hann'): raise ValueError('unknown window')
    y=y.astype(float); y=y-y.mean() if demean else y.copy()
    weights=np.ones(len(y)) if window=='rectangular' else np.hanning(len(y))
    energy=float(weights@weights)
    if energy==0: raise ValueError('window requires more samples')
    density=sample_interval/energy*abs(np.fft.rfft(weights*y))**2
    density[1:-1 if len(y)%2==0 else None]*=2
    return np.fft.rfftfreq(len(y),d=sample_interval),density
