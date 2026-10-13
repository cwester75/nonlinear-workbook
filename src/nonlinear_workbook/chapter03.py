"""Autonomous planar systems; dimensionless time unless otherwise specified."""
from dataclasses import dataclass
import numpy as np


def _state(state):
    a=np.asarray(state,dtype=float)
    if a.shape!=(2,) or not np.isfinite(a).all():
        raise ValueError('state must contain two finite real coordinates')
    return a


def _parameters(*values,positive=False):
    if not np.isfinite(values).all() or (positive and any(v<=0 for v in values)):
        raise ValueError('invalid parameters')


def van_der_pol(t,state,*,mu):
    x,v=_state(state); _parameters(t,mu)
    return np.array([v,-x+mu*(1-x*x)*v])


def van_der_pol_jacobian(t,state,*,mu):
    x,v=_state(state); _parameters(t,mu)
    return np.array([[0.,1.],[-1-2*mu*x*v,mu*(1-x*x)]])


def radial_equilibrium(t,state):
    a=_state(state); _parameters(t)
    return a*(1-a@a)


def radial_equilibrium_jacobian(t,state):
    a=_state(state); _parameters(t)
    return (1-a@a)*np.eye(2)-2*np.outer(a,a)


def homoclinic(t,state):
    x,v=_state(state); _parameters(t)
    return np.array([v,x/2-x*x-x**3])


def homoclinic_jacobian(t,state):
    x,v=_state(state); _parameters(t)
    return np.array([[0.,1.],[.5-2*x-3*x*x,0.]])


def homoclinic_energy(state):
    x,v=_state(state)
    return float(v*v/2-x*x/4+x**3/3+x**4/4)


def homoclinic_reference(t):
    """Time-centered positive separatrix, shape (...,2), stable for large |t|.

    With s=t/sqrt(2), sech(s)=2 exp(-|s|)/(1+exp(-2|s|)).
    q=3 sech(s)/(2 sech(s)+sqrt(13)); v=q'.
    """
    t=np.asarray(t,float)
    if not np.isfinite(t).all(): raise ValueError('finite times required')
    z=np.exp(-np.abs(t)/np.sqrt(2)); sech=2*z/(1+z*z)
    denominator=2*sech+np.sqrt(13)
    q=3*sech/denominator
    v=-3*np.sqrt(13)*sech*np.tanh(t/np.sqrt(2))/(np.sqrt(2)*denominator**2)
    return np.stack((q,v),axis=-1)


def pendulum(t,state):
    q,p=_state(state); _parameters(t)
    return np.array([p,-np.sin(q)])


def pendulum_jacobian(t,state):
    q,p=_state(state); _parameters(t)
    return np.array([[0.,1.],[-np.cos(q),0.]])


def pendulum_energy(state):
    q,p=_state(state)
    return float(p*p/2+1-np.cos(q))


def sine_damping(t,state,*,mu):
    x,v=_state(state); _parameters(t,mu)
    return np.array([v,-x-mu*np.sin(v)])


def sine_damping_jacobian(t,state,*,mu):
    x,v=_state(state); _parameters(t,mu)
    return np.array([[0.,1.],[-1.,-mu*np.cos(v)]])


def van_der_pol_listing_step(state,*,step,mu):
    """p.148 drift then frozen-position exponential damping/kick; not symplectic.

    Matches x_new=x+h*v, v_new=v*exp(-h*mu*(x_new**2-1))-h*x_new.
    """
    x,v=_state(state); _parameters(step,mu)
    if step<=0: raise ValueError('positive step required')
    x=x+step*v
    return np.array([x,v*np.exp(-step*mu*(x*x-1))-step*x])


def lotka_volterra(t,state,*,a,b,c):
    x,y=_state(state); _parameters(t); _parameters(a,b,c,positive=True)
    return np.array([a*x-b*x*y,-c*y+b*x*y])


def lotka_volterra_jacobian(t,state,*,a,b,c):
    x,y=_state(state); _parameters(t); _parameters(a,b,c,positive=True)
    return np.array([[a-b*y,-b*x],[b*y,-c+b*x]])


def lotka_volterra_invariant(state,*,a,b,c):
    x,y=_state(state); _parameters(a,b,c,positive=True)
    if x<=0 or y<=0: raise ValueError('logarithmic invariant requires positive populations')
    # Generalizes p.150's log(x)+log(y)-x-y for a=b=c=1.
    return float(c*np.log(x)+a*np.log(y)-b*(x+y))


def equilibria(system,*,mu=None,a=1.,b=1.,c=1.,pendulum_indices=(-1,0,1),circle_angles=()):
    """Finite equilibrium fixtures; radial circle points are supplied, not exhaustive."""
    if system in ('van_der_pol','sine_damping'):
        if mu is None: raise ValueError('mu required')
        _parameters(mu)
        return np.array([[0.,0.]])
    if system=='homoclinic': return np.array([[0.,0.],[(-1-np.sqrt(3))/2,0.],[(-1+np.sqrt(3))/2,0.]])
    if system=='pendulum':
        indices=np.asarray(pendulum_indices)
        if indices.ndim!=1 or not len(indices) or indices.dtype.kind not in 'iu': raise ValueError('integer pendulum indices required')
        return np.column_stack((indices*np.pi,np.zeros(len(indices))))
    if system=='lotka_volterra':
        _parameters(a,b,c,positive=True)
        return np.array([[0.,0.],[c/b,a/b]])
    if system=='radial':
        angles=np.asarray(circle_angles,float)
        if angles.ndim!=1 or not np.isfinite(angles).all(): raise ValueError('finite angles required')
        return np.vstack(([[0.,0.]],np.column_stack((np.cos(angles),np.sin(angles)))))
    raise ValueError('unknown system')


@dataclass(frozen=True)
class PlanarClassification:
    eigenvalues: np.ndarray
    trace: float
    determinant: float
    discriminant: float
    stability: str
    geometric_type: str


def classify_planar(matrix,*,tolerance=1e-10):
    """Continuous-time linear classification; nonhyperbolic nonlinear cases inconclusive."""
    m=np.asarray(matrix,float)
    if m.shape!=(2,2) or not np.isfinite(m).all() or not np.isfinite(tolerance) or tolerance<0:
        raise ValueError('finite 2x2 matrix and nonnegative tolerance required')
    trace=float(np.trace(m)); det=float(np.linalg.det(m)); disc=trace*trace-4*det
    values=np.linalg.eigvals(m); real=values.real
    if np.any(np.abs(real)<=tolerance): stability='nonhyperbolic'
    elif np.all(real < -tolerance): stability='stable'
    else: stability='unstable'
    if np.any(np.abs(values)<=tolerance): kind='degenerate'
    elif np.any(values.imag>tolerance): kind='linear_center' if stability=='nonhyperbolic' else 'spiral'
    elif min(real)<-tolerance and max(real)>tolerance: kind='saddle'
    elif abs(values[0]-values[1])<=tolerance:
        kind='star_node' if np.max(np.abs(m-np.eye(2)*trace/2))<=tolerance else 'defective_node'
    else: kind='node'
    return PlanarClassification(values,trace,det,disc,stability,kind)


def section_crossings(times,states,*,coordinate,level,direction):
    """Linear interpolation between samples, explicit section direction +/-1.

    Returns times, interpolated states, successive intervals. No root refinement.
    """
    t=np.asarray(times,float); y=np.asarray(states,float)
    if t.ndim!=1 or len(t)<2 or y.shape!=(len(t),2) or not np.isfinite(t).all() or not np.isfinite(y).all() or np.any(np.diff(t)<=0) or coordinate not in (0,1) or direction not in (-1,1) or not np.isfinite(level):
        raise ValueError('invalid section inputs')
    z=direction*(y[:,coordinate]-level)
    indices=np.where((z[:-1]<0)&(z[1:]>=0))[0]
    fractions=-z[indices]/(z[indices+1]-z[indices])
    crossing_times=t[indices]+fractions*(t[indices+1]-t[indices])
    crossing_states=y[indices]+fractions[:,None]*(y[indices+1]-y[indices])
    return crossing_times,crossing_states,np.diff(crossing_times)
