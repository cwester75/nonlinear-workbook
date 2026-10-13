"""Nonlinear dissipative systems from Chapter 5, with named parameters."""
from dataclasses import dataclass
import numpy as np


def _state(state,n):
    y=np.asarray(state,float)
    if y.shape!=(n,) or not np.isfinite(y).all(): raise ValueError(f'expected {n} finite coordinates')
    return y


def _positive(*parameters):
    if not np.isfinite(parameters).all() or any(p<=0 for p in parameters): raise ValueError('positive finite parameters required')


def _lorenz_parameters(sigma,r,b):
    _positive(sigma,b)
    if not np.isfinite(r) or r<0: raise ValueError('r must be finite and nonnegative')


def lorenz(t,state,*,sigma,r,b):
    x,y,z=_state(state,3); _lorenz_parameters(sigma,r,b)
    return np.array([sigma*(y-x),r*x-y-x*z,x*y-b*z])


def lorenz_jacobian(t,state,*,sigma,r,b):
    x,y,z=_state(state,3); _lorenz_parameters(sigma,r,b)
    return np.array([[-sigma,sigma,0],[r-z,-1,-x],[y,x,-b]])


def lorenz_equilibria(*,sigma,r,b):
    _lorenz_parameters(sigma,r,b)
    if r<=1: return np.zeros((1,3))
    x=np.sqrt(b)*np.sqrt(r-1)
    return np.array([[0,0,0],[x,x,r-1],[-x,-x,r-1]])


def lorenz_hopf_threshold(*,sigma,b):
    _positive(sigma,b)
    if sigma<=b+1: raise ValueError('threshold requires sigma>b+1')
    return float(sigma*(sigma+b+3)/(sigma-b-1))


def generalized_lotka_volterra(t,state,*,a,b,c):
    x,y,z=_state(state,3); _positive(a,b,c)
    return np.array([x-x*y+c*x*x-a*z*x*x,-y+x*y,-b*z+a*x*x*z])


def generalized_lotka_volterra_jacobian(t,state,*,a,b,c):
    x,y,z=_state(state,3); _positive(a,b,c)
    return np.array([[1-y+2*c*x-2*a*z*x,-x,-a*x*x],[y,x-1,0],[2*a*x*z,0,-b+a*x*x]])


def hyperchaotic(t,state,*,a,b,c,d):
    x,y,z,w=_state(state,4)
    if not np.isfinite([a,b,c,d]).all(): raise ValueError('finite parameters required')
    return np.array([-y-z,x+a*y+w,b+x*z,-c*z+d*w])


def hyperchaotic_jacobian(t,state,*,a,b,c,d):
    x,y,z,w=_state(state,4)
    if not np.isfinite([a,b,c,d]).all(): raise ValueError('finite parameters required')
    return np.array([[0,-1,-1,0],[1,a,0,1],[z,0,x,0],[0,0,-c,d]])


def belousov_zhabotinsky(t,state,*,s,q,f,w):
    x,y,z=_state(state,3); _positive(s,q,f,w)
    return np.array([s*(y-x*y+x-q*x*x),(f*z-y-x*y)/s,w*(x-z)])


def belousov_zhabotinsky_jacobian(t,state,*,s,q,f,w):
    x,y,z=_state(state,3); _positive(s,q,f,w)
    return np.array([[s*(1-y-2*q*x),s*(1-x),0],[-y/s,(-1-x)/s,f/s],[w,0,-w]])


def chemical_equilibria(*,s,q,f,w):
    """Origin and both real nonzero branches; positive mask is explicit."""
    _positive(s,q,f,w)
    coefficient=q+f-1
    root=np.hypot(coefficient,2*np.sqrt(q)*np.sqrt(1+f))
    # Stable quadratic roots: q*x^2+(q+f-1)*x-(1+f)=0.
    numerator=-.5*(coefficient+np.copysign(root,coefficient))
    roots=(numerator/q,-(1+f)/numerator)
    states=np.array([[0.,0.,0.]]+[[x,f*x/(1+x),x] for x in roots])
    if not np.isfinite(states).all(): raise ValueError('nonfinite equilibria')
    return states,np.all(states>0,axis=1)


def lorenz_time_integral(t,state,*,sigma,r,b):
    _,y,z=_state(state,3); _lorenz_parameters(sigma,r,b)
    if r!=0 or b!=1 or not np.isfinite(t): raise ValueError('integral requires r=0,b=1 and finite time')
    with np.errstate(over='ignore',invalid='ignore'):
        value=(y*y+z*z)*np.exp(2*t)
    if not np.isfinite(value): raise ValueError('nonfinite integral')
    return float(value)


@dataclass(frozen=True)
class EquilibriumClassification:
    eigenvalues: np.ndarray
    stable_dimension: int
    unstable_dimension: int
    center_dimension: int
    hyperbolic: bool
    stability: str


def classify_equilibrium(matrix,*,tolerance=1e-10):
    m=np.asarray(matrix,float)
    if m.ndim!=2 or m.shape[0]!=m.shape[1] or not len(m) or not np.isfinite(m).all() or not np.isfinite(tolerance) or tolerance<0: raise ValueError('invalid Jacobian/tolerance')
    values=np.linalg.eigvals(m); real=values.real
    stable=int(np.sum(real < -tolerance)); unstable=int(np.sum(real > tolerance)); center=len(values)-stable-unstable
    return EquilibriumClassification(values,stable,unstable,center,center==0,
        'unstable' if unstable else ('nonhyperbolic' if center else 'stable'))


def hopf_candidate(jacobian,branch,*,bracket,tolerance=1e-8,max_iterations=100,eigen_tolerance=1e-8):
    """jacobian(parameter,state), branch(parameter); one isolated complex pair.

    Caller owns branch selection and equilibrium verification. Multiple pairs are
    ambiguous and rejected; real crossings are not Hopf candidates.
    """
    lo,hi=map(float,bracket)
    if not np.isfinite([lo,hi,tolerance,eigen_tolerance]).all() or lo>=hi or tolerance<=0 or eigen_tolerance<=0 or not isinstance(max_iterations,int) or max_iterations<1: raise ValueError('invalid crossing controls')
    def eigen(p):
        state=np.asarray(branch(p),float)
        if state.ndim!=1 or not len(state) or not np.isfinite(state).all(): raise ValueError('invalid equilibrium branch')
        m=np.asarray(jacobian(p,state),float)
        if m.shape!=(len(state),len(state)): raise ValueError('branch/Jacobian shape mismatch')
        values=classify_equilibrium(m,tolerance=eigen_tolerance).eigenvalues
        pairs=values[values.imag>eigen_tolerance]
        if len(pairs)>1: raise ValueError('ambiguous complex eigenvalue branches')
        if len(pairs)==0: return None
        others=values[np.abs(values.imag)<=eigen_tolerance]
        if np.any(np.abs(others.real)<=eigen_tolerance): raise ValueError('additional center eigenvalue')
        return pairs[0]
    left,right=eigen(lo),eigen(hi)
    if left is None or right is None: return dict(status='no_complex_pair',parameter=None)
    if left.real*right.real>0: return dict(status='not_bracketed',parameter=None)
    iterations=0
    while hi-lo>tolerance and iterations<max_iterations:
        mid=(lo+hi)/2; value=eigen(mid); iterations+=1
        if value is None: raise ValueError('complex branch disappears within bracket')
        if left.real*value.real<=0: hi=mid
        else: lo=mid; left=value
    parameter=(lo+hi)/2; value=eigen(parameter)
    return dict(status='hopf_candidate' if hi-lo<=tolerance else 'max_iterations',parameter=parameter,
                eigenvalue=value,bracket=[lo,hi],iterations=iterations)
