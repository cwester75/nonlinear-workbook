"""Initial-value solvers and Chapter 10 discretization demonstrations.

Real vector states; rhs(t,y); time and state units are caller supplied.
"""
from dataclasses import dataclass
import math
import numpy as np


def _state(y):
    a=np.asarray(y,float)
    if a.ndim==0: a=a.reshape(1)
    if a.ndim!=1 or not len(a) or not np.isfinite(a).all(): raise ValueError('finite vector required')
    return a.copy()


def _rhs(f,t,y):
    # Detect invalid trial stages before a model's state validator sees them.
    # Do not catch its ValueError: that may identify invalid model parameters.
    if not np.isfinite(t) or not np.isfinite(y).all():
        raise FloatingPointError('nonfinite trial time/state')
    a=np.asarray(f(t,y.copy()),float)
    if a.ndim==0 and y.shape==(1,): a=a.reshape(1)
    if a.shape!=y.shape: raise ValueError('derivative shape mismatch')
    if not np.isfinite(a).all(): raise FloatingPointError('nonfinite derivative')
    return a


def euler_step(rhs,t,y,h):
    y=_state(y)
    if not np.isfinite([t,h]).all() or h==0: raise ValueError('finite nonzero step required')
    return y+h*_rhs(rhs,t,y)


def taylor_step(y,h,derivatives):
    """Truncated Lie/Taylor expansion from supplied time derivatives at y.

    derivatives[k-1] is d**k y/dt**k. No symbolic differentiation is inferred.
    """
    y=_state(y)
    if not np.isfinite(h) or not len(derivatives): raise ValueError('invalid expansion')
    result=y.copy()
    for k,d in enumerate(derivatives,1):
        d=_state(d)
        if d.shape!=y.shape: raise ValueError('derivative shape mismatch')
        result += h**k/math.factorial(k)*d
    return result


# Fehlberg 4(5), pp.322-323. Stages use derivatives, not h-scaled increments.
_A=((),(1/4,), (3/32,9/32), (1932/2197,-7200/2197,7296/2197),
    (439/216,-8,3680/513,-845/4104),(-8/27,2,-3544/2565,1859/4104,-11/40))
_C=(0,1/4,3/8,12/13,1,1/2)
_B4=np.array([25/216,0,1408/2565,2197/4104,-1/5,0])
_B5=np.array([16/135,0,6656/12825,28561/56430,-9/50,2/55])


def rkf45_step(rhs,t,y,h):
    """Return fifth-order state and embedded fifth-minus-fourth error; six calls."""
    y=_state(y)
    if not np.isfinite([t,h]).all() or h==0: raise ValueError('finite nonzero step required')
    stages=[]
    for c,row in zip(_C,_A):
        stage=y+h*sum((a*k for a,k in zip(row,stages)),np.zeros_like(y))
        stages.append(_rhs(rhs,t+c*h,stage))
    k=np.array(stages)
    return y+h*(_B5@k),h*((_B5-_B4)@k)


@dataclass
class IntegrationResult:
    times: np.ndarray
    states: np.ndarray
    step_sizes: np.ndarray
    error_norms: np.ndarray
    accepted_steps: int
    rejected_steps: int
    evaluations: int
    status: str


def integrate(rhs,y0,*,t_span,method,step,max_steps=100000,escape_radius=np.inf,
              atol=1e-9,rtol=1e-6,min_step=1e-12,max_step=np.inf):
    """Euler/fixed Fehlberg/adaptive Fehlberg, including initial and terminal states.

    Forward integration. Adaptive norm is max(error/(atol+rtol*max(|old|,|new|))).
    Stop statuses: completed, max_steps, step_underflow, nonfinite, escaped.
    Only accepted states enter the trajectory; rejected trials enter counters.
    """
    y=_state(y0)
    if len(t_span)!=2: raise ValueError('two endpoint times required')
    t,end=map(float,t_span)
    if not np.isfinite([t,end,step,min_step]).all() or end<=t or step<=0 or min_step<=0 or max_step<min_step or np.isnan(max_step): raise ValueError('invalid times/steps')
    if not isinstance(max_steps,(int,np.integer)) or isinstance(max_steps,bool) or max_steps<1: raise ValueError('positive integer max_steps required')
    if method not in ('euler','rkf45','rkf45_adaptive'): raise ValueError('unknown method')
    if escape_radius<=0 or np.isnan(escape_radius): raise ValueError('invalid escape radius')
    atol=np.asarray(atol,float)
    if atol.ndim>1 or (atol.ndim==1 and atol.shape!=y.shape) or not np.isfinite(atol).all() or (atol<=0).any() or not np.isfinite(rtol) or rtol<0: raise ValueError('invalid tolerances')
    times=[t]; states=[y.copy()]; steps=[]; errors=[]; rejected=0; calls=0; status='max_steps'
    h=min(step,max_step)
    def counted_rhs(t,y):
        nonlocal calls
        calls+=1
        return rhs(t,y)
    if np.max(np.abs(y))>escape_radius: status='escaped'; max_steps=0
    for _ in range(max_steps):
        h=min(h,end-t,max_step)
        if t+h==t: status='step_underflow'; break
        try:
            with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
                if method=='euler': new=euler_step(counted_rhs,t,y,h); error=np.zeros_like(y)
                else: new,error=rkf45_step(counted_rhs,t,y,h)
        except (FloatingPointError,OverflowError):
            status='nonfinite'; break
        if not np.isfinite(new).all() or not np.isfinite(error).all(): status='nonfinite'; break
        norm=float(np.max(np.abs(error)/(atol+rtol*np.maximum(np.abs(y),np.abs(new)))))
        if method=='rkf45_adaptive' and norm>1:
            rejected+=1
            next_h=h*max(.1,.9*norm**(-.2))
            if next_h<min_step: status='step_underflow'; break
            h=next_h; continue
        t=end if h==end-t else t+h
        y=new; times.append(t); states.append(y.copy()); steps.append(h); errors.append(norm)
        if np.max(np.abs(y))>escape_radius: status='escaped'; break
        if t==end: status='completed'; break
        if method=='rkf45_adaptive': h=max(min_step,h*min(5.,max(.1,.9*norm**(-.2))) if norm else h*5)
    return IntegrationResult(np.array(times),np.array(states),np.array(steps),np.array(errors),len(steps),rejected,calls,status)


def symplectic_step(q,p,h,*,grad_potential,grad_kinetic,method):
    """Separable H=T(p)+V(q); kick-drift Euler or kick-drift-kick Verlet."""
    q,p=_state(q),_state(p)
    if q.shape!=p.shape or not np.isfinite(h) or h==0 or method not in ('euler','verlet'): raise ValueError('invalid canonical state/method')
    def grad(f,x):
        a=_state(f(x.copy()))
        if a.shape!=x.shape: raise ValueError('gradient shape mismatch')
        return a
    if method=='euler':
        p=p-h*grad(grad_potential,q); q=q+h*grad(grad_kinetic,p)
    else:
        p=p-h/2*grad(grad_potential,q); q=q+h*grad(grad_kinetic,p); p=p-h/2*grad(grad_potential,q)
    return q,p


def central_difference(rhs,u0,u1,*,step,n,escape_radius=np.inf):
    """Autonomous u[n+1]=u[n-1]+2*h*f(u[n]); includes two supplied states."""
    a,b=_state(u0),_state(u1)
    if a.shape!=b.shape or not np.isfinite(step) or step<=0 or not isinstance(n,(int,np.integer)) or n<2 or escape_radius<=0 or np.isnan(escape_radius): raise ValueError('invalid recurrence controls')
    values=[a,b]; status='completed'
    if max(np.max(np.abs(a)),np.max(np.abs(b)))>escape_radius:
        return np.arange(2)*step,np.array(values),'escaped'
    for _ in range(n-2):
        with np.errstate(over='ignore',invalid='ignore'): new=a+2*step*np.asarray(rhs(b.copy()),float)
        if new.shape!=b.shape: raise ValueError('derivative shape mismatch')
        if not np.isfinite(new).all(): status='nonfinite'; break
        values.append(new.copy())
        if np.max(np.abs(new))>escape_radius: status='escaped'; break
        a,b=b,new
    return np.arange(len(values))*step,np.array(values),status


def quartic_discrete_step(state):
    """p.335 scaled map (U_n,U_{n-1}) -> (2U_n-U_{n-1}-U_n**3,U_n)."""
    u,v=_state(state)
    return np.array([2*u-v-u**3,u])
