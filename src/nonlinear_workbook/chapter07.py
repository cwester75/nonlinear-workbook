"""Control and synchronization, printed pp.243–274; natural logs/time or update."""
from dataclasses import dataclass
import numpy as np
from .chapter01.maps import logistic_step, logistic_derivative
from .chapter05 import lorenz, lorenz_jacobian


def _state(y,n=None):
    y=np.asarray(y,float)
    if y.ndim!=1 or not len(y) or (n is not None and y.shape!=(n,)) or not np.isfinite(y).all():
        raise ValueError('finite vector of matching size required')
    return y


def _finite(*v):
    if not np.isfinite(v).all(): raise ValueError('finite parameters required')


def henon_additive_step(state,*,a,b):
    x,y=_state(state,2); _finite(a,b)
    return np.array([a-x*x+b*y,x])


def henon_additive_jacobian(state,*,a,b):
    x,y=_state(state,2); _finite(a,b)
    return np.array([[-2*x,b],[1.,0.]])


def henon_additive_fixed_points(*,a,b):
    _finite(a,b); c=(1-b)/2; rad=c*c+a
    if rad<0: return np.empty((0,2))
    roots=np.unique([-c-np.sqrt(rad),-c+np.sqrt(rad)])
    return np.column_stack((roots,roots))


def place_poles_2d(A,B,*,poles):
    """Real SISO gain K for A-BK; controllability required (Ackermann solve)."""
    A=np.asarray(A,float); B=_state(B,2); poles=np.asarray(poles,complex)
    if A.shape!=(2,2) or not np.isfinite(A).all() or poles.shape!=(2,) or not np.isfinite(poles).all():
        raise ValueError('finite 2D system and two poles required')
    coefficients=np.poly(poles)
    if np.max(abs(coefficients.imag))>1e-12: raise ValueError('real or conjugate poles required')
    C=np.column_stack((B,A@B))
    if np.linalg.matrix_rank(C)<2: raise ValueError('uncontrollable system')
    row=np.linalg.solve(C.T,[0.,1.])
    K=row@(A@A+coefficients[1].real*A+coefficients[2].real*np.eye(2))
    return K


def logistic_targets(*,r,period):
    _finite(r)
    if not 0<r<=4 or period not in (1,2): raise ValueError('r in (0,4] and period 1/2 required')
    if period==1:
        if r<1: raise ValueError('nonzero fixed-point branch must lie in [0,1]')
        return np.array([1-1/r])
    if r<=3: raise ValueError('distinct real period-two orbit requires r>3')
    root=np.sqrt((r-3)*(r+1))
    return np.array([(r+1-root)/(2*r),(r+1+root)/(2*r)])


@dataclass
class ControlResult:
    states: np.ndarray
    measured_states: np.ndarray
    parameters: np.ndarray
    proposed_controls: np.ndarray
    controls: np.ndarray
    phases: np.ndarray
    errors: np.ndarray
    active: np.ndarray
    reasons: list
    status: str


def _simulate(step,initial,proposal,*,nominal,updates,max_perturbation,escape_radius,noise_std,seed):
    y=_state(initial).copy(); _finite(nominal,max_perturbation,escape_radius,noise_std)
    if not isinstance(updates,(int,np.integer)) or isinstance(updates,bool) or updates<1 or max_perturbation<0 or escape_radius<=0 or noise_std<0:
        raise ValueError('invalid control limits')
    if noise_std>0 and seed is None: raise ValueError('explicit noise seed required')
    rng=np.random.default_rng(seed); states=[y.copy()]; measured=[]; parameters=[]; proposed=[]
    applied=[]; phases=[]; errors=[]; active=[]; reasons=[]; status='completed'
    if np.max(abs(y))>escape_radius: status='escaped'
    for _ in range(0 if status=='escaped' else updates):
        observation=y+rng.normal(0,noise_std,size=y.shape)
        raw,phase,error,reason=proposal(observation)
        if not np.isfinite(raw): reason='unavailable'; raw=0.
        enabled=reason=='eligible' and abs(raw)<=max_perturbation
        if reason=='eligible' and not enabled: reason='perturbation_bound'
        delta=raw if enabled else 0.
        with np.errstate(over='ignore',invalid='ignore'):
            new=np.asarray(step(y,nominal+delta),float)
        if new.shape!=y.shape: raise ValueError('map shape mismatch')
        if not np.isfinite(new).all(): status='nonfinite'; break
        measured.append(observation); parameters.append(nominal+delta); proposed.append(raw); applied.append(delta)
        phases.append(phase); errors.append(error); active.append(enabled); reasons.append('active' if enabled else reason)
        y=new; states.append(y.copy())
        if np.max(abs(y))>escape_radius: status='escaped'; break
    return ControlResult(np.array(states),np.array(measured).reshape(-1,len(y)),np.array(parameters),
        np.array(proposed),np.array(applied),np.array(phases,dtype=int),np.array(errors),np.array(active,dtype=bool),reasons,status)


def logistic_ogy(initial,*,r,period,updates,neighborhood,max_perturbation,noise_std=0.,seed=None):
    targets=logistic_targets(r=r,period=period); _finite(initial,neighborhood)
    if not 0<=initial<=1 or neighborhood<=0 or max_perturbation>min(r,4-r):
        raise ValueError('invalid logistic state/neighborhood/perturbation range')
    def proposal(y):
        phase=int(np.argmin(abs(targets-y[0]))); target=targets[phase]; error=y[0]-target
        denominator=target*(1-target)
        if denominator==0: return 0.,phase,abs(error),'singular_direction'
        delta=-r*(1-2*target)*error/denominator
        return delta,phase,abs(error),'eligible' if abs(error)<neighborhood else 'outside_neighborhood'
    return _simulate(lambda y,p:np.array([logistic_step(y[0],r=p)]),[initial],proposal,nominal=r,updates=updates,
        max_perturbation=max_perturbation,escape_radius=1.,noise_std=noise_std,seed=seed)


def henon_ogy(initial,*,a,b,target,gain,updates,neighborhood,max_perturbation,
              escape_radius,noise_std=0.,seed=None):
    target=_state(target,2); gain=_state(gain,2); _finite(a,b,neighborhood)
    if neighborhood<=0 or np.max(abs(henon_additive_step(target,a=a,b=b)-target))>1e-10:
        raise ValueError('fixed point and positive neighborhood required')
    def proposal(y):
        error=y-target; norm=float(np.max(abs(error)))
        return float(-gain@error),0,norm,'eligible' if norm<neighborhood else 'outside_neighborhood'
    return _simulate(lambda y,p:henon_additive_step(y,a=p,b=b),_state(initial,2),proposal,
        nominal=a,updates=updates,max_perturbation=max_perturbation,escape_radius=escape_radius,noise_std=noise_std,seed=seed)


def unstable_projection_control(J,B,error,left_next,*,denominator_tolerance):
    """delta=-l_(n+1) J_n error/(l_(n+1) B); caller supplies the next covector."""
    error=_state(error); B=_state(B,len(error)); left=_state(left_next,len(error)); J=np.asarray(J,float)
    if J.shape!=(len(error),len(error)) or not np.isfinite(J).all() or not np.isfinite(denominator_tolerance) or denominator_tolerance<=0:
        raise ValueError('invalid projection')
    norm=np.linalg.norm(left)
    if norm==0: return 0.,'singular_direction'
    left=left/norm; denominator=left@B
    if abs(denominator)<=denominator_tolerance: return 0.,'singular_direction'
    return float(-(left@J@error)/denominator),'eligible'


def henon_synchronize(driver,response,*,a,b,updates,neighborhood,max_perturbation,
                      escape_radius,denominator_tolerance=1e-12,noise_std=0.,seed=None):
    _finite(a,b,neighborhood,denominator_tolerance)
    if b<=0 or neighborhood<=0 or denominator_tolerance<=0: raise ValueError('positive b, neighborhood and denominator tolerance required')
    def proposal(y):
        master,slave=y[:2],y[2:]; error=slave-master; norm=float(np.sum(abs(error)))
        if norm>=neighborhood: return 0.,-1,norm,'outside_neighborhood'
        J=henon_additive_jacobian(master,a=a,b=b)
        next_master=henon_additive_step(master,a=a,b=b)
        next_J=henon_additive_jacobian(next_master,a=a,b=b)
        values,vectors=np.linalg.eig(next_J.T); unstable=np.flatnonzero(abs(values)>1)
        if len(unstable)!=1: return 0.,-1,norm,'unavailable_direction'
        raw,reason=unstable_projection_control(J,[1,0],error,vectors[:,unstable[0]].real,
            denominator_tolerance=denominator_tolerance)
        return raw,0,norm,reason
    return _simulate(lambda y,p:np.r_[henon_additive_step(y[:2],a=a,b=b),henon_additive_step(y[2:],a=p,b=b)],
        np.r_[_state(driver,2),_state(response,2)],proposal,nominal=a,updates=updates,max_perturbation=max_perturbation,
        escape_radius=escape_radius,noise_std=noise_std,seed=seed)


def modulated_lorenz(t,state,*,sigma,r,b,k,omega,phase=0.):
    _finite(t,k,omega,phase)
    if omega<=0: raise ValueError('positive drive frequency required')
    y=_state(state,3); result=lorenz(t,y,sigma=sigma,r=r,b=b)
    result[1]+=k*np.cos(omega*t+phase)*y[0]
    return result


def modulated_lorenz_jacobian(t,state,*,sigma,r,b,k,omega,phase=0.):
    modulated_lorenz(t,state,sigma=sigma,r=r,b=b,k=k,omega=omega,phase=phase)
    J=lorenz_jacobian(t,state,sigma=sigma,r=r,b=b)
    J[1,0]+=k*np.cos(omega*t+phase)
    return J


def feedforward_force(rhs,target,target_derivative,t):
    _finite(t); y=_state(target(t)); derivative=_state(target_derivative(t),len(y))
    return derivative-_state(rhs(t,y.copy()),len(y))


def rossler(t,state,*,a,b,c):
    _finite(t,a,b,c); x,y,z=_state(state,3)
    if min(a,b,c)<=0: raise ValueError('positive Rossler parameters required')
    return np.array([-y-z,x+a*y,b+z*(x-c)])


def rossler_jacobian(t,state,*,a,b,c):
    rossler(t,state,a=a,b=b,c=c); x,y,z=_state(state,3)
    return np.array([[0,-1,-1],[1,a,0],[z,0,x-c]])


def rossler_drive_response(t,state,*,a,b,c):
    y=_state(state,5); master=rossler(t,y[:3],a=a,b=b,c=c); u,w=y[3:]
    return np.r_[master,-y[1]-w,b+w*(u-c)]


def rossler_drive_response_jacobian(t,state,*,a,b,c):
    rossler_drive_response(t,state,a=a,b=b,c=c); y=_state(state,5)
    J=np.zeros((5,5)); J[:3,:3]=rossler_jacobian(t,y[:3],a=a,b=b,c=c)
    J[3,1]=-1; J[3,4]=-1; J[4,3]=y[4]; J[4,4]=y[3]-c
    return J


def rossler_response_jacobian(t,state,*,a,b,c):
    rossler_drive_response(t,state,a=a,b=b,c=c); y=_state(state,5)
    return np.array([[0.,-1.],[y[4],y[3]-c]])


def diffusive_rhs(rhs,t,state,*,coupling):
    y=_state(state); _finite(t,coupling)
    if len(y)%2 or coupling<0: raise ValueError('paired state and nonnegative coupling required')
    n=len(y)//2; u,v=y[:n],y[n:]; diff=coupling*(v-u)
    return np.r_[_state(rhs(t,u.copy()),n)+diff,_state(rhs(t,v.copy()),n)-diff]


def diffusive_jacobian(jacobian,t,state,*,coupling):
    y=_state(state); _finite(t,coupling)
    if len(y)%2 or coupling<0: raise ValueError('paired state and nonnegative coupling required')
    n=len(y)//2; A=np.asarray(jacobian(t,y[:n].copy()),float); B=np.asarray(jacobian(t,y[n:].copy()),float)
    if A.shape!=(n,n) or B.shape!=(n,n) or not np.isfinite([A,B]).all(): raise ValueError('Jacobian shape/nonfinite')
    C=coupling*np.eye(n)
    return np.block([[A-C,C],[C,B-C]])


def logistic_transverse_curve(observations,*,r,coupling):
    """Along a supplied synchronized orbit; rates are per map update."""
    x=_state(observations); _finite(r,coupling)
    if not 0<r<=4 or coupling<0 or np.any((x<0)|(x>1)): raise ValueError('invalid orbit/parameters')
    factors=np.array([logistic_derivative(s,r=r)-2*coupling for s in x])
    with np.errstate(divide='ignore'): running=np.cumsum(np.log(abs(factors)))/np.arange(1,len(x)+1)
    return dict(factors=factors,running_estimates=running,exponent=float(running[-1]),samples=len(x))
