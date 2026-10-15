"""Hamiltonian examples, printed pp.153–176. States are (q..., p...)."""
from dataclasses import dataclass
import numpy as np
from .chapter10 import integrate, symplectic_step, IntegrationResult
from .chapter06 import variational_map, classify_multipliers


def _vector(x, n):
    x = np.asarray(x, float)
    if x.shape != (n,) or not np.isfinite(x).all():
        raise ValueError(f'finite {n}-vector required')
    return x


def _r(r):
    if not np.isfinite(r) or not 0 <= r <= 1:
        raise ValueError('r must lie in [0,1]')
    return r


def henon_heiles_potential(q):
    x, y = _vector(q, 2)
    return (x*x+y*y)/2+x*x*y-y**3/3


def henon_heiles_gradient(q):
    x, y = _vector(q, 2)
    return np.array([x+2*x*y, y+x*x-y*y])


def henon_heiles_hessian(q):
    x, y = _vector(q, 2)
    return np.array([[1+2*y, 2*x], [2*x, 1-2*y]])


def quartic_potential(q, *, r):
    x, y = _vector(q, 2); r = _r(r)
    return (1-r)*(x**4+y**4)/12+x*x*y*y/2


def quartic_gradient(q, *, r):
    x, y = _vector(q, 2); r = _r(r)
    return np.array([(1-r)*x**3/3+x*y*y, (1-r)*y**3/3+x*x*y])


def quartic_hessian(q, *, r):
    x, y = _vector(q, 2); r = _r(r)
    return np.array([[(1-r)*x*x+y*y, 2*x*y], [2*x*y, (1-r)*y*y+x*x]])


def _toda(q, periodic):
    if not isinstance(periodic,(bool,np.bool_)): raise ValueError('explicit boolean periodic required')
    q = _vector(q, 3)
    bonds = np.array([[1., -1., 0.], [0., 1., -1.], [-1., 0., 1.]])[:3 if periodic else 2]
    with np.errstate(over='ignore', invalid='ignore'):
        e = np.exp(bonds@q)
    if not np.isfinite(e).all():
        raise FloatingPointError('Toda exponential overflow')
    return bonds, e


def toda_potential(q, *, periodic):
    return float(np.sum(_toda(q, periodic)[1]))


def toda_gradient(q, *, periodic):
    b, e = _toda(q, periodic)
    return b.T@e


def toda_hessian(q, *, periodic):
    b, e = _toda(q, periodic)
    return b.T@(e[:, None]*b)


def _energy(state, n, potential):
    y = _vector(state, 2*n)
    return float(y[n:]@y[n:]/2+potential(y[:n]))


def henon_heiles_energy(state):
    return _energy(state, 2, henon_heiles_potential)


def quartic_energy(state, *, r):
    return _energy(state, 2, lambda q: quartic_potential(q, r=r))


def toda_energy(state, *, periodic):
    return _energy(state, 3, lambda q: toda_potential(q, periodic=periodic))


def _rhs(t, state, n, gradient):
    if not np.isfinite(t): raise ValueError('finite time required')
    y = _vector(state, 2*n)
    return np.concatenate((y[n:], -gradient(y[:n])))


def _jacobian(t, state, n, hessian):
    if not np.isfinite(t): raise ValueError('finite time required')
    y = _vector(state, 2*n)
    return np.block([[np.zeros((n,n)), np.eye(n)], [-hessian(y[:n]), np.zeros((n,n))]])


def henon_heiles_rhs(t, state):
    return _rhs(t, state, 2, henon_heiles_gradient)


def henon_heiles_jacobian(t, state):
    return _jacobian(t, state, 2, henon_heiles_hessian)


def quartic_rhs(t, state, *, r):
    return _rhs(t, state, 2, lambda q: quartic_gradient(q, r=r))


def quartic_jacobian(t, state, *, r):
    return _jacobian(t, state, 2, lambda q: quartic_hessian(q, r=r))


def toda_rhs(t, state, *, periodic):
    return _rhs(t, state, 3, lambda q: toda_gradient(q, periodic=periodic))


def toda_jacobian(t, state, *, periodic):
    return _jacobian(t, state, 3, lambda q: toda_hessian(q, periodic=periodic))


def quartic_integral(state, *, r=0):
    if _r(r) != 0: raise ValueError('extra integral applies only at r=0')
    x, y, p, v = _vector(state, 4)
    return 3*p*v+x*y*(x*x+y*y)


def quartic_rotation(state):
    """Canonical orthogonal rotation; H=sum(P**2/2+Q**4/6) at r=0."""
    x, y, p, v = _vector(state, 4)
    return np.array([x+y, x-y, p+v, p-v])/np.sqrt(2)


def periodic_toda_integrals(state):
    y = _vector(state, 6); p = y[3:]; _, e = _toda(y[:3], True)
    return np.array([p.sum(), p.prod()-p[0]*e[1]-p[1]*e[2]-p[2]*e[0]])


def open_toda_lax(state):
    """L'=A L-L A, a=exp((q_j-q_{j+1})/2)/2, b=p/2."""
    y = _vector(state, 6); _, e = _toda(y[:3], False)
    a = np.sqrt(e)/2; b = y[3:]/2
    L = np.diag(b)+np.diag(a,1)+np.diag(a,-1)
    A = np.diag(-a,1)+np.diag(a,-1)
    da = a*(b[:-1]-b[1:]); db = -toda_gradient(y[:3], periodic=False)/2
    derivative = np.diag(db)+np.diag(da,1)+np.diag(da,-1)
    return dict(a=a, b=b, L=L, A=A, derivative=derivative,
                residual=derivative-(A@L-L@A), eigenvalues=np.linalg.eigvalsh(L),
                invariants=np.array([np.trace(L), np.trace(L@L), np.linalg.det(L)]))


def henon_heiles_shell(*, energy, q2, p2, momentum_sign):
    if not np.isfinite([energy,q2,p2]).all() or momentum_sign not in (-1,1):
        raise ValueError('finite shell and sign +/-1 required')
    radicand = 2*(energy-henon_heiles_potential([0,q2]))-p2*p2
    if radicand < 0: raise ValueError('inaccessible energy shell')
    return np.array([0.,q2,momentum_sign*np.sqrt(radicand),p2])


def mathieu_rhs(t, state, *, delta, epsilon):
    u, v = _vector(state, 2)
    if not np.isfinite([t,delta,epsilon]).all(): raise ValueError('finite parameters required')
    return np.array([v,-(delta+2*epsilon*np.cos(2*t))*u])


def mathieu_jacobian(t, state, *, delta, epsilon):
    mathieu_rhs(t, state, delta=delta, epsilon=epsilon)
    return np.array([[0.,1.],[-(delta+2*epsilon*np.cos(2*t)),0.]])


def mathieu_floquet(*, delta, epsilon, solver_parameters, tolerance=1e-7):
    if not np.isfinite(tolerance) or tolerance <= 0: raise ValueError('positive tolerance required')
    result = variational_map(lambda t,y: mathieu_rhs(t,y,delta=delta,epsilon=epsilon),
        lambda t,y: mathieu_jacobian(t,y,delta=delta,epsilon=epsilon), [0,0],
        initial_time=0., period=np.pi, solver_parameters=solver_parameters)
    m = result.monodromy
    classification = classify_multipliers(m, tolerance=tolerance)
    trace = float(np.trace(m)); defect = abs(float(np.linalg.det(m))-1)
    if result.status != 'completed': band = 'integration_failure'
    elif defect > tolerance: band = 'unresolved'
    elif abs(trace) < 2-tolerance: band = 'bounded_oscillatory'
    elif abs(trace) > 2+tolerance: band = 'exponentially_unstable'
    else: band = 'boundary_unresolved'
    scalar_boundary = min(np.linalg.norm(m-np.eye(2)),np.linalg.norm(m+np.eye(2))) <= tolerance
    return dict(monodromy=m, classification=classification, determinant_error=defect,
                trace=trace, band=band, scalar_boundary=bool(scalar_boundary),
                status=result.status, evaluations=result.evaluations)


def integrate_separable(state, *, grad_potential, t_span, step, method,
                       max_steps=1000000, escape_radius=np.inf):
    """Unit-mass kinetic energy; counts gradient evaluations, including failed trials."""
    y = np.asarray(state,float)
    if y.ndim != 1 or len(y)%2 or not len(y) or not np.isfinite(y).all(): raise ValueError('canonical state required')
    if len(t_span)!=2 or not np.isfinite([*t_span,step]).all() or t_span[1]<=t_span[0] or step<=0:
        raise ValueError('invalid time controls')
    if not isinstance(max_steps,(int,np.integer)) or isinstance(max_steps,bool) or max_steps<1 or method not in ('euler','verlet'):
        raise ValueError('invalid method/step limit')
    if np.isnan(escape_radius) or escape_radius<=0: raise ValueError('invalid escape radius')
    n=len(y)//2; t,end=map(float,t_span); times=[t]; states=[y.copy()]; steps=[]; calls=0
    def gradient(q):
        nonlocal calls
        calls+=1
        a=np.asarray(grad_potential(q),float)
        if a.shape!=(n,): raise ValueError('gradient shape mismatch')
        if not np.isfinite(a).all(): raise FloatingPointError('nonfinite gradient')
        return a
    status='escaped' if np.max(abs(y))>escape_radius else 'max_steps'
    for _ in range(0 if status=='escaped' else max_steps):
        h=min(step,end-t)
        if t+h==t: status='step_underflow'; break
        try:
            with np.errstate(over='raise',invalid='raise'):
                q,p=symplectic_step(y[:n],y[n:],h,grad_potential=gradient,grad_kinetic=lambda p:p,method=method)
                new=np.concatenate((q,p))
        except FloatingPointError:
            status='nonfinite'; break
        if not np.isfinite(new).all(): status='nonfinite'; break
        t=end if h==end-t else t+h; y=new
        times.append(t); states.append(y.copy()); steps.append(h)
        if np.max(abs(y))>escape_radius: status='escaped'; break
        if t==end: status='completed'; break
    return IntegrationResult(np.array(times),np.array(states),np.array(steps),np.zeros(len(steps)),len(steps),0,calls,status)


@dataclass
class SectionResult:
    times: np.ndarray
    states: np.ndarray
    brackets: np.ndarray
    residuals: np.ndarray
    failures: list
    evaluations: int
    status: str


def refined_sections(rhs, times, states, *, coordinate, level, direction,
                     max_crossings, solver_parameters, tolerance=1e-10, max_iterations=60):
    """Refine transverse crossings; right endpoints included, left endpoints excluded.

    Adaptive reintegration does not remove error in the supplied left state.
    A bracket lost under reintegration is reported, never silently interpolated.
    """
    times=np.asarray(times,float); states=np.asarray(states,float)
    if times.ndim!=1 or len(times)<2 or states.ndim!=2 or states.shape[0]!=len(times) or not states.shape[1] or not np.isfinite(times).all() or not np.isfinite(states).all() or np.any(np.diff(times)<=0): raise ValueError('finite increasing trajectory required')
    if not isinstance(coordinate,(int,np.integer)) or not 0<=coordinate<states.shape[1] or direction not in (-1,1) or not np.isfinite([level,tolerance]).all() or tolerance<=0: raise ValueError('invalid section controls')
    if any(not isinstance(v,(int,np.integer)) or isinstance(v,bool) or v<1 for v in (max_crossings,max_iterations)) or 't_span' in solver_parameters: raise ValueError('invalid limits/solver controls')
    found=[]; values=[]; brackets=[]; residuals=[]; failures=[]; calls=0; status='completed'
    for i in range(len(times)-1):
        a,b=states[i:i+2,coordinate]-level
        if not (direction*a<0 and direction*b>=0): continue
        left,right=times[i:i+2]; base=states[i]; lo=left; hi=right; best=None; error=np.inf
        for iteration in range(max_iterations):
            target=right if iteration==0 else (lo+hi)/2
            try:
                trial=integrate(rhs,base,t_span=(left,target),**solver_parameters)
            except FloatingPointError:
                failures.append((i,'nonfinite')); break
            calls+=trial.evaluations
            if trial.status!='completed': failures.append((i,trial.status)); break
            v=trial.states[-1]; error=float(v[coordinate]-level)
            derivative=np.asarray(rhs(target,v),float)
            if derivative.shape!=v.shape: raise ValueError('RHS shape mismatch')
            calls+=1
            if not np.isfinite(derivative).all():
                failures.append((i,'nonfinite')); break
            if iteration==0 and direction*error < -tolerance:
                failures.append((i,'lost_bracket')); break
            if abs(error)<=tolerance and direction*derivative[coordinate]>0:
                best=(target,v.copy()); break
            if direction*error>=0: hi=target
            else: lo=target
            if hi-lo<=np.finfo(float).eps*max(1,abs(target)):
                failures.append((i,'grazing_or_resolution')); break
        else: failures.append((i,'max_iterations'))
        if best is not None:
            found.append(best[0]); values.append(best[1]); brackets.append([left,right]); residuals.append(abs(error))
            if len(found)==max_crossings: status='crossing_limit'; break
    if failures and status=='completed': status='refinement_failures'
    return SectionResult(np.array(found),np.array(values).reshape(-1,states.shape[1]),
        np.array(brackets).reshape(-1,2),np.array(residuals),failures,calls,status)
