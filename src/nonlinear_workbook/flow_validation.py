"""Known-equation continuous-flow variational diagnostics (natural logs/time)."""
from dataclasses import dataclass
import itertools
import numpy as np
from .chapter10 import integrate


def wedge(u,v):
    u,v=np.asarray(u,float),np.asarray(v,float)
    if u.ndim!=1 or v.shape!=u.shape or len(u)<2 or not np.isfinite(u).all() or not np.isfinite(v).all(): raise ValueError('finite matching vectors required')
    return np.array([u[i]*v[j]-u[j]*v[i] for i,j in itertools.combinations(range(len(u)),2)])


def second_compound(matrix):
    """Additive compound in lexicographic pairs (01,02,...); a'=C2(J)a."""
    j=np.asarray(matrix,float)
    if j.ndim!=2 or j.shape[0]!=j.shape[1] or len(j)<2: raise ValueError('square Jacobian of dimension>=2 required')
    pairs=list(itertools.combinations(range(len(j)),2)); out=np.empty((len(pairs),len(pairs)))
    for k,(a,b) in enumerate(pairs):
        basis=np.zeros_like(j); basis[a,b]=1; basis[b,a]=-1
        evolved=j@basis+basis@j.T
        out[:,k]=[evolved[i,l] for i,l in pairs]
    return out


@dataclass
class FlowSpectrumResult:
    exponents: np.ndarray
    running_estimates: np.ndarray
    times: np.ndarray
    final_state: np.ndarray
    final_basis: np.ndarray
    elapsed: float
    trace_average: float
    evaluations: int
    accepted_steps: int
    rejected_steps: int
    status: str


@dataclass
class AreaGrowthResult:
    rate: float
    running_estimates: np.ndarray
    times: np.ndarray
    final_state: np.ndarray
    final_wedge: np.ndarray
    elapsed: float
    evaluations: int
    accepted_steps: int
    rejected_steps: int
    status: str


def _controls(y0,burn_in,duration,interval,solver_parameters):
    y=np.asarray(y0,float)
    if y.ndim!=1 or not len(y) or not np.isfinite(y).all() or not np.isfinite([burn_in,duration,interval]).all() or burn_in<0 or duration<=0 or interval<=0: raise ValueError('invalid state/time controls')
    parameters=dict(solver_parameters)
    if parameters.get('method')!='rkf45_adaptive' or 't_span' in parameters: raise ValueError('adaptive Fehlberg parameters without t_span required')
    return y.copy(),parameters


def _functions(rhs,jacobian,n):
    def checked(t,y):
        f=np.asarray(rhs(t,y.copy()),float); j=np.asarray(jacobian(t,y.copy()),float)
        if f.shape!=(n,) or j.shape!=(n,n): raise ValueError('RHS/Jacobian shape mismatch')
        return f,j
    return checked


def _segments(burn_in,duration,interval):
    # Endpoint arithmetic, rather than repeatedly adding interval, bounds the horizon.
    for phase,length,start in (('burn',burn_in,0.),('measure',duration,burn_in)):
        count=int(np.ceil(length/interval))
        for i in range(count):
            lo=start+min(i*interval,length); hi=start+min((i+1)*interval,length)
            if hi>lo: yield phase,lo,hi


def flow_lyapunov_spectrum(rhs,jacobian,y0,*,burn_in,duration,qr_interval,
                           initial_basis,solver_parameters):
    """Integrate y'=f, Q'=JQ and trace integral; QR at specified endpoints.

    Exponents remain in column order. Burn-in aligns the tangent basis without
    accumulating growth. Incomplete intervals are not bridged or measured.
    Evaluation count includes all augmented RHS calls, each evaluating f and J.
    """
    raw=np.asarray(y0,float)
    if raw.ndim!=1: raise ValueError('vector state required')
    return conditional_flow_spectrum(rhs,jacobian,y0,burn_in=burn_in,duration=duration,
        qr_interval=qr_interval,initial_basis=initial_basis,solver_parameters=solver_parameters,
        response_dimension=len(raw))


def conditional_flow_spectrum(rhs,response_jacobian,y0,*,response_dimension,burn_in,
                              duration,qr_interval,initial_basis,solver_parameters):
    """Full state integration with a separately sized conditional tangent system.

    response_jacobian(t,full_state) describes infinitesimal response perturbations
    with the drive held fixed. It is not the full coupled-system Jacobian.
    Column order, burn alignment, trace accounting and failures match flow QR.
    """
    y,parameters=_controls(y0,burn_in,duration,qr_interval,solver_parameters); state_size=len(y)
    n=response_dimension
    if not isinstance(n,(int,np.integer)) or isinstance(n,bool) or not 1<=n<=state_size:
        raise ValueError('valid response dimension required')
    basis=np.asarray(initial_basis,float)
    if basis.shape!=(n,n) or not np.isfinite(basis).all() or np.linalg.matrix_rank(basis)!=n: raise ValueError('full-rank initial basis required')
    q,r=np.linalg.qr(basis); signs=np.where(np.diag(r)<0,-1.,1.); q=q*signs
    def checked(t,y):
        f=np.asarray(rhs(t,y.copy()),float); j=np.asarray(response_jacobian(t,y.copy()),float)
        if f.shape!=(state_size,) or j.shape!=(n,n): raise ValueError('RHS/response Jacobian shape mismatch')
        return f,j
    def augmented(t,z):
        f,j=checked(t,z[:state_size]); tangent=z[state_size:-1].reshape(n,n)
        return np.concatenate((f,(j@tangent).ravel(),[np.trace(j)]))
    logs=np.zeros(n); elapsed=0.; trace=0.; times=[]; running=[]; calls=accepted=rejected=0; status='completed'
    for phase,lo,hi in _segments(burn_in,duration,qr_interval):
        result=integrate(augmented,np.concatenate((y,q.ravel(),[0.])),t_span=(lo,hi),**parameters)
        calls+=result.evaluations; accepted+=result.accepted_steps; rejected+=result.rejected_steps
        if result.status!='completed': status=result.status; break
        z=result.states[-1]; tangent=z[state_size:-1].reshape(n,n)
        if np.linalg.matrix_rank(tangent)!=n: status='singular_tangent'; break
        newq,r=np.linalg.qr(tangent); diagonal=np.diag(r)
        if not np.isfinite(diagonal).all() or np.any(diagonal==0): status='singular_tangent'; break
        q=newq*np.where(diagonal<0,-1.,1.); y=z[:state_size]
        if phase=='measure':
            logs+=np.log(np.abs(diagonal)); trace+=z[-1]; elapsed+=hi-lo
            times.append(hi); running.append(logs/elapsed)
    exponents=logs/elapsed if elapsed else np.full(n,np.nan)
    return FlowSpectrumResult(exponents,np.array(running).reshape(-1,n),np.array(times),y,q,elapsed,
        trace/elapsed if elapsed else np.nan,calls,accepted,rejected,status)


def exterior_area_growth(rhs,jacobian,y0,*,burn_in,duration,normalization_interval,
                         initial_vectors,solver_parameters):
    """Independent second-compound propagation; generic area gives lambda1+lambda2.

    A special invariant initial two-plane can yield another pair sum.
    Euclidean norm in wedge coordinates; no QR is used.
    """
    y,parameters=_controls(y0,burn_in,duration,normalization_interval,solver_parameters); n=len(y)
    pair=np.asarray(initial_vectors,float)
    if pair.shape!=(2,n): raise ValueError('two initial tangent vectors required')
    area=wedge(*pair); norm=np.linalg.norm(area)
    if not np.isfinite(norm) or norm==0: raise ValueError('nonzero finite initial area required')
    area=area/norm; checked=_functions(rhs,jacobian,n)
    def augmented(t,z):
        f,j=checked(t,z[:n]); return np.concatenate((f,second_compound(j)@z[n:]))
    total=elapsed=0.; times=[]; running=[]; calls=accepted=rejected=0; status='completed'
    for phase,lo,hi in _segments(burn_in,duration,normalization_interval):
        result=integrate(augmented,np.concatenate((y,area)),t_span=(lo,hi),**parameters)
        calls+=result.evaluations; accepted+=result.accepted_steps; rejected+=result.rejected_steps
        if result.status!='completed': status=result.status; break
        z=result.states[-1]; norm=np.linalg.norm(z[n:])
        if not np.isfinite(norm) or norm==0: status='zero_area'; break
        y=z[:n]; area=z[n:]/norm
        if phase=='measure':
            total+=np.log(norm); elapsed+=hi-lo; times.append(hi); running.append(total/elapsed)
    return AreaGrowthResult(total/elapsed if elapsed else np.nan,np.array(running),np.array(times),y,area,elapsed,calls,accepted,rejected,status)
