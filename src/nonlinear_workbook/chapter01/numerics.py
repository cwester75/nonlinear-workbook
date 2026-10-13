"""Iteration, Fourier transforms, Newton iteration, and bifurcation roots."""
from dataclasses import dataclass
import numpy as np
from .maps import logistic_step, logistic_derivative


def integer(value, name, minimum=1):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def series(values, minimum=1, complex_ok=False):
    x=np.asarray(values,dtype=complex if complex_ok else float)
    if x.ndim != 1 or len(x)<minimum or not np.all(np.isfinite(x)):
        raise ValueError('Expected finite one-dimensional samples')
    return x


@dataclass(frozen=True)
class IterationResult:
    values: np.ndarray
    status: str
    updates: int


def iterate(step, x0, *, n, burn_in=0, exact=False, escape_radius=1e100):
    """n states beginning with f**burn_in(x0); premature termination is explicit.

    Boundary/overflow checks run on the initial state and each update. Exact mode
    retains Python objects; numeric mode returns float64 or complex128 arrays.
    A failed update is not appended. Invalid map parameters propagate ValueError.
    """
    integer(n,'n'); integer(burn_in,'burn_in',0)
    if not np.isfinite(escape_radius) or escape_radius<=0: raise ValueError('Invalid escape_radius')
    state=x0 if exact else np.asarray(x0,dtype=complex if np.iscomplexobj(x0) else float)
    values=[]; updates=0; status='completed'
    for t in range(burn_in+n):
        try:
            magnitude=max(abs(v) for v in np.asarray(state,dtype=object).flat) if exact else np.linalg.norm(np.atleast_1d(state))
            finite=True if exact else bool(np.all(np.isfinite(state)))
        except (OverflowError,TypeError): finite=False; magnitude=0
        if not finite: status='nonfinite'; break
        if magnitude>escape_radius: status='escaped'; break
        if t>=burn_in: values.append(state if exact else np.array(state,copy=True))
        if t == burn_in+n-1: break
        try:
            with np.errstate(over='ignore',invalid='ignore'):
                new=step(state)
            if not exact and np.shape(new)!=np.shape(state): raise ValueError('Map changed state shape')
            state=new; updates+=1
        except (OverflowError,FloatingPointError): status='nonfinite'; break
    return IterationResult(np.asarray(values,dtype=object if exact else (complex if np.iscomplexobj(x0) else float)),status,updates)


def dft(x, *, inverse=False):
    x=series(x,complex_ok=True); n=len(x); k=np.arange(n)
    sign=1 if inverse else -1
    return np.exp(sign*2j*np.pi*np.outer(k,k)/n) @ x / (1 if inverse else n)


def radix2_fft(x, *, inverse=False):
    x=series(x,complex_ok=True); n=len(x)
    if n & (n-1): raise ValueError('radix2_fft requires a power-of-two length')
    def transform(a):
        if len(a)==1: return a.copy()
        even,odd=transform(a[::2]),transform(a[1::2])
        twiddle=np.exp((1 if inverse else -1)*2j*np.pi*np.arange(len(a)//2)/len(a))*odd
        return np.concatenate([even+twiddle,even-twiddle])
    return transform(x)/(1 if inverse else n)


@dataclass(frozen=True)
class NewtonResult:
    root: object
    residuals: np.ndarray
    iterations: int
    status: str


def newton(f, derivative, x0, *, tolerance=1e-12, max_steps=50, escape_radius=1e100):
    """Real/complex Newton; stopping requires residual <= tolerance."""
    integer(max_steps,'max_steps')
    if not np.isfinite(tolerance) or tolerance<=0 or not np.isfinite(escape_radius) or escape_radius<=0:
        raise ValueError('Invalid tolerance/escape radius')
    x=x0; history=[]
    for t in range(max_steps+1):
        if not np.isfinite(x): return NewtonResult(x,np.array(history),t,'nonfinite')
        if abs(x)>escape_radius: return NewtonResult(x,np.array(history),t,'escaped')
        try: value=f(x)
        except OverflowError: return NewtonResult(x,np.array(history),t,'nonfinite')
        residual=abs(value); history.append(residual)
        if not np.isfinite(residual): return NewtonResult(x,np.array(history),t,'nonfinite')
        if residual<=tolerance: return NewtonResult(x,np.array(history),t,'converged')
        if t==max_steps: break
        slope=derivative(x)
        if not np.isfinite(slope): return NewtonResult(x,np.array(history),t,'nonfinite')
        if slope==0: return NewtonResult(x,np.array(history),t,'singular_derivative')
        x=x-value/slope
    return NewtonResult(x,np.array(history),max_steps,'max_steps')


def logistic_sweep(parameters, *, n=256, burn_in=2000, x0=.618):
    from ..chapter02 import logistic_map, lyapunov_map
    parameters=series(parameters)
    trajectories=np.array([logistic_map(n,r=float(r),x0=x0,burn_in=burn_in) for r in parameters])
    exponents=np.array([lyapunov_map(x,r=float(r)) for x,r in zip(trajectories,parameters)])
    return parameters,trajectories,exponents


def period_doubling(period, *, r_guess, x_guess=None, tolerance=1e-11, max_steps=40):
    """Solve f**period(x)=x and its multiplier=-1. Reject lower-period roots.

    Local two-variable Newton with centered finite differences; caller supplies
    continuation guesses. Returns NewtonResult whose root is [x,r].
    """
    integer(period,'period'); integer(max_steps,'max_steps')
    if period & (period-1): raise ValueError('period must be a power of two')
    if tolerance<=0 or not np.isfinite(tolerance) or not 2<r_guess<4: raise ValueError('Invalid root controls')
    if x_guess is None:
        x_guess=.618
        for _ in range(3000): x_guess=logistic_step(x_guess,r=r_guess-.0001)
    def equations(v):
        x,r=v; y=x; multiplier=1.
        for _ in range(period): multiplier*=r*(1-2*y); y=r*y*(1-y)
        return np.array([y-x,multiplier+1])
    v=np.array([x_guess,r_guess]); history=[]
    for t in range(max_steps+1):
        e=equations(v); history.append(float(np.linalg.norm(e)))
        if not np.all(np.isfinite(e)) or not 0<v[0]<1 or not 2<v[1]<4:
            return NewtonResult(v,np.array(history),t,'escaped')
        if history[-1]<=tolerance:
            if period>1:
                y=v[0]
                for _ in range(period//2): y=v[1]*y*(1-y)
                if abs(y-v[0])<1e-8: return NewtonResult(v,np.array(history),t,'lower_period')
            return NewtonResult(v,np.array(history),t,'converged')
        if t==max_steps: break
        h=1e-6
        jac=np.column_stack([(equations(v+h*np.eye(2)[j])-equations(v-h*np.eye(2)[j]))/(2*h) for j in range(2)])
        try: v-=np.linalg.solve(jac,e)
        except np.linalg.LinAlgError: return NewtonResult(v,np.array(history),t,'singular_jacobian')
    return NewtonResult(v,np.array(history),max_steps,'max_steps')


def feigenbaum_ratios(thresholds):
    thresholds=series(thresholds,3)
    differences=np.diff(thresholds)
    if np.any(differences<=0): raise ValueError('Thresholds must be increasing')
    return differences[:-1]/differences[1:]


def vector_newton(f, jacobian, x0, *, tolerance=1e-12, max_steps=50, escape_radius=1e100):
    """Newton for real vector systems; solve J*delta=f instead of inverting J."""
    integer(max_steps,'max_steps')
    if not np.isfinite(tolerance) or tolerance<=0 or not np.isfinite(escape_radius) or escape_radius<=0:
        raise ValueError('Invalid Newton controls')
    x=series(x0); history=[]
    for t in range(max_steps+1):
        if not np.all(np.isfinite(x)): return NewtonResult(x,np.array(history),t,'nonfinite')
        if np.linalg.norm(x)>escape_radius: return NewtonResult(x,np.array(history),t,'escaped')
        try:
            with np.errstate(over='ignore',invalid='ignore'): value=np.asarray(f(x),dtype=float)
        except OverflowError: return NewtonResult(x,np.array(history),t,'nonfinite')
        if value.shape!=x.shape: raise ValueError('Residual shape must match state')
        residual=float(np.linalg.norm(value)); history.append(residual)
        if not np.isfinite(residual): return NewtonResult(x,np.array(history),t,'nonfinite')
        if residual<=tolerance: return NewtonResult(x,np.array(history),t,'converged')
        if t==max_steps: break
        matrix=np.asarray(jacobian(x),dtype=float)
        if matrix.shape!=(len(x),len(x)): raise ValueError('Jacobian shape must match state')
        if not np.all(np.isfinite(matrix)): return NewtonResult(x,np.array(history),t,'nonfinite')
        try: x=x-np.linalg.solve(matrix,value)
        except np.linalg.LinAlgError: return NewtonResult(x,np.array(history),t,'singular_jacobian')
    return NewtonResult(x,np.array(history),max_steps,'max_steps')


@dataclass(frozen=True)
class BasinResult:
    labels: np.ndarray
    iterations: np.ndarray
    statuses: np.ndarray
    roots: np.ndarray


def complex_newton_basins(real, imaginary, *, tolerance=1e-10, max_steps=60, escape_radius=1e6):
    """z**4-1 basins. Labels 0..3 for [1,-1,i,-i], -1 for unresolved."""
    real=series(real); imaginary=series(imaginary)
    # Validate controls even if every supplied initial point is already a root.
    integer(max_steps,'max_steps')
    if tolerance<=0 or not np.isfinite(tolerance) or escape_radius<=0 or not np.isfinite(escape_radius):
        raise ValueError('Invalid basin controls')
    roots=np.array([1,-1,1j,-1j],dtype=complex)
    labels=np.full((len(imaginary),len(real)),-1,dtype=int)
    iterations=np.zeros_like(labels); statuses=np.empty(labels.shape,dtype='U24')
    for j,y in enumerate(imaginary):
        for i,x in enumerate(real):
            result=newton(lambda z:z**4-1,lambda z:4*z**3,complex(x,y),tolerance=tolerance,max_steps=max_steps,escape_radius=escape_radius)
            statuses[j,i]=result.status; iterations[j,i]=result.iterations
            if result.status=='converged': labels[j,i]=int(np.argmin(np.abs(result.root-roots)))
    return BasinResult(labels,iterations,statuses,roots)
