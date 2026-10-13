"""Finite-sample scalar diagnostics with explicit conventions."""
from dataclasses import dataclass
import numpy as np
from .numerics import series, integer, iterate


def stability(derivative, point, *, tolerance=1e-10):
    value=complex(derivative(point))
    if not np.isfinite(value) or tolerance<0 or not np.isfinite(tolerance): raise ValueError('Invalid derivative/tolerance')
    modulus=abs(value)
    return value, ('nonhyperbolic' if abs(modulus-1)<=tolerance else ('stable' if modulus<1 else 'unstable'))


def scalar_lyapunov(x, derivative, *, sample_interval=1.):
    x=series(x)
    if not np.isfinite(sample_interval) or sample_interval<=0: raise ValueError('Invalid sample interval')
    slopes=np.abs([derivative(v) for v in x])
    if not np.all(np.isfinite(slopes)): raise ValueError('Nonfinite derivative')
    if np.any(slopes==0): return float('-inf')
    return float(np.mean(np.log(slopes))/sample_interval)


@dataclass(frozen=True)
class DensityResult:
    edges: np.ndarray
    density: np.ndarray
    counts: np.ndarray
    retained: int


def invariant_density(x, *, edges):
    x=series(x); edges=series(edges,2)
    if np.any(np.diff(edges)<=0): raise ValueError('Edges must increase')
    counts,_=np.histogram(x,bins=edges); retained=int(counts.sum())
    if retained==0: raise ValueError('No samples inside histogram range')
    return DensityResult(edges,counts/(retained*np.diff(edges)),counts,retained)


def moments(x, *, orders=(1,2)):
    x=series(x)
    for p in orders: integer(p,'moment order',0)
    return np.array([np.mean(x**p) for p in orders])


def autocovariance(x, *, max_lag):
    x=series(x); integer(max_lag,'max_lag',0)
    if max_lag>=len(x): raise ValueError('Lag exceeds sample bounds')
    centered=x-x.mean(); pairs=len(x)-np.arange(max_lag+1)
    curve=np.array([np.dot(centered[:len(x)-k],centered[k:])/pairs[k] for k in range(max_lag+1)])
    return np.arange(max_lag+1),curve,pairs


@dataclass(frozen=True)
class MonteCarloResult:
    estimate: float
    standard_error: float
    samples: int
    seed: int
    method: str


def monte_carlo(f, *, n, seed, method='mean'):
    """Scalar integral on [0,1]; f returns one real scalar per sample.

    Hit/miss requires 0<=f<=1; IID NumPy generator.
    """
    integer(n,'n',2); integer(seed,'seed',0)
    if method not in ('mean','hit_or_miss'): raise ValueError('Unknown integration method')
    rng=np.random.default_rng(seed); values=np.array([f(v) for v in rng.random(n)])
    if values.shape != (n,) or not np.isrealobj(values):
        raise ValueError('Integrand must return one real scalar per sample')
    if not np.all(np.isfinite(values)): raise ValueError('Integrand returned nonfinite values')
    if method=='hit_or_miss':
        if np.any((values<0)|(values>1)): raise ValueError('Hit/miss requires integrand in [0,1]')
        values=(rng.random(n)<=values).astype(float)
    return MonteCarloResult(float(values.mean()),float(values.std(ddof=1)/np.sqrt(n)),n,seed,method)


def rotation_number(*, omega, strength, x0=.3, n=10000, burn_in=1000):
    from .maps import circle_step
    integer(n,'n')
    result=iterate(lambda x:circle_step(x,omega=omega,strength=strength),x0,n=n+1,burn_in=burn_in)
    if result.status!='completed': raise ValueError('Incomplete lift trajectory')
    return float((result.values[-1]-result.values[0])/n)


def symbolic_partition(x, *, boundaries):
    """Right interval owns equality; symbols are integer bin indices."""
    x=series(x); boundaries=np.asarray(boundaries,dtype=float)
    if boundaries.ndim!=1 or not np.all(np.isfinite(boundaries)) or np.any(np.diff(boundaries)<=0):
        raise ValueError('Boundaries must be finite and increasing')
    return np.searchsorted(boundaries,x,side='right')


def inverse_symbolic(symbols, terminal, *, r):
    """Quadratic map inverse composition: L=-sqrt((1-y)/r), R=+sqrt(...)."""
    if not np.isfinite(r) or not 0<r<=2 or not np.isfinite(terminal): raise ValueError('Invalid inverse controls')
    x=float(terminal)
    for symbol in reversed(list(symbols)):
        if symbol not in ('L','R'): raise ValueError('Symbols must be L or R')
        argument=(1-x)/r
        if argument<0: raise ValueError('Inverse branch is outside real domain')
        x=(-1 if symbol=='L' else 1)*np.sqrt(argument)
        if not -1<=x<=1: raise ValueError('Inverse branch leaves map domain')
    return float(x)


def states(values):
    values=np.asarray(values,dtype=float)
    if values.ndim==1: values=values[:,None]
    if values.ndim!=2 or len(values)<2 or values.shape[1]<1 or not np.all(np.isfinite(values)):
        raise ValueError('Expected finite sample-by-coordinate array')
    return values


def fixed_point_classification(jacobian, point, *, tolerance=1e-10):
    point=series(point)
    matrix=np.asarray(jacobian(point),dtype=float)
    if matrix.shape!=(len(point),len(point)) or not np.all(np.isfinite(matrix)) or not np.isfinite(tolerance) or tolerance<0:
        raise ValueError('Invalid Jacobian/tolerance')
    eigenvalues=np.linalg.eigvals(matrix); moduli=np.abs(eigenvalues)
    if np.any(np.abs(moduli-1)<=tolerance): classification='nonhyperbolic'
    elif np.all(moduli<1): classification='sink'
    elif np.all(moduli>1): classification='source'
    else: classification='saddle'
    return eigenvalues,classification


@dataclass(frozen=True)
class KnownSpectrumResult:
    exponents: np.ndarray
    running_estimates: np.ndarray
    usable_steps: int
    accumulated_steps: int
    status: str


def known_lyapunov_spectrum(trajectory, jacobian, *, sample_interval=1., qr_burn_in=0):
    """Known-equation Jacobian QR spectrum; retains QR direction order."""
    trajectory=states(trajectory); integer(qr_burn_in,'qr_burn_in',0)
    if sample_interval<=0 or not np.isfinite(sample_interval): raise ValueError('Invalid sample interval')
    dimension=trajectory.shape[1]; q=np.eye(dimension); sums=np.zeros(dimension)
    running=[]; usable=accumulated=0; status='completed'
    for x in trajectory:
        matrix=np.asarray(jacobian(x),dtype=float)
        if matrix.shape!=(dimension,dimension): raise ValueError('Invalid Jacobian shape')
        if not np.all(np.isfinite(matrix)): status='nonfinite_jacobian'; break
        q_new,r=np.linalg.qr(matrix@q); diagonal=np.abs(np.diag(r))
        if np.any(diagonal==0) or not np.all(np.isfinite(diagonal)):
            status='singular_tangent'; break
        q=q_new; usable+=1
        if usable>qr_burn_in:
            accumulated+=1; sums+=np.log(diagonal)
            running.append(sums.copy()/(accumulated*sample_interval))
    if not accumulated: raise ValueError(f'No post-burn-in tangent updates: {status}')
    return KnownSpectrumResult(sums/(accumulated*sample_interval),np.asarray(running),usable,accumulated,status)


@dataclass(frozen=True)
class CorrelationIntegralResult:
    radii: np.ndarray
    values: np.ndarray
    counts: np.ndarray
    eligible_pairs: int


def correlation_integral(points, *, radii, theiler=0):
    """Fraction of eligible unordered pairs with Euclidean distance <= radius.

    Streaming rows avoid allocating a full pair-distance matrix. Theiler exclusion
    operates on sample row indices. Duplicate states count when temporally eligible.
    """
    points=states(points); radii=series(radii); integer(theiler,'theiler',0)
    if np.any(radii<=0) or np.any(np.diff(radii)<=0): raise ValueError('Radii must be positive and increasing')
    counts=np.zeros(len(radii),dtype=np.int64); eligible=0
    for i in range(len(points)-1):
        neighbors=points[i+theiler+1:]
        distances=np.sort(np.linalg.norm(neighbors-points[i],axis=1))
        eligible+=len(distances); counts+=np.searchsorted(distances,radii,side='right')
    if not eligible: raise ValueError('No eligible pairs')
    return CorrelationIntegralResult(radii,counts/eligible,counts,eligible)


@dataclass(frozen=True)
class BoxCountResult:
    scales: np.ndarray
    counts: np.ndarray
    origin: np.ndarray


def box_counts(points, *, scales, origin):
    """Half-open Cartesian grid cells floor((point-origin)/scale)."""
    points=states(points); scales=series(scales); origin=series(origin)
    if len(origin)!=points.shape[1] or np.any(scales<=0): raise ValueError('Invalid grid origin/scales')
    if len(np.unique(scales))!=len(scales): raise ValueError('Scales must be distinct')
    counts=[]
    for scale in scales:
        grid=np.floor((points-origin)/scale)
        if not np.all(np.isfinite(grid)): raise ValueError('Grid scale overflow')
        counts.append(len(np.unique(grid,axis=0)))
    return BoxCountResult(scales,np.array(counts),origin)


@dataclass(frozen=True)
class ScalingFit:
    slope: float
    intercept: float
    r_squared: float
    indices: np.ndarray


def fit_scaling(scales, values, *, fit_slice, inverse_scale=False):
    scales=series(scales,2); values=series(values,2)
    if len(scales)!=len(values) or np.any(scales<=0) or np.any(values<=0): raise ValueError('Positive matching scales/values required')
    if not isinstance(fit_slice,slice) or fit_slice.start is None or fit_slice.stop is None or fit_slice.step not in (None,1):
        raise ValueError('Provide an explicit contiguous fit slice')
    if not 0<=fit_slice.start<fit_slice.stop<=len(scales): raise ValueError('Fit slice is out of bounds')
    indices=np.arange(len(scales))[fit_slice]
    if len(indices)<2 or len(np.unique(scales[indices]))<2: raise ValueError('At least two distinct scales required')
    x=np.log(scales[indices])*(-1 if inverse_scale else 1); y=np.log(values[indices])
    slope,intercept=np.polyfit(x,y,1); residual=np.sum((y-(slope*x+intercept))**2); total=np.sum((y-y.mean())**2)
    score=1-residual/total if total>0 else (1. if residual<1e-24 else 0.)
    return ScalingFit(float(slope),float(intercept),float(score),indices)


@dataclass(frozen=True)
class BoundednessResult:
    parameters: np.ndarray
    couplings: np.ndarray
    labels: np.ndarray
    updates: np.ndarray
    horizon: int
    escape_radius: float


def boundedness_scan(parameters, couplings, *, x0=(.7,.3), horizon=800, escape_radius=20.):
    """Finite-horizon coupled logistic scan; labels completed/escaped/nonfinite."""
    from .maps import coupled_logistic_step
    parameters=series(parameters); couplings=series(couplings); x0=series(x0)
    if len(x0)!=2: raise ValueError('Expected two initial coordinates')
    integer(horizon,'horizon')
    labels=np.empty((len(couplings),len(parameters)),dtype='U12'); updates=np.zeros(labels.shape,dtype=int)
    for j,e in enumerate(couplings):
        for i,r in enumerate(parameters):
            result=iterate(lambda x:coupled_logistic_step(x,r=r,coupling=e),x0,n=horizon+1,escape_radius=escape_radius)
            labels[j,i]=result.status; updates[j,i]=result.updates
    return BoundednessResult(parameters,couplings,labels,updates,horizon,escape_radius)
