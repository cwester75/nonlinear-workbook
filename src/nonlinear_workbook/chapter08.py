"""Finite fractal constructions and diagnostics; no finite image certifies a set."""
from dataclasses import dataclass
from fractions import Fraction
import numpy as np
from .chapter01.diagnostics import box_counts,correlation_integral


def _integer(n,name,minimum=0):
    if isinstance(n,bool) or not isinstance(n,(int,np.integer)) or n<minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(n)


def _allocation(base,depth,multiplier,extra,max_elements):
    depth=_integer(depth,'depth'); limit=_integer(max_elements,'max_elements',1)
    if depth>64: raise ValueError('depth exceeds work limit64')
    size=multiplier
    if size+extra>limit: raise ValueError('element limit exceeded')
    for _ in range(depth):
        if size>(limit-extra)//base: raise ValueError('element limit exceeded')
        size*=base
    return size+extra


def cantor_intervals(depth,*,max_elements=1000000):
    """Exact closed intervals; bound counts both endpoints of every interval."""
    _allocation(2,depth,2,0,max_elements)
    intervals=[(Fraction(0),Fraction(1))]
    for _ in range(depth):
        intervals=[pair for a,b in intervals for pair in ((a,a+(b-a)/3),(b-(b-a)/3,b))]
    return tuple(intervals)


def cantor_endpoints(intervals):
    """Explicit float conversion, sorted unique endpoint column for diagnostics."""
    points=np.asarray(intervals,dtype=float)
    if points.ndim!=2 or points.shape[1]!=2 or not len(points) or not np.isfinite(points).all() or np.any(points[:,1]<points[:,0]): raise ValueError('invalid intervals')
    return np.unique(points).reshape(-1,1)


def _ifs(matrices,translations):
    a,b=np.asarray(matrices,float),np.asarray(translations,float)
    if a.ndim!=3 or a.shape[1]!=a.shape[2] or not a.shape[0] or not a.shape[1] or b.shape!=a.shape[:2] or not np.isfinite(a).all() or not np.isfinite(b).all(): raise ValueError('invalid affine IFS')
    return a,b


def contraction_norms(matrices,translations):
    a,b=_ifs(matrices,translations)
    return np.linalg.svd(a,compute_uv=False)[:,0]


def deterministic_ifs(matrices,translations,initial_points,*,depth,max_elements=1000000):
    """Map-major generation of A*x+b; duplicate points are retained."""
    a,b=_ifs(matrices,translations); points=np.asarray(initial_points,float)
    if points.ndim!=2 or points.shape[1]!=a.shape[1] or not len(points) or not np.isfinite(points).all(): raise ValueError('invalid initial points')
    _allocation(len(a),depth,points.size,0,max_elements)
    points=points.copy()
    for _ in range(depth):
        points=np.concatenate([points@matrix.T+offset for matrix,offset in zip(a,b)])
        if not np.isfinite(points).all(): raise ValueError('nonfinite IFS iteration')
    return points


@dataclass
class ChaosGameResult:
    points: np.ndarray
    map_indices: np.ndarray
    burn_in: int
    seed: int
    contraction_norms: np.ndarray


def chaos_game(matrices,translations,*,probabilities,initial_point,samples,burn_in,seed,max_elements=1000000):
    a,b=_ifs(matrices,translations); n=_integer(samples,'samples',1); burn=_integer(burn_in,'burn_in')
    _allocation(1,0,n*(a.shape[1]+1)+burn,0,max_elements)
    y=np.asarray(initial_point,float); p=np.asarray(probabilities,float)
    if y.shape!=(a.shape[1],) or not np.isfinite(y).all() or p.shape!=(len(a),) or not np.isfinite(p).all() or np.any(p<0) or not np.isclose(p.sum(),1,rtol=0,atol=1e-12): raise ValueError('invalid initial point/probabilities')
    rng=np.random.default_rng(seed); choices=rng.choice(len(a),n+burn,p=p/p.sum())
    points=np.empty((n,len(y)))
    for i,k in enumerate(choices):
        y=a[k]@y+b[k]
        if not np.isfinite(y).all(): raise ValueError('nonfinite chaos game')
        if i>=burn: points[i-burn]=y
    return ChaosGameResult(points,choices[burn:],burn,seed,contraction_norms(a,b))


def gasket_ifs(*,printed=False):
    """Normalized equilateral fixture or exact p.284 affine translations."""
    vertices=np.array([[0.,0.],[1.,0.],[.5,np.sqrt(3)/2]])
    offsets=np.array([[1.,1.],[1.,50.],[50.,50.]]) if printed else vertices/2
    return np.repeat((np.eye(2)/2)[None,:,:],3,axis=0),offsets


def sierpinski_triangles(depth,*,vertices=None,max_elements=1000000):
    _allocation(3,depth,6,0,max_elements)
    triangle=np.array([[0.,0.],[1.,0.],[.5,np.sqrt(3)/2]]) if vertices is None else np.asarray(vertices,float)
    if triangle.shape!=(3,2) or not np.isfinite(triangle).all() or abs(np.linalg.det(np.column_stack((triangle[1]-triangle[0],triangle[2]-triangle[0]))))==0: raise ValueError('nondegenerate triangle required')
    triangles=triangle[None,:,:].copy()
    for _ in range(depth):
        next_triangles=[]
        for v in triangles:
            ab,ac,bc=(v[0]+v[1])/2,(v[0]+v[2])/2,(v[1]+v[2])/2
            next_triangles.extend(([v[0],ab,ac],[ab,v[1],bc],[ac,bc,v[2]]))
        triangles=np.asarray(next_triangles)
    return triangles


def dragon(depth,*,max_elements=1000000):
    _allocation(2,depth,1,1,max_elements)
    vertices=np.array([0j,1+0j])
    for _ in range(depth):
        first=(1+1j)/2*vertices
        second=1+(-1+1j)/2*vertices[::-1]
        vertices=np.concatenate((first,second[1:]))
    return vertices


def _vertices(vertices,minimum=2):
    z=np.asarray(vertices,complex)
    if z.ndim!=1 or len(z)<minimum or not np.isfinite(z).all(): raise ValueError('finite complex vertices required')
    return z


def polyline_length(vertices):
    return float(np.sum(np.abs(np.diff(_vertices(vertices)))))


def polygon_area(vertices,*,signed=False):
    z=_vertices(vertices,3)
    value=float(np.sum(z.real*np.roll(z.imag,-1)-z.imag*np.roll(z.real,-1))/2)
    return value if signed else abs(value)


def koch_curve(depth,*,start=0j,end=1+0j,side=1,max_elements=1000000):
    if side not in (-1,1): raise ValueError('side must be -1 or +1')
    z=_vertices([start,end])
    if z[0]==z[1]: raise ValueError('distinct endpoints required')
    _allocation(4,depth,1,1,max_elements)
    for _ in range(depth): z=_koch_refine(z,side)
    return z


def _koch_refine(z,side):
    a=z[:-1]; d=np.diff(z)/3
    return np.concatenate((np.column_stack((a,a+d,a+d+d*np.exp(side*1j*np.pi/3),a+2*d)).ravel(),z[-1:]))


def koch_snowflake(depth,*,vertices=None,max_elements=1000000):
    triangle=np.array([0j,1+0j,.5+1j*np.sqrt(3)/2]) if vertices is None else _vertices(vertices,3)
    if len(triangle)!=3 or polygon_area(triangle)==0: raise ValueError('nondegenerate three-vertex triangle required')
    _allocation(4,depth,3,1,max_elements)
    side=-1 if polygon_area(triangle,signed=True)>0 else 1
    z=np.concatenate((triangle,triangle[:1]))
    for _ in range(depth): z=_koch_refine(z,side)
    return z


@dataclass
class WeierstrassResult:
    values: np.ndarray
    frequencies: np.ndarray
    amplitudes: np.ndarray
    tail_bound: float


def weierstrass(x,*,s,lam,terms,max_elements=1000000):
    n=_integer(terms,'terms',1); x=np.asarray(x,float)
    _allocation(1,0,x.size+2*n,0,max_elements)
    if not x.size or not np.isfinite(x).all() or not np.isfinite([s,lam]).all() or not 1<s<2 or lam<=1: raise ValueError('invalid Weierstrass controls')
    j=np.arange(1,n+1)
    with np.errstate(over='ignore',invalid='ignore'): frequency=float(lam)**j; amplitudes=float(lam)**(-(2-s)*j)
    if not np.isfinite(frequency).all(): raise ValueError('nonfinite retained frequencies')
    values=np.zeros_like(x)
    for f,a in zip(frequency,amplitudes):
        with np.errstate(over='ignore',invalid='ignore'): phase=f*x
        if not np.isfinite(phase).all(): raise ValueError('nonfinite series phase')
        values+=a*np.sin(phase)
    ratio=lam**(-(2-s))
    if ratio>=1: raise ValueError('tail bound is numerically unresolved')
    return WeierstrassResult(values,frequency,amplitudes,float(ratio**(n+1)/(1-ratio)))


def levy_distribution(*,lam,b,max_level,max_elements=1000000):
    level=_integer(max_level,'max_level'); _allocation(1,0,3*(level+1),0,max_elements)
    if not np.isfinite([lam,b]).all() or lam<=1 or b<=1: raise ValueError('lam and b must exceed one')
    j=np.arange(level+1)
    with np.errstate(over='ignore'): sizes=float(b)**j
    if not np.isfinite(sizes).all(): raise ValueError('nonfinite jump sizes')
    weights=np.exp(-j*np.log(lam)); weights=weights/weights.sum()
    return sizes,weights,float(lam**(-(level+1)))


@dataclass
class LevyFlightResult:
    increments: np.ndarray
    positions: np.ndarray
    levels: np.ndarray
    signs: np.ndarray
    probabilities: np.ndarray
    jump_sizes: np.ndarray
    omitted_probability: float
    seed: int


def levy_flight(*,steps,lam,b,max_level,seed,max_elements=1000000):
    n=_integer(steps,'steps',1); _allocation(1,0,4*n+1,0,max_elements)
    sizes,probabilities,omitted=levy_distribution(lam=lam,b=b,max_level=max_level,max_elements=max_elements)
    rng=np.random.default_rng(seed); levels=rng.choice(len(sizes),n,p=probabilities); signs=rng.choice([-1,1],n)
    increments=signs*sizes[levels]; positions=np.concatenate(([0.],np.cumsum(increments)))
    if not np.isfinite(positions).all(): raise ValueError('nonfinite walk')
    return LevyFlightResult(increments,positions,levels,signs,probabilities,sizes,omitted,seed)


def levy_characteristic(k,*,lam,b,max_level,max_elements=1000000):
    sizes,p,_=levy_distribution(lam=lam,b=b,max_level=max_level,max_elements=max_elements)
    k=np.asarray(k,float)
    _allocation(1,0,k.size+len(sizes)*2,0,max_elements)
    if not np.isfinite(k).all(): raise ValueError('finite frequencies required')
    result=np.zeros_like(k)
    for size,weight in zip(sizes,p):
        with np.errstate(over='ignore',invalid='ignore'): phase=k*size
        if not np.isfinite(phase).all(): raise ValueError('nonfinite characteristic phase')
        result+=weight*np.cos(phase)
    return result


@dataclass
class EscapeResult:
    escaped: np.ndarray
    counts: np.ndarray
    nonfinite: np.ndarray
    final_values: np.ndarray
    max_iterations: int
    bailout_radius: float


def _escape(initial,parameters,*,max_iterations,radius):
    n=_integer(max_iterations,'max_iterations',1)
    z,parameters=np.broadcast_arrays(initial,parameters); shape=z.shape
    z=z.astype(complex,copy=True).ravel(); parameters=parameters.ravel()
    escaped=np.abs(z)>radius; nonfinite=np.zeros(z.shape,bool); counts=np.zeros(z.shape,np.int64)
    for _ in range(n):
        indices=np.flatnonzero(~escaped & ~nonfinite)
        if not len(indices): break
        with np.errstate(over='ignore',invalid='ignore'): z[indices]=z[indices]*z[indices]+parameters[indices]
        counts[indices]+=1
        nonfinite[indices]=~np.isfinite(z[indices]); escaped[indices]=(~nonfinite[indices]) & (np.abs(z[indices])>radius)
    return EscapeResult(escaped.reshape(shape),counts.reshape(shape),nonfinite.reshape(shape),z.reshape(shape),n,radius)


def mandelbrot(grid,*,max_iterations,max_elements=1000000):
    grid=np.asarray(grid,complex)
    if not grid.size or grid.size>_integer(max_elements,'max_elements',1) or not np.isfinite(grid).all(): raise ValueError('invalid complex grid')
    return _escape(np.zeros_like(grid),grid,max_iterations=max_iterations,radius=2.)


def filled_julia(grid,*,c,max_iterations,max_elements=1000000):
    grid=np.asarray(grid,complex)
    if not grid.size or grid.size>_integer(max_elements,'max_elements',1) or not np.isfinite(grid).all() or not np.isscalar(c) or not np.isfinite(c): raise ValueError('invalid Julia grid/parameter')
    return _escape(grid,np.asarray(complex(c)),max_iterations=max_iterations,radius=max(2.,abs(c)))


@dataclass
class DimensionFit:
    slope: float
    intercept: float
    residuals: np.ndarray
    indices: np.ndarray
    scales: np.ndarray
    values: np.ndarray
    log_abscissa: np.ndarray
    log_values: np.ndarray
    kind: str


def fit_dimension(scales,values,*,fit_slice,kind):
    """Explicit log-log fit; box uses log(1/scale), correlation log(radius).

    Zero values outside the slice are retained as -inf logs, never fitted.
    """
    scales,values=np.asarray(scales,float),np.asarray(values,float)
    if scales.ndim!=1 or values.shape!=scales.shape or len(scales)<2 or not np.isfinite(scales).all() or not np.isfinite(values).all() or (scales<=0).any() or (values<0).any() or len(np.unique(scales))!=len(scales) or kind not in ('box','correlation'): raise ValueError('invalid diagnostic curve')
    if not isinstance(fit_slice,slice) or fit_slice.step not in (None,1) or fit_slice.start is None or fit_slice.stop is None or not 0<=fit_slice.start<fit_slice.stop<=len(scales): raise ValueError('explicit bounded fit slice required')
    indices=np.arange(len(scales))[fit_slice]
    if len(indices)<2 or np.any(values[indices]<=0): raise ValueError('at least two positive fit values required')
    x=-np.log(scales) if kind=='box' else np.log(scales)
    with np.errstate(divide='ignore'): y=np.log(values)
    slope,intercept=np.polyfit(x[indices],y[indices],1); residuals=y[indices]-(slope*x[indices]+intercept)
    return DimensionFit(float(slope),float(intercept),residuals,indices,scales.copy(),values.copy(),x,y,kind)
