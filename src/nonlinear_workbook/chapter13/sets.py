"""Finite fuzzy sets. Membership grades are not probabilities."""
from dataclasses import dataclass
import numpy as np


def finite(x):
    if np.iscomplexobj(x): raise ValueError('real values required')
    a=np.asarray(x,dtype=float)
    if not np.isfinite(a).all(): raise ValueError('finite values required')
    return a


def grades(x):
    a=finite(x)
    if np.any((a<0)|(a>1)): raise ValueError('grades must lie in [0,1]')
    return a


def _positive(x):
    if np.ndim(x) or not np.isfinite(x) or x<=0: raise ValueError('positive finite scalar required')
    return float(x)


def triangle(x,left,center,right):
    if not np.isfinite([left,center,right]).all() or not left<=center<=right: raise ValueError('ordered finite vertices required')
    x=finite(x); out=np.zeros_like(x)
    if center>left: out=np.maximum(out,np.where((x>=left)&(x<=center),(x-left)/(center-left),0))
    if right>center: out=np.maximum(out,np.where((x>=center)&(x<=right),(right-x)/(right-center),0))
    return np.where(x==center,1.,out)


def trapezoid(x,a,b,c,d):
    if not np.isfinite([a,b,c,d]).all() or not a<=b<=c<=d: raise ValueError('ordered finite vertices required')
    x=finite(x); out=np.where((x>=b)&(x<=c),1.,0.)
    if b>a: out=np.maximum(out,np.where((x>=a)&(x<b),(x-a)/(b-a),0))
    if d>c: out=np.maximum(out,np.where((x>c)&(x<=d),(d-x)/(d-c),0))
    return out


def rectangle(x,left,right):
    if not np.isfinite([left,right]).all() or left>right: raise ValueError('ordered bounds required')
    x=finite(x); return ((x>=left)&(x<=right)).astype(float)


def ramp(x,left,right,*,increasing=True):
    _positive(right-left); finite([left,right]); y=np.clip((finite(x)-left)/(right-left),0,1)
    return y if increasing else 1-y


def _normalized_difference(x,center,width):
    width=_positive(width)
    x,center=np.broadcast_arrays(finite(x),finite(center))
    # Subtract first to preserve cancellation accuracy on the ordinary path.
    # Only opposite-sign extremes can overflow finite subtraction. Scaling their
    # operands first avoids that overflow; their signs prevent inf-inf here.
    with np.errstate(over='ignore',under='ignore'):
        difference=x-center
        result=np.array(difference/width,copy=True)
        overflow=~np.isfinite(difference)
        if np.any(overflow):
            result[overflow]=x[overflow]/width-center[overflow]/width
    return result


def gaussian(x,center,sigma):
    z=_normalized_difference(x,center,sigma)
    with np.errstate(over='ignore',under='ignore'): return np.exp(-.5*z**2)


def logistic(x,center,width):
    z=_normalized_difference(x,center,width)
    # logaddexp avoids overflow in either tail; underflow is valid saturation.
    with np.errstate(under='ignore'): return np.exp(-np.logaddexp(0,-z))


def spread(z,*,kind='linear',power=2.):
    z=finite(z); _positive(power)
    if np.any(z<0): raise ValueError('spread argument must be nonnegative')
    if kind=='linear': return np.maximum(0,1-z)
    if kind=='exponential': return np.exp(-z)
    if kind=='gaussian':
        with np.errstate(over='ignore'): return np.exp(-z*z)
    if kind=='polynomial_outer': return np.maximum(0,1-z)**power
    if kind=='polynomial':
        with np.errstate(over='ignore'): return np.maximum(0,1-z**power)
    raise ValueError('unknown spread')


def combine(a,b,*,operator='min',gamma=.5):
    a=grades(a); b=grades(b)
    if a.shape!=b.shape: raise ValueError('matching shapes required')
    if operator=='min': return np.minimum(a,b)
    if operator=='max': return np.maximum(a,b)
    if operator=='product': return a*b
    if operator=='algebraic_sum': return a+b-a*b
    if operator=='bounded_sum': return np.minimum(1,a+b)
    if operator=='bounded_difference': return np.maximum(0,a+b-1)
    if operator in ('fuzzy_and','fuzzy_or'):
        if not np.isfinite(gamma) or not 0<=gamma<=1: raise ValueError('gamma outside [0,1]')
        extreme=np.minimum(a,b) if operator=='fuzzy_and' else np.maximum(a,b)
        return gamma*extreme+(1-gamma)*(a/2+b/2)
    raise ValueError('unknown operator')


def complement(a): return 1-grades(a)

def hedge(a,exponent): return grades(a)**_positive(exponent)

def contrast(a):
    a=grades(a); return np.where(a<.5,2*a*a,1-2*(1-a)**2)


def entropy(a,*,base=2.):
    a=grades(a)
    if not np.isfinite(base) or base<=1: raise ValueError('base must exceed one')
    t=a[(a>0)&(a<1)]
    return float(np.sum(-t*np.log(t)-(1-t)*np.log1p(-t))/np.log(base))


def weighted_mean(sets,weights=None):
    a=grades(sets)
    if a.ndim<2 or not len(a): raise ValueError('nonempty collection required')
    w=np.ones(len(a)) if weights is None else finite(weights)
    if w.shape!=(len(a),) or np.any(w<0) or not np.any(w>0): raise ValueError('nonnegative nonzero weights required')
    w=w/w.max(); return np.average(a,axis=0,weights=w)


def cartesian_min(*arrays,max_elements=1_000_000):
    if isinstance(max_elements,bool) or not isinstance(max_elements,(int,np.integer)) or max_elements<1: raise ValueError('positive allocation bound required')
    if not arrays: raise ValueError('at least one set required')
    from math import prod
    try: sizes=[len(a) for a in arrays]
    except TypeError as exc: raise ValueError('sized Cartesian vectors required') from exc
    if len(arrays)>32 or any(n<1 for n in sizes) or prod(sizes)>max_elements: raise ValueError('invalid Cartesian allocation')
    arrays=[grades(a) for a in arrays]
    if any(a.ndim!=1 for a in arrays): raise ValueError('Cartesian vectors required')
    grids=np.meshgrid(*arrays,indexing='ij');return np.minimum.reduce(grids)


def gaussian_compatibility(a,sa,b,sb):
    """sup_x min(exp(-((x-a)/sa)^2),exp(-((x-b)/sb)^2))."""
    _positive(sa);_positive(sb);finite([a,b])
    scale=max(sa,sb);wa=sa/scale;wb=sb/scale
    x=(wb/(wa+wb))*a+(wa/(wa+wb))*b
    if a==b: value=1.
    else:
        with np.errstate(over='ignore',invalid='raise'):
            difference=np.float64(a)-np.float64(b)
            distance=(difference/scale if np.isfinite(difference) else a/scale-b/scale)/(wa+wb)
            value=np.exp(-distance**2)
    if not np.isfinite([value,x]).all(): raise ValueError('nonfinite compatibility')
    return dict(value=float(value),location=float(x))


@dataclass(frozen=True)
class FuzzySet:
    universe: np.ndarray
    membership: np.ndarray

    def __post_init__(self):
        try: size=len(self.universe)
        except TypeError as exc: raise ValueError('sized universe required') from exc
        if not 0<size<=1_000_000: raise ValueError('universe exceeds one million elements or is empty')
        try: membership_size=len(self.membership)
        except TypeError as exc: raise ValueError('membership vector required') from exc
        if membership_size!=size: raise ValueError('matching membership length required')
        x=finite(self.universe).copy();m=grades(self.membership).copy()
        if x.ndim!=1 or not len(x) or m.shape!=x.shape or np.any(np.diff(x)<=0): raise ValueError('strictly increasing finite universe required')
        x.flags.writeable=False;m.flags.writeable=False
        object.__setattr__(self,'universe',x);object.__setattr__(self,'membership',m)

    @property
    def support(self): return self.universe[self.membership>0]
    @property
    def height(self): return float(self.membership.max())
    @property
    def cardinality(self): return float(self.membership.sum())
    @property
    def relative_cardinality(self): return self.cardinality/len(self.universe)
    def normalize(self):
        if self.height==0: raise ValueError('zero height')
        return FuzzySet(self.universe,self.membership/self.height)
    def alpha_cut(self,alpha):
        if np.ndim(alpha): raise ValueError('scalar alpha required')
        grades(alpha);return self.universe[self.membership>=alpha]
    def clipped(self,ceiling):
        if np.ndim(ceiling): raise ValueError('scalar ceiling required')
        grades(ceiling);return FuzzySet(self.universe,np.minimum(self.membership,ceiling))
    def sampled_quasiconcave(self):
        peak=int(np.argmax(self.membership))
        return bool(np.all(np.diff(self.membership[:peak+1])>=0) and np.all(np.diff(self.membership[peak:])<=0))
    def top_beta(self,beta):
        if isinstance(beta,bool) or not isinstance(beta,(int,np.integer)) or not 0<=beta<=len(self.universe): raise ValueError('invalid beta')
        # Stable ties: smaller universe coordinate first.
        indices=np.argsort(-self.membership,kind='stable')[:beta]
        return self.universe[np.sort(indices)]
    def at(self,x,*,outside='raise'):
        x=finite(x)
        if outside not in ('raise','zero'): raise ValueError('unknown outside policy')
        if outside=='raise' and np.any((x<self.universe[0])|(x>self.universe[-1])): raise ValueError('outside universe')
        return np.interp(x,self.universe,self.membership,left=0,right=0)
    def _aligned(self,other):
        if not isinstance(other,FuzzySet) or not np.array_equal(self.universe,other.universe): raise ValueError('universes must match; resample explicitly')
    def subset(self,other):
        self._aligned(other);return bool(np.all(self.membership<=other.membership))
    def operation(self,other,**kwargs):
        self._aligned(other);return FuzzySet(self.universe,combine(self.membership,other.membership,**kwargs))
    def centroid(self,*,mode='discrete'):
        x=self.universe;m=self.membership
        if mode=='discrete': weights=m
        elif mode=='continuous':
            if len(x)<2: raise ValueError('continuous centroid requires at least two points')
            dx=np.diff(x);weights=m*np.r_[dx[0]/2,(dx[:-1]+dx[1:])/2,dx[-1]/2]
        else: raise ValueError('unknown centroid mode')
        if not np.any(weights>0): raise ValueError('zero membership mass')
        result=float(np.sum((weights/weights.sum())*x))
        if not np.isfinite(result): raise ValueError('nonfinite centroid')
        return result
    def extremum(self,*,kind='max',ties='first'):
        if kind not in ('min','max') or ties not in ('first','last','mean'): raise ValueError('invalid extremum policy')
        target=self.membership.max() if kind=='max' else self.membership.min()
        x=self.universe[self.membership==target]
        return float(x[0] if ties=='first' else x[-1] if ties=='last' else np.mean(x))


def compare(a,b,*,operator):
    a=grades(a);b=grades(b)
    if a.shape!=b.shape: raise ValueError('matching shapes required')
    operations={'lt':np.less,'le':np.less_equal,'eq':np.equal,'ne':np.not_equal,'ge':np.greater_equal,'gt':np.greater}
    if operator not in operations: raise ValueError('unknown comparison')
    return operations[operator](a,b).astype(float)
