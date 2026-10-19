"""Exact finite extension arithmetic and explicitly approximate LR compression."""
from dataclasses import dataclass
from itertools import product
from math import prod
import numpy as np
from .sets import finite,grades,triangle,spread


def extension(universes,memberships,function,*,max_combinations=1_000_000):
    """Exact output keys; NumPy integer inputs become Python ints before callbacks.

    No rounding/binning. Fraction inputs and keys stay exact.
    """
    if isinstance(max_combinations,bool) or not isinstance(max_combinations,(int,np.integer)) or max_combinations<1: raise ValueError('positive combination bound required')
    if not callable(function): raise ValueError('extension function must be callable')
    try:
        if not 0<len(universes)<=32 or len(universes)!=len(memberships): raise ValueError('one to 32 input sets required')
        sizes=[len(x) for x in universes]
        grade_sizes=[len(m) for m in memberships]
    except TypeError as exc: raise ValueError('sized finite input collections required') from exc
    if any(n<1 for n in sizes) or sizes!=grade_sizes or prod(sizes)>max_combinations: raise ValueError('invalid finite sets or extension combination limit exceeded')
    xs=[tuple(int(v) if isinstance(v,np.integer) else v for v in x) for x in universes]
    ms=[grades(m) for m in memberships]
    if any(m.shape!=(n,) for m,n in zip(ms,sizes)): raise ValueError('membership vectors required')
    for x in xs:
        finite(x)
        if len(set(x))!=len(x): raise ValueError('input universe has duplicates')
    result={}
    for indices in product(*(range(len(x)) for x in xs)):
        grade=min(float(m[i]) for m,i in zip(ms,indices))
        y=function(*(x[i] for x,i in zip(xs,indices)))
        if np.ndim(y) or not np.isfinite(float(y)): raise ValueError('finite scalar output required')
        result[y]=max(result.get(y,0.),grade)
    return dict(sorted(result.items()))


@dataclass(frozen=True)
class CutIntervals:
    levels: np.ndarray
    bounds: np.ndarray

    def __post_init__(self):
        try: size=len(self.levels)
        except TypeError as exc: raise ValueError('sized cut levels required') from exc
        if not 0<size<=1_000_000: raise ValueError('cut allocation limit exceeded or empty levels')
        try: bounds_size=len(self.bounds)
        except TypeError as exc: raise ValueError('cut bounds required') from exc
        if bounds_size!=size: raise ValueError('matching bounds required')
        a=grades(self.levels).copy();b=finite(self.bounds).copy()
        if a.ndim!=1 or not len(a) or np.any(np.diff(a)<=0) or b.shape!=(len(a),2) or np.any(b[:,0]>b[:,1]): raise ValueError('ordered levels and intervals required')
        if np.any(np.diff(b[:,0])<0) or np.any(np.diff(b[:,1])>0): raise ValueError('alpha cuts must be nested')
        a.flags.writeable=False;b.flags.writeable=False
        object.__setattr__(self,'levels',a);object.__setattr__(self,'bounds',b)

    def operation(self,other,*,operator):
        if not isinstance(other,CutIntervals) or not np.array_equal(self.levels,other.levels): raise ValueError('matching cut levels required')
        a,b=self.bounds,other.bounds
        if operator=='add': c=a+b
        elif operator=='subtract': c=np.column_stack((a[:,0]-b[:,1],a[:,1]-b[:,0]))
        elif operator in ('multiply','divide'):
            if operator=='divide':
                if np.any((b[:,0]<=0)&(b[:,1]>=0)): raise ValueError('denominator cut contains zero')
                b=1/b[:,::-1]
            with np.errstate(over='raise',invalid='raise'):
                pairs=np.stack([a[:,i]*b[:,j] for i in (0,1) for j in (0,1)],axis=1)
            c=np.column_stack((pairs.min(axis=1),pairs.max(axis=1)))
        else: raise ValueError('unknown arithmetic operation')
        return CutIntervals(self.levels,c)


@dataclass(frozen=True)
class Triangular:
    mode: float
    left: float
    right: float

    def __post_init__(self):
        finite([self.mode,self.left,self.right])
        if self.left<0 or self.right<0: raise ValueError('nonnegative spreads required')
        finite([self.mode-self.left,self.mode+self.right])
    def membership(self,x):return triangle(x,self.mode-self.left,self.mode,self.mode+self.right)
    def cuts(self,levels):
        a=grades(levels);return CutIntervals(a,np.column_stack((self.mode-self.left*(1-a),self.mode+self.right*(1-a))))
    def negate(self):return Triangular(-self.mode,self.right,self.left)
    def scale(self,value):
        finite(value)
        return Triangular(self.mode*value,abs(value)*(self.left if value>=0 else self.right),abs(value)*(self.right if value>=0 else self.left))
    def add(self,other):return Triangular(self.mode+other.mode,self.left+other.left,self.right+other.right)
    def subtract(self,other):return self.add(other.negate())
    def approximate_product(self,other):
        """First-order shape compression; exact only in special cases."""
        # Derivative signs select which input flank contributes to each flank.
        a=self.scale(other.mode);b=other.scale(self.mode)
        return Triangular(self.mode*other.mode,a.left+b.left,a.right+b.right)
    def approximate_reciprocal(self):
        """Match support endpoints and mode; interior membership is approximated."""
        lo=self.mode-self.left;hi=self.mode+self.right
        if lo<=0<=hi: raise ValueError('support contains zero')
        center=1/self.mode
        return Triangular(center,center-1/hi,1/lo-center)
    def approximate_divide(self,other):return self.approximate_product(other.approximate_reciprocal())


@dataclass(frozen=True)
class LRInterval:
    core_left: float
    core_right: float
    left: float
    right: float
    left_kind: str='linear'
    right_kind: str='linear'
    power: float=2.

    def __post_init__(self):
        finite([self.core_left,self.core_right,self.left,self.right,self.power])
        if self.core_left>self.core_right or self.left<0 or self.right<0: raise ValueError('invalid LR interval')
        spread([0.],kind=self.left_kind,power=self.power);spread([0.],kind=self.right_kind,power=self.power)
    def membership(self,x):
        x=finite(x);out=np.ones_like(x)
        if self.left>0: out=np.where(x<self.core_left,spread(np.maximum(0,(self.core_left-x)/self.left),kind=self.left_kind,power=self.power),out)
        else: out=np.where(x<self.core_left,0,out)
        if self.right>0: out=np.where(x>self.core_right,spread(np.maximum(0,(x-self.core_right)/self.right),kind=self.right_kind,power=self.power),out)
        else: out=np.where(x>self.core_right,0,out)
        return out

    def cuts(self,levels):
        """Finite support-closure cuts for linear LR flanks only."""
        if self.left_kind!='linear' or self.right_kind!='linear': raise ValueError('cuts require linear flanks; use explicit CutIntervals for other shapes')
        a=grades(levels)
        return CutIntervals(a,np.column_stack((self.core_left-self.left*(1-a),self.core_right+self.right*(1-a))))
