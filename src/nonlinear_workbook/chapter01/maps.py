"""Chapter 1 scalar maps. Scalar algebra preserves int/Fraction inputs."""
import math
import numpy as np
from ..chapter02 import logistic_map  # Compatibility: same generator, no copied body.


def _finite(value, name):
    if not np.isfinite(float(value)):
        raise ValueError(f'{name} must be finite')


def _unit(x):
    _finite(x, 'x')
    if not 0 <= x <= 1:
        raise ValueError('x must be in [0,1]')


def collatz_step(x):
    if isinstance(x, bool) or not isinstance(x, int) or x < 1:
        raise ValueError('Collatz requires a positive Python integer')
    return x // 2 if x % 2 == 0 else 3*x + 1


def logistic_step(x, *, r=4):
    _unit(x)
    _finite(r, 'r')
    if not 0 < r <= 4:
        raise ValueError('r must be in (0,4]')
    return r*x*(1-x)


def logistic_derivative(x, *, r=4):
    return r*(1-2*x)


def bernoulli_step(x):
    _unit(x)
    return (2*x) % 1


def tent_step(x):
    _unit(x)
    return 2*x if x < .5 else 2*(1-x)


def bungalow_step(x, *, a):
    _unit(x)
    _finite(a, 'a')
    if not 0 < a < .5:
        raise ValueError('a must be in (0,.5)')
    if x < a: return (1-a)*x/a
    if x < .5: return 2*a*x/(1-2*a) + (1-3*a)/(1-2*a)
    if x < 1-a: return 2*a*(1-x)/(1-2*a) + (1-3*a)/(1-2*a)
    return (1-a)*(1-x)/a


def gauss_step(x):
    _unit(x)
    if x == 0: return x
    # Integer division for rational values retains exact reciprocals.
    reciprocal = 1/x
    return reciprocal - math.floor(reciprocal)


def sine_step(x):
    _unit(x)
    return float(np.sin(np.pi*x))


def sine_derivative(x):
    return float(np.pi*np.cos(np.pi*x))


def circle_step(x, *, omega, strength, wrap=False):
    """Lift by default; strength is the chapter's r parameter."""
    for name,v in [('x',x),('omega',omega),('strength',strength)]: _finite(v,name)
    y = x + omega - strength*np.sin(2*np.pi*x)/(2*np.pi)
    return float(y % 1 if wrap else y)


def polynomial_random_step(x):
    """Book demonstration frac((x+pi)**5), not an IID RNG."""
    _unit(x)
    return float((x+np.pi)**5 % 1)


def lcg_step(x, *, multiplier, increment, modulus):
    for name,v in [('x',x),('multiplier',multiplier),('increment',increment),('modulus',modulus)]:
        if isinstance(v,bool) or not isinstance(v,int): raise ValueError(f'{name} must be an integer')
    if modulus <= 0 or not 0 <= x < modulus or not 0 <= increment < modulus or not 0 <= multiplier < modulus:
        raise ValueError('Require modulus>0 and x,multiplier,increment in [0,modulus)')
    return (multiplier*x + increment) % modulus


def quadratic_step(x, *, r):
    """Symbolic-dynamics map 1-r*x**2 on [-1,1], pp.64-66."""
    _finite(x,'x'); _finite(r,'r')
    if not -1 <= x <= 1 or not 0 < r <= 2: raise ValueError('Invalid quadratic map state/parameter')
    return 1-r*x*x

# Multidimensional map coordinates follow the book's update order.
def _state2(state):
    state=np.asarray(state,dtype=float)
    if state.shape!=(2,) or not np.all(np.isfinite(state)): raise ValueError('Expected finite state [x,y]')
    return state


def _parameters(**values):
    for name,value in values.items(): _finite(value,name)


def henon_step(state, *, a=1.4, b=.3):
    x,y=_state2(state); _parameters(a=a,b=b)
    return np.array([1+y-a*x*x,b*x])


def henon_jacobian(state, *, a=1.4, b=.3):
    x,y=_state2(state); _parameters(a=a,b=b)
    return np.array([[-2*a*x,1],[b,0]])


def lozi_step(state, *, a=1.7, b=.4):
    x,y=_state2(state); _parameters(a=a,b=b)
    return np.array([1+y-a*abs(x),b*x])


def lozi_jacobian(state, *, a=1.7, b=.4):
    x,y=_state2(state); _parameters(a=a,b=b)
    if x==0: raise ValueError('Lozi map is nondifferentiable at x=0')
    return np.array([[-a*np.sign(x),1],[b,0]])


def standard_step(state, *, k=.8):
    """State [momentum,angle]; only angle wraps to [0,2*pi)."""
    momentum,angle=_state2(state); _parameters(k=k)
    new_momentum=momentum+k*np.sin(angle)
    return np.array([new_momentum,(angle+new_momentum)%(2*np.pi)])


def standard_jacobian(state, *, k=.8):
    momentum,angle=_state2(state); _parameters(k=k)
    c=k*np.cos(angle)
    return np.array([[1,c],[1,1+c]])


def ikeda_step(state, *, rho=.85, c1=.4, c2=.9, c3=9.):
    x,y=_state2(state); _parameters(rho=rho,c1=c1,c2=c2,c3=c3)
    phase=c1-c3/(1+x*x+y*y); c,s=np.cos(phase),np.sin(phase)
    return np.array([rho+c2*(x*c-y*s),c2*(x*s+y*c)])


def ikeda_jacobian(state, *, rho=.85, c1=.4, c2=.9, c3=9.):
    v=_state2(state); _parameters(rho=rho,c1=c1,c2=c2,c3=c3)
    phase=c1-c3/(1+v@v); c,s=np.cos(phase),np.sin(phase)
    rotation=np.array([[c,-s],[s,c]])
    grad=2*c3*v/(1+v@v)**2
    return c2*rotation@(np.eye(2)+np.outer([-v[1],v[0]],grad))


def coupled_logistic_step(state, *, r=3.7, coupling=.06):
    x,y=_state2(state); _parameters(r=r,coupling=coupling)
    return np.array([r*x*(1-x)+coupling*(y-x),r*y*(1-y)+coupling*(x-y)])


def coupled_logistic_jacobian(state, *, r=3.7, coupling=.06):
    x,y=_state2(state); _parameters(r=r,coupling=coupling)
    return np.array([[r*(1-2*x)-coupling,coupling],[coupling,r*(1-2*y)-coupling]])


def rtn_step(state, *, r=1.0834):
    """Book pp.104-106: sequential x update, then y uses new x."""
    x,y=_state2(state); _parameters(r=r)
    new_x=r*(3*y+1)*x*(1-x)
    return np.array([new_x,r*(3*new_x+1)*y*(1-y)])


def rtn_jacobian(state, *, r=1.0834):
    x,y=_state2(state); _parameters(r=r)
    new_x=r*(3*y+1)*x*(1-x)
    row1=np.array([r*(3*y+1)*(1-2*x),3*r*x*(1-x)])
    row2=3*r*y*(1-y)*row1+np.array([0,r*(3*new_x+1)*(1-2*y)])
    return np.vstack([row1,row2])
