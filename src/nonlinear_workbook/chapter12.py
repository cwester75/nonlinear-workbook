"""Chapter 12: bounded, reproducible evolutionary search and annealing.

Search results are finite-budget outcomes, not certificates of optimality.
"""
from dataclasses import dataclass
import itertools
import math
import numpy as np


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _bits(bits):
    a = np.asarray(bits)
    if a.ndim != 1 or not len(a) or not np.all((a == 0) | (a == 1)):
        raise ValueError('expected a nonempty binary vector')
    return a.astype(np.int8)


def decode_binary(bits, lower, upper):
    """MSB first; inclusive endpoints, lower + integer*(upper-lower)/(2**N-1)."""
    a = _bits(bits)
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError('finite increasing bounds required')
    m = 0
    for bit in a: m = 2*m + int(bit)
    maximum = 2**len(a)-1
    if m == 0: return float(lower)
    if m == maximum: return float(upper)
    fraction = m/maximum
    # Opposite signs can overflow in upper-lower; convex interpolation avoids it.
    if lower < 0 < upper:
        return float((1-fraction)*lower + fraction*upper)
    return float(lower + fraction*(upper-lower))


def schema_matches(bits, schema):
    a = _bits(bits)
    if len(schema) != len(a) or any(c not in '01*' for c in schema):
        raise ValueError('schema must match vector length and use 0, 1, *')
    return all(c == '*' or int(c) == b for b, c in zip(a, schema))


def schema_properties(schema):
    if not schema or any(c not in '01*' for c in schema): raise ValueError('invalid schema')
    fixed = [i for i,c in enumerate(schema) if c != '*']
    return len(fixed), (fixed[-1]-fixed[0] if fixed else 0)


def roulette_indices(weights, count, *, rng):
    w = np.asarray(weights, dtype=float)
    count = _integer(count, 'count', 0)
    if w.ndim != 1 or not len(w) or not np.isfinite(w).all() or (w < 0).any() or w.max() <= 0:
        raise ValueError('roulette needs finite nonnegative weights with positive total')
    w = w/w.max()
    return rng.choice(len(w), size=count, p=w/w.sum())


def binary_crossover(a, b, cut):
    a,b = _bits(a),_bits(b)
    cut = _integer(cut, 'cut', 0)
    if len(a) != len(b) or cut > len(a): raise ValueError('invalid crossover')
    return np.concatenate((a[:cut],b[cut:])), np.concatenate((b[:cut],a[cut:]))


def mutate_bits(bits, probability, *, rng):
    a = _bits(bits)
    if not np.isfinite(probability) or not 0 <= probability <= 1: raise ValueError('invalid probability')
    return a ^ (rng.random(len(a)) < probability).astype(np.int8)


@dataclass
class SearchResult:
    best: np.ndarray
    best_value: float
    best_history: np.ndarray
    mean_history: np.ndarray
    diversity_history: np.ndarray
    evaluations: int
    population: np.ndarray
    termination: str = 'budget_exhausted'


def binary_ga(objective, *, bits, population_size, generations, seed,
              selection, crossover_probability, mutation_probability, elitism,
              initial_population=None):
    """Maximize objective(binary_vector). Explicit roulette or tournament policy.

    Roulette uses raw objective weights and rejects negative/all-zero fitness.
    Elitism preserves one best chromosome; reported best is best ever evaluated.
    """
    bits = _integer(bits,'bits'); size = _integer(population_size,'population_size',2)
    generations = _integer(generations,'generations',0)
    if selection not in ('roulette','tournament'): raise ValueError('invalid selection')
    if not isinstance(elitism,bool): raise ValueError('elitism must be bool')
    for p in (crossover_probability,mutation_probability):
        if not np.isfinite(p) or not 0 <= p <= 1: raise ValueError('invalid probability')
    rng = np.random.default_rng(seed)
    if initial_population is None: pop = rng.integers(0,2,(size,bits),dtype=np.int8)
    else:
        pop=np.asarray(initial_population)
        if pop.shape!=(size,bits) or not np.all((pop==0)|(pop==1)): raise ValueError('invalid initial population')
        pop=pop.astype(np.int8,copy=True)
    best = None; best_value = -np.inf; histories = [[],[],[]]
    for generation in range(generations+1):
        values = np.array([float(objective(a.copy())) for a in pop])
        if not np.isfinite(values).all(): raise ValueError('objective must be finite')
        winner = int(np.argmax(values))
        if values[winner] > best_value: best_value,best = float(values[winner]),pop[winner].copy()
        histories[0].append(best_value); histories[1].append(values.mean())
        histories[2].append(len(np.unique(pop,axis=0))/size)
        if generation == generations: break
        if selection == 'roulette': indices = roulette_indices(values,size,rng=rng)
        else:
            pairs = rng.integers(0,size,(size,2))
            indices = np.array([p[np.argmax(values[p])] for p in pairs])
        new = pop[indices].copy()
        for i in range(0,size-1,2):
            if rng.random() < crossover_probability:
                new[i],new[i+1] = binary_crossover(new[i],new[i+1],int(rng.integers(1,bits+1)))
        for i in range(size): new[i] = mutate_bits(new[i],mutation_probability,rng=rng)
        if elitism: new[0] = best
        pop = new
    return SearchResult(best,best_value,*map(np.asarray,histories),size*(generations+1),pop)


KNAPSACK_WEIGHTS = (11,7,5,4,3,3,3,2,2,2,2,1)
KNAPSACK_VALUES = (20,10,11,5,25,50,15,12,6,4,5,30)


def knapsack_score(bits, weights, values, capacity, *, infeasible_value):
    a = _bits(bits); w,v = np.asarray(weights,float),np.asarray(values,float)
    if w.shape != a.shape or v.shape != a.shape or not np.isfinite(w).all() or not np.isfinite(v).all() or (w <= 0).any() or (v < 0).any():
        raise ValueError('invalid items')
    if not np.isfinite([capacity,infeasible_value]).all() or capacity < 0: raise ValueError('invalid capacity/score')
    weight,value = float(a@w),float(a@v)
    return (value if weight <= capacity else float(infeasible_value)),weight,weight <= capacity


def exact_knapsack(weights, values, capacity, *, max_items=20):
    n = len(weights)
    if n > _integer(max_items,'max_items') or n < 1: raise ValueError('enumeration limit')
    candidates = itertools.product((0,1),repeat=n)
    best,value = None,-np.inf
    for bits in candidates:
        score,_,feasible = knapsack_score(bits,weights,values,capacity,infeasible_value=-1)
        if feasible and score > value: best,value = np.array(bits,dtype=np.int8),score
    return best,value


def _permutations(a,b):
    a,b = np.asarray(a),np.asarray(b)
    if a.ndim != 1 or b.shape != a.shape or not len(a) or a.dtype.kind not in 'iu' or b.dtype.kind not in 'iu' or len(set(a)) != len(a) or set(a) != set(b):
        raise ValueError('matching unique integer permutations required')
    return a.copy(),b.copy()


def pmx(a,b,start,stop):
    """PMX with half-open segment [start, stop); preserves arbitrary integer labels."""
    a,b = _permutations(a,b)
    start,stop = _integer(start,'start',0),_integer(stop,'stop',0)
    if not 0 <= start <= stop <= len(a): raise ValueError('invalid segment')
    def child(p,q):
        out = p.copy(); out[start:stop] = q[start:stop]
        mapping = dict(zip(q[start:stop],p[start:stop]))
        for i in list(range(start))+list(range(stop,len(p))):
            value = p[i]
            while value in mapping: value = mapping[value]
            out[i] = value
        return out
    return child(a,b),child(b,a)


def permutation_composition(a,b):
    """Book p.470 composition b[a[i]], for zero-based permutations."""
    a,b = _permutations(a,b)
    if set(a) != set(range(len(a))): raise ValueError('composition requires zero-based labels')
    return b[a],a[b]


def tsp_ga(distances, *, population_size, generations, seed, crossover):
    """Greedy parent replacement, swap mutation, and PMX or composition."""
    d = np.asarray(distances,float)
    if d.ndim != 2 or d.shape[0] != d.shape[1] or len(d)<2 or not np.isfinite(d).all() or (d<0).any(): raise ValueError('invalid distance matrix')
    size = _integer(population_size,'population_size',2); generations = _integer(generations,'generations',0)
    if crossover not in ('pmx','composition'): raise ValueError('invalid crossover')
    rng = np.random.default_rng(seed); n=len(d)
    pop = np.array([rng.permutation(n) for _ in range(size)])
    def length(p): return float(d[p,np.roll(p,-1)].sum())
    history=[]; means=[]; diversity=[]; evaluations=size
    scores=np.array([length(p) for p in pop])
    for g in range(generations+1):
        winner=int(scores.argmin())
        history.append(scores[winner]); means.append(scores.mean()); diversity.append(len(np.unique(pop,axis=0))/size)
        if g==generations: break
        for i in range(size):
            candidate=pop[i].copy(); j,k=rng.choice(n,2,replace=False)
            candidate[j],candidate[k]=candidate[k],candidate[j]
            value=length(candidate); evaluations+=1
            if value<scores[i]: pop[i],scores[i]=candidate,value
        for _ in range(size//2):
            i,j=rng.choice(size,2,replace=False)
            if crossover=='pmx':
                start,stop=sorted(rng.choice(n+1,2,replace=False)); children=pmx(pop[i],pop[j],int(start),int(stop))
            else: children=permutation_composition(pop[i],pop[j])
            for index,child in zip((i,j),children):
                value=length(child); evaluations+=1
                if value<scores[index]: pop[index],scores[index]=child,value
    winner=int(scores.argmin())
    return SearchResult(pop[winner].copy(),float(scores[winner]),np.array(history),np.array(means),np.array(diversity),evaluations,pop)


def metropolis_probability(delta, temperature):
    if not np.isfinite([delta,temperature]).all() or temperature <= 0: raise ValueError('finite delta and positive temperature required')
    return 1.0 if delta<=0 else math.exp(-delta/temperature)


@dataclass
class AnnealingResult:
    best: np.ndarray
    best_value: float
    final: np.ndarray
    final_value: float
    values: np.ndarray
    best_history: np.ndarray
    accepted: np.ndarray
    temperatures: np.ndarray
    evaluations: int


def simulated_annealing(objective, initial, *, proposal, temperatures, steps_per_temperature, seed):
    """Minimize with proposal(state_copy, rng) and an explicit temperature schedule."""
    schedule=np.asarray(temperatures,float)
    if schedule.ndim!=1 or not len(schedule) or not np.isfinite(schedule).all() or (schedule<=0).any() or (np.diff(schedule)>0).any(): raise ValueError('positive nonincreasing schedule required')
    steps=_integer(steps_per_temperature,'steps_per_temperature')
    state=np.asarray(initial,float).copy()
    if not state.size or not np.isfinite(state).all(): raise ValueError('finite initial state required')
    def evaluate(x):
        value=float(objective(x.copy()))
        if not np.isfinite(value): raise ValueError('finite objective required')
        return value
    value=evaluate(state); best=state.copy(); best_value=value
    values=[value]; history=[value]; accepted=[]; ts=[]; rng=np.random.default_rng(seed)
    for temperature in schedule:
        for _ in range(steps):
            candidate=np.asarray(proposal(state.copy(),rng),float)
            if candidate.shape!=state.shape or not np.isfinite(candidate).all(): raise ValueError('invalid proposal')
            new=evaluate(candidate); take=rng.random()<metropolis_probability(new-value,float(temperature))
            if take: state,value=candidate.copy(),new
            if value<best_value: best,best_value=state.copy(),value
            values.append(value); history.append(best_value); accepted.append(take); ts.append(temperature)
    return AnnealingResult(best,best_value,state,value,np.array(values),np.array(history),np.array(accepted),np.array(ts),len(values))


def island_ga(objective, *, islands, migration_interval, rounds, seed, **ga_parameters):
    """Serial island model; ring migration copies each island's best to next worst.

    Uses independently spawned streams. This models migration, not parallel speed.
    """
    islands=_integer(islands,'islands',2); rounds=_integer(rounds,'rounds'); interval=_integer(migration_interval,'migration_interval')
    if 'generations' in ga_parameters or 'initial_population' in ga_parameters: raise ValueError('round length is migration_interval')
    streams=np.random.SeedSequence(seed).spawn(islands)
    rngs=[np.random.default_rng(s) for s in streams]
    populations=[None]*islands; history=[]; evaluations=0; results=None; best=None
    for _ in range(rounds):
        results=[binary_ga(objective, generations=interval,seed=int(r.integers(0,2**63)),
                  initial_population=p,**ga_parameters) for r,p in zip(rngs,populations)]
        evaluations+=sum(r.evaluations for r in results)
        winner=max(results,key=lambda r:r.best_value)
        if best is None or winner.best_value>best.best_value: best=winner
        history.append([r.best_value for r in results])
        populations=[r.population.copy() for r in results]
        for i,result in enumerate(results):
            target=populations[(i+1)%islands]
            scores=[float(objective(a.copy())) for a in target]; evaluations+=len(target)
            worst=int(np.argmin(scores))
            if result.best_value>scores[worst]: target[worst]=result.best
    return dict(populations=np.array(populations),best_history=np.array(history),
                evaluations=evaluations,migrations=rounds*islands,
                best=best.best.copy(),best_value=best.best_value)
