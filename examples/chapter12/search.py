"""Run: PYTHONPATH=src python examples/chapter12/search.py (headless)."""
import argparse
import json
import math
import platform
from pathlib import Path
from dataclasses import asdict,is_dataclass
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nonlinear_workbook import chapter12 as c
from nonlinear_workbook.chapter11.fixtures import CITIES
from nonlinear_workbook.chapter11 import exact_tsp


def clean(x):
    if is_dataclass(x): return clean(asdict(x))
    if isinstance(x,dict): return {k:clean(v) for k,v in x.items()}
    if isinstance(x,np.ndarray): return clean(x.tolist())
    if isinstance(x,(tuple,list)): return [clean(v) for v in x]
    if isinstance(x,np.generic): return x.item()
    return x


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',default='outputs/chapter12')
    out=Path(parser.parse_args().output); out.mkdir(parents=True,exist_ok=True)
    seeds=[42,43,44,45,46]
    common=dict(population_size=32,generations=100,selection='tournament',
                crossover_probability=.7,mutation_probability=.03,elitism=True)
    objective=lambda a: math.cos(c.decode_binary(a,0,2*math.pi))-math.sin(2*c.decode_binary(a,0,2*math.pi))
    grid=np.arange(1024)*2*math.pi/1023; gridvalues=np.cos(grid)-np.sin(2*grid)
    scalar=[c.binary_ga(objective,bits=10,seed=s,**common) for s in seeds]
    # Book Gaussian objective, [-2,2]^2; maximum one at (1,0).
    def gaussian(a):
        x=c.decode_binary(a[:12],-2,2); y=c.decode_binary(a[12:],-2,2)
        return math.exp(-(x-1)**2-y*y/2)
    two=c.binary_ga(gaussian,bits=24,seed=42,**dict(common,selection='roulette'))
    knap=lambda a: c.knapsack_score(a,c.KNAPSACK_WEIGHTS,c.KNAPSACK_VALUES,20,infeasible_value=-1)[0]
    exactbits,exactvalue=c.exact_knapsack(c.KNAPSACK_WEIGHTS,c.KNAPSACK_VALUES,20)
    knapsack=[c.binary_ga(knap,bits=12,seed=s,**common) for s in seeds]
    d=np.linalg.norm(CITIES[:,None]-CITIES[None,:],axis=-1)
    tsp={operator:[c.tsp_ga(d,population_size=20,generations=100,seed=s,crossover=operator) for s in seeds] for operator in ('pmx','composition')}
    reference=exact_tsp(CITIES)
    energy=lambda x: math.sin(float(x[0]))*float(x[0])**2*math.exp(-float(x[0])/15)
    # Independent derivative-bracket reference by bisection, plus endpoints.
    derivative=lambda x: x*((2-x/15)*math.sin(x)+x*math.cos(x))
    roots=[0.,100.]
    intervals=np.linspace(.00001,100,10001)
    for lo,hi in zip(intervals[:-1],intervals[1:]):
        if derivative(lo)*derivative(hi)>=0: continue
        for _ in range(60):
            mid=(lo+hi)/2
            if derivative(lo)*derivative(mid)<=0: hi=mid
            else: lo=mid
        roots.append((lo+hi)/2)
    target=min((energy([x]),x) for x in roots)
    schedule=np.geomspace(100,.01,100)
    anneal=[c.simulated_annealing(energy,[50.],proposal=lambda x,rng: rng.uniform(0,100,1),
              temperatures=schedule,steps_per_temperature=20,seed=s) for s in seeds]
    islands=c.island_ga(objective,islands=3,migration_interval=20,rounds=5,seed=42,bits=10,
                **{k:v for k,v in common.items() if k!='generations'})
    # Explicit sensitivity experiment: fixed budgets, no inferred optimal controls.
    sweep=[]
    for bits in (6,10,14):
        for mutation in (0.,.03,.15):
            runs=[c.binary_ga(objective,bits=bits,seed=s,**dict(common,mutation_probability=mutation)) for s in seeds]
            sweep.append(dict(bits=bits,mutation=mutation,values=[r.best_value for r in runs]))
    report=dict(python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__,
        seeds=seeds,conventions='MSB first, inclusive bounds; maximize GA, minimize annealing; half-open PMX',
        parameters=common,scalar=dict(bits=10,bounds=[0,2*math.pi],grid_target=float(gridvalues.max()),continuous_target=1.760172593046087,runs=scalar),
        gaussian=dict(bits_per_coordinate=12,bounds=[-2,2],analytic_target=1.,run=two),
        knapsack=dict(weights=c.KNAPSACK_WEIGHTS,values=c.KNAPSACK_VALUES,capacity=20,exact_bits=exactbits,exact_value=exactvalue,runs=knapsack),
        tsp=dict(coordinates=CITIES,reference=reference,population_size=20,generations=100,runs=tsp),
        annealing=dict(bounds=[0,100],initial=[50.],proposal='independent uniform',steps_per_temperature=20,target=target,runs=anneal),
        islands=dict(islands=3,migration_interval=20,rounds=5,result=islands),sensitivity=sweep)
    (out/'report.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for r in scalar: axes[0,0].plot(r.best_history)
    axes[0,0].axhline(gridvalues.max(),color='black',linestyle='--'); axes[0,0].set_title('Scalar best-so-far fitness')
    for r in knapsack: axes[0,1].plot(r.best_history)
    axes[0,1].axhline(149,color='black',linestyle='--'); axes[0,1].set_title('Knapsack value; exact target 149')
    for operator,runs in tsp.items():
        axes[1,0].plot(runs[0].best_history,label=operator)
    axes[1,0].set_title('TSP tour length'); axes[1,0].legend()
    for r in anneal: axes[1,1].plot(r.best_history)
    axes[1,1].axhline(target[0],color='black',linestyle='--'); axes[1,1].set_title('Annealing best-so-far energy')
    for ax in axes.flat: ax.set_xlabel('Generation / proposal'); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out/'search.png',dpi=150,metadata={'Software':'nonlinear-workbook'}); plt.close(fig)
    np.savetxt(out/'sensitivity.csv',[[r['bits'],r['mutation'],s,v] for r in sweep for s,v in zip(seeds,r['values'])],delimiter=',',header='bits,mutation,seed,best_value',comments='')
    print('scalar', [r.best_value for r in scalar]); print('gaussian',two.best_value)
    print('knapsack',[r.best_value for r in knapsack]); print('TSP',{k:[r.best_value for r in v] for k,v in tsp.items()})
    print('annealing target',target,'observed',[r.best_value for r in anneal])
    print('Saved',out)


if __name__=='__main__': main()
