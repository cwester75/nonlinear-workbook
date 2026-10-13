"""Headless solver validation and sampled-observation sensitivity sweeps."""
import json
import platform
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nonlinear_workbook import chapter10 as c
from nonlinear_workbook.chapter02 import kantz_curve,hurst_rs,lz_complexity
from nonlinear_workbook.validation import sampled_observations


def main():
    out=Path('outputs/chapter10'); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for method in ('euler','rkf45','rkf45_adaptive'):
        for h in (.2,.1,.05):
            r=c.integrate(lambda t,y:y*(1-y),[.1],t_span=(0,10),method=method,step=h,atol=1e-10,rtol=1e-10)
            target=1/(1+9*np.exp(-r.times))
            rows.append(dict(method=method,step=h,max_error=float(np.max(abs(r.states[:,0]-target))),evaluations=r.evaluations,status=r.status))
    t,ghost,status=c.central_difference(lambda y:y*(1-y),.5,.5125,step=.05,n=1000,escape_radius=10)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].plot(t,ghost[:,0],label='central difference'); axes[0].plot(t,1/(1+np.exp(-t)),label='exact'); axes[0].legend(); axes[0].set_xlabel('Time'); axes[0].set_title('Ghost solution')
    energies={}
    for method in ('euler','symplectic_euler','verlet'):
        q,p=np.array([1.]),np.array([0.]); energy=[.5]
        for _ in range(3000):
            if method=='euler': q,p=q+.1*p,p-.1*q
            else: q,p=c.symplectic_step(q,p,.1,grad_potential=lambda x:x,grad_kinetic=lambda x:x,method='euler' if method=='symplectic_euler' else method)
            energy.append(float((q@q+p@p)/2))
        energies[method]=dict(max_error=float(np.max(abs(np.array(energy)-.5))),final=energy[-1])
        axes[1].semilogy(np.arange(len(energy))*.1,np.maximum(abs(np.array(energy)-.5),1e-16),label=method)
    axes[1].legend(); axes[1].set_title('Harmonic oscillator energy error'); axes[1].set_xlabel('Time')
    fig.tight_layout(); fig.savefig(out/'artifacts.png',dpi=150); plt.close(fig)
    # Periodic known equation: q'=p, p'=-q, exact frequency1 and no positive exponent.
    # Fit-window sensitivity is reported, never classified as chaos.
    sweep=[]
    for n in (256,512):
        for dt in (.05,.2):
            for noise in (0.,.02):
                data=sampled_observations(lambda t,y:np.array([y[1],-y[0]]),[1,0],samples=n,
                    sample_interval=dt,burn_in=0,noise_std=noise,seed=20261013,
                    observation=lambda y:y[0],solver_parameters=dict(method='rkf45_adaptive',step=.05,atol=1e-10,rtol=1e-10))
                for dimension in (2,3):
                    curve=kantz_curve(data['observed'],dimension=dimension,delay=2,epsilon=.15,horizon=8,theiler=20)
                    # Existing API returns horizon, log curve and eligible counts.
                    horizons,values,counts=curve
                    fits=[]
                    for start,stop in ((0,3),(1,6)):
                        eligible=np.isfinite(values[start:stop])
                        slope=float(np.polyfit(horizons[start:stop][eligible]*dt,values[start:stop][eligible],1)[0]) if eligible.sum()>=2 else None
                        fits.append(dict(slice=[start,stop],slope_per_time=slope))
                    sweep.append(dict(samples=n,interval=dt,noise=noise,dimension=dimension,delay=2,theiler=20,epsilon=.15,
                        horizons=horizons.tolist(),curve=values.tolist(),counts=int(counts),fits=fits,
                        clean_max_error=float(np.max(abs(data['clean']-np.cos(data['times'])))),evaluations=data['evaluations']))
    def clean(x):
        if isinstance(x,dict): return {k:clean(v) for k,v in x.items()}
        if isinstance(x,list): return [clean(v) for v in x]
        if isinstance(x,float) and not np.isfinite(x): return None
        return x
    report=dict(python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__,seed=20261013,
        solver_checks=rows,ghost=dict(step=.05,initial=[.5,.5125],status=status,max_error=float(np.max(abs(ghost[:,0]-1/(1+np.exp(-t)))))),
        energy=energies,sampled_periodic_sweeps=sweep)
    (out/'report.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    np.savetxt(out/'solver-errors.csv',[[r['step'],r['max_error'],r['evaluations']] for r in rows],delimiter=',',header='step,max_error,evaluations',comments='')
    print('solver checks',rows); print('energy',energies); print('Saved',out)


if __name__=='__main__': main()
