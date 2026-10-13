"""Reproducible Chapter 5 systems, continuous spectra and scalar-observation checks."""
import argparse
import csv
import json
import platform
from pathlib import Path
from dataclasses import asdict,is_dataclass
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nonlinear_workbook import chapter05 as c
from nonlinear_workbook.chapter10 import integrate
from nonlinear_workbook.validation import flow_lyapunov_spectrum,exterior_area_growth,kantz_sampling_sweep


SEED=20261013
SOLVER=dict(method='rkf45_adaptive',step=.02,max_step=.02,atol=1e-9,rtol=1e-9,escape_radius=1e6)


def clean(x):
    if is_dataclass(x): return clean(asdict(x))
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,np.ndarray): return clean(x.tolist())
    if isinstance(x,(tuple,list)): return [clean(v) for v in x]
    if isinstance(x,np.generic): return clean(x.item())
    if isinstance(x,complex): return dict(real=x.real,imag=x.imag)
    if isinstance(x,float) and not np.isfinite(x): return None
    return x


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',default='outputs/chapter05')
    out=Path(parser.parse_args().output); out.mkdir(parents=True,exist_ok=True)
    report=dict(seed=SEED,python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__,
        units='natural logarithms per dimensionless physical time',solver=SOLVER)
    def savefig(fig,name):
        fig.tight_layout(); fig.savefig(out/(name+'.png'),dpi=140,metadata={'Software':'nonlinear-workbook'}); plt.close(fig)
    def dump_csv(name,header,rows):
        with (out/(name+'.csv')).open('w',newline='',encoding='utf-8') as f:
            writer=csv.writer(f); writer.writerow(header); writer.writerows(rows)
    systems={
        'lorenz':(c.lorenz,c.lorenz_jacobian,dict(sigma=16,r=40,b=4),[.8,.8,.8]),
        'lotka':(c.generalized_lotka_volterra,c.generalized_lotka_volterra_jacobian,dict(a=2.9851,b=3,c=7),[1,1.5,2.5]),
        'hyperchaotic':(c.hyperchaotic,c.hyperchaotic_jacobian,dict(a=.25,b=3,c=.5,d=.05),[-19,0,0,15])}
    report['systems']={}
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for column,(name,(function,jacobian,parameters,initial)) in enumerate(systems.items()):
        rhs=lambda t,y,fun=function,p=parameters:fun(t,y,**p)
        jac=lambda t,y,fun=jacobian,p=parameters:fun(t,y,**p)
        trajectory=integrate(rhs,initial,t_span=(0,70),**SOLVER)
        dump_csv(name+'_trajectory',['time']+['state'+str(i) for i in range(len(initial))],np.column_stack((trajectory.times,trajectory.states)))
        mask=trajectory.times>=20
        axes[0,column].plot(trajectory.times[mask],trajectory.states[mask,0]); axes[0,column].set_xlabel('time'); axes[0,column].set_ylabel('state0')
        axes[1,column].plot(trajectory.states[mask,0],trajectory.states[mask,1],linewidth=.5); axes[1,column].set_xlabel('state0'); axes[1,column].set_ylabel('state1')
        axes[0,column].set_title(name)
        runs=[]
        for duration in (100,200):
            spectrum=flow_lyapunov_spectrum(rhs,jac,initial,burn_in=20,duration=duration,qr_interval=.1,initial_basis=np.eye(len(initial)),solver_parameters=SOLVER)
            runs.append(dict(duration=duration,qr_interval=.1,solver=SOLVER,result=spectrum))
            print(name,duration,spectrum.status,spectrum.exponents,'sum-trace',spectrum.exponents.sum()-spectrum.trace_average,flush=True)
            dump_csv(name+'_spectrum_'+str(duration),['time']+['lambda'+str(i) for i in range(len(initial))],np.column_stack((spectrum.times,spectrum.running_estimates)))
        report['systems'][name]=dict(parameters=parameters,initial=initial,trajectory_status=trajectory.status,trajectory_evaluations=trajectory.evaluations,
            minimum_coordinate=np.min(trajectory.states,axis=0),spectra=runs)
    savefig(fig,'trajectories')

    function,jacobian,parameters,initial=systems['lorenz']; rhs=lambda t,y:function(t,y,**parameters); jac=lambda t,y:jacobian(t,y,**parameters)
    sensitivity=[]
    for interval,tolerance in ((.05,1e-9),(.2,1e-9),(.1,1e-10)):
        solver=dict(SOLVER,atol=tolerance,rtol=tolerance)
        result=flow_lyapunov_spectrum(rhs,jac,initial,burn_in=20,duration=100,qr_interval=interval,initial_basis=np.eye(3),solver_parameters=solver)
        sensitivity.append(dict(qr_interval=interval,duration=100,solver=solver,result=result))
        print('Lorenz sensitivity',interval,tolerance,result.exponents,flush=True)
    report['lorenz_sensitivity']=sensitivity
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,(name,system) in zip(axes,report['systems'].items()):
        for run in system['spectra']:
            result=run['result']
            for i in range(result.running_estimates.shape[1]):
                ax.plot(result.times-20,result.running_estimates[:,i],label=f"column{i}, T={run['duration']}",alpha=.7)
        ax.set_title(name); ax.set_xlabel('measurement time'); ax.set_ylabel('running exponent'); ax.legend(fontsize=7)
    savefig(fig,'spectra')

    fun,jfun,p,initial=systems['hyperchaotic']; rhs=lambda t,y:fun(t,y,**p); jac=lambda t,y:jfun(t,y,**p)
    areas=[]; fig,ax=plt.subplots()
    for duration in (100,200):
        result=exterior_area_growth(rhs,jac,initial,burn_in=20,duration=duration,normalization_interval=.1,
            initial_vectors=[[1,2,3,4],[4,-1,2,0]],solver_parameters=SOLVER)
        spectrum=report['systems']['hyperchaotic']['spectra'][(duration==200)]['result']
        target=float(np.sort(spectrum.exponents)[-2:].sum())
        areas.append(dict(duration=duration,normalization_interval=.1,initial_vectors=[[1,2,3,4],[4,-1,2,0]],result=result,qr_largest_two_sum=target,difference=result.rate-target))
        ax.plot(result.times-20,result.running_estimates,label=f'exterior T={duration}')
        print('area',duration,result.status,result.rate,'QR pair sum',target,flush=True)
    ax.set_xlabel('measurement time'); ax.set_ylabel('area growth / time'); ax.legend(); savefig(fig,'exterior_area'); report['exterior_area']=areas

    threshold=c.lorenz_hopf_threshold(sigma=16,b=4)
    grid=np.unique(np.concatenate((np.linspace(.5,1.5,21),np.linspace(threshold-2,threshold+2,41))))
    fixed=[]
    for r in grid:
        for index,state in enumerate(c.lorenz_equilibria(sigma=16,r=r,b=4)):
            fixed.append(dict(r=r,branch=index,state=state,classification=c.classify_equilibrium(c.lorenz_jacobian(0,state,sigma=16,r=r,b=4))))
    crossing=c.hopf_candidate(lambda r,y:c.lorenz_jacobian(0,y,sigma=16,r=r,b=4),lambda r:c.lorenz_equilibria(sigma=16,r=r,b=4)[1],bracket=(threshold-1,threshold+1),tolerance=1e-8)
    report['lorenz_equilibria']=dict(scan=fixed,analytic_threshold=threshold,candidate=crossing)

    scan=[]; candidates=[]; ws=np.geomspace(.001,10,81); branch=lambda w:c.chemical_equilibria(s=10,q=.01,f=1,w=w)[0][c.chemical_equilibria(s=10,q=.01,f=1,w=w)[1]][0]
    for w in ws:
        state=branch(w); classification=c.classify_equilibrium(c.belousov_zhabotinsky_jacobian(0,state,s=10,q=.01,f=1,w=w))
        pair=classification.eigenvalues[classification.eigenvalues.imag>1e-8]
        scan.append(dict(w=w,state=state,classification=classification,pair_real=float(pair[0].real) if len(pair)==1 else None))
    for left,right in zip(scan[:-1],scan[1:]):
        if left['pair_real'] is not None and right['pair_real'] is not None and left['pair_real']*right['pair_real']<0:
            candidates.append(c.hopf_candidate(lambda w,y:c.belousov_zhabotinsky_jacobian(0,y,s=10,q=.01,f=1,w=w),branch,bracket=(left['w'],right['w']),tolerance=1e-8))
    report['chemical']=dict(parameters=dict(s=10,q=.01,f=1),scan=scan,candidates=candidates,status='candidates_found' if candidates else 'no_candidate_found')
    dump_csv('chemical_scan',['w','complex_pair_real','largest_real'],[[r['w'],r['pair_real'],max(r['classification'].eigenvalues.real)] for r in scan])
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for branch_index in (0,1):
        records=[r for r in fixed if r['branch']==branch_index]
        axes[0].plot([r['r'] for r in records],[max(r['classification'].eigenvalues.real) for r in records],'.',label='origin' if branch_index==0 else 'nonzero')
    axes[0].axhline(0,color='black'); axes[0].axvline(threshold,linestyle='--',color='gray'); axes[0].set_xlabel('Lorenz r'); axes[0].set_ylabel('largest real eigenvalue'); axes[0].legend()
    axes[1].semilogx(ws,[max(r['classification'].eigenvalues.real) for r in scan]); axes[1].axhline(0,color='black'); axes[1].set_xlabel('chemical w'); axes[1].set_ylabel('largest real eigenvalue'); savefig(fig,'equilibria')

    integral_parameters=dict(sigma=10,r=0,b=1)
    r=integrate(lambda t,y:c.lorenz(t,y,**integral_parameters),[1,1,1],t_span=(0,5),**dict(SOLVER,atol=1e-11,rtol=1e-11))
    values=np.array([c.lorenz_time_integral(t,y,**integral_parameters) for t,y in zip(r.times,r.states)])
    report['integral']=dict(parameters=integral_parameters,initial=[1,1,1],horizon=5,solver=dict(SOLVER,atol=1e-11,rtol=1e-11),status=r.status,max_relative_drift=float(np.max(abs(values/values[0]-1))))
    dump_csv('time_integral',['time','integral','relative_drift'],np.column_stack((r.times,values,values/values[0]-1)))
    fig,ax=plt.subplots(); ax.plot(r.times,values/values[0]-1); ax.set_xlabel('time'); ax.set_ylabel('relative integral drift'); savefig(fig,'time_integral')

    sweeps={}; fitrows=[]
    sweep_controls=dict(sample_counts=(256,512),sample_intervals=(.02,.1),noise_levels=(0,.02),dimensions=(2,3),delay=2,theiler=20,horizon=8,fit_slices=((0,3),(1,6)),seed=SEED,observation=lambda y:y[0],epsilon=lambda clean:.1*np.std(clean),burn_in=20,solver_parameters=SOLVER)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,name in zip(axes,('lorenz','lotka')):
        fun,jfun,p,initial=systems[name]
        records=kantz_sampling_sweep(lambda t,y:fun(t,y,**p),initial,**sweep_controls); sweeps[name]=records
        for record in records:
            if record['curve'] is not None and record['samples']==512 and record['dimension']==2:
                ax.plot(record['lags']*record['sample_interval'],record['curve'],label=f"dt={record['sample_interval']}, noise={record['noise_std']}")
            for fit in record['fits']: fitrows.append([name,record['samples'],record['sample_interval'],record['noise_std'],record['dimension'],record['epsilon'],*fit['slice'],fit['slope_per_time'],record['reference_count']])
        ax.set_title(name+' N512, dimension2'); ax.set_xlabel('physical lag'); ax.set_ylabel('mean log separation'); ax.legend(fontsize=8)
    savefig(fig,'observation_sweeps')
    report['sampling']=dict(radius_rule='.1*clean.std(ddof=0)',burn_in=20,systems=sweeps)
    dump_csv('fits',['system','samples','sample_interval','noise','dimension','epsilon','start','stop','slope_per_time','reference_count'],fitrows)
    (out/'report.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('chemical',report['chemical']['status'],'integral drift',report['integral']['max_relative_drift'],flush=True)
    print('Saved',out,flush=True)


if __name__=='__main__': main()
