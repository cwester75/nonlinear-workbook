"""Headless Chapter 6 models, exact-time sampling and signal diagnostics."""
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
from nonlinear_workbook import chapter06 as c
from nonlinear_workbook.chapter10 import integrate
from nonlinear_workbook.validation import flow_lyapunov_spectrum,sampled_observations,kantz_sampling_sweep


SEED=20261014
SOLVER=dict(method='rkf45_adaptive',step=.02,max_step=.1,atol=1e-9,rtol=1e-9,escape_radius=1e6)
FIXTURES={
    'anharmonic':(dict(a=1,b=-10,c=100,k1=0,k2=1.2,omega=3.5),[1,0]),
    'driven_pendulum':(dict(a=.2,b=1,k=1.1,omega=.8),[.8,.6]),
    'parametric_pendulum':(dict(a=.15,k=.94,omega=1.56),[.1,.2]),
    'driven_van_der_pol':(dict(a=5,k=5,omega=2.466),[.8,.6]),
    'two_drive_pendulum':(dict(a=.1,k1=.3,k2=.3,omega1=.618034,omega2=1.618034),[1,0])}


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
    parser=argparse.ArgumentParser(); parser.add_argument('--output',default='outputs/chapter06')
    out=Path(parser.parse_args().output); out.mkdir(parents=True,exist_ok=True)
    report=dict(seed=SEED,python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__,
        conventions='absolute time; unwrapped integrated angles; zero initial forcing phases; natural logs/time',solver=SOLVER,models={})
    def savefig(fig,name):
        fig.tight_layout(); fig.savefig(out/(name+'.png'),dpi=140,metadata={'Software':'nonlinear-workbook'}); plt.close(fig)
    def csvfile(name,header,rows):
        with (out/(name+'.csv')).open('w',newline='',encoding='utf-8') as f:
            writer=csv.writer(f); writer.writerow(header); writer.writerows(rows)
    fig,axes=plt.subplots(2,5,figsize=(17,7))
    for column,(name,(p,initial)) in enumerate(FIXTURES.items()):
        rhs=lambda t,y,fun=getattr(c,name),p=p:fun(t,y,**p)
        double=name=='two_drive_pendulum'; frequencies=[p['omega1'],p['omega2']] if double else [p['omega']]
        period=2*np.pi/frequencies[0]
        # One continuous integration supplies both burn-in views, no repeated work.
        sample=c.stroboscopic_sample(rhs,initial,initial_time=0,period=period,burn_in_periods=100,
            samples=512 if double else 612,drive_frequencies=frequencies,drive_phases=np.zeros(len(frequencies)),solver_parameters=SOLVER)
        if sample.status!='completed': raise RuntimeError(name+': '+sample.status)
        first=sample.states[:512].copy(); displayed=first.copy()
        if 'pendulum' in name: displayed[:,0]=c.wrap_angle(displayed[:,0])
        axes[1,column].scatter(displayed[:,0],displayed[:,1],s=2,c=sample.phases[:512,-1] if double else None)
        axes[1,column].set_xlabel('q wrapped' if 'pendulum' in name else 'x'); axes[1,column].set_ylabel('velocity')
        trajectory=integrate(rhs,sample.states[0],t_span=(sample.times[0],sample.times[0]+100),**SOLVER)
        if trajectory.status!='completed': raise RuntimeError('trajectory incomplete')
        axes[0,column].plot(trajectory.states[:,0],trajectory.states[:,1],linewidth=.5); axes[0,column].set_title(name.replace('_',' ')); axes[0,column].set_xlabel('unwrapped coordinate'); axes[0,column].set_ylabel('velocity')
        csvfile(name+'_section',['time','coordinate','velocity']+['phase'+str(i) for i in range(len(frequencies))],np.column_stack((sample.times,sample.states,sample.phases)))
        csvfile(name+'_trajectory',['time','coordinate','velocity'],np.column_stack((trajectory.times,trajectory.states)))
        model=dict(parameters=p,initial=initial,period=period,section=sample,burn100=first,
            section_kind='phase-aware driven sample; projected map is not closed' if double else 'single-drive stroboscopic map',trajectory_status=trajectory.status,trajectory_horizon=100)
        if not double:
            second=sample.states[100:612]
            model.update(burn200=second,burn_view_convention='burn100 and burn200 are overlapping windows of one trajectory',
                section_mean_difference=np.mean(second,axis=0)-np.mean(first,axis=0))
        report['models'][name]=model
        print('sections',name,sample.status,sample.evaluations,flush=True)
    savefig(fig,'phase_sections')

    report['spectra']={}; spectrum_solver=dict(SOLVER,max_step=.02)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,name in zip(axes[:2],('anharmonic','driven_van_der_pol')):
        p,initial=FIXTURES[name]; function=getattr(c,name); jacobian=getattr(c,name+'_jacobian')
        rhs=lambda t,y:function(t,y,**p); jac=lambda t,y:jacobian(t,y,**p)
        runs=[]
        for duration,tolerance in ((100,1e-9),(200,1e-9),(100,1e-10)):
            solver=dict(spectrum_solver,atol=tolerance,rtol=tolerance)
            result=flow_lyapunov_spectrum(rhs,jac,initial,burn_in=100,duration=duration,qr_interval=.1,initial_basis=np.eye(2),solver_parameters=solver)
            runs.append(dict(duration=duration,tolerance=tolerance,solver=solver,result=result))
            for i in range(2): ax.plot(result.times-100,result.running_estimates[:,i],label=f'T{duration},tol{tolerance},col{i}',alpha=.7)
            print('spectrum',name,duration,tolerance,result.status,result.exponents,'sum-trace',result.exponents.sum()-result.trace_average,flush=True)
        report['spectra'][name]=runs; ax.set_title(name); ax.legend(fontsize=6); ax.set_xlabel('measurement time')
    p,initial=FIXTURES['anharmonic']; lift=c.phase_lift('anharmonic',**p)
    result=flow_lyapunov_spectrum(lift.rhs,lift.jacobian,lift.initial_state(initial,time=0),burn_in=100,duration=100,qr_interval=.1,initial_basis=np.eye(3),solver_parameters=spectrum_solver)
    report['lifted_spectrum']=dict(model='anharmonic',duration=100,solver=spectrum_solver,result=result)
    for i in range(3): axes[2].plot(result.times-100,result.running_estimates[:,i],label='column'+str(i))
    axes[2].set_title('anharmonic phase lift'); axes[2].set_xlabel('measurement time'); axes[2].legend(); savefig(fig,'spectra')
    print('lifted',result.status,result.exponents,flush=True)

    report['signals']={}; fig,axes=plt.subplots(2,2,figsize=(11,7))
    for column,name in enumerate(('anharmonic','driven_pendulum')):
        p,initial=FIXTURES[name]; rhs=lambda t,y,fun=getattr(c,name),p=p:fun(t,y,**p)
        signals=[]
        for dt in (.02,.05):
            data=sampled_observations(rhs,initial,samples=8192,sample_interval=dt,burn_in=100,noise_std=0,seed=SEED,observation=lambda y:y[0],solver_parameters=SOLVER)
            data['forcing_phase']=(p['omega']*data['times'])%(2*np.pi)
            lags,correlation,counts=c.normalized_autocorrelation(data['observed'],sample_interval=dt,max_lag=1000)
            spectra={}
            for window in ('rectangular','hann'):
                f,density=c.periodogram(data['observed'],sample_interval=dt,demean=True,window=window)
                spectra[window]=dict(frequency=f,density=density)
                axes[1,column].semilogy(f,density,label=f'dt{dt}, {window}',alpha=.7)
            axes[0,column].plot(lags,correlation,label=f'dt{dt}')
            signals.append(dict(data=data,lags=lags,autocorrelation=correlation,pair_counts=counts,periodograms=spectra))
            csvfile(name+'_signal_'+str(dt),['time','observation','forcing_phase'],np.column_stack((data['times'],data['observed'],data['forcing_phase'])))
        report['signals'][name]=signals
        axes[0,column].set_title(name); axes[0,column].set_xlabel('physical lag'); axes[0,column].set_ylabel('normalized autocorrelation'); axes[0,column].legend()
        axes[1,column].set_xlim(0,5); axes[1,column].set_xlabel('cycles/time'); axes[1,column].set_ylabel('density'); axes[1,column].legend(fontsize=7)
    savefig(fig,'signals')
    # Off-bin deterministic sinusoid isolates leakage from chaotic signal shape.
    n=1024; dt=.02; time=np.arange(n)*dt; sinusoid=np.sin(2*np.pi*.713*time)
    fig,ax=plt.subplots(); leakage={}
    for window in ('rectangular','hann'):
        f,density=c.periodogram(sinusoid,sample_interval=dt,demean=True,window=window)
        leakage[window]=dict(frequency=f,density=density); ax.semilogy(f,density,label=window)
    ax.set_xlim(0,3); ax.set_xlabel('cycles/time'); ax.set_ylabel('density'); ax.legend(); savefig(fig,'leakage'); report['leakage']=dict(frequency=.713,sample_interval=dt,samples=n,periodograms=leakage)

    report['sampling']={}; fig,axes=plt.subplots(1,2,figsize=(11,4)); rows=[]
    for ax,name in zip(axes,('anharmonic','driven_pendulum')):
        p,initial=FIXTURES[name]; rhs=lambda t,y,fun=getattr(c,name),p=p:fun(t,y,**p)
        records=kantz_sampling_sweep(rhs,initial,sample_counts=(256,512),sample_intervals=(.02,.1),noise_levels=(0,.02),dimensions=(2,3),delay=2,theiler=20,epsilon=lambda x:.1*x.std(),horizon=8,fit_slices=((0,3),(1,6)),burn_in=100,seed=SEED,observation=lambda y:y[0],solver_parameters=SOLVER)
        for r in records:
            r['forcing_phase']=(p['omega']*r['times'])%(2*np.pi)
            if r['curve'] is not None and r['samples']==512 and r['dimension']==2:
                ax.plot(r['lags']*r['sample_interval'],r['curve'],label=f"dt{r['sample_interval']},noise{r['noise_std']}")
            for fit in r['fits']: rows.append([name,r['samples'],r['sample_interval'],r['noise_std'],r['dimension'],r['epsilon'],*fit['slice'],fit['slope_per_time'],r['reference_count']])
        report['sampling'][name]=records; ax.set_title(name+' N512,m2'); ax.set_xlabel('physical lag'); ax.set_ylabel('mean log separation'); ax.legend(fontsize=7)
    savefig(fig,'sampling'); csvfile('fits',['system','samples','interval','noise','dimension','radius','fit_start','fit_stop','slope_per_time','references'],rows)

    # Runnable analytical forced-linear periodic-orbit demonstration.
    p=dict(a=2,b=2,c=0,k1=0,k2=1,omega=3)
    shot=c.shoot_periodic(lambda t,y:c.anharmonic(t,y,**p),lambda t,y:c.anharmonic_jacobian(t,y,**p),[1,1],initial_time=.4,period=2*np.pi/3,period_multiple=1,tolerance=1e-9,max_iterations=5,escape_radius=100,solver_parameters=dict(SOLVER,atol=1e-11,rtol=1e-11,max_step=.02))
    report['shooting']=dict(parameters=p,initial_time=.4,period_multiple=1,result=shot,classification=c.classify_multipliers(shot.monodromy))
    (out/'report.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('shooting',shot.status,shot.residual_history,flush=True); print('Saved',out,flush=True)


if __name__=='__main__': main()
