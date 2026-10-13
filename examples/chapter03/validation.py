"""Chapter 3 phase portraits, invariants and data-only diagnostic controls."""
import argparse
import json
import platform
from pathlib import Path
from dataclasses import asdict,is_dataclass
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nonlinear_workbook import chapter03 as c
from nonlinear_workbook.chapter10 import integrate,symplectic_step
from nonlinear_workbook.validation import kantz_sampling_sweep


SEED=20261013
SOLVER=dict(method='rkf45_adaptive',step=.05,atol=1e-10,rtol=1e-10,max_step=.05)


def clean(value):
    if is_dataclass(value): return clean(asdict(value))
    if isinstance(value,dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,np.ndarray): return clean(value.tolist())
    if isinstance(value,(tuple,list)): return [clean(v) for v in value]
    if isinstance(value,np.generic): return clean(value.item())
    if isinstance(value,complex): return dict(real=value.real,imag=value.imag)
    if isinstance(value,float) and not np.isfinite(value): return None
    return value


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',default='outputs/chapter03')
    out=Path(parser.parse_args().output); out.mkdir(parents=True,exist_ok=True)
    report=dict(seed=SEED,python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__,solver=SOLVER,units='dimensionless time, pendulum radians',systems={},sampling_controls={})
    def savefig(fig,name):
        fig.tight_layout(); fig.savefig(out/(name+'.png'),dpi=140,metadata={'Software':'nonlinear-workbook'}); plt.close(fig)
    def trajectory(name,rhs,initial,end):
        result=integrate(rhs,initial,t_span=(0,end),**SOLVER)
        if result.status!='completed': raise RuntimeError(name+': '+result.status)
        np.savetxt(out/(name+'.csv'),np.column_stack((result.times,result.states)),delimiter=',',header='time,coordinate0,coordinate1',comments='')
        return result

    matrices=dict(stable_node=[[-1,0],[0,-2]],unstable_node=[[1,0],[0,2]],saddle=[[1,0],[0,-1]],spiral=[[-.2,-1],[1,-.2]],center=[[0,-1],[1,0]])
    fig,axes=plt.subplots(2,3,figsize=(11,7))
    classifications={}
    for ax,(name,matrix) in zip(axes.flat,matrices.items()):
        m=np.array(matrix,float); classifications[name]=c.classify_planar(m)
        for initial in ([.8,.3],[-.8,.3],[.2,-.8]):
            r=integrate(lambda t,y:m@y,initial,t_span=(0,3),**SOLVER,escape_radius=20)
            ax.plot(r.states[:,0],r.states[:,1])
        ax.set_title(name)
    ax=axes.flat[-1]
    for initial in ([.2,.3],[1.4,.4],[-.4,-.1]):
        r=trajectory('radial_'+str(initial[0]),c.radial_equilibrium,initial,10)
        ax.plot(r.states[:,0],r.states[:,1])
    theta=np.linspace(0,2*np.pi,200); ax.plot(np.cos(theta),np.sin(theta),'k--'); ax.set_title('Equilibrium circle')
    for ax in axes.flat: ax.set_aspect('equal',adjustable='datalim'); ax.set_xlabel('coordinate0'); ax.set_ylabel('coordinate1')
    savefig(fig,'fixed_points'); report['classification']=classifications

    initial=c.homoclinic_reference(-10)
    r=trajectory('homoclinic',c.homoclinic,initial,20)
    reference=c.homoclinic_reference(r.times-10)
    error=float(np.max(abs(r.states-reference)))
    fig,ax=plt.subplots(); ax.plot(r.states[:,0],r.states[:,1],label='numerical'); ax.plot(reference[:,0],reference[:,1],'--',label='analytic'); ax.legend(); ax.set_xlabel('x'); ax.set_ylabel('v'); savefig(fig,'homoclinic')
    report['systems']['homoclinic']=dict(initial=initial,max_error=error,energy_drift=max(abs(c.homoclinic_energy(y)) for y in r.states))

    fig,axes=plt.subplots(1,2,figsize=(10,4)); pendulum_results=[]
    for name,initial in (('libration',[1.,0.]),('rotation',[0.,2.5]),('near_separatrix',[0.,1.999])):
        r=trajectory('pendulum_'+name,c.pendulum,initial,40)
        axes[0].plot(r.states[:,0],r.states[:,1],label=name)
        crossings,_,periods=c.section_crossings(r.times,r.states,coordinate=0,level=0,direction=1)
        pendulum_results.append(dict(name=name,initial=initial,energy_drift=max(abs(c.pendulum_energy(y)-c.pendulum_energy(initial)) for y in r.states),crossing_times=crossings,periods=periods))
    energy_results=[]
    h=.05; steps=4000; initial=[1.,0.]
    for method in ('rkf45','symplectic_euler','verlet'):
        if method=='rkf45':
            fixed=integrate(c.pendulum,initial,t_span=(0,steps*h),method=method,step=h)
            times,states=fixed.times,fixed.states
        else:
            q,p=np.array([1.]),np.array([0.]); values=[initial]
            for _ in range(steps):
                q,p=symplectic_step(q,p,h,grad_potential=lambda q:np.sin(q),grad_kinetic=lambda p:p,method='euler' if method=='symplectic_euler' else method)
                values.append([q[0],p[0]])
            times=np.arange(steps+1)*h; states=np.array(values)
        drift=np.array([c.pendulum_energy(y)-c.pendulum_energy(initial) for y in states])
        axes[1].plot(times,drift,label=method); energy_results.append(dict(method=method,step=h,horizon=steps*h,max_drift=float(np.max(abs(drift)))))
        np.savetxt(out/('pendulum_energy_'+method+'.csv'),np.column_stack((times,drift)),delimiter=',',header='time,energy_drift',comments='')
    axes[0].set_xlabel('unwrapped q'); axes[0].set_ylabel('p'); axes[1].set_xlabel('time'); axes[1].set_ylabel('energy drift')
    for ax in axes: ax.legend()
    savefig(fig,'pendulum'); report['systems']['pendulum']=dict(trajectories=pendulum_results,energy_comparison=energy_results)

    fig,axes=plt.subplots(1,3,figsize=(12,4)); cycle_results=[]
    for ax,mu in zip(axes[:2],(1.,5.)):
        for initial in ([.1,.2],[3.,0.]):
            r=trajectory('vdp_'+str(mu)+'_'+str(initial[0]),lambda t,y:c.van_der_pol(t,y,mu=mu),initial,120)
            settled=r.times>=60
            times,states,periods=c.section_crossings(r.times[settled],r.states[settled],coordinate=0,level=0,direction=1)
            ax.plot(r.states[settled,0],r.states[settled,1],label=str(initial))
            cycle_results.append(dict(mu=mu,initial=initial,burn_in=60,crossing_times=times,crossing_states=states,periods=periods,recurrence_distances=np.linalg.norm(np.diff(states,axis=0),axis=1)))
        ax.set_title('Van der Pol mu='+str(mu)); ax.legend()
    sine_results=[]
    for initial in ([.05,.01],[8.,0.]):
        r=trajectory('sine_'+str(initial[0]),lambda t,y:c.sine_damping(t,y,mu=5),initial,120)
        axes[2].plot(r.states[:,0],r.states[:,1],label=str(initial))
        mask=r.times>=60
        times,states,periods=c.section_crossings(r.times[mask],r.states[mask],coordinate=0,level=0,direction=1)
        sine_results.append(dict(initial=initial,mu=5,crossing_times=times,crossing_states=states,periods=periods,final=r.states[-1]))
    axes[2].set_title('Printed sine damping'); axes[2].legend()
    for ax in axes: ax.set_xlabel('x'); ax.set_ylabel('v')
    savefig(fig,'limit_cycles')
    # Preserve listing result separately, without describing it as a sine-damping orbit.
    state=np.array([.05,.01]); listing=[state.copy()]
    for _ in range(10000):
        state=c.van_der_pol_listing_step(state,step=.005,mu=5); listing.append(state.copy())
    np.savetxt(out/'vdp_listing.csv',np.column_stack((np.arange(10001)*.005,listing)),delimiter=',',header='time,x,v',comments='')
    report['systems']['limit_cycles']=dict(van_der_pol=cycle_results,sine_damping=sine_results,listing=dict(initial=[.05,.01],step=.005,mu=5,horizon=50,final=state))

    rhs=lambda t,y:c.lotka_volterra(t,y,a=1,b=1,c=1)
    r=trajectory('lotka_volterra',rhs,[2,1.5],40)
    invariant=np.array([c.lotka_volterra_invariant(y,a=1,b=1,c=1) for y in r.states]); drift=invariant-invariant[0]
    fig,axes=plt.subplots(1,2,figsize=(9,4)); axes[0].plot(r.states[:,0],r.states[:,1]); axes[0].set_xlabel('prey'); axes[0].set_ylabel('predator'); axes[1].plot(r.times,drift); axes[1].set_xlabel('time'); axes[1].set_ylabel('invariant drift'); savefig(fig,'lotka_volterra')
    report['systems']['lotka_volterra']=dict(a=1,b=1,c=1,initial=[2,1.5],horizon=40,min_population=float(r.states.min()),max_invariant_drift=float(np.max(abs(drift))))

    sweep_parameters=dict(sample_counts=(256,512),sample_intervals=(.05,.2),noise_levels=(0.,.02),dimensions=(2,3),delay=2,theiler=20,epsilon=.15,horizon=8,fit_slices=((0,3),(1,6)),seed=SEED,observation=lambda y:y[0],solver_parameters=SOLVER)
    sweeps={}
    for name,rhs,initial,burn in (('pendulum',c.pendulum,[1,0],0),('lotka_volterra',lambda t,y:c.lotka_volterra(t,y,a=1,b=1,c=1),[2,1.5],0),('van_der_pol',lambda t,y:c.van_der_pol(t,y,mu=1),[.1,.2],60)):
        sweeps[name]=kantz_sampling_sweep(rhs,initial,burn_in=burn,**sweep_parameters)
    report['sampling_controls']=dict(parameters={k:v for k,v in sweep_parameters.items() if k not in ('observation','solver_parameters')},initial_states={'pendulum':[1,0],'lotka_volterra':[2,1.5],'van_der_pol':[.1,.2]},burn_in={'pendulum':0,'lotka_volterra':0,'van_der_pol':60},systems=sweeps)
    fig,axes=plt.subplots(1,3,figsize=(12,4)); fitrows=[]
    for ax,(name,records) in zip(axes,sweeps.items()):
        for record in records:
            if record['curve'] is not None and record['samples']==512 and record['dimension']==2:
                ax.plot(record['lags']*record['sample_interval'],record['curve'],label=f"dt={record['sample_interval']}, noise={record['noise_std']}")
            for fit in record['fits']: fitrows.append([name,record['samples'] if 'samples' in record else len(record['observed']),record['sample_interval'],record['noise_std'],record['dimension'],*fit['slice'],fit['slope_per_time'],record['reference_count']])
        ax.set_title(name+' (N=512, m=2)'); ax.set_xlabel('physical lag'); ax.set_ylabel('mean log separation'); ax.legend(fontsize=8)
    savefig(fig,'kantz_controls')
    import csv
    with (out/'fits.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f); writer.writerow(['system','samples','sample_interval','noise_std','dimension','fit_start','fit_stop','slope_per_time','reference_count']); writer.writerows(fitrows)
    (out/'report.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('homoclinic error',error); print('pendulum energy',energy_results); print('Lotka invariant drift',float(np.max(abs(drift)))); print('Saved',out)


if __name__=='__main__': main()
