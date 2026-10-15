"""Headless Chapter 7 controls, synchronization and conditional spectra."""
import argparse
import csv
import json
import platform
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nonlinear_workbook import chapter07 as c
from nonlinear_workbook.chapter03 import van_der_pol,van_der_pol_jacobian
from nonlinear_workbook.chapter05 import lorenz,lorenz_jacobian
from nonlinear_workbook.chapter06 import anharmonic,stroboscopic_sample
from nonlinear_workbook.chapter10 import integrate
from nonlinear_workbook.chapter01.maps import coupled_logistic_step,logistic_step
from nonlinear_workbook.flow_validation import conditional_flow_spectrum,flow_lyapunov_spectrum

SEED=20261015


def solver(tol):
    return dict(method='rkf45_adaptive',step=.01,max_step=.02,atol=tol,rtol=tol,max_steps=1000000,escape_radius=1e6)


def write_csv(path,header,rows):
    with path.open('w',newline='') as f:
        w=csv.writer(f); w.writerow(header); w.writerows(rows)


def figure(path):
    plt.tight_layout(); plt.savefig(path,dpi=140,metadata={'Software':'nonlinear-workbook'}); plt.close()


def capture(errors,tolerance=1e-6):
    """Finite-run entries/exits and final 100-sample persistence; no asymptotic claim."""
    inside=np.asarray(errors)<tolerance
    return dict(tolerance=tolerance,first_entry=int(np.flatnonzero(inside)[0]) if inside.any() else None,
        entries=int(np.sum(inside & ~np.r_[False,inside[:-1]])),losses=int(np.sum(~inside & np.r_[False,inside[:-1]])),
        tail_max_error=float(np.max(errors[-100:])),last_100_within=bool(len(errors)>=100 and inside[-100:].all()))


def control_output(out,name,result,truth_error):
    n=len(result.controls); d=result.states.shape[1]
    rows=[[*result.states[i],*result.measured_states[i],result.parameters[i],result.proposed_controls[i],result.controls[i],result.phases[i],result.errors[i],truth_error[i],int(result.active[i]),result.reasons[i]] for i in range(n)]
    write_csv(out/f'{name}.csv',[*[f'true{j}' for j in range(d)],*[f'measured{j}' for j in range(d)],'parameter','proposal','applied','phase','measured_error','true_error','active','reason'],rows)
    return dict(status=result.status,updates=n,final_state=result.states[-1].tolist(),activations=int(result.active.sum()),
        max_applied=float(np.max(abs(result.controls),initial=0)),reasons=dict(Counter(result.reasons)),capture=capture(truth_error))


def spectrum_output(out,name,result):
    write_csv(out/f'{name}.csv',['time',*[f'column{i}' for i in range(len(result.exponents))]],np.column_stack((result.times,result.running_estimates)))
    return dict(exponents=result.exponents.tolist(),trace_average=result.trace_average,elapsed=result.elapsed,
        status=result.status,evaluations=result.evaluations,accepted_steps=result.accepted_steps,rejected_steps=result.rejected_steps)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',default='outputs/chapter07')
    out=Path(parser.parse_args().output); out.mkdir(parents=True,exist_ok=True)
    report=dict(python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__,seed=SEED,
        map_updates=20000,noise_standard_deviations=[0,1e-6],solver_controls=[solver(t) for t in (1e-9,1e-10)],
        model_parameters=dict(logistic=dict(r=3.8,initial=.28,periods=[1,2],neighborhood=.001,perturbation_bound=.01),
            henon_sync=dict(a=1.4,b=.3,driver=[.5,-.8],response=[.001,.001],neighborhood=.01,perturbation_bound=.01,escape_radius=100),
            modulated_lorenz=dict(sigma=10,r=80,b=.4,initial=[.8,.8,.8],omega=6.28318,phase=0,k=[0,2,11,36.8]),
            entrainment=dict(a=1,b=1,c=.1,target_amplitude=.5,target_frequency=1,horizon=100),
            rossler=dict(a=.2,b=.2,c=9,initial=[.8,.4,.8,1.1,2.3],horizon=220),
            coupled_lorenz=dict(sigma=16,r=40,b=4,initial=[.8,.8,.8,1,.4,.3],couplings=[0,.45,.70,.75,1],horizon=220)),
        spectrum_controls=dict(burn_in=20,durations=[100,200],qr_interval=.1),maps={},modulated_lorenz=[],entrainment=[],rossler=[],coupled_lorenz=[],benchmarks={})
    # Map controls: all decisions use current noisy measurements; dynamics use true states.
    print('Map controls',flush=True)
    plt.figure(figsize=(10,7))
    for period in (1,2):
        targets=c.logistic_targets(r=3.8,period=period)
        for noise in (0.,1e-6):
            for bound in (0.,.01):
                name=f'logistic_p{period}_noise{noise}_bound{bound}'
                r=c.logistic_ogy(.28,r=3.8,period=period,updates=20000,neighborhood=.001,max_perturbation=bound,noise_std=noise,seed=SEED)
                error=np.min(abs(r.states[:-1,0,None]-targets),axis=1)
                report['maps'][name]=control_output(out,name,r,error)
                plt.subplot(2,1,period); plt.semilogy(np.maximum(error,1e-16),lw=.6,label=f'noise={noise}, bound={bound}')
        plt.xlabel('map update'); plt.ylabel('distance to target cycle'); plt.legend(fontsize=7)
    figure(out/'logistic_control.png')
    point=c.henon_additive_fixed_points(a=1.4,b=.3)[-1]; A=c.henon_additive_jacobian(point,a=1.4,b=.3)
    gain=c.place_poles_2d(A,[1,0],poles=[.2,.3])
    report['henon_design']=dict(a=1.4,b=.3,fixed_point=point.tolist(),gain=gain.tolist(),assigned_poles=[.2,.3],
        neighborhood=.005,bound=.01,printed_gain=[-.1,1.3],printed_pole_moduli=abs(np.linalg.eigvals(A-np.outer([1,0],[-.1,1.3]))).tolist())
    plt.figure(figsize=(10,7))
    for index,initial in enumerate(([-.760343,-1.40007],[.6,.7],point+[1e-4,-1e-4])):
        for noise in (0.,1e-6):
            name=f'henon_initial{index}_noise{noise}'
            r=c.henon_ogy(initial,a=1.4,b=.3,target=point,gain=gain,updates=20000,neighborhood=.005,max_perturbation=.01,escape_radius=100,noise_std=noise,seed=SEED)
            error=np.max(abs(r.states[:-1]-point),axis=1)
            report['maps'][name]=dict(initial=np.asarray(initial).tolist(),**control_output(out,name,r,error))
            plt.subplot(2,1,1); plt.semilogy(np.maximum(error,1e-16),lw=.6,label=f'initial{index}, noise={noise}')
    plt.xlabel('map update'); plt.ylabel('fixed-point error'); plt.legend(fontsize=7)
    for noise in (0.,1e-6):
        name=f'henon_sync_noise{noise}'
        r=c.henon_synchronize([.5,-.8],[.001,.001],a=1.4,b=.3,updates=20000,neighborhood=.01,max_perturbation=.01,escape_radius=100,noise_std=noise,seed=SEED)
        error=np.sum(abs(r.states[:-1,:2]-r.states[:-1,2:]),axis=1)
        report['maps'][name]=control_output(out,name,r,error)
        plt.subplot(2,1,2); plt.semilogy(np.maximum(error,1e-16),lw=.6,label=f'noise={noise}')
    plt.xlabel('map update'); plt.ylabel('driver–response L1 error'); plt.legend(); figure(out/'henon_control_sync.png')

    print('Periodically modulated Lorenz',flush=True)
    plt.figure(figsize=(11,8))
    omega=6.28318; T=2*np.pi/omega
    for k_index,k in enumerate((0,2,11,36.8)):
        for tol in (1e-9,1e-10):
            for burn in (100,200):
                rhs=lambda t,y:c.modulated_lorenz(t,y,sigma=10,r=80,b=.4,k=k,omega=omega)
                r=stroboscopic_sample(rhs,[.8,.8,.8],initial_time=0,period=T,burn_in_periods=burn,samples=512,
                    drive_frequencies=[omega],drive_phases=[0],solver_parameters=solver(tol))
                name=f'lorenz_mod_k{k}_tol{tol}_burn{burn}'
                write_csv(out/f'{name}.csv',['time','x','y','z','phase'],np.column_stack((r.times,r.states,r.phases)))
                residuals={str(p):float(np.max(np.linalg.norm(r.states[p:]-r.states[:-p],axis=1))) for p in (1,2,4)} if len(r.states)>4 else {}
                report['modulated_lorenz'].append(dict(k=k,omega=omega,period=T,burn_in_periods=burn,tolerance=tol,samples=len(r.times),status=r.status,evaluations=r.evaluations,accepted_steps=r.accepted_steps,rejected_steps=r.rejected_steps,recurrence_max_residuals=residuals))
                if tol==1e-9:
                    plt.subplot(2,2,k_index+1); plt.scatter(r.states[:,0],r.states[:,2],s=2,label=f'burn={burn}')
                    plt.xlabel('x'); plt.ylabel('z'); plt.title(f'k={k} stroboscopic samples'); plt.legend(fontsize=7)
    figure(out/'lorenz_modulation.png')

    print('Feedforward entrainment',flush=True)
    target=lambda t:np.array([.5*np.cos(t),-.5*np.sin(t)])
    derivative=lambda t:np.array([-.5*np.sin(t),-.5*np.cos(t)])
    base=lambda t,y:anharmonic(t,y,a=1,b=1,c=.1,k1=0,k2=0,omega=1)
    controlled=lambda t,y:base(t,y)+c.feedforward_force(base,target,derivative,t)
    plt.figure(figsize=(9,4))
    for tol in (1e-9,1e-10):
        for offset in (0.,-1.,-.1,.1,1.):
            r=integrate(controlled,target(0)+[offset,0],t_span=(0,100),**solver(tol))
            error=np.linalg.norm(r.states-np.array([target(t) for t in r.times]),axis=1)
            name=f'entrainment_tol{tol}_offset{offset}'
            force=np.array([c.feedforward_force(base,target,derivative,t)[1] for t in r.times])
            write_csv(out/f'{name}.csv',['time','q','p','error','mechanical_force'],np.column_stack((r.times,r.states,error,force)))
            report['entrainment'].append(dict(offset=offset,tolerance=tol,status=r.status,evaluations=r.evaluations,max_error=float(max(error)),final_error=float(error[-1])))
            if tol==1e-9: plt.semilogy(r.times,np.maximum(error,1e-16),label=f'offset={offset}')
    plt.xlabel('time'); plt.ylabel('distance to prescribed trajectory'); plt.legend(); figure(out/'entrainment.png')

    print('Rossler conditional spectra',flush=True)
    rrhs=lambda t,y:c.rossler_drive_response(t,y,a=.2,b=.2,c=9)
    rjac=lambda t,y:c.rossler_response_jacobian(t,y,a=.2,b=.2,c=9)
    plt.figure(figsize=(10,4))
    for tol in (1e-9,1e-10):
        y0=[.8,.4,.8,1.1,2.3]; trajectory=integrate(rrhs,y0,t_span=(0,220),**solver(tol))
        errors=np.linalg.norm(trajectory.states[:,[0,2]]-trajectory.states[:,3:],axis=1)
        write_csv(out/f'rossler_trajectory_tol{tol}.csv',['time','x','y','z','response_x','response_z','error'],np.column_stack((trajectory.times,trajectory.states,errors)))
        if tol==1e-9:
            plt.subplot(121); plt.semilogy(trajectory.times,np.maximum(errors,1e-16)); plt.xlabel('time'); plt.ylabel('response error')
        for duration in (100,200):
            r=conditional_flow_spectrum(rrhs,rjac,y0,response_dimension=2,burn_in=20,duration=duration,qr_interval=.1,initial_basis=np.eye(2),solver_parameters=solver(tol))
            name=f'rossler_conditional_tol{tol}_duration{duration}'
            report['rossler'].append(dict(tolerance=tol,duration=duration,trajectory_status=trajectory.status,trajectory_evaluations=trajectory.evaluations,final_response_error=float(errors[-1]),**spectrum_output(out,name,r)))
            if tol==1e-9:
                plt.subplot(122); plt.plot(r.times,r.running_estimates[:,0],label=f'column 0, T={duration}'); plt.plot(r.times,r.running_estimates[:,1],label=f'column 1, T={duration}')
    plt.xlabel('time'); plt.ylabel('conditional rate / time'); plt.legend(fontsize=7); figure(out/'rossler.png')

    print('Coupled Lorenz and transverse spectra',flush=True)
    base=lambda t,y:lorenz(t,y,sigma=16,r=40,b=4)
    jac=lambda t,y:lorenz_jacobian(t,y,sigma=16,r=40,b=4)
    plt.figure(figsize=(11,5))
    for tol in (1e-9,1e-10):
        for coupling in (0.,.45,.70,.75,1.):
            rhs=lambda t,y:c.diffusive_rhs(base,t,y,coupling=coupling)
            trajectory=integrate(rhs,[.8,.8,.8,1,.4,.3],t_span=(0,220),**solver(tol))
            errors=np.linalg.norm(trajectory.states[:,:3]-trajectory.states[:,3:],axis=1)
            write_csv(out/f'coupled_lorenz_c{coupling}_tol{tol}.csv',['time','x1','y1','z1','x2','y2','z2','error'],np.column_stack((trajectory.times,trajectory.states,errors)))
            if tol==1e-9:
                plt.subplot(121); plt.semilogy(trajectory.times,np.maximum(errors,1e-16),lw=.7,label=f'c={coupling}')
            for duration in (100,200):
                r=conditional_flow_spectrum(base,lambda t,y:jac(t,y)-2*coupling*np.eye(3),[.8,.8,.8],response_dimension=3,burn_in=20,duration=duration,qr_interval=.1,initial_basis=np.eye(3),solver_parameters=solver(tol))
                name=f'lorenz_transverse_c{coupling}_tol{tol}_duration{duration}'
                report['coupled_lorenz'].append(dict(coupling=coupling,tolerance=tol,duration=duration,trajectory_status=trajectory.status,trajectory_evaluations=trajectory.evaluations,final_response_error=float(errors[-1]),**spectrum_output(out,name,r)))
                if tol==1e-9 and duration==200:
                    plt.subplot(122); plt.plot(r.times,r.running_estimates[:,0],label=f'c={coupling}')
    plt.subplot(121); plt.xlabel('time'); plt.ylabel('complete-state error'); plt.legend(fontsize=7)
    plt.subplot(122); plt.xlabel('time'); plt.ylabel('transverse column 0 / time'); plt.axhline(0,color='black',lw=.5); plt.legend(fontsize=7); figure(out/'coupled_lorenz.png')

    print('Van der Pol and logistic coupling benchmarks',flush=True)
    report['benchmarks']['van_der_pol']=[]; report['benchmarks']['logistic']=[]
    plt.figure(figsize=(10,4))
    for tol in (1e-9,1e-10):
        f=lambda t,y:van_der_pol(t,y,mu=1)
        for coupling in (0.,.2):
            tr=integrate(lambda t,y:c.diffusive_rhs(f,t,y,coupling=coupling),[1,0,.5,.5],t_span=(0,100),**solver(tol))
            error=np.linalg.norm(tr.states[:,:2]-tr.states[:,2:],axis=1)
            report['benchmarks']['van_der_pol'].append(dict(mu=1,coupling=coupling,tolerance=tol,status=tr.status,evaluations=tr.evaluations,final_error=float(error[-1])))
            write_csv(out/f'vdp_c{coupling}_tol{tol}.csv',['time','x1','v1','x2','v2','error'],np.column_stack((tr.times,tr.states,error)))
            if tol==1e-9:
                plt.subplot(121); plt.semilogy(tr.times,np.maximum(error,1e-16),label=f'c={coupling}')
    for coupling in (0.,.05,.1,.15):
        r=2.8; fixed=1-1/r; values=[np.array([fixed+1e-4,fixed-1e-4])]; status='completed'
        for i in range(1000):
            new=coupled_logistic_step(values[-1],r=r,coupling=coupling)
            values.append(new)
            if not np.isfinite(new).all() or np.max(abs(new))>100: status='escaped'; break
        values=np.array(values); rate=c.logistic_transverse_curve(np.full(1000,fixed),r=r,coupling=coupling)
        report['benchmarks']['logistic'].append(dict(r=r,coupling=coupling,status=status,transverse_factor=2-r-2*coupling,transverse_exponent=rate['exponent']))
        write_csv(out/f'logistic_coupling_c{coupling}.csv',['update','x','y','error'],np.column_stack((np.arange(len(values)),values,abs(values[:,0]-values[:,1]))))
        plt.subplot(122); plt.semilogy(np.maximum(abs(values[:,0]-values[:,1]),1e-16),label=f'c={coupling}')
    plt.subplot(121); plt.xlabel('time'); plt.ylabel('Van der Pol error'); plt.legend()
    plt.subplot(122); plt.xlabel('map update'); plt.ylabel('logistic error'); plt.legend(); figure(out/'coupling_benchmarks.png')
    report['notes']='Capture counts use explicit 1e-6 finite-run tolerance, not asymptotic classifications. Conditional/transverse spectra differ from full-system spectra; rates are natural logs per time (maps per update).'
    def clean(value):
        if isinstance(value,dict): return {k:clean(v) for k,v in value.items()}
        if isinstance(value,list): return [clean(v) for v in value]
        if isinstance(value,(float,np.floating)) and not np.isfinite(value): return None
        return value
    (out/'results.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False))
    print('Wrote',out,flush=True)


if __name__=='__main__': main()
