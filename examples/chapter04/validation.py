"""Run: python examples/chapter04/validation.py [--output outputs/chapter04]."""
import argparse
import csv
import json
import platform
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nonlinear_workbook import chapter04 as c
from nonlinear_workbook.chapter10 import integrate

SOLVER=dict(method='rkf45_adaptive',step=.01,max_step=.02,atol=1e-10,rtol=1e-10,max_steps=1000000)


def csv_write(path, header, rows):
    with path.open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(header); writer.writerows(rows)


def savefig(path):
    plt.tight_layout(); plt.savefig(path,dpi=140,metadata={'Software':'nonlinear-workbook'}); plt.close()


def drift(values):
    a=np.asarray(values)
    return float(np.max(abs(a-a[0])/np.maximum(1,abs(a[0]))))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',default='outputs/chapter04')
    out=Path(parser.parse_args().output); out.mkdir(parents=True,exist_ok=True)
    report=dict(python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__,
        seed=None,randomness='none',state_order='q...,p...',time_units='dimensionless physical time',
        solver=SOLVER,section=dict(horizon=2000,steps=[.02,.01],coordinate=0,level=0,direction=-1,max_crossings=2048,tolerance=1e-10),
        tolerances=dict(jacobian=1e-6,lax=1e-12,invariant_drift_20=1e-6,mathieu_monodromy=1e-7,mathieu_determinant=1e-7),
        sections=[],comparisons=[])
    # Book trajectory, independently integrated at high accuracy.
    initial=np.array([.5,.1,.1,.1])
    book=integrate(c.henon_heiles_rhs,initial,t_span=(0,124),**SOLVER)
    e=np.array([c.henon_heiles_energy(y) for y in book.states])
    report['henon_heiles_book']=dict(initial=initial.tolist(),energy=float(e[0]),horizon=124,status=book.status,evaluations=book.evaluations,relative_energy_drift=drift(e))
    csv_write(out/'henon_heiles_book.csv',['time','q1','q2','p1','p2','energy'],np.column_stack((book.times,book.states,e)))
    plt.figure(figsize=(9,4)); plt.subplot(121); plt.plot(book.states[:,0],book.states[:,1],lw=.5); plt.xlabel('q1'); plt.ylabel('q2')
    plt.subplot(122); plt.plot(book.times,e-e[0]); plt.xlabel('time'); plt.ylabel('H(t)-H(0)'); savefig(out/'henon_heiles_book.png')

    fixtures=[('hh_E1_12_I',np.array([np.sqrt(1/8),0,0,np.sqrt(1/24)]),c.henon_heiles_rhs,c.henon_heiles_gradient,c.henon_heiles_energy,None),
        ('hh_E1_12_II',np.array([0,0,np.sqrt(1/6),0]),c.henon_heiles_rhs,c.henon_heiles_gradient,c.henon_heiles_energy,None)]
    for E in (1/8,.16):
        fixtures.append((f'hh_E{E}',c.henon_heiles_shell(energy=E,q2=.1,p2=0,momentum_sign=1),c.henon_heiles_rhs,c.henon_heiles_gradient,c.henon_heiles_energy,None))
    for r in (0.,.15,1.):
        fixtures.append((f'quartic_r{r}',np.array([1,1,1.5,1]),lambda t,y,r=r:c.quartic_rhs(t,y,r=r),lambda q,r=r:c.quartic_gradient(q,r=r),lambda y,r=r:c.quartic_energy(y,r=r),r))
    energy_figure,energy_axes=plt.subplots(4,2,figsize=(11,12))
    for fixture_index,(name,y0,rhs,grad,energy,r) in enumerate(fixtures):
        print('sections',name,flush=True)
        plt.figure(figsize=(10,4))
        for k,h in enumerate((.02,.01)):
            trajectory=c.integrate_separable(y0,grad_potential=grad,t_span=(0,2000),step=h,method='verlet',escape_radius=1e6)
            sections=c.refined_sections(rhs,trajectory.times,trajectory.states,coordinate=0,level=0,direction=-1,
                max_crossings=2048,solver_parameters=SOLVER,tolerance=1e-10)
            energies=np.array([energy(y) for y in trajectory.states])
            entry=dict(name=name,initial=y0.tolist(),r=r,step=h,status=trajectory.status,accepted_steps=trajectory.accepted_steps,
                gradient_evaluations=trajectory.evaluations,initial_energy=float(energies[0]),relative_energy_drift=drift(energies),
                section_status=sections.status,crossings=len(sections.times),refinement_evaluations=sections.evaluations,
                failures=sections.failures,max_section_residual=float(max(sections.residuals,default=0)))
            if r==0:
                iv=np.array([c.quartic_integral(y) for y in trajectory.states]); entry['relative_extra_integral_drift']=drift(iv)
            report['sections'].append(entry)
            section_energy=np.array([energy(y) for y in sections.states])
            csv_write(out/f'{name}_h{h}_sections.csv',['time','q1','q2','p1','p2','section_residual','energy'],np.column_stack((sections.times,sections.states,sections.residuals,section_energy)))
            # Decimation only affects the exported trajectory, not energy maxima or crossing searches.
            idx=np.unique(np.r_[np.arange(0,len(trajectory.times),20),len(trajectory.times)-1])
            csv_write(out/f'{name}_h{h}_trajectory.csv',['time','q1','q2','p1','p2','energy'],np.column_stack((trajectory.times[idx],trajectory.states[idx],energies[idx])))
            axis=energy_axes.flat[fixture_index]
            axis.semilogy(trajectory.times[idx],np.maximum(abs(energies[idx]-energies[0])/max(1,abs(energies[0])),1e-16),label=f'h={h}')
            axis.set_title(name); axis.set_xlabel('time'); axis.set_ylabel('|H-H0| / max(1,|H0|)'); axis.legend()
            plt.subplot(1,2,k+1); plt.scatter(sections.states[:,1],sections.states[:,3],s=1)
            plt.xlabel('q2'); plt.ylabel('p2')
            plt.title(f'{name}, h={h}, N={len(sections.times)}\n{trajectory.status}; energy error={entry["relative_energy_drift"]:.2g}')
        savefig(out/f'{name}_sections.png')
        reference=integrate(rhs,y0,t_span=(0,20),**SOLVER)
        ref_energy=np.array([energy(y) for y in reference.states])
        comparison=dict(name=name,adaptive_status=reference.status,adaptive_evaluations=reference.evaluations,adaptive_energy_drift=drift(ref_energy),methods=[])
        for method in ('euler','verlet'):
            for h in (.02,.01):
                short=c.integrate_separable(y0,grad_potential=grad,t_span=(0,20),step=h,method=method)
                comparison['methods'].append(dict(method=method,step=h,status=short.status,endpoint_difference=float(np.linalg.norm(short.states[-1]-reference.states[-1])),energy_drift=drift([energy(y) for y in short.states])))
        report['comparisons'].append(comparison)
    energy_axes.flat[-1].axis('off'); energy_figure.tight_layout()
    energy_figure.savefig(out/'section_energy_errors.png',dpi=140,metadata={'Software':'nonlinear-workbook'})
    plt.close(energy_figure)

    print('Toda comparisons',flush=True)
    y0=np.array([-.3,.1,.2,.2,-.1,-.1]); report['toda']=[]
    plt.figure(figsize=(10,4))
    for periodic in (True,False):
        rhs=lambda t,y:c.toda_rhs(t,y,periodic=periodic)
        result=integrate(rhs,y0,t_span=(0,100),**SOLVER)
        vals=np.array([[c.toda_energy(y,periodic=periodic),*(c.periodic_toda_integrals(y) if periodic else c.open_toda_lax(y)['invariants'])] for y in result.states])
        entry=dict(periodic=periodic,initial=y0.tolist(),horizon=100,status=result.status,evaluations=result.evaluations,relative_invariant_drift=drift(vals))
        if not periodic:
            eigenvalues=np.array([c.open_toda_lax(y)['eigenvalues'] for y in result.states]); entry['max_eigenvalue_drift']=float(np.max(abs(eigenvalues-eigenvalues[0])))
            csv_write(out/'toda_open_eigenvalues.csv',['time','lambda1','lambda2','lambda3'],np.column_stack((result.times,eigenvalues)))
        report['toda'].append(entry)
        csv_write(out/f'toda_{periodic}.csv',['time','q1','q2','q3','p1','p2','p3']+[f'invariant{i}' for i in range(vals.shape[1])],np.column_stack((result.times,result.states,vals)))
        plt.subplot(121); plt.plot(result.times,result.states[:,0],label='periodic' if periodic else 'open'); plt.xlabel('time'); plt.ylabel('q1'); plt.legend()
        plt.subplot(122); plt.plot(result.times,vals[:,0]-vals[0,0],label='periodic' if periodic else 'open'); plt.xlabel('time'); plt.ylabel('energy drift'); plt.legend()
    savefig(out/'toda.png')

    print('Mathieu scan',flush=True)
    fixture=c.mathieu_floquet(delta=1,epsilon=.5,solver_parameters=SOLVER)
    report['mathieu_fixture']=dict(delta=1,epsilon=.5,monodromy=fixture['monodromy'].tolist(),
        multipliers=[[float(z.real),float(z.imag)] for z in fixture['classification']['multipliers']],
        band=fixture['band'],determinant_error=fixture['determinant_error'],evaluations=fixture['evaluations'],status=fixture['status'])
    ds=np.linspace(0,5,41); es=np.linspace(0,2,31); codes=[]; rows=[]; max_det=0
    labels={'bounded_oscillatory':0,'boundary_unresolved':1,'exponentially_unstable':2,'unresolved':3,'integration_failure':4}
    for eps in es:
        row=[]
        for delta in ds:
            f=c.mathieu_floquet(delta=delta,epsilon=eps,solver_parameters=SOLVER)
            row.append(labels[f['band']]); max_det=max(max_det,f['determinant_error'])
            rows.append([delta,eps,f['trace'],f['determinant_error'],f['band'],f['evaluations'],f['status']])
        codes.append(row)
    csv_write(out/'mathieu_scan.csv',['delta','epsilon','trace','determinant_error','band','evaluations','status'],rows)
    report['mathieu_scan']=dict(delta=[0,5,41],epsilon=[0,2,31],max_determinant_error=max_det,labels=labels,
        counts={label:sum(row[4]==label for row in rows) for label in labels})
    plt.figure(figsize=(8,4)); plt.imshow(codes,origin='lower',extent=[0,5,0,2],aspect='auto',vmin=0,vmax=4,cmap='viridis',interpolation='nearest')
    plt.xlabel('delta'); plt.ylabel('epsilon'); plt.title('Mathieu: 0 oscillatory, 1 boundary, 2 unstable, 3/4 unresolved'); plt.colorbar(); savefig(out/'mathieu.png')
    report['export']='Trajectory CSVs for sections retain every twentieth state plus the endpoint; energy maxima and sections use every step.'
    with (out/'results.json').open('w') as f: json.dump(report,f,indent=2,allow_nan=False)
    print('Wrote',out,flush=True)


if __name__=='__main__': main()
