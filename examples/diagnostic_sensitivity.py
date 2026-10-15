"""Synthetic controls and baseline feature comparisons; never held-out market data."""
import argparse
import csv
import itertools
import json
import platform
from pathlib import Path
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nonlinear_workbook.diagnostic_adapter import WindowConfig,diagnose_prefix

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',default='outputs/diagnostic-sensitivity'); out=Path(parser.parse_args().output); out.mkdir(parents=True,exist_ok=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    records=[]; seeds=(11,23,37,41,53)
    for seed in seeds:
        for n,dt,burn in itertools.product((256,512),(.05,.2),(0,100)):
            t=(np.arange(n)+burn)*dt; rng=np.random.default_rng(seed)
            signals=dict(periodic=np.sin(2*np.pi*t),iid=rng.normal(size=n),
                ar1=np.zeros(n),transient_periodic=np.sin(2*np.pi*t)+2*np.exp(-t/2))
            innovation=rng.normal(size=n+burn); ar=np.zeros(n+burn)
            for i in range(1,len(ar)): ar[i]=.8*ar[i-1]+innovation[i]
            signals['ar1']=ar[burn:]
            for name,clean in signals.items():
                epsilon=.2*float(np.std(clean))
                for noise,dimension,fit in itertools.product((0.,.02),(2,3),((0,3),(1,6))):
                    observed=clean+np.random.default_rng(seed+1000).normal(0,noise,n)
                    cfg=WindowConfig(window=n,sample_interval=dt,time_unit='dimensionless time',value_unit='synthetic amplitude',epsilon=epsilon,dimension=dimension,fit_start=fit[0],fit_stop=fit[1])
                    record=diagnose_prefix(observed,end_index=n,config=cfg,source_id=f'{name}-seed{seed}-n{n}-dt{dt}-burn{burn}-noise{noise}',workbook_commit=commit)
                    record.update(control=name,seed=seed,burn_samples=burn,noise_std=noise,split='development' if seed in seeds[:3] else 'reserved_synthetic')
                    # Preserve observations once per setting, not once per estimator configuration.
                    if dimension==2 and fit==(0,3):
                        path=out/f'{name}_seed{seed}_n{n}_dt{dt}_burn{burn}_noise{noise}.csv'
                        with path.open('w',newline='') as f:
                            w=csv.writer(f); w.writerow(['time','clean','observed']); w.writerows(zip(t,clean,observed))
                    records.append(record)
    features=['std','mad','mean_abs_increment','lag1','hurst','lz','divergence']
    with (out/'features.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['control','seed','split','samples','dt','burn','noise','dimension','fit_start','fit_stop','status',*features])
        for r in records:
            cfg=r['config']; w.writerow([r['control'],r['seed'],r['split'],r['samples'],cfg['sample_interval'],r['burn_samples'],r['noise_std'],cfg['dimension'],cfg['fit_start'],cfg['fit_stop'],r['status'],*[r['features'].get(k) for k in features]])
    dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip())
    summary=dict(python=platform.python_version(),numpy=np.__version__,commit=commit,working_tree_dirty=dirty,seeds=seeds,records=len(records),
        interpretation='Synthetic controls only. Reserved seeds are not an outlierhunter held-out data evaluation. No feature admission or detector thresholds.',controls={})
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for name in ('periodic','iid','ar1','transient_periodic'):
        selected=[r for r in records if r['control']==name]; summary['controls'][name]={}
        for axis,feature in zip(axes,('hurst','lz','divergence')):
            values=[r['features'].get(feature) for r in selected if r['features'].get(feature) is not None]
            summary['controls'][name][feature]=dict(available=len(values),unavailable=len(selected)-len(values),minimum=min(values) if values else None,maximum=max(values) if values else None)
            axis.scatter(np.full(len(values),('periodic','iid','ar1','transient_periodic').index(name)),values,s=3,alpha=.25)
            axis.set_title(feature); axis.set_xticks(range(4),('periodic','iid','ar1','transient'),rotation=25)
    # Descriptive within-control correlations, grouped to avoid treating mixed regimes as incremental predictive evidence.
    summary['baseline_correlations']={}
    for name in summary['controls']:
        matrix=np.array([[r['features'].get(k) if r['features'].get(k) is not None else np.nan for k in features] for r in records if r['control']==name])
        correlation={}
        for i,j in itertools.combinations(range(len(features)),2):
            valid=np.isfinite(matrix[:,i])&np.isfinite(matrix[:,j])
            correlation[features[i]+'__'+features[j]]=float(np.corrcoef(matrix[valid,i],matrix[valid,j])[0,1]) if valid.sum()>2 and np.std(matrix[valid,i])>0 and np.std(matrix[valid,j])>0 else None
        summary['baseline_correlations'][name]=correlation
    (out/'records.json').write_text(json.dumps(records,indent=2,allow_nan=False))
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    plt.tight_layout(); plt.savefig(out/'sensitivity.png',dpi=140); plt.close()
    print('Synthetic sensitivity records:',len(records))

if __name__=='__main__': main()
