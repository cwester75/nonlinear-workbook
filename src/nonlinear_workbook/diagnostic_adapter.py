"""Causal, shadow-only window diagnostics. No detection or trading decisions."""
from dataclasses import asdict,dataclass
import hashlib
import platform
import re
import numpy as np
from .chapter02 import hurst_rs,lz_complexity,kantz_curve


@dataclass(frozen=True)
class WindowConfig:
    window: int
    sample_interval: float
    time_unit: str
    value_unit: str
    epsilon: float
    dimension: int = 2
    delay: int = 2
    theiler: int = 20
    horizon: int = 8
    fit_start: int = 0
    fit_stop: int = 3
    rs_scales: tuple = (8,16,32,64)
    minimum_references: int = 10

    def __post_init__(self):
        for name,minimum in [('window',32),('dimension',1),('delay',1),('theiler',0),('horizon',2),('fit_start',0),('fit_stop',2),('minimum_references',1)]:
            value=getattr(self,name)
            if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<minimum: raise ValueError('invalid '+name)
            object.__setattr__(self,name,int(value))
        if not np.isfinite([self.sample_interval,self.epsilon]).all() or self.sample_interval<=0 or self.epsilon<=0:
            raise ValueError('positive finite interval/radius required')
        if not isinstance(self.time_unit,str) or not self.time_unit.strip() or not isinstance(self.value_unit,str) or not self.value_unit.strip():
            raise ValueError('explicit time and value units required')
        if not self.fit_start+2<=self.fit_stop<=self.horizon: raise ValueError('invalid explicit fit slice')
        if self.window-(self.dimension-1)*self.delay-self.horizon<=self.theiler+1: raise ValueError('window too short for embedding/exclusion')
        if len(self.rs_scales)<2 or len(set(self.rs_scales))!=len(self.rs_scales) or any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) or not 8<=v<=self.window//2 for v in self.rs_scales):
            raise ValueError('explicit distinct R/S scales required within window')
        object.__setattr__(self,'rs_scales',tuple(int(v) for v in self.rs_scales))


def diagnose_prefix(values,*,end_index,config,source_id,workbook_commit):
    """Only values[max(0,end-window):end] are validated or measured.

    Commit is caller-supplied provenance, not an integrity attestation. Consumers
    must verify their installed VCS commit against their pin before calling.
    Unavailable estimates are None with reasons, never filled or classified.
    """
    if not isinstance(config,WindowConfig): raise ValueError('WindowConfig required')
    a=np.asarray(values)
    if a.ndim!=1 or a.dtype.kind not in 'fiu' or isinstance(end_index,(bool,np.bool_)) or not isinstance(end_index,(int,np.integer)) or not 0<=end_index<=len(a):
        raise ValueError('real numeric series and valid exclusive endpoint required')
    if not isinstance(source_id,str) or not source_id or not isinstance(workbook_commit,str) or not re.fullmatch('[0-9a-f]{40}',workbook_commit):
        raise ValueError('source ID and full lowercase commit hash required')
    start=max(0,end_index-config.window); x=np.array(a[start:end_index],dtype='<f8',copy=True)
    result=dict(status='available',start_index=int(start),end_index=int(end_index),samples=len(x),
        features={},failures={},curves={},config=asdict(config),
        provenance=dict(source_id=source_id,workbook_commit=workbook_commit,commit_verification='caller responsibility',
            window_sha256=hashlib.sha256(x.tobytes()).hexdigest(),python=platform.python_version(),numpy=np.__version__),
        feature_units=dict(mean=config.value_unit,std=config.value_unit,mad=config.value_unit,
            mean_abs_increment=config.value_unit,lag1='dimensionless',hurst='dimensionless',lz='dimensionless',divergence='1/'+config.time_unit),
        conventions=dict(symbolization='x > median(current causal window); ties=0',divergence='natural log separation per time unit',rs_fit='all explicitly supplied valid scales',lag1='whole-window centering; lagged dot product divided by vector norms',shadow_only=True))
    if len(x)!=config.window or not np.isfinite(x).all():
        result['status']='insufficient_window' if len(x)!=config.window else 'invalid_window'
        result['failures']['window']=result['status']; return result
    with np.errstate(over='ignore',invalid='ignore'):
        std=float(np.std(x)); centered=x-np.mean(x)
        basics=dict(mean=float(np.mean(x)),std=std,mad=float(np.median(abs(x-np.median(x)))),mean_abs_increment=float(np.mean(abs(np.diff(x)))))
        denominator=float(np.linalg.norm(centered[:-1])*np.linalg.norm(centered[1:]))
        basics['lag1']=float(centered[:-1]@centered[1:]/denominator) if denominator>0 else None
    for key,value in basics.items():
        result['features'][key]=value if value is not None and np.isfinite(value) else None
        if result['features'][key] is None: result['failures'][key]='undefined or nonfinite baseline'
    for name in ('hurst','lz','divergence'):
        try:
            if name=='hurst':
                estimate,scales,rs=hurst_rs(x,sizes=config.rs_scales)
                if not np.isfinite(rs).all(): raise FloatingPointError('nonfinite R/S curve')
                result['curves']['rs']=dict(scales=scales.tolist(),mean_rs=rs.tolist())
            elif name=='lz':
                count,estimate=lz_complexity((x>np.median(x)).astype(int))
                result['curves']['lz']=dict(phrases=count,alphabet=int(len(np.unique(x>np.median(x)))))
            else:
                lag,curve,count=kantz_curve(x,dimension=config.dimension,delay=config.delay,epsilon=config.epsilon,horizon=config.horizon,theiler=config.theiler)
                times=lag*config.sample_interval
                if not np.isfinite(curve).all() or not np.isfinite(times).all(): raise FloatingPointError('nonfinite divergence curve')
                result['curves']['kantz']=dict(times=times.tolist(),log_separation=curve.tolist(),eligible_references=count)
                if count<config.minimum_references: raise ValueError('insufficient eligible references')
                fit=slice(config.fit_start,config.fit_stop)
                estimate=float(np.polyfit(times[fit],curve[fit],1)[0])
            if not np.isfinite(estimate): raise ValueError('nonfinite estimate')
            result['features'][name]=float(estimate)
        except (ValueError,FloatingPointError,OverflowError) as exc:
            result['features'][name]=None; result['failures'][name]=str(exc)
    if result['failures']: result['status']='partial'
    return result
