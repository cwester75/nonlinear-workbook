"""Shared equations -> uniformly sampled observations -> diagnostic harness."""
import numpy as np
from .chapter10 import integrate


def sampled_observations(rhs,y0,*,samples,sample_interval,burn_in,noise_std,seed,
                         observation,solver_parameters):
    """Integrate to each requested observation time; avoid interpolation artifacts.

    Solver is restarted at sampling boundaries; internal adaptive step sizes are
    independent of observation spacing. Noise is IID Gaussian in observation units.
    """
    if not isinstance(samples,(int,np.integer)) or isinstance(samples,bool) or samples<2 or not np.isfinite([sample_interval,burn_in,noise_std]).all() or sample_interval<=0 or burn_in<0 or noise_std<0: raise ValueError('invalid sampling controls')
    if 't_span' in solver_parameters: raise ValueError('sampling controls t_span')
    state=np.atleast_1d(np.asarray(y0,float)); t=0.; calls=0; accepted=0; rejected=0
    def advance(end):
        nonlocal state,t,calls,accepted,rejected
        result=integrate(rhs,state,t_span=(t,end),**solver_parameters)
        if result.status!='completed': raise ValueError('integration failed: '+result.status)
        state=result.states[-1]; t=end; calls+=result.evaluations; accepted+=result.accepted_steps; rejected+=result.rejected_steps
    if burn_in: advance(burn_in)
    times=burn_in+sample_interval*np.arange(samples)
    observed=[]
    for i,time in enumerate(times):
        if i: advance(float(time))
        value=np.asarray(observation(state.copy()),float)
        if value.ndim!=0 or not np.isfinite(value): raise ValueError('observation must be one finite scalar')
        observed.append(float(value))
    clean=np.array(observed); noisy=clean+np.random.default_rng(seed).normal(0,noise_std,samples)
    return dict(times=times,clean=clean,observed=noisy,sample_interval=sample_interval,
                noise_std=noise_std,seed=seed,evaluations=calls,accepted_steps=accepted,rejected_steps=rejected)
