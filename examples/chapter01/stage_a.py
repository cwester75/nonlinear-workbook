"""Chapter 1 Stage A, run from repository root with PYTHONPATH=src."""
from fractions import Fraction
import numpy as np
from nonlinear_workbook.chapter01 import *
from common import setup,save_report,save_plot,csv,plt,SEED


def main():
    out,report=setup('stage_a')
    report['conventions']=dict(numeric_dtype='float64',first_state='after burn_in updates',fourier='forward 1/N; inverse unscaled',sample_interval=1.)
    report['parameters']=dict(logistic_r=4.,logistic_x0=.618,logistic_n=100000,logistic_burn_in=1000,histogram_bins=40,
        sweep_grid='linspace(2,4,500)',sweep_n=128,sweep_burn_in=2000,rotation_n=4000,rotation_burn_in=1000)
    exact=iterate(logistic_step,Fraction(1,3),n=12,exact=True).values
    approximate=iterate(logistic_step,1/3,n=12).values
    report['exact_logistic']=exact
    report['exact_vs_float_absolute_error']=np.abs(np.array([float(v) for v in exact])-approximate)
    report['exact_book_maps']={name:iterate(step,x0,n=12,exact=True) for name,step,x0 in [
        ('collatz',collatz_step,56),('bernoulli',bernoulli_step,Fraction(1,17)),
        ('tent',tent_step,Fraction(1,17)),('gauss',gauss_step,Fraction(23,101)),
        ('bungalow',lambda x:bungalow_step(x,a=Fraction(1,7)),Fraction(1,17))]}
    float_orbits={name:iterate(step,.618,n=80).values for name,step in [('bernoulli',bernoulli_step),('tent',tent_step)]}
    report['float_collapse']={name:int(np.flatnonzero(x==0)[0]) for name,x in float_orbits.items()}
    x=logistic_map(100000); edges=np.linspace(0,1,41)
    density=invariant_density(x,edges=edges); target=np.diff(2/np.pi*np.arcsin(np.sqrt(edges)))
    report['logistic_density']=density
    report['logistic_density_max_mass_error']=float(np.max(np.abs(density.counts/density.retained-target)))
    report['logistic_lyapunov']=scalar_lyapunov(x,logistic_derivative)
    report['logistic_moments']=moments(x,orders=(1,2,3))
    # Long exact tent orbit with a seeded 4096-bit rational initial state.
    bits=np.random.default_rng(SEED).integers(0,2,4096)
    numerator=int(''.join(map(str,bits)),2)
    rational=Fraction(numerator,2**4096-1)
    tent=iterate(tent_step,rational,n=4096,exact=True).values
    tent_density=invariant_density([float(v) for v in tent],edges=edges)
    report['exact_tent_density']=tent_density
    report['exact_tent_initial']=rational
    report['other_map_diagnostics']={}
    for name,step in [('sine',sine_step),('bungalow',lambda x:bungalow_step(x,a=.2)),
                      ('gauss',gauss_step),('polynomial_random',polynomial_random_step)]:
        orbit=iterate(step,.618,n=10000,burn_in=1000)
        report['other_map_diagnostics'][name]=dict(status=orbit.status,density=invariant_density(orbit.values,edges=edges))
    report['sine_fixed_point']=newton(lambda x:x-sine_step(x),lambda x:1-sine_derivative(x),.7)
    sine=iterate(sine_step,.618,n=10000,burn_in=1000).values
    report['sine_lyapunov']=scalar_lyapunov(sine,sine_derivative)
    report['lcg']=iterate(lambda x:lcg_step(x,multiplier=7,increment=7,modulus=10),7,n=12,exact=True)
    report['monte_carlo']=[monte_carlo(np.sin,n=100000,seed=SEED,method=m) for m in ('mean','hit_or_miss')]
    report['integral_reference']=float(1-np.cos(1))
    random_orbit=iterate(polynomial_random_step,.5,n=20000).values
    report['book_random_map_integral']=float(np.mean(np.sin(random_orbit)))
    parameters,orbits,exponents=logistic_sweep(np.linspace(2,4,500),n=128,burn_in=2000)
    csv(out,'logistic_exponents',np.column_stack([parameters,exponents]),'r,exponent')
    csv(out,'bifurcation',np.column_stack([np.repeat(parameters,orbits.shape[1]),orbits.ravel()]),'r,x')
    report['period_doubling']=[period_doubling(p,r_guess=r) for p,r in zip([1,2,4,8,16],[3.,3.45,3.544,3.564,3.5687])]
    if any(v.status!='converged' for v in report['period_doubling']): raise RuntimeError('Bifurcation solve failed')
    thresholds=[v.root[1] for v in report['period_doubling']]
    report['feigenbaum_ratios']=feigenbaum_ratios(thresholds)
    omega=np.linspace(0,1,101)
    rotations=np.array([rotation_number(omega=o,strength=1,n=4000) for o in omega])
    csv(out,'rotation',np.column_stack([omega,rotations]),'omega,rotation_number')
    report['rotation_zero_forcing']=rotation_number(omega=.37,strength=0,n=4000)
    report['rotation_locked_zero']=rotation_number(omega=0,strength=.8,n=4000)
    quadratic=iterate(lambda x:quadratic_step(x,r=1.85),.2,n=10).values
    symbols=['L' if v<0 else 'R' for v in quadratic[:-1]]
    report['symbolic']=dict(states=quadratic,symbols=symbols,reconstructed=inverse_symbolic(symbols,quadratic[-1],r=1.85))
    lags,cov,pairs=autocovariance(x,max_lag=20)
    csv(out,'autocovariance',np.column_stack([lags,cov,pairs]),'lag,covariance,pairs')
    cosine=np.cos(2*np.pi*np.arange(64)/64); transformed=radix2_fft(cosine)
    report['fft_reference_max_error']=float(np.max(np.abs(transformed-np.fft.fft(cosine)/64)))
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    axes[0,0].stairs(density.counts/density.retained,edges,label='trajectory mass'); axes[0,0].stairs(target,edges,label='analytic mass')
    axes[0,0].set(title='Logistic invariant bin masses',xlabel='x',ylabel='probability'); axes[0,0].legend()
    axes[0,1].stairs(tent_density.density,edges); axes[0,1].axhline(1,color='black',ls='--')
    axes[0,1].set(title='Exact rational tent trajectory',xlabel='x',ylabel='density')
    axes[1,0].plot(lags,cov,'o-'); axes[1,0].set(title='Logistic autocovariance',xlabel='lag',ylabel='covariance')
    axes[1,1].stem(np.arange(64),np.abs(transformed)); axes[1,1].set(title='Cosine Fourier coefficients',xlabel='frequency bin',ylabel='magnitude')
    save_plot(out,'density_and_spectrum',fig)
    fig,axes=plt.subplots(2,1,figsize=(11,8),layout='constrained',sharex=True)
    axes[0].scatter(np.repeat(parameters,128),orbits.ravel(),s=.12,color='black'); axes[0].set(ylabel='x',title='Logistic bifurcation diagram')
    axes[1].plot(parameters,exponents); axes[1].axhline(0,color='black',lw=.5); axes[1].set(xlabel='r',ylabel='exponent per step')
    save_plot(out,'bifurcation',fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    axes[0].plot(omega,rotations); axes[0].set(title='Circle rotation, strength=1',xlabel='omega',ylabel='rotation per step')
    for name,orbit in float_orbits.items(): axes[1].plot(orbit,label=name)
    axes[1].set(title='Float64 trajectory collapse',xlabel='iteration',ylabel='x'); axes[1].legend()
    save_plot(out,'rotation_and_precision',fig)
    save_report(out,report)

if __name__=='__main__': main()
