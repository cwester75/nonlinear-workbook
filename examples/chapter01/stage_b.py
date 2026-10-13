"""Chapter 1 Stage B: equations, finite-horizon diagnostics, and sensitivity."""
import numpy as np
from nonlinear_workbook.chapter01 import *
from common import setup,save_report,save_plot,csv,plt


def main():
    out,report=setup('stage_b')
    configurations=[
        ('henon',henon_step,henon_jacobian,[.1,.3],dict(a=1.4,b=.3)),
        ('lozi',lozi_step,lozi_jacobian,[.1,.3],dict(a=1.7,b=.4)),
        ('standard',standard_step,standard_jacobian,[.5,.8],dict(k=.8)),
        ('ikeda',ikeda_step,ikeda_jacobian,[.5,.5],dict(rho=.85,c1=.4,c2=.9,c3=9.)),
        ('coupled_logistic',coupled_logistic_step,coupled_logistic_jacobian,[.1,.2],dict(r=3.7,coupling=.06)),
        ('rtn',rtn_step,rtn_jacobian,[.4,.5],dict(r=1.0834))]
    trajectories={}; report['maps']={}
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    for axis,(name,step,jac,initial,parameters) in zip(axes.flat,configurations):
        result=iterate(step,initial,n=5000,burn_in=1000,escape_radius=1e6)
        trajectories[name]=result.values
        spectrum=known_lyapunov_spectrum(result.values,jac,qr_burn_in=100) if len(result.values)>100 else None
        report['maps'][name]=dict(parameters=parameters,initial=initial,n=5000,burn_in=1000,escape_radius=1e6,
            trajectory_status=result.status,spectrum=spectrum,
            coordinate_minimum=result.values.min(axis=0) if len(result.values) else None,
            coordinate_maximum=result.values.max(axis=0) if len(result.values) else None)
        if len(result.values):
            axis.scatter(result.values[:,0],result.values[:,1],s=.3,color='black')
            csv(out,name+'_trajectory',result.values,'coordinate_0,coordinate_1')
        axis.set(title=name.replace('_',' '),xlabel='momentum' if name=='standard' else 'x',ylabel='angle' if name=='standard' else 'y')
    save_plot(out,'phase_portraits',fig)
    report['henon_exponent_sum_reference']=float(np.log(.3))
    fixed=[]
    for sign in (-1,1):
        x=(-.7+sign*np.sqrt(.7**2+4*1.4))/(2*1.4); point=np.array([x,.3*x])
        eigenvalues,kind=fixed_point_classification(henon_jacobian,point)
        fixed.append(dict(point=point,residual=float(np.linalg.norm(henon_step(point)-point)),eigenvalues=eigenvalues,classification=kind))
    report['henon_fixed_points']=fixed
    radii=np.geomspace(.002,.2,16); scales=np.geomspace(.2,.002,16)
    report['geometry']={'parameters':dict(radii=radii,scales=scales,theiler_values=[0,20],sample_sizes=[1000,2500],fit_slices=[[2,9],[4,12]],box_origin=[-2,-1]),'correlation':[],'box':[]}
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for n in (1000,2500):
        for theiler in (0,20):
            result=correlation_integral(trajectories['henon'][:n],radii=radii,theiler=theiler)
            fits=[fit_scaling(radii,result.values,fit_slice=slice(a,b)) for a,b in [(2,9),(4,12)]]
            report['geometry']['correlation'].append(dict(n=n,theiler=theiler,curve=result,fits=fits))
            axes[0].loglog(radii,result.values,'o-',ms=3,label=f'n={n}, exclusion={theiler}')
            csv(out,f'correlation_n{n}_t{theiler}',np.column_stack([radii,result.values,result.counts]),'radius,probability,pairs_within_radius')
    for origin in ([-2,-1],[-1.999,-.999]):
        result=box_counts(trajectories['henon'],scales=scales,origin=origin)
        fits=[fit_scaling(scales,result.counts,fit_slice=slice(a,b),inverse_scale=True) for a,b in [(2,9),(4,12)]]
        report['geometry']['box'].append(dict(curve=result,fits=fits))
        axes[1].loglog(1/scales,result.counts,'o-',ms=3,label=f'origin={origin}')
    axes[0].set(title='Hénon correlation-integral sensitivity',xlabel='radius',ylabel='eligible-pair fraction'); axes[0].legend(fontsize=8)
    axes[1].set(title='Hénon box-count sensitivity',xlabel='inverse scale',ylabel='occupied cells'); axes[1].legend(fontsize=8)
    save_plot(out,'dimension_curves',fig)
    parameter_values=np.linspace(3.2,3.7,31); spectra=[]
    for r in parameter_values:
        orbit=iterate(lambda x:coupled_logistic_step(x,r=r,coupling=.06),[.7,.3],n=2000,burn_in=1000)
        if orbit.status=='completed':
            spectrum=known_lyapunov_spectrum(orbit.values,lambda x:coupled_logistic_jacobian(x,r=r,coupling=.06),qr_burn_in=100)
            spectra.append(spectrum.exponents)
        else: spectra.append([np.nan,np.nan])
    spectra=np.asarray(spectra)
    report['coupled_spectrum_sweep']=dict(parameters=parameter_values,coupling=.06,n=2000,burn_in=1000,qr_burn_in=100,initial=[.7,.3],exponents=spectra)
    csv(out,'coupled_spectra',np.column_stack([parameter_values,spectra]),'r,qr_exponent_0,qr_exponent_1')
    uncoupled=iterate(lambda x:coupled_logistic_step(x,r=4,coupling=0),[.618,.317],n=10000,burn_in=1000)
    report['uncoupled_reference']=known_lyapunov_spectrum(uncoupled.values,lambda x:coupled_logistic_jacobian(x,r=4,coupling=0))
    scan=boundedness_scan(np.linspace(3.5,4,31),np.linspace(.04,.09,21),x0=[.7,.3],horizon=400,escape_radius=20)
    report['boundedness']=scan
    report['boundedness_initial']=[.7,.3]
    codes={'completed':0,'escaped':1,'nonfinite':2}; scan_codes=np.vectorize(codes.get)(scan.labels)
    csv(out,'boundedness',np.column_stack([np.tile(scan.parameters,len(scan.couplings)),np.repeat(scan.couplings,len(scan.parameters)),scan_codes.ravel(),scan.updates.ravel()]),'r,coupling,status_code,updates')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    axes[0].plot(parameter_values,spectra[:,0],label='QR direction 0'); axes[0].plot(parameter_values,spectra[:,1],label='QR direction 1'); axes[0].axhline(0,color='black',lw=.5)
    axes[0].set(title='Coupled-logistic finite-time spectra',xlabel='r',ylabel='exponent per step'); axes[0].legend()
    from matplotlib.colors import ListedColormap,BoundaryNorm
    colormap=ListedColormap(['#23648a','#e1a444','#333333'])
    image=axes[1].imshow(scan_codes,origin='lower',extent=[3.5,4,.04,.09],aspect='auto',interpolation='nearest',cmap=colormap,norm=BoundaryNorm([-.5,.5,1.5,2.5],3))
    colorbar=fig.colorbar(image,ax=axes[1],ticks=[0,1,2]); colorbar.ax.set_yticklabels(['completed','escaped','nonfinite'])
    axes[1].set(title='400-update boundedness scan',xlabel='r',ylabel='coupling')
    save_plot(out,'spectra_and_boundedness',fig)
    coordinates=np.linspace(-2,2,121)
    basins=complex_newton_basins(coordinates,coordinates,max_steps=60,tolerance=1e-10,escape_radius=1e6)
    report['newton_basins']=dict(parameters=dict(real_range=[-2,2],imaginary_range=[-2,2],grid_size=121,max_steps=60,tolerance=1e-10,escape_radius=1e6),
        roots=basins.roots,status_counts={str(s):int(np.sum(basins.statuses==s)) for s in np.unique(basins.statuses)},labels=basins.labels,iterations=basins.iterations)
    np.savetxt(out/'newton_labels.csv',basins.labels,delimiter=',',fmt='%d')
    report['vector_newton']=vector_newton(lambda p:np.array([p@p-1,p[0]-p[1]]),lambda p:np.array([[2*p[0],2*p[1]],[1,-1]]),[3.5,-20.3])
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    colors=ListedColormap(['#333333','#23648a','#bd5635','#51a58b','#bca74b'])
    image=axes[0].imshow(basins.labels,origin='lower',extent=[-2,2,-2,2],interpolation='nearest',cmap=colors,norm=BoundaryNorm([-1.5,-.5,.5,1.5,2.5,3.5],5))
    colorbar=fig.colorbar(image,ax=axes[0],ticks=[-1,0,1,2,3]); colorbar.ax.set_yticklabels(['unresolved','1','-1','i','-i'])
    axes[0].set(title='Newton basins: z⁴ − 1',xlabel='Re(z)',ylabel='Im(z)')
    image=axes[1].imshow(basins.iterations,origin='lower',extent=[-2,2,-2,2],interpolation='nearest',cmap='viridis'); fig.colorbar(image,ax=axes[1],label='updates')
    axes[1].set(title='Newton iteration counts',xlabel='Re(z)',ylabel='Im(z)')
    save_plot(out,'newton_basins',fig)
    save_report(out,report)

if __name__=='__main__': main()
