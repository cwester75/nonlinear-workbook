"""Chapter 8 constructions, finite escape images and dimension sensitivity."""
import argparse
import csv
import json
import platform
from pathlib import Path
from dataclasses import asdict,is_dataclass
from fractions import Fraction
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from nonlinear_workbook import chapter08 as c


SEED=20261014
JULIA_C=-.12256117+.74486177j


def clean(x):
    if is_dataclass(x): return clean(asdict(x))
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,np.ndarray): return clean(x.tolist())
    if isinstance(x,(tuple,list)): return [clean(v) for v in x]
    if isinstance(x,np.generic): return clean(x.item())
    if isinstance(x,complex): return dict(real=x.real,imag=x.imag)
    if isinstance(x,Fraction): return str(x)
    if isinstance(x,float) and not np.isfinite(x): return None
    return x


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',default='outputs/chapter08')
    out=Path(parser.parse_args().output); out.mkdir(parents=True,exist_ok=True)
    report=dict(seed=SEED,python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__,constructions={})
    def savefig(fig,name):
        fig.tight_layout(); fig.savefig(out/(name+'.png'),dpi=140,metadata={'Software':'nonlinear-workbook'}); plt.close(fig)
    def csvfile(name,header,rows):
        with (out/(name+'.csv')).open('w',newline='',encoding='utf-8') as f:
            writer=csv.writer(f); writer.writerow(header); writer.writerows(rows)

    fig,axes=plt.subplots(2,2,figsize=(11,8))
    cantor=[]
    for depth in range(7):
        intervals=c.cantor_intervals(depth); cantor.append(dict(depth=depth,intervals=intervals,measure=sum((b-a for a,b in intervals),Fraction(0))))
        for a,b in intervals: axes[0,0].plot([float(a),float(b)],[-depth,-depth],color='black')
    axes[0,0].set_title('Exact Cantor stages0..6'); axes[0,0].set_xlabel('x'); axes[0,0].set_ylabel('-depth')
    dragons=[]
    for depth in range(13):
        z=c.dragon(depth); dragons.append(dict(depth=depth,segments=len(z)-1,length=c.polyline_length(z)))
        if depth in (0,4,8,12): axes[0,1].plot(z.real,z.imag,label=str(depth),linewidth=.7)
    axes[0,1].set_title('Dragon; selected depths'); axes[0,1].legend(); axes[0,1].set_aspect('equal')
    triangles=c.sierpinski_triangles(6); axes[1,0].add_collection(PolyCollection(triangles,facecolors='black',edgecolors='none')); axes[1,0].autoscale_view(); axes[1,0].set_aspect('equal'); axes[1,0].set_title('Gasket depth6')
    koch=[]
    for depth in range(6):
        z=c.koch_snowflake(depth); koch.append(dict(depth=depth,perimeter=c.polyline_length(z),area=c.polygon_area(z)))
        if depth in (0,2,5): axes[1,1].plot(z.real,z.imag,label=str(depth),linewidth=.7)
    axes[1,1].set_title('Snowflake'); axes[1,1].set_aspect('equal'); axes[1,1].legend(); savefig(fig,'constructions')
    report['constructions']=dict(cantor=cantor,dragon=dragons,gasket=[dict(depth=d,triangles=3**d,side=2.**-d,area=np.sqrt(3)/4*(3/4)**d) for d in range(7)],koch=koch)
    # Full small-multiple plots retain every requested stage, not just selected overlays.
    fig,axes=plt.subplots(3,5,figsize=(13,8))
    for depth,ax in enumerate(axes.flat):
        if depth>12: ax.axis('off'); continue
        z=c.dragon(depth); ax.plot(z.real,z.imag,linewidth=.5); ax.set_title('depth'+str(depth)); ax.set_aspect('equal'); ax.axis('off')
    savefig(fig,'dragon_stages')
    fig,axes=plt.subplots(2,7,figsize=(17,5))
    for depth in range(7):
        ax=axes[0,depth]; triangles=c.sierpinski_triangles(depth); ax.add_collection(PolyCollection(triangles,facecolors='black',edgecolors='none')); ax.autoscale_view(); ax.set_aspect('equal'); ax.set_title('gasket'+str(depth)); ax.axis('off')
        ax=axes[1,depth]
        if depth<6:
            z=c.koch_snowflake(depth); ax.plot(z.real,z.imag,linewidth=.6); ax.set_aspect('equal'); ax.set_title('Koch'+str(depth))
        ax.axis('off')
    savefig(fig,'gasket_koch_stages')

    a,b=c.gasket_ifs(); deterministic=c.deterministic_ifs(a,b,[[0,0],[1,0],[.5,np.sqrt(3)/2]],depth=6)
    game=c.chaos_game(a,b,probabilities=[1/3]*3,initial_point=[0,0],samples=20000,burn_in=100,seed=SEED)
    pa,pb=c.gasket_ifs(printed=True)
    printed=c.chaos_game(pa,pb,probabilities=[1/3]*3,initial_point=[0,0],samples=20000,burn_in=100,seed=SEED)
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,points,title in zip(axes,[deterministic,game.points,printed.points],['deterministic','seeded chaos game','printed affine coordinates']):
        ax.scatter(points[:,0],points[:,1],s=.1); ax.set_aspect('equal'); ax.set_title(title)
    savefig(fig,'ifs'); report['ifs']=dict(matrices=a,translations=b,norms=game.contraction_norms,samples=20000,burn_in=100,deterministic_depth=6,printed_matrices=pa,printed_translations=pb)
    csvfile('ifs_points',['x','y','map'],np.column_stack((game.points,game.map_indices)))

    report['escape']={}; fig,axes=plt.subplots(2,3,figsize=(12,7))
    for row,(name,x,y,parameter) in enumerate((('mandelbrot',np.linspace(-2,1,600),np.linspace(-1,1,400),None),('filled_julia',np.linspace(-1.5,1.5,500),np.linspace(-1.5,1.5,500),JULIA_C))):
        grid=x[None,:]+1j*y[:,None]; runs=[]
        for ax,budget in zip(axes[row],(32,128,512)):
            result=c.mandelbrot(grid,max_iterations=budget) if parameter is None else c.filled_julia(grid,c=parameter,max_iterations=budget)
            image=np.where(result.escaped,np.log1p(result.counts),0)
            ax.imshow(image,origin='lower',extent=(x[0],x[-1],y[0],y[-1]),cmap='magma',vmin=0,vmax=np.log1p(512),interpolation='nearest'); ax.set_title(name+' budget'+str(budget)); ax.set_xlabel('real'); ax.set_ylabel('imaginary')
            runs.append(dict(budget=budget,escaped=int(result.escaped.sum()),not_escaped_within_budget=int((~result.escaped & ~result.nonfinite).sum()),nonfinite=int(result.nonfinite.sum())))
            # Full grids retained as plain CSV; no image-membership ambiguity.
            np.savetxt(out/(name+'_counts_'+str(budget)+'.csv'),result.counts,fmt='%d',delimiter=',')
            np.savetxt(out/(name+'_escaped_'+str(budget)+'.csv'),result.escaped,fmt='%d',delimiter=',')
        report['escape'][name]=dict(real_bounds=[float(x[0]),float(x[-1])],imag_bounds=[float(y[0]),float(y[-1])],shape=grid.shape,parameter=parameter,runs=runs)
    fig.suptitle('Black: not escaped within budget; common log(1+updates) color scale')
    savefig(fig,'escape_images')

    report['weierstrass']=[]; fig,axes=plt.subplots(2,1,figsize=(11,7))
    coarse=np.linspace(0,2*np.pi,1025)
    for terms in (3,6,9):
        result=c.weierstrass(coarse,s=1.5,lam=4,terms=terms)
        axes[0].plot(coarse,result.values,label='terms'+str(terms),alpha=.7)
        report['weierstrass'].append(dict(terms=terms,s=1.5,lam=4,samples=len(coarse),spacing=coarse[1]-coarse[0],highest_frequency=result.frequencies[-1],tail_bound=result.tail_bound))
    # 32 samples per shortest wavelength at retained frequency4^6.
    dense=np.linspace(0,2*np.pi,131073); reference=c.weierstrass(dense,s=1.5,lam=4,terms=6)
    axes[0].set_title('Coarse grid: high terms alias'); axes[0].legend(); axes[1].plot(dense,reference.values,linewidth=.5); axes[1].set_title('Six terms,32 samples per shortest wavelength')
    for ax in axes: ax.set_xlabel('x'); ax.set_ylabel('finite sum')
    savefig(fig,'weierstrass'); report['weierstrass_reference']=dict(terms=6,samples=len(dense),spacing=dense[1]-dense[0],highest_frequency=4096)
    csvfile('weierstrass_dense',['x','value'],zip(dense,reference.values))

    walk=c.levy_flight(steps=10000,lam=2,b=2,max_level=12,seed=SEED)
    k=np.linspace(0,1,101); exact=c.levy_characteristic(k,lam=2,b=2,max_level=12)
    empirical=np.array([np.mean(np.cos(v*walk.increments)) for v in k]); empirical_imag=np.array([np.mean(np.sin(v*walk.increments)) for v in k])
    counts=np.bincount(walk.levels,minlength=13)
    report['levy']=dict(lam=2,b=2,max_level=12,steps=10000,omitted_probability=walk.omitted_probability,probabilities=walk.probabilities,counts=counts,k=k,characteristic=exact,empirical_real=empirical,empirical_imaginary=empirical_imag)
    csvfile('levy_walk',['step','increment','position','level','sign'],zip(np.arange(10000),walk.increments,walk.positions[1:],walk.levels,walk.signs))
    fig,axes=plt.subplots(1,3,figsize=(12,4)); axes[0].plot(walk.positions); axes[0].set_xlabel('step'); axes[0].set_ylabel('position'); axes[1].semilogy(np.arange(13),counts/10000,'o',label='observed'); axes[1].semilogy(np.arange(13),walk.probabilities,label='truncated target'); axes[1].legend(); axes[1].set_xlabel('level'); axes[2].plot(k,exact,label='exact'); axes[2].plot(k,empirical,'--',label='empirical real'); axes[2].plot(k,empirical_imag,label='empirical imaginary'); axes[2].set_xlabel('k'); axes[2].legend(); savefig(fig,'levy')

    # Geometry -> sampled/noisy cloud -> explicit box and correlation fits.
    rng=np.random.default_rng(SEED); records=[]; rows=[]
    def points_for(name,depth):
        if name=='cantor': return c.cantor_endpoints(c.cantor_intervals(depth))
        if name=='gasket': return np.unique(c.sierpinski_triangles(depth).reshape(-1,2),axis=0)
        if name=='koch':
            z=c.koch_curve(depth); return np.column_stack((z.real,z.imag))
        if name=='line': return np.column_stack((np.linspace(0,1,2048),np.zeros(2048)))
        grid=np.linspace(0,1,65); x,y=np.meshgrid(grid,grid); return np.column_stack((x.ravel(),y.ravel()))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name,depths in (('cantor',(6,9)),('gasket',(4,6)),('koch',(3,5)),('line',(0,)),('square',(0,))):
        for depth in depths:
            source=points_for(name,depth)
            for count in (512,2048):
                count_actual=min(count,len(source)); selection=rng.choice(len(source),count_actual,replace=False); base=source[selection]
                for noise in (0.,1e-4,1e-2):
                    points=base+rng.normal(0,noise,base.shape)
                    for scales in (2.**(-np.arange(1,9)),3.**(-np.arange(1,7))):
                        for shift in (0.,.137):
                            origin=np.full(points.shape[1],shift)
                            boxes=c.box_counts(points,scales=scales,origin=origin)
                            fit=c.fit_dimension(scales,boxes.counts,fit_slice=slice(1,5),kind='box')
                            records.append(dict(object=name,depth=depth,requested_samples=count,samples=count_actual,noise_std=noise,origin=origin,method='box',curve=boxes,fit=fit))
                            rows.append([name,depth,count_actual,noise,shift,'box',fit.slope])
                            if depth==depths[-1] and count==2048 and noise==0 and shift==0: axes[0].plot(fit.log_abscissa,fit.log_values,label=name+' '+str(len(scales)))
                    radii=2.**np.arange(-8,0,dtype=float)
                    pairs=c.correlation_integral(points,radii=radii,theiler=0)
                    # Explicit slice; unavailable zero counts are recorded, never auto-shifted.
                    try: fit=c.fit_dimension(radii,pairs.values,fit_slice=slice(2,6),kind='correlation')
                    except ValueError: fit=None
                    records.append(dict(object=name,depth=depth,requested_samples=count,samples=count_actual,noise_std=noise,theiler=0,method='correlation',curve=pairs,fit=fit))
                    rows.append([name,depth,count_actual,noise,0,'correlation',fit.slope if fit else None])
                    if fit is not None and depth==depths[-1] and count==2048 and noise==0: axes[1].plot(fit.log_abscissa,fit.log_values,label=name)
    axes[0].set_xlabel('log(1/scale)'); axes[0].set_ylabel('log(box count)'); axes[1].set_xlabel('log(radius)'); axes[1].set_ylabel('log(pair fraction)')
    for ax in axes: ax.legend(fontsize=7)
    savefig(fig,'dimensions'); csvfile('dimension_fits',['object','depth','samples','noise','origin_shift','method','slope'],rows)
    report['dimensions']=records
    (out/'report.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('escape budgets',report['escape'],flush=True); print('dimension records',len(records),flush=True); print('Saved',out,flush=True)


if __name__=='__main__': main()
