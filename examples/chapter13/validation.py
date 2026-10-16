"""Chapter 13 demonstrations. All outputs are synthetic/book fixtures."""
import argparse
import csv
from fractions import Fraction
import json
from pathlib import Path
import platform
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nonlinear_workbook import chapter13 as c

SEED=20261016

def write_csv(path,header,rows):
    with path.open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(header);writer.writerows(rows)


def sets_demo(out):
    x=np.linspace(0,40,401);hot=c.logistic(x,24,2);pleasant=c.gaussian(x,24,1)
    report={}
    a=c.FuzzySet(np.arange(1,6),[.4,.8,1,.7,.7]);b=c.FuzzySet(np.arange(1,6),[0,.1,1,.2,.1]);third=c.FuzzySet(np.arange(1,6),[.1,.2,.5,.2,.4]).normalize()
    report['discrete']={op:a.operation(b,operator=op).membership.tolist() for op in ('min','max','product','algebraic_sum','bounded_sum','bounded_difference','fuzzy_and','fuzzy_or')}
    report['discrete_inputs']=dict(universe=a.universe.tolist(),a=a.membership.tolist(),b=b.membership.tolist())
    report['complement']=c.complement(a.membership).tolist()
    report['means']=dict(weighted=c.weighted_mean([a.membership,b.membership],[.2,.8]).tolist(),three=c.weighted_mean([a.membership,b.membership,third.membership]).tolist())
    report['third']=dict(membership=third.membership.tolist(),discrete_centroid=third.centroid(),continuous_grid_centroid=third.centroid(mode='continuous'),cardinality=third.cardinality,relative_cardinality=third.relative_cardinality,entropy=c.entropy(third.membership),contrast_entropy=c.entropy(c.contrast(third.membership)),alpha_065=third.alpha_cut(.65).tolist(),top3=third.top_beta(3).tolist())
    report['cruising_speed']=dict(grid=[80,100,110,120,130,140],triangle=c.triangle([80,100,110,120,130,140],100,110,120).tolist(),trapezoid=c.trapezoid([80,100,110,120,130,140],80,100,120,130).tolist(),units='km/hour')
    ram=[4,8,16,32,64,128,256,512];efficient=np.array([0,0,.2,.5,1,.8,.5,.2]);powerful=np.array([0,0,.2,.3,.4,.5,.8,1])
    report['computer_sets']=dict(ram_MB=ram,efficient=efficient.tolist(),powerful=powerful.tolist(),both=np.minimum(efficient,powerful).tolist(),either=np.maximum(efficient,powerful).tolist(),not_powerful=(1-powerful).tolist())
    report['max_min_vectors']=float(np.max(np.min([[3.1,4.5,1.7],[1.1,6.7,5.1],[.9,.5,4.9],[1.1,.7,6.1]],axis=1)))
    report['compatibility']=c.gaussian_compatibility(1,2,4,1)
    report['temperature_membership_parameters']=dict(domain=[0,40],points=401,hot_logistic_center=24,hot_logistic_width=2,pleasant_gaussian_center=24,pleasant_gaussian_sigma=1)
    report['temperature_units']='degree Celsius';report['degrees_are_not_probabilities']=True
    write_csv(out/'temperature_membership.csv',['temperature','hot','pleasant','and','or'],zip(x,hot,pleasant,np.minimum(hot,pleasant),np.maximum(hot,pleasant)))
    plt.figure(figsize=(12,4));plt.subplot(131)
    for label,y in [('hot',hot),('pleasant',pleasant),('AND',np.minimum(hot,pleasant))]:plt.plot(x,y,label=label)
    plt.xlabel('temperature (degree Celsius)');plt.ylabel('membership');plt.legend()
    plt.subplot(132)
    t=np.linspace(0,1,301)
    for exponent in (.25,.5,1,2,4):plt.plot(t,c.hedge(t,exponent),label=str(exponent))
    plt.xlabel('original grade');plt.ylabel('hedged grade');plt.legend(title='exponent')
    plt.subplot(133)
    for kind in ('linear','exponential','gaussian','polynomial','polynomial_outer'):plt.plot(np.linspace(0,3,301),c.spread(np.linspace(0,3,301),kind=kind),label=kind)
    plt.xlabel('normalized distance');plt.ylabel('LR spread membership');plt.legend()
    plt.tight_layout();plt.savefig(out/'memberships.png',dpi=140);plt.close()
    report['centroid_refinement']=[]
    for n in (101,401,1601):
        grid=np.linspace(0,3,n);s=c.FuzzySet(grid,c.triangle(grid,0,1,3));value=s.centroid(mode='continuous')
        report['centroid_refinement'].append(dict(points=n,value=value,target=4/3,error=abs(value-4/3)))
    # Fixed seed supplies deterministic bounded operator property diagnostics, not data evidence.
    rng=np.random.default_rng(SEED);aa,bb=rng.random((2,1000))
    report['random_operator_check']=dict(seed=SEED,samples=1000,max_de_morgan_error=float(np.max(abs(1-np.minimum(aa,bb)-np.maximum(1-aa,1-bb)))))
    return report


def arithmetic_demo(out):
    report={}
    fixtures=[('square',[[-1,0,1,2]],[[.5,.8,1,.4]],lambda x:x*x),('sum',[[1,2,3,4],[3,4,5]],[[.2,1,.5,.3],[.9,1,.4]],lambda x,y:x+y),('product',[[1,2,3],[2,3,4]],[[.3,1,.4],[.7,1,.2]],lambda x,y:x*y)]
    for name,x,m,fn in fixtures:report[name]=list(c.extension(x,m,fn).items())
    levels=np.linspace(0,1,101);rows=[];report['approximation_sensitivity']=[]
    plt.figure(figsize=(12,4));plt.subplot(131)
    for width in (.5,.25,.125):
        a=c.Triangular(2,width,width);b=c.Triangular(3,width,width)
        exact=a.cuts(levels).operation(b.cuts(levels),operator='multiply');approx=a.approximate_product(b).cuts(levels)
        error=float(np.max(abs(exact.bounds-approx.bounds)))
        report['approximation_sensitivity'].append(dict(width=width,max_endpoint_error=error))
        rows.extend([['product',width,float(alpha),*e,*p] for alpha,e,p in zip(levels,exact.bounds,approx.bounds)])
        if width==.5:
            plt.plot(exact.bounds[:,0],levels,'b-',label='exact cut endpoints');plt.plot(exact.bounds[:,1],levels,'b-');plt.plot(approx.bounds[:,0],levels,'r--',label='LR approximation');plt.plot(approx.bounds[:,1],levels,'r--')
    plt.xlabel('product value');plt.ylabel('alpha');plt.legend()
    a=c.Triangular(2,.5,.5);b=c.Triangular(3,.5,.5);q=a.approximate_divide(b)
    report['quotient']=dict(mode=q.mode,left=q.left,right=q.right,printed_right=.2619,corrected_right=.3)
    ex=a.cuts(levels).operation(b.cuts(levels),operator='divide');ap=q.cuts(levels)
    rows.extend([['division',.5,float(alpha),*e,*p] for alpha,e,p in zip(levels,ex.bounds,ap.bounds)])
    plt.subplot(132);plt.plot(ex.bounds[:,0],levels,'b-',label='exact cuts');plt.plot(ex.bounds[:,1],levels,'b-');plt.plot(ap.bounds[:,0],levels,'r--',label='approximate triangle');plt.plot(ap.bounds[:,1],levels,'r--');plt.xlabel('quotient');plt.ylabel('alpha');plt.legend()
    voltage=c.Triangular(120,10,10);current=c.Triangular(10,.2,.2);power=voltage.approximate_product(current)
    exact_power=voltage.cuts(levels).operation(current.cuts(levels),operator='multiply')
    report['power']=dict(units='watt',voltage_triangle=[120,10,10],voltage_unit='volt',current_triangle=[10,.2,.2],current_unit='ampere',approximate=[power.mode,power.left,power.right],exact_support=exact_power.bounds[0].tolist(),assumption='voltage and current possibilities combine independently')
    # pV constant, but p1,V1,V2 are independently varying fuzzy inputs here.
    v1=[Fraction(i,4) for i in range(2,7)];p1=[Fraction(1,2),Fraction(7,4),Fraction(2),Fraction(9,4),Fraction(5,2)]
    v2=[Fraction(i,20) for i in range(8,13)];m=[0,.5,1,.5,0]
    gas=c.extension([v1,p1,v2],[m,m,m],lambda v,p,w:p*v/w,max_combinations=125)
    report['gas']=dict(v1=[str(v) for v in v1],p1=[str(v) for v in p1],v2=[str(v) for v in v2],input_grades=m,volume_unit='common unspecified book volume unit',output=[[str(k),v] for k,v in gas.items()],units='same pressure unit as p1',assumption='independent input possibilities; no extra dependence information')
    plt.subplot(133);plt.scatter([float(k) for k in gas],list(gas.values()),s=12);plt.xlabel('pressure');plt.ylabel('finite extension grade')
    plt.tight_layout();plt.savefig(out/'arithmetic.png',dpi=140);plt.close()
    write_csv(out/'arithmetic_cuts.csv',['operation','width','alpha','exact_left','exact_right','approx_left','approx_right'],rows)
    return report


def decision_demo(out):
    report={};looks=np.array([.7,1,.4,.7,.5]);comfort=np.array([.4,0,.8,.6,.9]);valid=np.array([1,0,1,1,0])
    report['shoes']=[]
    for name,objective in [('equal_preferences',(looks+comfort)/2),('hedged',(np.sqrt(looks)+comfort**2)/2)]:
        scores=np.minimum(valid,objective);report['shoes'].append(dict(scenario=name,scores=scores.tolist(),winner=int(np.argmax(scores))))
    report['pricing']=[];rows=[]
    scenarios=[(2,1.5,.25,4,2),(2,4,.25,4,4)]
    for n in (101,401,1601,3001):
        x=np.linspace(2000,5000,n)
        base=[c.ramp(x,2000,5000),c.ramp(x,2000,5000,increasing=False),c.gaussian(x,3250,300),c.logistic(x,2500,300),1-c.logistic(x,4500,300)]
        for scenario,exponents in enumerate(scenarios):
            objectives=[c.hedge(b,e) for b,e in zip(base,exponents)]
            for gamma in (0,.5,1):
                value=objectives[0]
                for objective in objectives[1:]:value=c.combine(value,objective,operator='fuzzy_and',gamma=gamma)
                best=int(np.argmax(value));record=dict(points=n,scenario=scenario,gamma=gamma,price=float(x[best]),grade=float(value[best]));report['pricing'].append(record)
                rows.append([n,scenario,gamma,x[best],value[best]])
    report['pricing_conventions']=dict(unit='Rand (book fixture)',range=[2000,5000],cost=2500,competitor=4500,gaussian_sigma=300,scenario_hedges=scenarios,aggregation='gamma=1 reproduces book logical AND; other gamma values use left-associated fuzzy AND, which is not generally associative')
    # Language listing uses an interpolated 200-point universe [0,10].
    report['language_parameters']=dict(domain=[0,10],semicolon_center=5,semicolon_width=1,curly_hedge=.5,extend_hedge=.25,AND='minimum',NOT='one minus grade',interpolation='linear',outside='raise')
    report['language']=[];report['language_note']='p.576 third extend token is malformed (0.0.5); both 0 and .5 are shown explicitly, not silently repaired'
    for n in (101,200,401,1601):
        grid=np.linspace(0,10,n);many=c.ramp(grid,0,10);ext=c.FuzzySet(grid,c.hedge(many,.25));curly=c.FuzzySet(grid,c.hedge(many,.5));semi=c.FuzzySet(grid,c.logistic(grid,5,1));beg=c.FuzzySet(grid,1-many)
        for inputs in [(7,5,.2,.1),(7,5,.2,1.5),(4,3,.2,0.),(4,3,.2,.5)]:
            se,cu,be,ex=inputs;java=float(ext.at(ex));cpp=min(float(semi.at(se)),float(curly.at(cu)),float(beg.at(be)),1-java)
            report['language'].append(dict(points=n,inputs=inputs,java_grade=java,cpp_grade=cpp))
    write_csv(out/'pricing_sensitivity.csv',['points','scenario','gamma','price','grade'],rows)
    plt.figure(figsize=(11,4));plt.subplot(121)
    for i,s in enumerate(report['shoes']):plt.plot(range(5),s['scores'],'o-',label=s['scenario'])
    plt.xlabel('shoe index');plt.ylabel('score');plt.legend()
    plt.subplot(122)
    for scenario in (0,1):
        vals=[r for r in report['pricing'] if r['scenario']==scenario and r['points']==3001];plt.plot([r['gamma'] for r in vals],[r['price'] for r in vals],'o-',label=f'scenario {scenario}')
    plt.xlabel('gamma');plt.ylabel('selected price (Rand)');plt.legend();plt.tight_layout();plt.savefig(out/'decisions.png',dpi=140);plt.close()
    return report


def rule_demo(out):
    report={};grid=np.linspace(-3,3,401)
    simple_terms={'angle':{'small_positive':lambda x:c.triangle(x,0,15,30)}}
    simple_output={'small_negative':c.FuzzySet(grid,c.triangle(grid,-2,-1,0))}
    simple_rules=[c.Rule((('angle','small_positive'),),'small_negative')]
    report['single_pendulum_rule']=[dict(angle=a,**c.infer({'angle':a},simple_terms,simple_output,simple_rules)) for a in (-5.,0.,7.5,15.,25.,30.)]
    names=['LN','SN','Z','SP','LP'];centers=[-30,-15,0,15,30]
    terms={'angle':{name:(lambda x,m=m:c.triangle(x,m-15,m,m+15)) for name,m in zip(names,centers)}}
    outputs={name:c.FuzzySet(grid,c.triangle(grid,m-1.5,m,m+1.5)) for name,m in zip(names,[-3,-1.5,0,1.5,3])}
    # Shoulder output terms at the finite actuator bounds are an explicit modelling choice.
    outputs['LN']=c.FuzzySet(grid,c.trapezoid(grid,-3,-3,-3,-1.5));outputs['LP']=c.FuzzySet(grid,c.trapezoid(grid,1.5,3,3,3))
    rules=[c.Rule((('angle',name),),opposite) for name,opposite in zip(names,reversed(names))]
    rows=[];report['pendulum']=[]
    for method in ('clip','product'):
        for angle in np.linspace(-50,50,201):
            r=c.infer({'angle':float(angle)},terms,outputs,rules,implication=method)
            rows.append([angle,method,r['status'],r['value']]);report['pendulum'].append(dict(angle=float(angle),**r))
    report['pendulum_parameters']=dict(angle_centers=centers,angle_halfwidth=15,force_centers=[-3,-1.5,0,1.5,3],force_halfwidth=1.5,force_extremes='shoulders at actuator bounds',rule_pairs=list(zip(names,reversed(names))),connective='min',aggregation='max',centroid='sampled trapezoidal integration',output_points=len(grid))
    report['pendulum_note']='force inference only, degrees and arbitrary force units; no plant dynamics or stability claim'
    # Temperature membership geometry is supplied, not specified by the source.
    temp=np.linspace(-10,10,201);duty=np.linspace(0,100,401)
    labels=['cold','cool','right','hot'];terms_t={'error':{name:(lambda x,m=m:c.triangle(x,m-7,m,m+7)) for name,m in zip(labels,[-10,-3,3,10])}}
    outputs_t={name:c.FuzzySet(duty,c.triangle(duty,max(0,m-30),m,min(100,m+30))) for name,m in zip(['large','medium','small','zero'],[100,65,30,0])}
    rules_t=[c.Rule((('error',a),),b) for a,b in zip(labels,['large','medium','small','zero'])]
    report['temperature']=[dict(error=float(e),**c.infer({'error':float(e)},terms_t,outputs_t,rules_t)) for e in temp]
    report['temperature_parameters']=dict(error_centers=[-10,-3,3,10],error_halfwidth=7,duty_centers=[100,65,30,0],duty_halfwidth=30,duty_bounds=[0,100],rule_pairs=list(zip(labels,['large','medium','small','zero'])),implication='clip',aggregation='max',output_points=len(duty))
    report['temperature_note']='synthetic membership choices; error = current minus setpoint, Celsius; duty percent; no derivative input or thermal plant supplied'
    write_csv(out/'force_inference.csv',['angle_degrees','implication','status','force'],rows)
    plt.figure(figsize=(11,4));plt.subplot(121)
    for method in ('clip','product'):
        vals=[r for r in report['pendulum'] if r['implication']==method];plt.plot([r['angle'] for r in vals],[np.nan if r['value'] is None else r['value'] for r in vals],label=method)
    plt.xlabel('angle (degree)');plt.ylabel('force (arbitrary units)');plt.legend()
    plt.subplot(122);plt.plot(temp,[r['value'] for r in report['temperature']]);plt.xlabel('temperature error (degree Celsius)');plt.ylabel('duty (%)');plt.tight_layout();plt.savefig(out/'rule_outputs.png',dpi=140);plt.close()
    report['singleton']=c.singleton_output([0,50,100],[.2,.6,.2])
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='outputs/chapter13');parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    report=dict(python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__,seed=SEED,source='printed pp.495-576; PDF not redistributed',conventions='membership grades, not probabilities; exact finite keys, explicit cut levels and inference policies')
    if args.smoke:
        assert c.extension([[0,1]],[[.5,1]],lambda x:x*x)=={0:.5,1:1.}
        assert abs(c.Triangular(2,.5,.5).approximate_divide(c.Triangular(3,.5,.5)).right-.3)<1e-12
        assert abs(c.FuzzySet([0,1,2],[0,1,0]).centroid()-1)<1e-12
        report['status']='smoke passed'
    else:
        report.update(sets=sets_demo(out),arithmetic=arithmetic_demo(out),decisions=decision_demo(out),inference=rule_demo(out))
    (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False));print('Wrote',out)

if __name__=='__main__':main()
