"""Bounded cross-chapter numerical smoke checks, suitable for CI."""
import argparse
import json
import platform
from pathlib import Path
import numpy as np
from nonlinear_workbook.chapter10 import integrate,central_difference
from nonlinear_workbook.chapter03 import van_der_pol
from nonlinear_workbook.chapter04 import toda_rhs,mathieu_floquet
from nonlinear_workbook.chapter05 import lorenz
from nonlinear_workbook.chapter06 import anharmonic
from nonlinear_workbook.chapter07 import rossler_drive_response
from nonlinear_workbook.flow_validation import conditional_flow_spectrum

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',default='outputs/release-smoke'); out=Path(p.parse_args().output); out.mkdir(parents=True,exist_ok=True)
    solver=dict(method='rkf45_adaptive',step=.01,max_step=.02,atol=1e-9,rtol=1e-9)
    fixtures=[('planar',lambda t,y:van_der_pol(t,y,mu=1),[1,0]),
        ('toda',lambda t,y:toda_rhs(t,y,periodic=True),[-.3,.1,.2,.2,-.1,-.1]),
        ('lorenz',lambda t,y:lorenz(t,y,sigma=16,r=40,b=4),[.8,.8,.8]),
        ('driven',lambda t,y:anharmonic(t,y,a=1,b=-10,c=100,k1=0,k2=1.2,omega=3.5),[1,0]),
        ('response',lambda t,y:rossler_drive_response(t,y,a=.2,b=.2,c=9),[.8,.4,.8,1.1,2.3])]
    results=[]
    for name,rhs,y in fixtures:
        r=integrate(rhs,y,t_span=(0,1),**solver)
        assert r.status=='completed' and np.isfinite(r.states).all()
        results.append(dict(name=name,status=r.status,evaluations=r.evaluations,final_state=r.states[-1].tolist()))
    overflow=integrate(lambda t,y:toda_rhs(t,y,periodic=True),[1000,-1000,0,0,0,0],t_span=(0,1),**solver)
    assert overflow.status=='nonfinite' and overflow.evaluations==1
    _,_,status=central_difference(lambda y:(_ for _ in ()).throw(AssertionError('escaped RHS evaluated')),100,100,step=.1,n=2,escape_radius=1)
    assert status=='escaped'
    f=mathieu_floquet(delta=0,epsilon=0,solver_parameters=solver)
    assert np.max(abs(f['monodromy']-[[1,np.pi],[0,1]]))<1e-7
    r=conditional_flow_spectrum(lambda t,y:np.r_[1.,-.5*y[1]],lambda t,y:np.array([[-.5]]),[0,1],response_dimension=1,burn_in=.1,duration=1,qr_interval=.1,initial_basis=np.eye(1),solver_parameters=solver)
    assert abs(r.exponents[0]+.5)<1e-7
    (out/'results.json').write_text(json.dumps(dict(python=platform.python_version(),numpy=np.__version__,fixtures=results,overflow_status=overflow.status,initial_escape=status),indent=2))
    print('Release smoke passed:',out)

if __name__=='__main__': main()
