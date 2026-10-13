"""Boolean classification, explicit XOR, and gradient-validated XOR learning."""
from dataclasses import fields
import numpy as np
from nonlinear_workbook.chapter11 import *
from nonlinear_workbook.chapter11.fixtures import (BOOLEAN_INPUTS,XOR_TARGETS,XOR_HIDDEN_WEIGHTS,
    XOR_HIDDEN_BIAS,XOR_OUTPUT_WEIGHTS,XOR_OUTPUT_BIAS)
from common import setup,save_report,save_plot,csv,plt,SEED


def main():
    out,report=setup('feedforward')
    report['and']=perceptron_train(BOOLEAN_INPUTS,[0,0,0,1],seed=SEED)
    points=np.array([[2,0],[1.5,1.5],[0,1],[1,0],[0,0]])
    report['separator']=perceptron_train(points,[1,1,0,0,0],seed=SEED)
    report['single_perceptron_xor']=perceptron_train(BOOLEAN_INPUTS,XOR_TARGETS[:,0],max_epochs=100,seed=SEED)
    report['explicit_xor']=xor_network(BOOLEAN_INPUTS)
    report['boolean_functions']=[]
    for code in range(16):
        targets=np.array([(code >> (3-j)) & 1 for j in range(4)])
        result=perceptron_train(BOOLEAN_INPUTS,targets,max_epochs=300,seed=SEED)
        report['boolean_functions'].append(dict(code=code,targets=targets,result=result))

    initial=MLPParameters(XOR_HIDDEN_WEIGHTS,XOR_HIDDEN_BIAS,XOR_OUTPUT_WEIGHTS,XOR_OUTPUT_BIAS)
    report['mlp']={}
    for mode in ('online','batch'):
        result=mlp_train(BOOLEAN_INPUTS,XOR_TARGETS,initial=initial,hidden_slope=10,output_slope=10,
            learning_rate=.05,seed=SEED,mode=mode,max_epochs=10000,tolerance=.00025)
        report['mlp'][mode]=result
        csv(out,mode+'_loss',np.column_stack([np.arange(len(result.losses)),result.losses]),'epoch,mean_half_squared_error')
        csv(out,mode+'_predictions',np.column_stack([BOOLEAN_INPUTS,XOR_TARGETS,result.predictions]),'x0,x1,target,prediction')
    rng=np.random.default_rng(SEED);x=rng.normal(size=(3,2));target=rng.random((3,2))
    weights=mlp_initialize(2,hidden_width=3,n_outputs=2,seed=SEED)
    _,grad=mlp_loss_and_gradients(x,target,weights,hidden_slope=1.3,output_slope=2.1)
    errors={}
    for field in fields(weights):
        analytic=getattr(grad,field.name);numeric=np.empty_like(analytic)
        for index in np.ndindex(analytic.shape):
            values={f.name:getattr(weights,f.name).copy() for f in fields(weights)}
            values[field.name][index]+=1e-6
            plus=mlp_loss_and_gradients(x,target,MLPParameters(**values),hidden_slope=1.3,output_slope=2.1)[0]
            values[field.name][index]-=2e-6
            minus=mlp_loss_and_gradients(x,target,MLPParameters(**values),hidden_slope=1.3,output_slope=2.1)[0]
            numeric[index]=(plus-minus)/2e-6
        errors[field.name]=float(np.max(np.abs(analytic-numeric)))
    report['gradient_check']=dict(absolute_errors=errors,finite_difference_step=1e-6,hidden_slope=1.3,output_slope=2.1,absolute_tolerance=1e-6,relative_tolerance=1e-5)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for mode,result in report['mlp'].items():
        axes[0].semilogy(np.arange(len(result.losses)),result.losses,label=f'{mode}: {result.status}')
        axes[1].plot(np.arange(4),result.predictions[:,0],'o-',label=mode)
    axes[0].axhline(.00025,color='black',ls='--',lw=.7);axes[0].set(title='XOR training, current-weight gradients',xlabel='epoch',ylabel='mean half-squared error');axes[0].legend(fontsize=8)
    axes[1].plot(np.arange(4),XOR_TARGETS[:,0],'s--',color='black',label='target');axes[1].set(title='Four XOR outputs',xlabel='Boolean input',ylabel='output',xticks=np.arange(4),xticklabels=['00','01','10','11']);axes[1].legend()
    save_plot(out,'xor_training',fig)
    gx,gy=np.meshgrid(np.linspace(-.25,2.25,120),np.linspace(-.25,2.25,120))
    grid=np.column_stack([gx.ravel(),gy.ravel()]);separator=report['separator']
    prediction=perceptron_predict(grid,separator.weights,bias=separator.bias).reshape(gx.shape)
    fig,axis=plt.subplots(figsize=(6,5),layout='constrained')
    axis.contourf(gx,gy,prediction,levels=[-.5,.5,1.5],colors=['#d4e3ee','#f1d7a4'])
    axis.scatter(points[:2,0],points[:2,1],marker='o',color='#a34225',label='positive')
    axis.scatter(points[2:,0],points[2:,1],marker='x',color='#235979',label='negative')
    axis.set(title='Learned geometric separator',xlabel='x0',ylabel='x1');axis.legend()
    save_plot(out,'perceptron_separator',fig)
    save_report(out,report)

if __name__=='__main__':main()
