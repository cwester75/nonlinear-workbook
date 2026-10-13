"""Kohonen worked updates, prototype-count sensitivity, and eight-city TSP."""
import numpy as np
from nonlinear_workbook.chapter11 import *
from nonlinear_workbook.chapter11.fixtures import KOHONEN_INPUTS,KOHONEN_INITIAL,CITIES,BOOK_TOUR
from common import setup,save_report,save_plot,csv,plt,SEED


def main():
    out,report=setup('kohonen')
    weights=KOHONEN_INITIAL.copy(); worked=[]
    for x in KOHONEN_INPUTS:
        weights,winner=kohonen_update(weights,x,learning_rate=.6)
        worked.append(dict(observation=x,winner=winner,prototypes=weights.copy()))
    report['book_first_epoch']=dict(initial=KOHONEN_INITIAL,learning_rate=.6,updates=worked)
    report['prototype_counts']=[]
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for n in range(1,6):
        result=kohonen_train(KOHONEN_INPUTS,n_neurons=n,updates=4000,learning_rate=(.5,.01),seed=SEED,check_every=40)
        report['prototype_counts'].append(result)
        axes[0].plot(result.checkpoint_updates,result.quantization_errors,label=f'{n} prototype' + ('s' if n != 1 else ''))
        csv(out,f'quantization_n{n}',np.column_stack([result.checkpoint_updates,result.quantization_errors]),'updates,mean_distance')
    axes[0].set(title='Prototype count and quantization',xlabel='online updates',ylabel='mean nearest distance');axes[0].legend(fontsize=8)
    two=report['prototype_counts'][1]
    image=axes[1].imshow(two.prototypes,aspect='auto',vmin=0,vmax=1,cmap='viridis')
    axes[1].set(title='Two learned four-dimensional prototypes',xlabel='feature index',ylabel='prototype index',xticks=np.arange(4),yticks=np.arange(2));fig.colorbar(image,ax=axes[1],label='weight')
    save_plot(out,'clustering',fig)
    angle=2*np.pi*np.arange(24)/24
    initial=np.column_stack([10+10*np.cos(angle),10+10*np.sin(angle)])
    result=kohonen_train(CITIES,n_neurons=24,updates=15000,learning_rate=(.8,.05),
        width=(12,.5),topology='ring',initial=initial,seed=SEED,check_every=100)
    tour=decode_ring_tour(CITIES,result.prototypes)
    optimum=exact_tsp(CITIES)
    length=tour_length(CITIES,tour);book_length=tour_length(CITIES,BOOK_TOUR)
    report['tsp']=dict(cities=CITIES,training=result,tour=tour,length=length,
        optimum=optimum,gap=length-optimum.length,gap_percent=100*(length/optimum.length-1),
        book_tour=BOOK_TOUR,book_length=book_length,book_gap=book_length-optimum.length,
        initialization=dict(center=[10,10],radius=10,neurons=24),decode='winner index, city-index ties; city 0 fixed')
    csv(out,'tour',np.column_stack([tour,CITIES[tour]]),'city,x,y')
    csv(out,'tsp_quantization',np.column_stack([result.checkpoint_updates,result.quantization_errors]),'updates,mean_distance')
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for axis,path,title in [(axes[0],tour,f'Ring heuristic: {length:.4f}'),(axes[1],optimum.tour,f'Exact optimum: {optimum.length:.4f}')]:
        axis.plot(CITIES[path,0],CITIES[path,1],'o-',lw=1.5)
        for i,(x,y) in enumerate(CITIES):axis.annotate(str(i),(x,y),xytext=(5,5),textcoords='offset points')
        axis.set(title=title,xlabel='x',ylabel='y');axis.set_aspect('equal')
    ring=np.vstack([result.prototypes,result.prototypes[0]])
    axes[0].plot(ring[:,0],ring[:,1],'--',alpha=.5,label='learned neuron ring');axes[0].legend()
    save_plot(out,'tsp_comparison',fig)
    save_report(out,report)

if __name__=='__main__':main()
