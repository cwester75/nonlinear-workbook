"""Hopfield recall, finite-state basins, similarity, and Fourier invariance."""
import itertools
import numpy as np
from nonlinear_workbook.chapter11 import *
from nonlinear_workbook.chapter11.fixtures import FIVE_PATTERNS,DIGIT_PATTERNS,DIGIT_LABELS,DIGIT_CORRUPTED
from common import setup,save_report,save_plot,csv,plt,SEED


def main():
    out,report = setup('memory')
    weights = hebbian_weights(FIVE_PATTERNS)
    report['five_neuron'] = dict(patterns=FIVE_PATTERNS,weights=weights,
        energies=[hopfield_energy(weights,s) for s in FIVE_PATTERNS],
        recall=hopfield_recall(weights,np.ones(5),mode='synchronous'))
    enumeration = []
    for mode in ('synchronous','asynchronous'):
        for state in itertools.product((-1,1),repeat=5):
            result = hopfield_recall(weights,state,mode=mode)
            enumeration.append(dict(initial=state,result=result,**classify_recall(result.state,FIVE_PATTERNS)))
    report['enumerated_states'] = enumeration
    report['basin_radii'] = []
    for mode in ('synchronous','asynchronous'):
        subset = [v for v in enumeration if v['result'].parameters['mode'] == mode]
        for index,pattern in enumerate(FIVE_PATTERNS):
            long_failures = [hamming_distance(v['initial'],pattern) for v in subset
                if v['result'].status != 'converged' or not np.array_equal(v['result'].state,pattern)]
            direct_failures = [hamming_distance(v['initial'],pattern) for v in subset
                if not np.array_equal(v['result'].states[1],pattern)]
            report['basin_radii'].append(dict(mode=mode,pattern=index,
                long_term_radius=min(long_failures)-1 if long_failures else 5,
                direct_radius=min(direct_failures)-1 if direct_failures else 5,
                units='Hamming distance',enumerated_configurations=32))
    net = float(np.array([-.5,.7,-1.1,4.5,1.5]) @ np.array([1,.7,1.2,1.5,-4.5]))
    report['activation_fixture'] = dict(printed_page=341,net=net,heaviside=threshold(net),
        bipolar=bipolar_activation(net),logistic=logistic(net),tanh=tanh_activation(net))

    csv(out,'basins',np.array([[0 if v['result'].parameters['mode']=='synchronous' else 1,
        *v['initial'],*v['result'].state,v['result'].iterations,0 if v['result'].status=='converged' else 1] for v in enumeration]),
        'mode,s0,s1,s2,s3,s4,final0,final1,final2,final3,final4,iterations,status_0converged_1cycle')
    report['synchronous_two_cycle'] = hopfield_recall([[0,1],[1,0]],[1,-1],mode='synchronous')
    digit_weights = hebbian_weights(DIGIT_PATTERNS)
    report['digits'] = dict(labels=DIGIT_LABELS,patterns=DIGIT_PATTERNS,
        book_corrupted=DIGIT_CORRUPTED,book_recall=hopfield_recall(digit_weights,DIGIT_CORRUPTED,mode='synchronous'),
        parameters=dict(shape=[8,5],flipped_indices=[0,5,10],max_sweeps=100,zero_field='hold'),recalls=[])
    fig,axes = plt.subplots(3,3,figsize=(8,10),layout='constrained')
    for row,(pattern,label) in enumerate(zip(DIGIT_PATTERNS,DIGIT_LABELS)):
        corrupted = pattern.copy(); corrupted[[0,5,10]] *= -1
        result = hopfield_recall(digit_weights,corrupted,mode='asynchronous')
        report['digits']['recalls'].append(dict(label=label,corrupted=corrupted,result=result,**classify_recall(result.state,DIGIT_PATTERNS)))
        for axis,image,title in zip(axes[row],[pattern,corrupted,result.state],[f'Stored {label}','Three bit flips',f'Recall: {result.status}']):
            axis.imshow(image.reshape(8,5),cmap='gray_r',vmin=-1,vmax=1,interpolation='nearest'); axis.set_title(title); axis.set_xticks([]); axis.set_yticks([])
    save_plot(out,'digit_recall',fig)
    fig,axes = plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for mode in ('synchronous','asynchronous'):
        result = hopfield_recall(weights,[1,-1,1,-1,1],mode=mode)
        axes[0].plot(result.energies,'o-',label=mode)
    axes[0].set(title='Five-neuron energy histories',xlabel='recorded update',ylabel='energy'); axes[0].legend()
    cycle = report['synchronous_two_cycle']
    axes[1].plot(cycle.states[:,0],'o-',label='neuron 0'); axes[1].plot(cycle.states[:,1],'o-',label='neuron 1')
    axes[1].set(title='Synchronous two-cycle',xlabel='simultaneous step',ylabel='state'); axes[1].legend()
    save_plot(out,'energy_and_cycle',fig)
    image = DIGIT_PATTERNS[0].reshape(8,5)
    shifted = np.roll(image,(2,1),axis=(0,1)); reflected = image[::-1,::-1]
    features = [fourier_magnitude(v) for v in (image,shifted,reflected)]
    report['fourier'] = dict(parameters=dict(shift=[2,1],normalization='forward 1/(rows*columns)',boundary='circular'),
        shift_pixel_distance=euclidean_distance(image.ravel(),shifted.ravel()),
        shift_feature_error=float(np.max(np.abs(features[0]-features[1]))),
        reflection_feature_error=float(np.max(np.abs(features[0]-features[2]))))
    report['similarity_example'] = dict(hamming=hamming_distance([1,-1,1,1],[-1,1,1,-1]),
        directed_hausdorff=directed_hausdorff([[0],[1],[4]],[[0],[1]]),
        symmetric_hausdorff=hausdorff_distance([[0],[1],[4]],[[0],[1]]))
    fig,axes = plt.subplots(2,3,figsize=(10,7),layout='constrained')
    for col,(pixels,magnitude,title) in enumerate(zip((image,shifted,reflected),features,('Original','Circular shift','Reflection'))):
        axes[0,col].imshow(pixels,cmap='gray_r',vmin=-1,vmax=1,interpolation='nearest'); axes[0,col].set_title(title)
        axes[1,col].imshow(magnitude,cmap='viridis',interpolation='nearest'); axes[1,col].set_title('Fourier magnitude')
        for row in (0,1): axes[row,col].set_xticks([]); axes[row,col].set_yticks([])
    save_plot(out,'fourier_invariance',fig)
    save_report(out,report)

if __name__ == '__main__': main()
