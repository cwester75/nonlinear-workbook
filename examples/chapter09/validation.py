"""Headless exact Chapter 9 references; optional puzzle UI is puzzle.py."""
import argparse
import csv
import json
import platform
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nonlinear_workbook import chapter09 as c

SEED=20261015


def write_csv(path,header,rows):
    with path.open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(header); writer.writerows(rows)


def save(path):
    plt.tight_layout(); plt.savefig(path,dpi=140,metadata={'Software':'nonlinear-workbook'}); plt.close()


def history_csv(path,history):
    write_csv(path,['generation',*[f'cell{i}' for i in range(history[0].size)]],
              ((t,*a.ravel()) for t,a in enumerate(history)))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',default='outputs/chapter09')
    out=Path(parser.parse_args().output); out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(SEED)
    report=dict(python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__,seed=SEED,
        units='one synchronous update per generation; dimensionless lattice cells',
        conventions=dict(initial_in_history=True,binary_dtype='uint8',rule_index='4*left+2*center+right',outer_bit='2*neighbors+center',max_elements=1000000),
        elementary=[],rings=[],life=[],totalistic=[])
    single=np.zeros(257,dtype=np.uint8); single[128]=1
    random=rng.integers(0,2,257,dtype=np.uint8)
    fig,axes=plt.subplots(6,2,figsize=(12,14))
    for row,rule in enumerate((0,30,56,90,110,255)):
        for col,(name,initial) in enumerate((('single',single),('bernoulli_half',random))):
            h=c.evolve(lambda a:c.elementary_step(a,rule=rule,boundary='fixed_zero'),initial,updates=128)
            history_csv(out/f'elementary_rule{rule}_{name}.csv',h)
            report['elementary'].append(dict(rule=rule,initial=name,width=257,updates=128,boundary='fixed_zero',table=c.decode_rule(rule,entries=8).tolist(),populations=h.sum(axis=1).tolist()))
            axes[row,col].imshow(h,cmap='binary',interpolation='nearest',aspect='auto'); axes[row,col].set_title(f'Rule {rule}: {name}')
            axes[row,col].set_xlabel('site'); axes[row,col].set_ylabel('generation')
    save(out/'elementary.png')
    spin=c.spin_to_binary([-1,-1,1,-1,-1]); step=lambda a:c.elementary_step(a,rule=90,boundary='periodic')
    h=c.evolve(step,spin,updates=7); cycle=c.detect_cycle(step,spin,max_steps=100)
    history_csv(out/'book_spin_binary.csv',h); history_csv(out/'book_spin.csv',c.binary_to_spin(h))
    report['book_spin']=dict(initial=[-1,-1,1,-1,-1],updates=7,printed_S6_equals_S1=bool(np.array_equal(h[6],h[1])),
        S4_equals_S1=bool(np.array_equal(h[4],h[1])),S7_equals_S1=bool(np.array_equal(h[7],h[1])),preperiod=cycle.preperiod,period=cycle.period)
    fig,axes=plt.subplots(2,8,figsize=(15,6))
    for j,width in enumerate((2,4,8,16,32,5,7,9)):
        initial=rng.integers(0,2,width,dtype=np.uint8)
        if not initial.any(): initial[width//2]=1
        for i,boundary in enumerate(('periodic','fixed_zero')):
            f=lambda a:c.elementary_step(a,rule=90,boundary=boundary)
            h=c.evolve(f,initial,updates=128); cycle=c.detect_cycle(f,initial,max_steps=256)
            history_csv(out/f'rule90_width{width}_{boundary}.csv',h)
            zeros=np.flatnonzero(~h.any(axis=1))
            entry=dict(width=width,initial=initial.tolist(),initial_method='seeded bits; all-zero draw replaced by centered cell',boundary=boundary,updates=128,first_zero=int(zeros[0]) if len(zeros) else None,
                preperiod=cycle.preperiod,period=cycle.period,examined_steps=cycle.examined_steps,status=cycle.status)
            if width in (2,4,8,16,32) and boundary=='periodic': entry['zero_at_half_width']=bool(not h[width//2].any()); assert entry['zero_at_half_width']
            report['rings'].append(entry)
            axes[i,j].imshow(h,cmap='binary',interpolation='nearest',aspect='auto'); axes[i,j].set_title(f'N={width}, {boundary}'); axes[i,j].set_ylabel('generation'); axes[i,j].set_xlabel('site')
    save(out/'rule90_boundaries.png')

    configurations={}
    block=np.zeros((64,64),dtype=np.uint8); block[30:32,30:32]=1; configurations['block']=block
    blinker=np.zeros_like(block); blinker[31,30:33]=1; configurations['blinker']=blinker
    glider=np.zeros_like(block); glider[60:63,60:63]=[[0,1,0],[0,0,1],[1,1,1]]; configurations['glider']=glider
    book=np.zeros((8,8),dtype=np.uint8); book[5:7,5:7]=1; configurations['book_block']=book
    configurations['random']=rng.integers(0,2,(64,64),dtype=np.uint8)
    fig,axes=plt.subplots(5,4,figsize=(11,13))
    populations=[]
    for i,(name,initial) in enumerate(configurations.items()):
        for j,boundary in enumerate(('periodic','fixed_zero')):
            h=c.evolve(lambda a:c.life_step(a,boundary=boundary),initial,updates=128)
            history_csv(out/f'life_{name}_{boundary}.csv',h)
            cycle=c.detect_cycle(lambda a:c.life_step(a,boundary=boundary),initial,max_steps=128)
            counts=h.sum(axis=(1,2)); populations.append((name,boundary,counts))
            report['life'].append(dict(name=name,shape=list(initial.shape),boundary=boundary,updates=128,
                outer_code=c.encode_outer_rule(c.life_table()),populations=counts.tolist(),cycle_status=cycle.status,preperiod=cycle.preperiod,period=cycle.period))
            for k,t in enumerate((0,128)):
                axes[i,2*j+k].imshow(h[t],cmap='binary',vmin=0,vmax=1,interpolation='nearest'); axes[i,2*j+k].set_title(f'{name}, {boundary}, t={t}',fontsize=8); axes[i,2*j+k].set_xticks([]); axes[i,2*j+k].set_yticks([])
    save(out/'life_snapshots.png')
    plt.figure(figsize=(10,5))
    for name,boundary,counts in populations:
        plt.plot(np.arange(129),counts,label=f'{name}, {boundary}',lw=.8)
    plt.xlabel('generation'); plt.ylabel('live cells'); plt.legend(fontsize=7,ncol=2); save(out/'life_populations.png')
    fig,axes=plt.subplots(2,2,figsize=(9,8))
    for i,neighborhood in enumerate(('von_neumann','moore')):
        neighbors=4 if neighborhood=='von_neumann' else 8
        table=np.arange(neighbors+2,dtype=np.uint8)%2
        initial=np.zeros((64,64),dtype=np.uint8); initial[32,32]=1
        for j,boundary in enumerate(('periodic','fixed_zero')):
            h=c.evolve(lambda a:c.totalistic_step(a,table=table,neighborhood=neighborhood,boundary=boundary),initial,updates=128)
            history_csv(out/f'totalistic_{neighborhood}_{boundary}.csv',h)
            report['totalistic'].append(dict(neighborhood=neighborhood,boundary=boundary,shape=[64,64],initial='centered single cell',updates=128,table=table.tolist(),code=c.encode_rule(table),populations=h.sum(axis=(1,2)).tolist()))
            axes[i,j].imshow(h[-1],cmap='binary',interpolation='nearest'); axes[i,j].set_title(f'{neighborhood}, {boundary}, t=128')
    save(out/'totalistic.png')

    goal=c.puzzle_goal(); shuffled=c.shuffle_puzzle(goal,moves=200,seed=SEED); replay=[shuffled.board]
    for coordinate in shuffled.inverse_moves:
        moved=c.puzzle_move(replay[-1],coordinate); assert moved.moved; replay.append(moved.board)
    assert c.puzzle_solved(replay[-1])
    history_csv(out/'puzzle_inverse_replay.csv',np.array(replay))
    write_csv(out/'puzzle_shuffle_moves.csv',['move','row','column'],((i,*move) for i,move in enumerate(shuffled.moves)))
    write_csv(out/'puzzle_inverse_moves.csv',['move','row','column'],((i,*move) for i,move in enumerate(shuffled.inverse_moves)))
    book=np.array([[8,4,15,5],[2,1,0,10],[7,6,9,12],[3,11,13,14]])
    book_after=c.puzzle_move(book,(1,3)).board
    report['puzzle']=dict(shape=[4,4],seed=SEED,shuffle_moves=200,avoid_immediate_reverse=True,initial=shuffled.board.tolist(),inverse_recovered_goal=c.puzzle_solved(replay[-1]),book_before=book.tolist(),book_click=[1,3],book_after=book_after.tolist(),uniform_reachable_sample=False)
    fig,axes=plt.subplots(1,4,figsize=(12,3))
    for ax,board,label in zip(axes,(book,book_after,shuffled.board,goal),('Book initial','Book legal move','Seeded legal shuffle','Inverse replay goal')):
        ax.imshow(board!=0,cmap='Pastel1',vmin=0,vmax=1)
        for (i,j),v in np.ndenumerate(board):
            if v: ax.text(j,i,str(v),ha='center',va='center')
        ax.set_title(label,fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    save(out/'puzzle.png')
    (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print('Wrote',out,flush=True)


if __name__=='__main__': main()
