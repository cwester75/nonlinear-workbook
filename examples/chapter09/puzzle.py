"""Explicit optional interface: python examples/chapter09/puzzle.py [--smoke]."""
import argparse
import numpy as np
from nonlinear_workbook.chapter09 import puzzle_goal,puzzle_move,puzzle_solved,shuffle_puzzle


class PuzzleController:
    """GUI-independent callbacks; reset restores this shuffle, reshuffle advances seed."""
    def __init__(self, *, seed, moves=200):
        if not isinstance(seed,int) or isinstance(seed,bool) or seed<0: raise ValueError('nonnegative integer seed required')
        self.seed=seed; self.shuffle_moves=moves; self.shuffle_index=0
        self.initial=shuffle_puzzle(puzzle_goal(),moves=moves,seed=seed).board
        self.reset()

    def click(self,row,column):
        result=puzzle_move(self.board,(row,column))
        if result.moved: self.board=result.board; self.move_count+=1
        return result.moved

    def reset(self):
        self.board=self.initial.copy(); self.move_count=0

    def reshuffle(self):
        self.shuffle_index+=1
        self.initial=shuffle_puzzle(puzzle_goal(),moves=self.shuffle_moves,seed=self.seed+self.shuffle_index).board
        self.reset()


def create_interface(controller):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
    from matplotlib.colors import ListedColormap
    fig,ax=plt.subplots(figsize=(5,6)); fig.subplots_adjust(bottom=.2)
    reset=Button(fig.add_axes([.15,.05,.25,.07]),'Reset')
    shuffle=Button(fig.add_axes([.60,.05,.25,.07]),'Reshuffle')
    def redraw():
        ax.clear(); ax.imshow(controller.board!=0,cmap=ListedColormap(['#a9d8ac','#eee8d5']),vmin=0,vmax=1)
        for (i,j),value in np.ndenumerate(controller.board):
            if value: ax.text(j,i,str(value),ha='center',va='center',fontsize=22)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xticks(np.arange(-.5,4,1),minor=True); ax.set_yticks(np.arange(-.5,4,1),minor=True)
        ax.grid(which='minor',color='white',lw=2); ax.tick_params(which='both',bottom=False,left=False,labelbottom=False,labelleft=False)
        ax.set_title(f'Moves: {controller.move_count}'+(' — solved' if puzzle_solved(controller.board) else ''))
        fig.canvas.draw_idle()
    def clicked(event):
        if event.inaxes is ax and event.xdata is not None and event.ydata is not None:
            row,column=int(np.floor(event.ydata+.5)),int(np.floor(event.xdata+.5))
            if 0<=row<4 and 0<=column<4 and controller.click(row,column): redraw()
    def reset_clicked(event): controller.reset(); redraw()
    def shuffle_clicked(event): controller.reshuffle(); redraw()
    fig.canvas.mpl_connect('button_press_event',clicked)
    reset.on_clicked(reset_clicked); shuffle.on_clicked(shuffle_clicked)
    # Retain widgets and callbacks for their lifetime and for a real-backend smoke check.
    fig._puzzle_widgets=(reset,shuffle); fig._puzzle_callbacks=(clicked,reset_clicked,shuffle_clicked)
    redraw(); return fig,ax


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--seed',type=int,default=20261015)
    parser.add_argument('--backend',default='TkAgg'); parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    import matplotlib
    matplotlib.use(args.backend)
    import matplotlib.pyplot as plt
    controller=PuzzleController(seed=args.seed); fig,ax=create_interface(controller)
    if args.smoke:
        from types import SimpleNamespace
        from nonlinear_workbook.chapter09 import legal_moves
        plt.show(block=False); plt.pause(.2)
        i,j=legal_moves(controller.board)[0]
        fig._puzzle_callbacks[0](SimpleNamespace(inaxes=ax,xdata=j,ydata=i))
        assert controller.move_count==1
        fig._puzzle_callbacks[1](None); assert controller.move_count==0
        fig._puzzle_callbacks[2](None); assert controller.shuffle_index==1
        fig.canvas.draw(); plt.close(fig)
        print('GUI smoke passed:',matplotlib.get_backend())
    else: plt.show()


if __name__=='__main__': main()
