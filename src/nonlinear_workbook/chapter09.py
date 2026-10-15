"""Binary synchronous cellular automata and sliding puzzles, pp.299–312."""
from dataclasses import dataclass
import numpy as np


def _integer(value, minimum=0):
    if not isinstance(value,(int,np.integer)) or isinstance(value,(bool,np.bool_)) or value<minimum:
        raise ValueError('integer outside allowed range')
    return int(value)


def _binary(state, ndim=None):
    a=np.asarray(state)
    if a.dtype.kind not in 'biu' or (ndim is not None and a.ndim!=ndim) or a.ndim not in (1,2) or not a.size or any(s==0 for s in a.shape) or not np.all((a==0)|(a==1)):
        raise ValueError('nonempty binary integer/bool array required')
    return a.astype(np.uint8,copy=True)


def _boundary(boundary):
    if boundary not in ('periodic','fixed_zero'): raise ValueError('unknown boundary')


def decode_rule(code, *, entries):
    """Bits in ascending index order; elementary index is 4*left+2*center+right."""
    code=_integer(code); entries=_integer(entries,1)
    if entries>1024 or code.bit_length()>entries: raise ValueError('rule exceeds lookup width')
    return np.array([(code>>i)&1 for i in range(entries)],dtype=np.uint8)


def encode_rule(table):
    a=_binary(table,1)
    if len(a)>1024: raise ValueError('lookup width exceeds 1024')
    return sum(int(v)<<i for i,v in enumerate(a))


def elementary_step(state, *, rule, boundary):
    a=_binary(state,1); _boundary(boundary); table=decode_rule(rule,entries=8)
    if boundary=='periodic': left=np.roll(a,1); right=np.roll(a,-1)
    else:
        padded=np.pad(a,1); left=padded[:-2]; right=padded[2:]
    return table[4*left+2*a+right]


def binary_to_spin(state):
    return 2*_binary(state).astype(np.int8)-1


def spin_to_binary(state):
    a=np.asarray(state)
    if a.dtype.kind not in 'iu' or a.ndim not in (1,2) or not a.size or not np.all((a==-1)|(a==1)):
        raise ValueError('nonempty integer spin array required')
    return ((a+1)//2).astype(np.uint8)


def _offsets(neighborhood):
    if neighborhood=='von_neumann': return ((-1,0),(1,0),(0,-1),(0,1))
    if neighborhood=='moore': return tuple((i,j) for i in (-1,0,1) for j in (-1,0,1) if (i,j)!=(0,0))
    raise ValueError('unknown neighborhood')


def neighbor_counts(state, *, neighborhood, boundary):
    """Count directional offsets, including multiplicities on tiny periodic grids."""
    a=_binary(state,2); _boundary(boundary); offsets=_offsets(neighborhood)
    counts=np.zeros(a.shape,dtype=np.int64)
    if boundary=='periodic':
        for i,j in offsets: counts+=np.roll(a,(i,j),axis=(0,1))
    else:
        pad=np.pad(a,1); rows,cols=a.shape
        for i,j in offsets: counts+=pad[1+i:1+i+rows,1+j:1+j+cols]
    return counts


def totalistic_step(state, *, table, neighborhood, boundary):
    a=_binary(state,2); lookup=_binary(table,1); size=len(_offsets(neighborhood))
    if len(lookup)!=size+2: raise ValueError('totalistic table includes center: entries=neighbors+2')
    return lookup[neighbor_counts(a,neighborhood=neighborhood,boundary=boundary)+a]


def encode_outer_rule(table):
    lookup=_binary(table,2)
    if lookup.shape[0]!=2 or lookup.shape[1] not in (5,9): raise ValueError('outer table must have shape (2,5) or (2,9)')
    return encode_rule(lookup.T.ravel())


def decode_outer_rule(code, *, neighborhood):
    count=len(_offsets(neighborhood))+1
    return decode_rule(code,entries=2*count).reshape(count,2).T.copy()


def outer_totalistic_step(state, *, table, neighborhood, boundary):
    a=_binary(state,2); lookup=_binary(table,2); size=len(_offsets(neighborhood))
    if lookup.shape!=(2,size+1): raise ValueError('outer table shape mismatch')
    return lookup[a,neighbor_counts(a,neighborhood=neighborhood,boundary=boundary)]


def life_table():
    table=np.zeros((2,9),dtype=np.uint8); table[0,3]=1; table[1,[2,3]]=1
    return table


def life_step(state, *, boundary):
    return outer_totalistic_step(state,table=life_table(),neighborhood='moore',boundary=boundary)


def evolve(step, initial, *, updates, max_elements=1000000):
    """History includes generation zero; allocation checked using Python integers."""
    updates=_integer(updates); maximum=_integer(max_elements,1)
    if (updates+1)*np.asarray(initial).size>maximum: raise ValueError('history allocation limit exceeded')
    a=_binary(initial)
    history=np.empty((updates+1,*a.shape),dtype=np.uint8); history[0]=a
    for i in range(updates):
        new=_binary(step(history[i].copy()))
        if new.shape!=a.shape: raise ValueError('step changed state shape')
        history[i+1]=new
    return history


@dataclass
class CycleResult:
    history: np.ndarray
    preperiod: int | None
    period: int | None
    examined_steps: int
    status: str


def detect_cycle(step, initial, *, max_steps, max_elements=1000000):
    """Exact byte equality, includes the repeated endpoint when a cycle is found."""
    limit=_integer(max_steps); maximum=_integer(max_elements,1)
    if (limit+1)*np.asarray(initial).size>maximum: raise ValueError('cycle history allocation limit exceeded')
    a=_binary(initial)
    history=np.empty((limit+1,*a.shape),dtype=np.uint8); history[0]=a; seen={a.tobytes():0}
    for i in range(1,limit+1):
        a=_binary(step(a.copy()))
        if a.shape!=history.shape[1:]: raise ValueError('step changed state shape')
        history[i]=a; key=a.tobytes()
        if key in seen:
            start=seen[key]
            return CycleResult(history[:i+1].copy(),start,i-start,i,'cycle_detected')
        seen[key]=i
    return CycleResult(history,None,None,limit,'limit_exhausted')


def _board(board):
    a=np.asarray(board)
    if a.ndim!=2 or min(a.shape)<2 or a.dtype.kind not in 'iu' or not np.array_equal(np.sort(a.ravel()),np.arange(a.size)):
        raise ValueError('rectangular puzzle with tiles 0..size-1 exactly once required')
    return a.astype(np.int64,copy=True)


def puzzle_goal(*, rows=4, columns=4, max_elements=1000000):
    rows=_integer(rows,2); columns=_integer(columns,2); maximum=_integer(max_elements,1)
    if rows*columns>maximum: raise ValueError('puzzle allocation limit exceeded')
    return np.r_[np.arange(1,rows*columns),0].reshape(rows,columns)


def puzzle_solved(board):
    a=_board(board)
    return bool(np.array_equal(a,puzzle_goal(rows=a.shape[0],columns=a.shape[1],max_elements=a.size)))


def legal_moves(board):
    a=_board(board); row,col=np.argwhere(a==0)[0]
    return tuple((int(row+i),int(col+j)) for i,j in _offsets('von_neumann') if 0<=row+i<a.shape[0] and 0<=col+j<a.shape[1])


@dataclass
class PuzzleMove:
    board: np.ndarray
    moved: bool
    blank_before: tuple
    blank_after: tuple


def puzzle_move(board, coordinate):
    a=_board(board)
    if not isinstance(coordinate,(tuple,list,np.ndarray)) or len(coordinate)!=2: raise ValueError('row/column pair required')
    row,col=(_integer(v) for v in coordinate)
    if row>=a.shape[0] or col>=a.shape[1]: raise ValueError('coordinate outside board')
    blank=tuple(int(v) for v in np.argwhere(a==0)[0]); moved=(row,col) in legal_moves(a)
    if moved: a[blank],a[row,col]=a[row,col],a[blank]
    return PuzzleMove(a,moved,blank,(row,col) if moved else blank)


@dataclass
class PuzzleShuffle:
    board: np.ndarray
    moves: np.ndarray
    inverse_moves: np.ndarray


def shuffle_puzzle(board, *, moves, seed, avoid_immediate_reverse=True, max_elements=1000000):
    a=_board(board); count=_integer(moves); maximum=_integer(max_elements,1)
    if seed is None or not isinstance(avoid_immediate_reverse,(bool,np.bool_)): raise ValueError('explicit seed and boolean reversal choice required')
    if a.size+4*count>maximum: raise ValueError('shuffle allocation limit exceeded')
    rng=np.random.default_rng(seed); forward=[]; reverse=[]; previous=None
    for _ in range(count):
        options=list(legal_moves(a))
        if avoid_immediate_reverse and previous in options and len(options)>1: options.remove(previous)
        chosen=options[int(rng.integers(len(options)))]; result=puzzle_move(a,chosen)
        forward.append(chosen); reverse.append(result.blank_before); previous=result.blank_before; a=result.board
    return PuzzleShuffle(a,np.array(forward,dtype=int).reshape(-1,2),np.array(reverse[::-1],dtype=int).reshape(-1,2))
