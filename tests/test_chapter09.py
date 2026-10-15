import itertools
import unittest
import importlib.util
from pathlib import Path
import numpy as np
from nonlinear_workbook import chapter09 as c


def scalar_step(a,rule,boundary):
    out=[]
    def get(i): return int(a[i%len(a)]) if boundary=='periodic' else int(a[i]) if 0<=i<len(a) else 0
    for i in range(len(a)): out.append((rule>>(4*get(i-1)+2*get(i)+get(i+1)))&1)
    return np.array(out)


def scalar_counts(a,neighborhood,boundary):
    out=np.zeros(a.shape,int)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            for di in (-1,0,1):
                for dj in (-1,0,1):
                    if di==dj==0 or (neighborhood=='von_neumann' and abs(di)+abs(dj)!=1): continue
                    r,s=i+di,j+dj
                    if boundary=='periodic': out[i,j]+=int(a[r%a.shape[0],s%a.shape[1]])
                    elif 0<=r<a.shape[0] and 0<=s<a.shape[1]: out[i,j]+=int(a[r,s])
    return out


class Chapter09Tests(unittest.TestCase):
    def test_all_elementary_tables_and_neighborhoods(self):
        for rule in range(256):
            table=c.decode_rule(rule,entries=8); self.assertEqual(c.encode_rule(table),rule)
            for left,center,right in itertools.product((0,1),repeat=3):
                self.assertEqual(c.elementary_step([left,center,right],rule=rule,boundary='fixed_zero')[1],(rule>>(4*left+2*center+right))&1)
        np.testing.assert_array_equal(c.decode_rule(56,entries=8)[::-1],[0,0,1,1,1,0,0,0])
        for bits in itertools.product((0,1),repeat=3):
            self.assertEqual(c.elementary_step(bits,rule=90,boundary='fixed_zero')[1],bits[0]^bits[2])

    def test_independent_scalar_and_synchronous_updates(self):
        rng=np.random.default_rng(20261015)
        for width in (1,2,3,17):
            a=rng.integers(0,2,width,dtype=np.uint8); original=a.copy()
            for rule in (0,30,56,90,110,255):
                for boundary in ('periodic','fixed_zero'):
                    np.testing.assert_array_equal(c.elementary_step(a,rule=rule,boundary=boundary),scalar_step(a,rule,boundary))
            np.testing.assert_array_equal(a,original)
        history=c.evolve(lambda a:c.elementary_step(a,rule=90,boundary='fixed_zero'),[0,0,1,0,0],updates=2)
        np.testing.assert_array_equal(history,[[0,0,1,0,0],[0,1,0,1,0],[1,0,0,0,1]])

    def test_rule90_exhaustive_rings_and_spin(self):
        for width in range(1,9):
            for bits in itertools.product((0,1),repeat=width):
                initial=np.array(bits); step=lambda a:c.elementary_step(a,rule=90,boundary='periodic')
                new=step(initial); spin=c.binary_to_spin(initial)
                np.testing.assert_array_equal(c.binary_to_spin(new),-np.roll(spin,1)*np.roll(spin,-1))
                np.testing.assert_array_equal(c.spin_to_binary(spin),initial)
                self.assertEqual(int(np.prod(c.binary_to_spin(new))),(-1)**width)
                if width in (2,4,8): self.assertFalse(c.evolve(step,initial,updates=width//2)[-1].any())
        initial=c.spin_to_binary([-1,-1,1,-1,-1])
        history=c.evolve(lambda a:c.elementary_step(a,rule=90,boundary='periodic'),initial,updates=7)
        self.assertFalse(np.array_equal(history[6],history[1]))
        np.testing.assert_array_equal(history[4],history[1]); np.testing.assert_array_equal(history[7],history[1])
        reference=[initial]
        for _ in range(7): reference.append(scalar_step(reference[-1],90,'periodic'))
        np.testing.assert_array_equal(history,reference)

    def test_neighbors_and_totalistic_scalar_references(self):
        rng=np.random.default_rng(20261015)
        for shape in ((1,1),(1,2),(2,2),(3,5),(5,3)):
            a=rng.integers(0,2,shape); original=a.copy()
            for neighborhood in ('von_neumann','moore'):
                size=4 if neighborhood=='von_neumann' else 8
                table=np.arange(size+2)%2; outer=np.arange(2*(size+1)).reshape(2,size+1)%2
                code=c.encode_outer_rule(outer)
                np.testing.assert_array_equal(c.decode_outer_rule(code,neighborhood=neighborhood),outer)
                for boundary in ('periodic','fixed_zero'):
                    counts=scalar_counts(a,neighborhood,boundary)
                    np.testing.assert_array_equal(c.neighbor_counts(a,neighborhood=neighborhood,boundary=boundary),counts)
                    np.testing.assert_array_equal(c.totalistic_step(a,table=table,neighborhood=neighborhood,boundary=boundary),table[counts+a])
                    np.testing.assert_array_equal(c.outer_totalistic_step(a,table=outer,neighborhood=neighborhood,boundary=boundary),outer[a,counts])
            np.testing.assert_array_equal(a,original)
        self.assertEqual(c.neighbor_counts([[1]],neighborhood='moore',boundary='periodic')[0,0],8)
        self.assertEqual(c.neighbor_counts([[1]],neighborhood='moore',boundary='fixed_zero')[0,0],0)

    def test_life_all_local_cases_and_patterns(self):
        self.assertEqual(c.encode_outer_rule(c.life_table()),224)
        positions=[(i,j) for i in range(3) for j in range(3) if (i,j)!=(1,1)]
        for center in (0,1):
            for neighbors in range(9):
                a=np.zeros((3,3),int); a[1,1]=center
                for coordinate in positions[:neighbors]: a[coordinate]=1
                expected=int(neighbors==3 or (center==1 and neighbors==2))
                self.assertEqual(c.life_step(a,boundary='fixed_zero')[1,1],expected)
        block=np.zeros((8,8),int); block[5:7,5:7]=1
        for boundary in ('periodic','fixed_zero'): np.testing.assert_array_equal(c.life_step(block,boundary=boundary),block)
        blinker=np.zeros((7,7),int); blinker[3,2:5]=1
        history=c.evolve(lambda a:c.life_step(a,boundary='fixed_zero'),blinker,updates=2)
        np.testing.assert_array_equal(history[2],blinker); self.assertFalse(np.array_equal(history[1],blinker))
        glider=np.zeros((12,12),int); glider[2:5,2:5]=[[0,1,0],[0,0,1],[1,1,1]]
        history=c.evolve(lambda a:c.life_step(a,boundary='fixed_zero'),glider,updates=4)
        np.testing.assert_array_equal(history[4],np.roll(glider,(1,1),axis=(0,1)))

    def test_cycles_history_and_limits(self):
        step=lambda a:c.elementary_step(a,rule=90,boundary='periodic')
        result=c.detect_cycle(step,[0,0,1,0,0],max_steps=20)
        self.assertEqual((result.preperiod,result.period,result.examined_steps),(1,3,4))
        self.assertEqual(result.status,'cycle_detected'); self.assertEqual(len(result.history),5)
        result=c.detect_cycle(step,[0,0,1,0,0],max_steps=1)
        self.assertEqual(result.status,'limit_exhausted'); self.assertIsNone(result.period)
        np.testing.assert_array_equal(c.evolve(step,[1,0],updates=0),[[1,0]])
        self.assertEqual(c.detect_cycle(step,[1,0],max_steps=0).examined_steps,0)
        calls=[]
        for call in (lambda:c.evolve(lambda a:calls.append(a),[1,0],updates=10**12),lambda:c.detect_cycle(lambda a:calls.append(a),[1,0],max_steps=10**12)):
            with self.assertRaises(ValueError): call()
        self.assertEqual(calls,[])

    def test_puzzle_book_move_and_invalid_clicks(self):
        board=np.array([[8,4,15,5],[2,1,0,10],[7,6,9,12],[3,11,13,14]])
        original=board.copy(); moved=c.puzzle_move(board,(1,3))
        expected=board.copy(); expected[1,2],expected[1,3]=10,0
        np.testing.assert_array_equal(moved.board,expected); self.assertTrue(moved.moved)
        np.testing.assert_array_equal(board,original)
        for coordinate in ((0,1),(1,2),(3,2)):
            result=c.puzzle_move(board,coordinate); self.assertFalse(result.moved); np.testing.assert_array_equal(result.board,board)
        edge=c.puzzle_goal(); result=c.puzzle_move(edge,(3,0)); self.assertFalse(result.moved)

    def test_shuffle_inverse_and_reachable_2x2(self):
        goal=c.puzzle_goal(); a=c.shuffle_puzzle(goal,moves=200,seed=20261015); b=c.shuffle_puzzle(goal,moves=200,seed=20261015)
        np.testing.assert_array_equal(a.board,b.board); np.testing.assert_array_equal(a.moves,b.moves)
        board=a.board
        for move in a.inverse_moves:
            result=c.puzzle_move(board,move); self.assertTrue(result.moved); board=result.board
        self.assertTrue(c.puzzle_solved(board)); np.testing.assert_array_equal(board,goal)
        goal=c.puzzle_goal(rows=2,columns=2); seen={goal.tobytes()}; pending=[goal]
        while pending:
            board=pending.pop()
            # Independent legal swaps from Manhattan adjacency.
            blank=tuple(np.argwhere(board==0)[0]); expected=set()
            for i,j in np.ndindex(board.shape):
                if abs(i-blank[0])+abs(j-blank[1])==1: expected.add((i,j))
            self.assertEqual(set(c.legal_moves(board)),expected)
            for move in expected:
                reference=board.copy(); reference[blank],reference[move]=reference[move],reference[blank]
                result=c.puzzle_move(board,move); np.testing.assert_array_equal(result.board,reference)
                if reference.tobytes() not in seen: seen.add(reference.tobytes()); pending.append(reference)
        self.assertEqual(len(seen),12)

    def test_gui_controller_callbacks(self):
        path=Path(__file__).resolve().parents[1]/'examples/chapter09/puzzle.py'
        spec=importlib.util.spec_from_file_location('chapter09_puzzle',path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        controller=module.PuzzleController(seed=20261015); initial=controller.board.copy()
        self.assertTrue(controller.click(*c.legal_moves(controller.board)[0])); self.assertEqual(controller.move_count,1)
        controller.reset(); np.testing.assert_array_equal(controller.board,initial); self.assertEqual(controller.move_count,0)
        blank=tuple(np.argwhere(controller.board==0)[0]); self.assertFalse(controller.click(*blank))
        controller.reshuffle(); self.assertEqual(controller.shuffle_index,1); self.assertEqual(controller.move_count,0)
        np.testing.assert_array_equal(controller.board,c.shuffle_puzzle(c.puzzle_goal(),moves=200,seed=20261016).board)

    def test_invalid_inputs_and_allocations(self):
        calls=[lambda:c.elementary_step([0,2],rule=90,boundary='periodic'),lambda:c.elementary_step([0.,1.],rule=90,boundary='periodic'),lambda:c.decode_rule(256,entries=8),lambda:c.decode_rule(-1,entries=8),lambda:c.decode_rule(True,entries=8),lambda:c.neighbor_counts([],neighborhood='moore',boundary='periodic'),lambda:c.life_step([[1]],boundary='bad'),lambda:c.totalistic_step([[0]],table=[0,1],neighborhood='moore',boundary='periodic'),lambda:c.decode_outer_rule(2**18,neighborhood='moore'),lambda:c.spin_to_binary([-1,0,1]),lambda:c.evolve(lambda a:a,[1],updates=-1),lambda:c.evolve(lambda a:a,[1],updates=1.5),lambda:c.evolve(lambda a:[1,0],[1],updates=1),lambda:c.puzzle_goal(rows=10**12,columns=10**12),lambda:c.puzzle_move(c.puzzle_goal(),(-1,0)),lambda:c.puzzle_move(c.puzzle_goal(),(4,0)),lambda:c.puzzle_move([[0,1],[1,2]],(0,0)),lambda:c.shuffle_puzzle(c.puzzle_goal(),moves=1,seed=None),lambda:c.shuffle_puzzle(c.puzzle_goal(),moves=10**12,seed=1)]
        for call in calls:
            with self.assertRaises(ValueError): call()


if __name__=='__main__': unittest.main()
