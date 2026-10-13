import unittest
import math
import numpy as np
from nonlinear_workbook import chapter12 as c


PARAMS=dict(bits=10,population_size=32,generations=40,seed=42,
            selection='tournament',crossover_probability=.7,mutation_probability=.05,elitism=True)


class Chapter12Tests(unittest.TestCase):
    def test_encoding_schema(self):
        self.assertAlmostEqual(c.decode_binary([1,0,1,0,1,1,0,1],-1,1),91/255)
        self.assertEqual(c.decode_binary([0,0],2,4),2)
        self.assertEqual(c.decode_binary([1,1],2,4),4)
        self.assertEqual(c.schema_properties('011*1**'),(4,4))
        matches=[a for a in np.ndindex(*(2,)*5) if c.schema_matches(a,'*101*')]
        self.assertEqual(len(matches),4)

    def test_binary_operators(self):
        a,b=c.binary_crossover([0,0,0],[1,1,1],1)
        np.testing.assert_array_equal(a,[0,1,1]); np.testing.assert_array_equal(b,[1,0,0])
        np.testing.assert_array_equal(c.mutate_bits(a,1,rng=np.random.default_rng(0)),1-a)
        with self.assertRaises(ValueError): c.roulette_indices([-1,2],1,rng=np.random.default_rng(0))
        with self.assertRaises(ValueError): c.roulette_indices([0,0],1,rng=np.random.default_rng(0))
        np.testing.assert_array_equal(c.roulette_indices([0,1],10,rng=np.random.default_rng(0)),np.ones(10))

    def test_ga_accounting_reproducibility(self):
        a=c.binary_ga(np.sum,**PARAMS); b=c.binary_ga(np.sum,**PARAMS)
        self.assertEqual(a.best_value,10); self.assertEqual(a.evaluations,32*41)
        np.testing.assert_array_equal(a.population,b.population)
        self.assertTrue((np.diff(a.best_history)>=0).all())
        with self.assertRaises(ValueError): c.binary_ga(lambda a: -1,**dict(PARAMS,selection='roulette'))

    def test_knapsack_exact(self):
        a,value=c.exact_knapsack(c.KNAPSACK_WEIGHTS,c.KNAPSACK_VALUES,20)
        self.assertEqual(value,149)
        _,weight,feasible=c.knapsack_score(a,c.KNAPSACK_WEIGHTS,c.KNAPSACK_VALUES,20,infeasible_value=-1)
        self.assertEqual(weight,19); self.assertTrue(feasible)
        self.assertFalse(c.knapsack_score(np.ones(12),c.KNAPSACK_WEIGHTS,c.KNAPSACK_VALUES,20,infeasible_value=-1)[2])

    def test_pmx_book_and_invariants(self):
        a=np.arange(1,13); b=np.array([7,3,6,11,4,12,5,2,10,9,1,8])
        x,y=c.pmx(a,b,3,7)
        np.testing.assert_array_equal(x,[1,2,3,11,4,12,5,8,9,10,7,6])
        np.testing.assert_array_equal(y,[11,3,12,4,5,6,7,2,10,9,1,8])
        rng=np.random.default_rng(8)
        for _ in range(50):
            for child in c.pmx(rng.permutation(9),rng.permutation(9),2,7): self.assertEqual(set(child),set(range(9)))

    def test_tsp(self):
        points=np.array([[0,0],[1,0],[1,1],[0,1]])
        d=np.linalg.norm(points[:,None]-points[None,:],axis=-1)
        for operator in ('pmx','composition'):
            r=c.tsp_ga(d,population_size=12,generations=20,seed=5,crossover=operator)
            self.assertEqual(r.best_value,4); self.assertTrue((np.diff(r.best_history)<=0).all())
            for p in r.population: self.assertEqual(set(p),set(range(4)))

    def test_annealing(self):
        self.assertEqual(c.metropolis_probability(-1,2),1)
        self.assertAlmostEqual(c.metropolis_probability(2,2),math.exp(-1))
        r=c.simulated_annealing(lambda x: float(x@x),[3.],proposal=lambda x,rng: x*.5,
            temperatures=[1,.1],steps_per_temperature=5,seed=1)
        self.assertEqual(r.evaluations,11); self.assertTrue(r.accepted.all())
        self.assertAlmostEqual(r.best_value,9/4**10)
        with self.assertRaises(ValueError): c.metropolis_probability(1,0)
        with self.assertRaises(ValueError): c.simulated_annealing(np.sum,[1],proposal=lambda x,r:x,temperatures=[1,2],steps_per_temperature=1,seed=1)

    def test_islands(self):
        params={k:v for k,v in PARAMS.items() if k not in ('generations','seed')}
        r=c.island_ga(np.sum,islands=3,migration_interval=5,rounds=3,seed=2,**params)
        self.assertEqual(r['migrations'],9)
        self.assertEqual(r['populations'].shape,(3,32,10))
        self.assertTrue((np.diff(r['best_history'],axis=0)>=0).all())

    def test_invalid(self):
        for bits in ([],[2],[float('nan')]):
            with self.assertRaises(ValueError): c.decode_binary(bits,0,1)
        with self.assertRaises(ValueError): c.pmx([0,0],[0,1],0,1)
        with self.assertRaises(ValueError): c.binary_ga(np.sum,**dict(PARAMS,bits=0))
        with self.assertRaises(ValueError): c.exact_knapsack([1]*21,[1]*21,10)
        with self.assertRaises(ValueError): c.tsp_ga([[0,np.nan],[1,0]],population_size=2,generations=2,seed=0,crossover='pmx')


if __name__=='__main__': unittest.main()
