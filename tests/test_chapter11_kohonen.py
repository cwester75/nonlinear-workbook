import unittest
import numpy as np
from nonlinear_workbook.chapter11 import *
from nonlinear_workbook.chapter11.fixtures import KOHONEN_INPUTS,KOHONEN_INITIAL,CITIES,BOOK_TOUR


class KohonenTests(unittest.TestCase):
    def test_book_first_updates(self):
        weights, winner = kohonen_update(KOHONEN_INITIAL,KOHONEN_INPUTS[0],learning_rate=.6)
        self.assertEqual(winner,1)
        np.testing.assert_allclose(weights,[[.2,.6,.5,.9],[.92,.76,.28,.12]],atol=1e-14)
        weights,winner = kohonen_update(weights,KOHONEN_INPUTS[1],learning_rate=.6)
        self.assertEqual(winner,0)
        np.testing.assert_allclose(weights[0],[.08,.24,.20,.96],atol=1e-14)
        for observation in KOHONEN_INPUTS[2:]:
            weights,_ = kohonen_update(weights,observation,learning_rate=.6)
        np.testing.assert_allclose(weights,[[.032,.096,.680,.984],[.968,.304,.112,.048]],atol=1e-14)

    def test_topology_and_tie_breaking(self):
        initial=np.zeros((5,1))
        ring,winner=kohonen_update(initial,[1],learning_rate=.5,topology='ring',width=1)
        chain,_=kohonen_update(initial,[1],learning_rate=.5,topology='chain',width=1)
        self.assertEqual(winner,0)
        self.assertEqual(ring[1,0],ring[-1,0])
        self.assertLess(chain[-1,0],ring[-1,0])
        single,_=kohonen_update(initial,[1],learning_rate=.5)
        np.testing.assert_array_equal(single[:,0],[.5,0,0,0,0])

    def test_training_repeatability_unused_nodes_and_stopping(self):
        params=dict(n_neurons=2,updates=1000,initial=KOHONEN_INITIAL,seed=7)
        result=kohonen_train(KOHONEN_INPUTS,**params)
        repeat=kohonen_train(KOHONEN_INPUTS,**params)
        np.testing.assert_array_equal(result.prototypes,repeat.prototypes)
        self.assertLess(result.quantization_errors[-1],result.quantization_errors[0])
        self.assertEqual(result.status,'completed')
        fixed=kohonen_train([[1,1],[1,1]],n_neurons=3,updates=20,tolerance=1e-12)
        self.assertEqual(fixed.status,'converged')
        np.testing.assert_array_equal(fixed.unused_prototypes,[1,2])
        exhausted=kohonen_train(KOHONEN_INPUTS,n_neurons=2,updates=1,tolerance=1e-15)
        self.assertEqual(exhausted.status,'max_updates')

    def test_tours_and_independent_reference(self):
        square=np.array([[0,0],[1,0],[1,1],[0,1.]])
        exact=exact_tsp(square)
        self.assertEqual(exact.length,4.)
        self.assertEqual(exact.evaluated_tours,6)
        decoded=decode_ring_tour(square,square)
        self.assertEqual(tour_length(square,decoded),4.)
        optimum=exact_tsp(CITIES)
        self.assertAlmostEqual(optimum.length,64.85596440153056,places=10)
        self.assertEqual(optimum.evaluated_tours,5040)
        self.assertAlmostEqual(tour_length(CITIES,BOOK_TOUR),68.0193642526682,places=10)
        self.assertGreater(tour_length(CITIES,BOOK_TOUR),optimum.length)
        tied=decode_ring_tour(square,[[0,0]])
        np.testing.assert_array_equal(tied,[0,1,2,3,0])

    def test_invalid_controls(self):
        calls=[lambda:kohonen_update([[0,0]],[1],learning_rate=.5),
            lambda:kohonen_update([[0]],[1],learning_rate=2),
            lambda:kohonen_train([[0]],n_neurons=0),
            lambda:kohonen_train([[0]],n_neurons=1,learning_rate=(.5,0)),
            lambda:kohonen_train([[0]],n_neurons=1,width=(0,1)),
            lambda:kohonen_train([[0]],n_neurons=1,topology='unknown'),
            lambda:kohonen_train([[0]],n_neurons=1,initial=[[0,1]]),
            lambda:tour_length(CITIES,[0,1,0]),lambda:tour_length([[0],[1]],[0,0,0]),
            lambda:exact_tsp(np.zeros((11,2)))]
        for call in calls:
            with self.assertRaises(ValueError):call()

if __name__=='__main__': unittest.main()
