import itertools
import unittest
import numpy as np
from nonlinear_workbook.chapter11 import *
from nonlinear_workbook.chapter11.fixtures import FIVE_PATTERNS, DIGIT_PATTERNS


class MemoryTests(unittest.TestCase):
    def test_five_neuron_book_weights_energy_and_recall(self):
        weights = hebbian_weights(FIVE_PATTERNS)
        expected = [[0,0,2,0,-2], [0,0,0,2,0], [2,0,0,0,-2], [0,2,0,0,0], [-2,0,-2,0,0]]
        np.testing.assert_array_equal(weights, expected)
        for pattern in FIVE_PATTERNS:
            self.assertEqual(hopfield_energy(weights, pattern), -8.)
            np.testing.assert_array_equal(hopfield_recall(weights, pattern).state, pattern)
        result = hopfield_recall(weights, np.ones(5), mode='synchronous')
        np.testing.assert_array_equal(result.state, FIVE_PATTERNS[0])
        self.assertEqual(result.status, 'converged')
        self.assertEqual(classify_recall(-FIVE_PATTERNS[0], FIVE_PATTERNS)['classification'], 'inverted')
        self.assertEqual(classify_recall(np.ones(5), FIVE_PATTERNS)['classification'], 'unmatched')

    def test_all_32_states_asynchronous_energy_and_thresholds(self):
        weights = hebbian_weights(FIVE_PATTERNS)
        for state in itertools.product((-1,1), repeat=5):
            for mode in ('asynchronous', 'randomized'):
                result = hopfield_recall(weights, state, mode=mode, thresholds=[.2,0,0,0,0])
                self.assertEqual(result.status, 'converged')
                self.assertTrue(np.all(np.diff(result.energies) <= 1e-12))
                np.testing.assert_array_equal(hopfield_recall(weights,result.state,thresholds=[.2,0,0,0,0]).state,result.state)
        result = hopfield_recall(weights, [1,-1,1,-1,1], mode='randomized', seed=3)
        repeat = hopfield_recall(weights, [1,-1,1,-1,1], mode='randomized', seed=3)
        np.testing.assert_array_equal(result.states, repeat.states)

    def test_synchronous_cycle_hold_and_limit(self):
        result = hopfield_recall([[0,1],[1,0]], [1,-1], mode='synchronous')
        self.assertEqual(result.status, 'cycle')
        self.assertEqual(result.cycle_start, 0)
        self.assertEqual(result.cycle_length, 2)
        np.testing.assert_array_equal(result.states, [[1,-1],[-1,1],[1,-1]])
        hold = hopfield_recall(np.zeros((2,2)), [1,-1])
        np.testing.assert_array_equal(hold.state, [1,-1])
        self.assertEqual(hold.status, 'converged')
        limited = hopfield_recall([[0,1],[1,0]], [1,-1], mode='synchronous', max_sweeps=1)
        self.assertEqual(limited.status, 'max_sweeps')

    def test_digit_patterns_and_corrupted_recall(self):
        self.assertEqual(DIGIT_PATTERNS.shape, (3,40))
        weights = hebbian_weights(DIGIT_PATTERNS)
        for pattern in DIGIT_PATTERNS:
            self.assertEqual(hopfield_recall(weights,pattern).status, 'converged')
            np.testing.assert_array_equal(hopfield_recall(weights,pattern).state, pattern)
            corrupted = pattern.copy(); corrupted[[0,5,10]] *= -1
            np.testing.assert_array_equal(hopfield_recall(weights,corrupted).state, pattern)

    def test_activations_scores_and_distances(self):
        np.testing.assert_array_equal(threshold([-1,0,1]), [0,1,1])
        np.testing.assert_array_equal(bipolar_activation([-1,0,1]), [-1,1,1])
        np.testing.assert_allclose(logistic([-1e308,0,1e308],slope=10), [0,.5,1])
        np.testing.assert_allclose(tanh_activation([-1e308,0,1e308],slope=10), [-1,0,1])
        self.assertEqual(hamming_distance([1,-1,1,1],[-1,1,1,-1]), 3)
        self.assertEqual(euclidean_distance([0,0],[3,4]), 5.)
        self.assertEqual(binary_similarity([1,0],[1,1],kind='dot'), 1.)
        self.assertEqual(binary_similarity([1,1],[1,0],kind='dot'), .5)
        self.assertEqual(binary_similarity([1,0],[0,1],kind='hamming'), 0.)
        self.assertEqual(binary_similarity([1,0],[0,1],kind='euclidean'), 0.)
        self.assertEqual(directed_hausdorff([[0],[1],[4]],[[0],[1]]), 3.)
        self.assertEqual(directed_hausdorff([[0],[1]],[[0],[1],[4]]), 0.)
        self.assertEqual(hausdorff_distance([[0],[1],[4]],[[0],[1]]), 3.)

    def test_fourier_translation_and_nonuniqueness(self):
        image = np.zeros((7,9)); image[[1,2,4],[2,5,7]] = [1,2,3]
        shifted = np.roll(image, (2,3), axis=(0,1))
        reflected = image[::-1,::-1]
        self.assertFalse(np.array_equal(image, reflected))
        np.testing.assert_allclose(fourier_magnitude(image),fourier_magnitude(shifted),atol=1e-14)
        np.testing.assert_allclose(fourier_magnitude(image),fourier_magnitude(reflected),atol=1e-14)
        self.assertAlmostEqual(fourier_magnitude(image)[0,0],image.mean())

    def test_invalid_inputs(self):
        calls = [lambda: hebbian_weights([[0,1]]), lambda: hopfield_recall([[0,1],[0,0]],[1,-1]),
            lambda: hopfield_recall(np.eye(2),[1,-1]), lambda: hopfield_recall(np.zeros((2,2)),[1,-1],thresholds=[0]),
            lambda: hopfield_recall(np.zeros((2,2)),[1,-1],mode='unknown'),
            lambda: hopfield_recall(np.zeros((2,2)),[1,-1],max_sweeps=0),
            lambda: logistic([np.nan]), lambda: logistic([1],slope=0),
            lambda: hamming_distance([1],[1,2]), lambda: hausdorff_distance([],[[0]]),
            lambda: binary_similarity([0,0],[1,1],kind='dot'), lambda: fourier_magnitude([1,2])]
        for call in calls:
            with self.assertRaises(ValueError): call()

if __name__ == '__main__': unittest.main()
