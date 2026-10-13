"""Chapter 11 neural-network methods. See docs/chapter11.md."""
from .similarity import (threshold, bipolar_activation, logistic, tanh_activation,
    hamming_distance, euclidean_distance, binary_similarity, directed_hausdorff,
    hausdorff_distance, fourier_magnitude)
from .hopfield import (hebbian_weights, hopfield_energy, hopfield_recall,
    HopfieldResult, classify_recall)
from .kohonen import (nearest_prototypes, kohonen_update, kohonen_train, KohonenResult,
    tour_length, decode_ring_tour, exact_tsp, ExactTourResult)
from .feedforward import (perceptron_predict, perceptron_train, PerceptronResult,
    xor_network, MLPParameters, MLPGradients, MLPResult, mlp_initialize,
    mlp_forward, mlp_loss_and_gradients, mlp_train)
