"""Kohonen prototype learning and exact small-TSP references (pp.370-386)."""
from dataclasses import dataclass
import itertools
import math
import numpy as np
from ._validation import array, integer, matrix, positive


def nearest_prototypes(samples, prototypes):
    samples, prototypes = matrix(samples), matrix(prototypes, 'prototypes')
    if samples.shape[1] != prototypes.shape[1]:
        raise ValueError('Sample/prototype dimensions must agree')
    squared = np.sum((samples[:, None, :] - prototypes[None, :, :]) ** 2, axis=2)
    if not np.all(np.isfinite(squared)):
        raise ValueError('Distance overflow')
    assignments = np.argmin(squared, axis=1)  # first index wins a tie
    distances = np.sqrt(squared[np.arange(len(samples)), assignments])
    return assignments, distances


def kohonen_update(prototypes, observation, *, learning_rate, topology='winner', width=1.):
    """One online update. Prototype rows; Gaussian h=exp(-d_index^2/(2*width^2)).

    Chain/ring distances are neuron-index distances, not observation distances.
    Returns a new prototype array and the pre-update winner index.
    """
    prototypes = matrix(prototypes, 'prototypes')
    observation = array(observation, ndim=1, name='observation')
    positive(learning_rate, 'learning_rate')
    positive(width, 'width')
    if learning_rate > 1 or observation.shape != (prototypes.shape[1],):
        raise ValueError('Learning rate must be <=1 and observation must match features')
    if topology not in ('winner', 'chain', 'ring'):
        raise ValueError('Unknown Kohonen topology')
    winner = int(nearest_prototypes(observation[None, :], prototypes)[0][0])
    if topology == 'winner':
        neighborhood = (np.arange(len(prototypes)) == winner).astype(float)
    else:
        distance = np.abs(np.arange(len(prototypes)) - winner)
        if topology == 'ring':
            distance = np.minimum(distance, len(prototypes) - distance)
        neighborhood = np.exp(-.5 * (distance / width) ** 2)
    updated = prototypes + learning_rate * neighborhood[:, None] * (observation - prototypes)
    if not np.all(np.isfinite(updated)):
        raise ValueError('Nonfinite prototype update')
    return updated, winner


def _schedule(endpoints, name, updates, maximum=None):
    endpoints = array(endpoints, ndim=1, name=name)
    if len(endpoints) != 2 or np.any(endpoints <= 0) or (maximum is not None and np.any(endpoints > maximum)):
        raise ValueError(f'{name} must contain two positive valid endpoints')
    return np.geomspace(endpoints[0], endpoints[1], updates)


@dataclass(frozen=True)
class KohonenResult:
    prototypes: np.ndarray
    assignments: np.ndarray
    distances: np.ndarray
    unused_prototypes: np.ndarray
    checkpoint_updates: np.ndarray
    prototype_history: np.ndarray
    quantization_errors: np.ndarray
    updates: int
    status: str
    parameters: dict


def kohonen_train(samples, *, n_neurons, updates=1000, learning_rate=(.5, .05),
                  width=(1., .1), topology='winner', initial=None, seed=20261012,
                  order='random', check_every=None, tolerance=None):
    """Online training with geometrically interpolated per-update schedules.

    Random order samples with replacement; sequential order cycles through rows.
    Quantization error is mean Euclidean distance to a nearest prototype. Optional
    tolerance stops when maximum prototype displacement between checkpoints is
    small; this is numerical stabilization, not a global-optimality certificate.
    Without tolerance, a fully executed budget returns 'completed'.
    """
    samples = matrix(samples)
    integer(n_neurons, 'n_neurons'); integer(updates, 'updates'); integer(seed, 'seed', 0)
    if topology not in ('winner', 'chain', 'ring') or order not in ('random', 'sequential'):
        raise ValueError('Invalid topology/order')
    if check_every is None: check_every = len(samples)
    integer(check_every, 'check_every')
    if tolerance is not None: positive(tolerance, 'tolerance')
    rates = _schedule(learning_rate, 'learning_rate', updates, maximum=1)
    widths = _schedule(width, 'width', updates)
    rng = np.random.default_rng(seed)
    if initial is None:
        prototypes = rng.uniform(samples.min(axis=0), samples.max(axis=0), size=(n_neurons, samples.shape[1]))
    else:
        prototypes = matrix(initial, 'initial').copy()
        if prototypes.shape != (n_neurons, samples.shape[1]):
            raise ValueError('Initial prototype shape must match n_neurons/features')
    starting = prototypes.copy()
    history = [prototypes.copy()]; checkpoints = [0]
    errors = [float(nearest_prototypes(samples, prototypes)[1].mean())]
    status = 'completed' if tolerance is None else 'max_updates'
    for update in range(1, updates + 1):
        row = int(rng.integers(len(samples))) if order == 'random' else (update - 1) % len(samples)
        prototypes, _ = kohonen_update(prototypes, samples[row], learning_rate=rates[update - 1],
                                       topology=topology, width=widths[update - 1])
        if update % check_every == 0 or update == updates:
            displacement = np.max(np.linalg.norm(prototypes - history[-1], axis=1))
            checkpoints.append(update); history.append(prototypes.copy())
            errors.append(float(nearest_prototypes(samples, prototypes)[1].mean()))
            if tolerance is not None and displacement <= tolerance:
                status = 'converged'; break
    assignments, distances = nearest_prototypes(samples, prototypes)
    unused = np.setdiff1d(np.arange(n_neurons), assignments)
    parameters = dict(n_neurons=n_neurons, max_updates=updates, learning_rate=list(learning_rate),
        width=list(width), topology=topology, initial=starting, seed=seed, order=order,
        check_every=check_every, tolerance=tolerance, neighborhood='exp(-d_index^2/(2*width^2))')
    return KohonenResult(prototypes, assignments, distances, unused, np.array(checkpoints),
                         np.array(history), np.array(errors), update, status, parameters)


def tour_length(cities, tour):
    """Require a closed Hamiltonian tour of all supplied cities."""
    cities = matrix(cities, 'cities')
    tour = np.asarray(tour)
    n = len(cities)
    if n < 2 or tour.ndim != 1 or not np.issubdtype(tour.dtype, np.integer) or len(tour) != n + 1:
        raise ValueError('Tour must contain integer city indices and close all cities')
    if tour[0] != tour[-1] or sorted(tour[:-1].tolist()) != list(range(n)):
        raise ValueError('Tour must visit each city exactly once and close')
    length = float(np.linalg.norm(cities[tour[1:]] - cities[tour[:-1]], axis=1).sum())
    if not np.isfinite(length): raise ValueError('Tour length overflow')
    return length


def decode_ring_tour(cities, prototypes):
    """Sort city winner positions; ties use city index, then rotate to city 0."""
    cities = matrix(cities, 'cities')
    if len(cities) < 2: raise ValueError('At least two cities required')
    assignments, _ = nearest_prototypes(cities, prototypes)
    ordered = np.lexsort((np.arange(len(cities)), assignments))
    ordered = np.roll(ordered, -int(np.flatnonzero(ordered == 0)[0]))
    return np.r_[ordered, ordered[0]]


@dataclass(frozen=True)
class ExactTourResult:
    tour: np.ndarray
    length: float
    evaluated_tours: int


def exact_tsp(cities):
    """Exhaustive Euclidean reference, city 0 fixed; limited to 2..10 cities."""
    cities = matrix(cities, 'cities')
    if not 2 <= len(cities) <= 10:
        raise ValueError('Exhaustive reference supports 2..10 cities')
    distance = np.linalg.norm(cities[:, None, :] - cities[None, :, :], axis=2)
    if not np.all(np.isfinite(distance)): raise ValueError('Distance overflow')
    best_length, best = float('inf'), None
    for permutation in itertools.permutations(range(1, len(cities))):
        tour = (0,) + permutation + (0,)
        length = float(sum(distance[a, b] for a, b in zip(tour, tour[1:])))
        if length < best_length:
            best_length, best = length, tour
    return ExactTourResult(np.array(best), best_length, math.factorial(len(cities) - 1))
