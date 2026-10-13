"""Hopfield dynamics with explicit update schedules (printed pp.342-365)."""
from dataclasses import dataclass
import numpy as np
from ._validation import array, bipolar, integer, matrix


def hebbian_weights(patterns):
    """Unnormalized sum of bipolar outer products; diagonal is zero."""
    patterns = bipolar(patterns, 2)
    weights = patterns.T @ patterns
    np.fill_diagonal(weights, 0)
    return weights.astype(float)


def _network(weights, state, thresholds):
    weights = matrix(weights, 'weights')
    state = bipolar(state, 1)
    if weights.shape != (len(state), len(state)):
        raise ValueError('Weights must be square and match state length')
    if not np.allclose(weights, weights.T, rtol=0, atol=1e-12) or np.any(np.diag(weights) != 0):
        raise ValueError('Hopfield weights must be symmetric with zero diagonal')
    # Enforce exact symmetry after accepting numerical roundoff.
    weights = (weights + weights.T) / 2
    if thresholds is None:
        thresholds = np.zeros(len(state))
    else:
        thresholds = array(thresholds, ndim=1, name='thresholds')
        if thresholds.shape != state.shape:
            raise ValueError('Thresholds must match state')
    return weights, state.copy(), thresholds


def hopfield_energy(weights, state, *, thresholds=None):
    weights, state, thresholds = _network(weights, state, thresholds)
    return float(-.5 * state @ weights @ state + thresholds @ state)


@dataclass(frozen=True)
class HopfieldResult:
    state: np.ndarray
    states: np.ndarray
    energies: np.ndarray
    iterations: int
    status: str
    cycle_start: int | None
    cycle_length: int | None
    parameters: dict


def hopfield_recall(weights, initial, *, mode='asynchronous', thresholds=None,
                    max_sweeps=100, seed=20261012):
    """Recall with zero-field hold. An iteration is one step or complete sweep.

    modes: synchronous, asynchronous (index order), randomized (seeded sweeps).
    States are recorded at iteration boundaries. Energies are recorded after each
    individual asynchronous update, or each synchronous step. Repeated-state
    detection is applied at the same boundaries. No recalled pattern is guessed.
    """
    integer(max_sweeps, 'max_sweeps')
    integer(seed, 'seed', 0)
    if mode not in ('synchronous', 'asynchronous', 'randomized'):
        raise ValueError('Unknown Hopfield update mode')
    weights, state, thresholds = _network(weights, initial, thresholds)
    energy = lambda s: float(-.5 * s @ weights @ s + thresholds @ s)
    states, energies = [state.copy()], [energy(state)]
    seen = {state.tobytes(): 0}
    rng = np.random.default_rng(seed)
    parameters = dict(mode=mode, max_sweeps=max_sweeps, seed=seed,
                      thresholds=thresholds.copy(), zero_field='hold')
    for step in range(1, max_sweeps + 1):
        previous = state.copy()
        if mode == 'synchronous':
            field = weights @ state - thresholds
            state = np.where(field == 0, state, np.where(field > 0, 1, -1))
            energies.append(energy(state))
        else:
            order = rng.permutation(len(state)) if mode == 'randomized' else np.arange(len(state))
            for i in order:
                field = weights[i] @ state - thresholds[i]
                if field != 0:
                    state[i] = 1 if field > 0 else -1
                energies.append(energy(state))
        states.append(state.copy())
        if np.array_equal(previous, state):
            return HopfieldResult(state, np.array(states), np.array(energies), step,
                                  'converged', None, None, parameters)
        key = state.tobytes()
        if key in seen:
            return HopfieldResult(state, np.array(states), np.array(energies), step,
                                  'cycle', seen[key], step - seen[key], parameters)
        seen[key] = step
    return HopfieldResult(state, np.array(states), np.array(energies), max_sweeps,
                          'max_sweeps', None, None, parameters)


def classify_recall(state, patterns):
    """Report stored, inverted, or unmatched state; inversion is not success."""
    state, patterns = bipolar(state, 1), bipolar(patterns, 2)
    if patterns.shape[1] != len(state):
        raise ValueError('Pattern and state sizes must agree')
    matches = np.flatnonzero(np.all(patterns == state, axis=1))
    inverse = np.flatnonzero(np.all(patterns == -state, axis=1))
    return dict(classification='stored' if len(matches) else ('inverted' if len(inverse) else 'unmatched'),
                stored_indices=matches.tolist(), inverted_indices=inverse.tolist())
