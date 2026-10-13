"""Chapter 2 methods with explicit finite-sample conventions."""
import numpy as np


def _series(values, minimum=2):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size < minimum or not np.all(np.isfinite(x)):
        raise ValueError("Expected a finite one-dimensional series of sufficient length")
    return x


def correlation(x, y):
    """Pearson correlation; constant inputs are undefined."""
    x, y = _series(x), _series(y)
    if x.size != y.size:
        raise ValueError("Series lengths must agree")
    a, b = x - x.mean(), y - y.mean()
    scale = np.linalg.norm(a) * np.linalg.norm(b)
    if scale == 0:
        raise ValueError("Correlation is undefined for constant series")
    return float(np.clip(np.dot(a, b) / scale, -1, 1))


def logistic_map(n, r=4.0, x0=0.618, burn_in=1000):
    """Generate n post-transient samples; initial state is not returned."""
    if not isinstance(n, int) or n < 1 or not isinstance(burn_in, int) or burn_in < 0:
        raise ValueError("n must be positive and burn_in nonnegative integers")
    if not 0 < x0 < 1 or not 0 < r <= 4:
        raise ValueError("Require 0 < x0 < 1 and 0 < r <= 4")
    x = float(x0)
    out = np.empty(n)
    for i in range(n + burn_in):
        x = r * x * (1 - x)
        if i >= burn_in:
            out[i - burn_in] = x
    return out


def lyapunov_map(x, r=4.0):
    """Known-map reference exponent: mean log absolute derivative per step.

    This uses the known logistic equation, not a data-only estimator.
    A visited zero derivative gives negative infinity.
    """
    x = _series(x)
    if not np.isfinite(r) or r <= 0:
        raise ValueError("r must be finite and positive")
    derivative = np.abs(r * (1 - 2 * x))
    if np.any(derivative == 0):
        return float("-inf")
    return float(np.mean(np.log(derivative)))


def kantz_curve(x, dimension=1, delay=1, epsilon=0.01, horizon=6, theiler=10):
    """Mean log mean neighbor separation at each future lag.

    Backward delay vectors use only observations at or before their index.
    Exclude temporal neighbors within theiler samples. Use the same eligible
    references and neighbors at every lag; reject zero separations. Returns
    (lags, curve, reference_count). Quadratic work: use bounded samples.
    """
    x = _series(x)
    for value in (dimension, delay, horizon):
        if not isinstance(value, int) or value < 1:
            raise ValueError("dimension, delay, horizon must be positive integers")
    if not isinstance(theiler, int) or theiler < 0 or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Invalid temporal exclusion or neighborhood radius")
    start = (dimension - 1) * delay
    indices = np.arange(start, len(x) - horizon + 1)
    if len(indices) < 2:
        raise ValueError("Insufficient observations for embedding and horizon")
    vectors = np.column_stack([x[indices - j * delay] for j in range(dimension)])
    curves = []
    for i, t in enumerate(indices):
        distances = np.linalg.norm(vectors - vectors[i], axis=1)
        mask = (distances > 0) & (distances < epsilon) & (np.abs(indices - t) > theiler)
        neighbors = indices[mask]
        if not len(neighbors):
            continue
        separation = np.array([np.linalg.norm(
            np.column_stack([x[neighbors + lag - j * delay] for j in range(dimension)])
            - np.array([x[t + lag - j * delay] for j in range(dimension)]), axis=1
        ).mean() for lag in range(horizon)])
        if np.all(separation > 0):
            curves.append(np.log(separation))
    if not curves:
        raise ValueError("No valid neighbors; adjust epsilon or sample size")
    return np.arange(horizon), np.mean(curves, axis=0), len(curves)


def hurst_rs(x, sizes=None):
    """Classical rescaled-range slope on the supplied observations.

    For each scale, demean contiguous blocks, cumulatively sum deviations,
    divide their range by population standard deviation, then average R/S.
    Drop trailing incomplete blocks and constant blocks. Returns (H, scales,
    mean_rs). No assertion of long memory or independence follows from H alone.
    """
    x = _series(x, 32)
    if sizes is None:
        sizes = 2 ** np.arange(4, int(np.floor(np.log2(len(x) // 4))) + 1)
    scales, ratios = [], []
    for n in sizes:
        if not isinstance(n, (int, np.integer)) or n < 8 or n > len(x) // 2:
            raise ValueError("Scales must be integer block sizes from 8 to len(x)//2")
        blocks = x[:len(x) // n * n].reshape(-1, n)
        deviations = blocks - blocks.mean(axis=1, keepdims=True)
        std = np.sqrt(np.mean(deviations ** 2, axis=1))
        cumulative = np.column_stack([np.zeros(len(blocks)), np.cumsum(deviations, axis=1)])
        ranges = np.ptp(cumulative, axis=1)
        valid = std > 0
        if valid.any():
            scales.append(n)
            ratios.append(float(np.mean(ranges[valid] / std[valid])))
    if len(set(scales)) < 2:
        raise ValueError("At least two distinct nonconstant scales are required")
    order = np.argsort(scales)
    scales, ratios = np.asarray(scales)[order], np.asarray(ratios)[order]
    slope = np.polyfit(np.log(scales), np.log(ratios), 1)[0]
    return float(slope), scales, ratios


def lz_complexity(symbols):
    """LZ76 exhaustive-history phrase count and asymptotic normalization.

    Symbols must be a nonempty string or sequence of hashable discrete values.
    Copying may overlap the phrase being generated. Finite-length normalized
    values can exceed one; normalization is not a probability.
    """
    s = list(symbols)
    if not s:
        raise ValueError("Sequence must be nonempty")
    alphabet = len(set(s))
    n, count, position = len(s), 1, 1
    while position < n:
        longest = 0
        for source in range(position):
            length = 0
            while position + length < n and s[source + length] == s[position + length]:
                length += 1
            longest = max(longest, length)
        count += 1
        position += longest + 1
    normalized = count * np.log(n) / (n * np.log(alphabet)) if alphabet > 1 else 0.0
    return count, float(normalized)

from dataclasses import dataclass


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def _interval(value):
    if not np.isfinite(value) or value <= 0:
        raise ValueError('sample_interval must be finite and positive')


def delay_embedding(x, *, dimension, delay, horizon=0):
    """Backward vectors and original sample indices, reserving horizon successors."""
    x = _series(x)
    _integer(dimension, 'dimension')
    _integer(delay, 'delay')
    _integer(horizon, 'horizon', 0)
    indices = np.arange((dimension - 1) * delay, len(x) - horizon)
    if len(indices) < 2:
        raise ValueError('Insufficient embedding history or successor observations')
    return np.column_stack([x[indices - j * delay] for j in range(dimension)]), indices


def _neighbor_mask(indices, t, distances, theiler):
    return (np.abs(indices - t) > theiler) & (distances > 0)


@dataclass(frozen=True)
class WolfSegment:
    reference_index: int
    neighbor_index: int
    steps: int
    initial_distance: float
    final_distance: float
    log_growth: float


@dataclass(frozen=True)
class WolfResult:
    exponent: float
    segments: tuple
    replacements: int
    replacement_attempts: int
    elapsed_time: float
    termination_reason: str


@dataclass(frozen=True)
class SpectrumResult:
    exponents: np.ndarray
    running_estimates: np.ndarray
    fit_ranks: np.ndarray
    condition_numbers: np.ndarray
    usable_steps: int
    accumulated_steps: int
    termination_reason: str


def wolf_lyapunov(x, *, dimension, delay, theiler, min_distance,
                  max_distance, max_angle, max_steps, sample_interval=1.0):
    """Wolf largest-exponent estimate with directed replacement angles in radians.

    Distances use Euclidean norm in observation units. Initial/replacement
    distances are in [min_distance, max_distance); evolution stops on reaching
    max_distance, max_steps, or the data boundary. If no aligned replacement
    exists, retain the evolved pair and retry after another evolution segment.
    A zero/nonfinite evolved distance terminates without logging that segment.
    Quadratic neighbor searches; no automatic parameter selection.
    """
    _integer(theiler, 'theiler', 0)
    _integer(max_steps, 'max_steps')
    _interval(sample_interval)
    if not (np.isfinite(min_distance) and np.isfinite(max_distance)
            and 0 < min_distance < max_distance):
        raise ValueError('Require 0 < min_distance < max_distance, both finite')
    if not np.isfinite(max_angle) or not 0 <= max_angle <= np.pi:
        raise ValueError('max_angle must be in [0, pi] radians')
    vectors, indices = delay_embedding(x, dimension=dimension, delay=delay)
    # Start at the first reference having a neighbor with a valid successor.
    pair = None
    for i in range(len(vectors) - 1):
        ds = np.linalg.norm(vectors[:-1] - vectors[i], axis=1)
        mask = _neighbor_mask(indices[:-1], indices[i], ds, theiler)
        candidates = np.flatnonzero(mask & (ds >= min_distance) & (ds < max_distance))
        if candidates.size:
            pair = (i, candidates[np.argmin(ds[candidates])])
            break
    if pair is None:
        raise ValueError('No admissible initial neighbor with a successor')
    i, j = pair
    segments, replacements, attempts = [], 0, 0
    reason = 'data_boundary'
    while i < len(vectors) - 1 and j < len(vectors) - 1:
        i0, j0 = i, j
        initial = float(np.linalg.norm(vectors[j] - vectors[i]))
        steps = 0
        while steps < max_steps and i < len(vectors) - 1 and j < len(vectors) - 1:
            i += 1
            j += 1
            steps += 1
            final = float(np.linalg.norm(vectors[j] - vectors[i]))
            if not np.isfinite(final) or final == 0 or final >= max_distance:
                break
        if not np.isfinite(final) or final == 0:
            reason = 'zero_or_nonfinite_separation'
            break
        segments.append(WolfSegment(int(indices[i0]), int(indices[j0]), steps,
                                    initial, final, float(np.log(final / initial))))
        if i == len(vectors) - 1:
            break
        attempts += 1
        direction = vectors[j] - vectors[i]
        offsets = vectors[:-1] - vectors[i]
        ds = np.linalg.norm(offsets, axis=1)
        mask = _neighbor_mask(indices[:-1], indices[i], ds, theiler)
        candidates = np.flatnonzero(mask & (ds >= min_distance) & (ds < max_distance)
                                    & (np.arange(len(ds)) != j))
        if candidates.size:
            angles = np.arccos(np.clip(offsets[candidates] @ direction / (ds[candidates] * final), -1, 1))
            accepted = angles <= max_angle
            if accepted.any():
                choices = candidates[accepted]
                # Primary key angle; secondary distance; stable index tie break.
                order = np.lexsort((choices, ds[choices], angles[accepted]))
                j = int(choices[order[0]])
                replacements += 1
                continue
        if j == len(vectors) - 1:
            reason = 'no_replacement_at_boundary'
            break
    if not segments:
        raise ValueError('No usable evolution segments')
    elapsed = sum(s.steps for s in segments) * sample_interval
    return WolfResult(sum(s.log_growth for s in segments) / elapsed,
                      tuple(segments), replacements, attempts, elapsed, reason)


def local_lyapunov_spectrum(x, *, dimension, delay, theiler, neighbors,
                            sample_interval=1.0, qr_burn_in=0):
    """Local least-squares Jacobians with chronological QR tangent propagation.

    Scalar data use backward delay vectors; matrix data are observed states and
    must have dimension columns (delay must be one). Row-oriented least squares
    solves Y B = Z, hence the column-vector Jacobian is B.T (equivalent to AV=C).
    Stop at the first rank-deficient fit or singular tangent update. Exponents
    retain QR direction order, not sorted order. Burn-in updates Q but not sums.
    """
    _integer(dimension, 'dimension')
    _integer(delay, 'delay')
    _integer(theiler, 'theiler', 0)
    _integer(neighbors, 'neighbors')
    _integer(qr_burn_in, 'qr_burn_in', 0)
    _interval(sample_interval)
    values = np.asarray(x, dtype=float)
    if values.ndim == 1:
        states, indices = delay_embedding(values, dimension=dimension, delay=delay)
    elif values.ndim == 2 and values.shape[1] == dimension and len(values) >= 3 and np.all(np.isfinite(values)):
        if delay != 1:
            raise ValueError('Observed-state matrices require delay=1')
        states, indices = values, np.arange(len(values))
    else:
        raise ValueError('Expected finite scalar series or state matrix matching dimension')
    if neighbors < dimension or neighbors > len(states) - 2:
        raise ValueError('neighbors must cover dimension and fit available samples')
    q, sums = np.eye(dimension), np.zeros(dimension)
    ranks, conditions, running = [], [], []
    usable = accumulated = 0
    reason = 'data_boundary'
    for i in range(len(states) - 1):
        offsets = states[:-1] - states[i]
        distances = np.linalg.norm(offsets, axis=1)
        candidates = np.flatnonzero(_neighbor_mask(indices[:-1], indices[i], distances, theiler))
        if len(candidates) < neighbors:
            reason = 'insufficient_neighbors'
            break
        selected = candidates[np.argsort(distances[candidates], kind='stable')[:neighbors]]
        y = offsets[selected]
        z = states[selected + 1] - states[i + 1]
        b, _, rank, singular = np.linalg.lstsq(y, z, rcond=None)
        ranks.append(int(rank))
        conditions.append(float(singular[0] / singular[-1]) if singular[-1] > 0 else float('inf'))
        if rank < dimension:
            reason = 'rank_deficient_fit'
            break
        q_new, r = np.linalg.qr(b.T @ q)
        diagonal = np.abs(np.diag(r))
        if np.any(diagonal == 0) or not np.all(np.isfinite(diagonal)):
            reason = 'singular_tangent_update'
            break
        q = q_new
        usable += 1
        if usable > qr_burn_in:
            accumulated += 1
            sums += np.log(diagonal)
            running.append(sums.copy() / (accumulated * sample_interval))
    if not accumulated:
        raise ValueError(f'No usable post-burn-in updates: {reason}')
    return SpectrumResult(sums / (accumulated * sample_interval), np.asarray(running),
                          np.asarray(ranks), np.asarray(conditions), usable, accumulated, reason)


def direct_lyapunov(x, *, epsilon=None, theiler=0, sample_interval=1.0):
    """Chapter pp.113-117 one-step scalar divergence, with valid references only.

    epsilon=None chooses the nearest nonzero neighbor; otherwise average log
    growth over neighbors inside epsilon, then average eligible references.
    Returns (exponent, eligible_reference_count).
    """
    x = _series(x, 3)
    _integer(theiler, 'theiler', 0)
    _interval(sample_interval)
    if epsilon is not None and (not np.isfinite(epsilon) or epsilon <= 0):
        raise ValueError('epsilon must be finite and positive')
    indices = np.arange(len(x) - 1)
    logs = []
    for i in indices:
        distances = np.abs(x[:-1] - x[i])
        candidates = np.flatnonzero(_neighbor_mask(indices, i, distances, theiler))
        if epsilon is None and candidates.size:
            candidates = candidates[np.argmin(distances[candidates]):np.argmin(distances[candidates]) + 1]
        elif epsilon is not None:
            candidates = candidates[distances[candidates] < epsilon]
        evolved = np.abs(x[candidates + 1] - x[i + 1])
        valid = evolved > 0
        if valid.any():
            logs.append(np.mean(np.log(evolved[valid] / distances[candidates[valid]])))
    if not logs:
        raise ValueError('No valid successor neighbors')
    return float(np.mean(logs) / sample_interval), len(logs)
