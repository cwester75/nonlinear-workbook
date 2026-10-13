"""Activations, similarity scores, and Fourier features (pp.337-341, 364-369)."""
import numpy as np
from ._validation import array, matrix, positive


def threshold(x):
    """Heaviside activation: equality belongs to +1."""
    return (array(x) >= 0).astype(int)


def bipolar_activation(x):
    """Bipolar threshold with +1 at zero; Hopfield uses a separate hold rule."""
    return np.where(array(x) >= 0, 1, -1)


def logistic(x, *, slope=1.):
    """Stable logistic evaluation without exponential overflow."""
    positive(slope, 'slope')
    with np.errstate(over='ignore'):
        z = slope * array(x)
    # exp(-abs(z)) is safe even when finite inputs scale beyond float64.
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1 / (1 + e), e / (1 + e))


def tanh_activation(x, *, slope=1.):
    positive(slope, 'slope')
    with np.errstate(over='ignore'):
        return np.tanh(slope * array(x))


def _vectors(x, y):
    x, y = array(x, ndim=1), array(y, ndim=1)
    if x.shape != y.shape:
        raise ValueError('Vector lengths must agree')
    return x, y


def hamming_distance(x, y):
    """Number of unequal coordinates (no automatic normalization)."""
    x, y = _vectors(x, y)
    return int(np.count_nonzero(x != y))


def euclidean_distance(x, y):
    x, y = _vectors(x, y)
    return float(np.linalg.norm(x - y))


def binary_similarity(x, y, *, kind):
    """Book p.366 scores: hamming, euclidean, and asymmetric dot product.

    Dot-product denominator is the number of ones in x; zero is undefined.
    These scores are not all metrics. Inputs must use the 0/1 encoding.
    """
    x, y = _vectors(x, y)
    if not np.all((x == 0) | (x == 1)) or not np.all((y == 0) | (y == 1)):
        raise ValueError('Binary similarity requires 0/1 vectors')
    d = np.count_nonzero(x != y)
    if kind == 'hamming':
        return float(1 - d / len(x))
    if kind == 'euclidean':
        return float(1 - np.sqrt(d / len(x)))
    if kind == 'dot':
        count = np.count_nonzero(x)
        if count == 0:
            raise ValueError('Book dot-product score is undefined for an all-zero first vector')
        return float(x @ y / count)
    raise ValueError('Unknown similarity kind')


def directed_hausdorff(a, b):
    a, b = matrix(a, 'point set a'), matrix(b, 'point set b')
    if a.shape[1] != b.shape[1]:
        raise ValueError('Point dimensions must agree')
    return float(max(np.min(np.linalg.norm(b - point, axis=1)) for point in a))


def hausdorff_distance(a, b):
    return max(directed_hausdorff(a, b), directed_hausdorff(b, a))


def fourier_magnitude(image):
    """Forward-normalized FFT2 magnitude, invariant under circular translations."""
    image = matrix(image, 'image')
    return np.abs(np.fft.fft2(image, norm='forward'))
