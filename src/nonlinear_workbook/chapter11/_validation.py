"""Shared finite-array and control validation for Chapter 11."""
import numpy as np


def integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')


def array(value, *, ndim=None, name='array'):
    if np.iscomplexobj(value):
        raise ValueError(f'{name} must be real')
    result = np.asarray(value, dtype=float)
    if result.size == 0 or (ndim is not None and result.ndim != ndim) or not np.all(np.isfinite(result)):
        raise ValueError(f'{name} must be nonempty, finite, and have the required shape')
    return result


def matrix(value, name='samples'):
    result = array(value, ndim=2, name=name)
    if min(result.shape) == 0:
        raise ValueError(f'{name} must have nonempty rows and columns')
    return result


def bipolar(value, ndim):
    result = array(value, ndim=ndim, name='bipolar states')
    if not np.all((result == -1) | (result == 1)):
        raise ValueError('Bipolar states must contain only -1 and +1')
    return result.astype(int)
