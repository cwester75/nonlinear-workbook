"""Perceptrons and one-hidden-layer sigmoid backpropagation (pp.387-418)."""
from dataclasses import dataclass
import numpy as np
from ._validation import array, integer, matrix, positive
from .similarity import logistic, threshold


def _finite_scalar(value, name):
    if not np.isscalar(value) or not np.isfinite(value):
        raise ValueError(f'{name} must be a finite scalar')


def perceptron_predict(samples, weights, *, bias=0.):
    samples, weights = matrix(samples), array(weights, ndim=1, name='weights')
    _finite_scalar(bias, 'bias')
    if samples.shape[1] != len(weights): raise ValueError('Weights must match features')
    net = samples @ weights + bias
    if not np.all(np.isfinite(net)): raise ValueError('Perceptron net overflow')
    return threshold(net)


@dataclass(frozen=True)
class PerceptronResult:
    weights: np.ndarray
    bias: float
    predictions: np.ndarray
    error_counts: np.ndarray
    epochs: int
    updates: int
    status: str
    parameters: dict


def perceptron_train(samples, targets, *, learning_rate=.5, max_epochs=1000,
                     initial_weights=None, initial_bias=0., seed=20261012, shuffle=True):
    """Online error correction, including bias; verify the whole dataset each epoch."""
    samples, targets = matrix(samples), array(targets, ndim=1, name='targets')
    if len(targets) != len(samples) or not np.all((targets == 0) | (targets == 1)):
        raise ValueError('One 0/1 target per sample required')
    positive(learning_rate, 'learning_rate'); integer(max_epochs, 'max_epochs'); integer(seed, 'seed', 0)
    _finite_scalar(initial_bias, 'initial_bias')
    if not isinstance(shuffle, (bool, np.bool_)): raise ValueError('shuffle must be boolean')
    rng = np.random.default_rng(seed)
    weights = rng.uniform(-.01, .01, samples.shape[1]) if initial_weights is None else array(initial_weights, ndim=1).copy()
    if len(weights) != samples.shape[1]: raise ValueError('Initial weights must match features')
    bias = float(initial_bias); initial = weights.copy()
    predictions = perceptron_predict(samples, weights, bias=bias)
    errors = [int(np.count_nonzero(predictions != targets))]; updates = 0
    parameters = dict(learning_rate=learning_rate, max_epochs=max_epochs, seed=seed, shuffle=bool(shuffle),
                      initial_weights=initial, initial_bias=initial_bias, equality='positive')
    if errors[-1] == 0:
        return PerceptronResult(weights,bias,predictions,np.array(errors),0,0,'converged',parameters)
    for epoch in range(1, max_epochs + 1):
        order = rng.permutation(len(samples)) if shuffle else np.arange(len(samples))
        for i in order:
            prediction = perceptron_predict(samples[i:i+1], weights, bias=bias)[0]
            correction = learning_rate * (targets[i] - prediction)
            if correction != 0:
                weights += correction * samples[i]
                bias += correction; updates += 1
        predictions = perceptron_predict(samples, weights, bias=bias)
        errors.append(int(np.count_nonzero(predictions != targets)))
        if errors[-1] == 0:
            return PerceptronResult(weights,bias,predictions,np.array(errors),epoch,updates,'converged',parameters)
    return PerceptronResult(weights,bias,predictions,np.array(errors),max_epochs,updates,'max_epochs',parameters)


def xor_network(samples):
    """Book pp.401-405 explicit two-hidden-unit threshold XOR, no training."""
    samples = matrix(samples)
    if samples.shape[1] != 2 or not np.all((samples == 0) | (samples == 1)):
        raise ValueError('Explicit XOR requires two binary features')
    hidden = threshold(samples @ np.array([[1,-1],[-1,1]]).T - .5)
    return threshold(hidden.sum(axis=1) - .5)


@dataclass(frozen=True)
class MLPParameters:
    hidden_weights: np.ndarray   # hidden x input
    hidden_bias: np.ndarray      # hidden
    output_weights: np.ndarray   # output x hidden
    output_bias: np.ndarray      # output


@dataclass(frozen=True)
class MLPGradients:
    hidden_weights: np.ndarray
    hidden_bias: np.ndarray
    output_weights: np.ndarray
    output_bias: np.ndarray


def _parameters(parameters):
    if not isinstance(parameters, MLPParameters): raise ValueError('Expected MLPParameters')
    w = matrix(parameters.hidden_weights, 'hidden weights')
    b = array(parameters.hidden_bias, ndim=1, name='hidden bias')
    v = matrix(parameters.output_weights, 'output weights')
    c = array(parameters.output_bias, ndim=1, name='output bias')
    if len(b) != w.shape[0] or v.shape[1] != w.shape[0] or len(c) != v.shape[0]:
        raise ValueError('Incompatible MLP parameter shapes')
    return MLPParameters(w,b,v,c)


def mlp_initialize(n_features, *, hidden_width=2, n_outputs=1, seed=20261012):
    integer(n_features, 'n_features'); integer(hidden_width, 'hidden_width'); integer(n_outputs, 'n_outputs'); integer(seed, 'seed', 0)
    rng = np.random.default_rng(seed)
    return MLPParameters(rng.uniform(-.5,.5,(hidden_width,n_features)),rng.uniform(-.5,.5,hidden_width),
                         rng.uniform(-.5,.5,(n_outputs,hidden_width)),rng.uniform(-.5,.5,n_outputs))


def _forward(samples, parameters, hidden_slope, output_slope):
    positive(hidden_slope, 'hidden_slope'); positive(output_slope, 'output_slope')
    if samples.shape[1] != parameters.hidden_weights.shape[1]: raise ValueError('Input features do not match MLP')
    with np.errstate(over='ignore', invalid='ignore'):
        hidden_net = samples @ parameters.hidden_weights.T + parameters.hidden_bias
        hidden = logistic(hidden_net, slope=hidden_slope)
        output_net = hidden @ parameters.output_weights.T + parameters.output_bias
        output = logistic(output_net, slope=output_slope)
    return hidden, output


def mlp_forward(samples, parameters, *, hidden_slope=1., output_slope=1.):
    """Always returns a sample-by-output array; bias is represented separately."""
    samples, parameters = matrix(samples), _parameters(parameters)
    return _forward(samples, parameters, hidden_slope, output_slope)[1]


def mlp_loss_and_gradients(samples, targets, parameters, *, hidden_slope=1., output_slope=1.):
    """Loss = mean over samples of half the sum of squared output errors.

    Every gradient uses the same current weights. No gradient is evaluated after
    applying another layer's update. Output/hidden sigmoid slopes are included.
    """
    samples, targets, parameters = matrix(samples), matrix(targets,'targets'), _parameters(parameters)
    if targets.shape != (len(samples),len(parameters.output_bias)) or np.any((targets<0)|(targets>1)):
        raise ValueError('Targets must match samples/outputs and be in [0,1]')
    hidden, output = _forward(samples, parameters, hidden_slope, output_slope)
    difference = output - targets
    loss = float(.5 * np.mean(np.sum(difference ** 2, axis=1)))
    with np.errstate(over='ignore', invalid='ignore'):
        output_delta = difference * output_slope * output * (1-output) / len(samples)
        hidden_delta = (output_delta @ parameters.output_weights) * hidden_slope * hidden * (1-hidden)
        gradients = MLPGradients(hidden_delta.T @ samples, hidden_delta.sum(axis=0),
                                 output_delta.T @ hidden, output_delta.sum(axis=0))
    if not np.isfinite(loss) or not all(np.all(np.isfinite(v)) for v in
            (gradients.hidden_weights, gradients.hidden_bias, gradients.output_weights, gradients.output_bias)):
        raise ValueError('Nonfinite MLP loss or gradients')
    return loss, gradients


@dataclass(frozen=True)
class MLPResult:
    weights: MLPParameters
    predictions: np.ndarray
    losses: np.ndarray
    epochs: int
    updates: int
    status: str
    parameters: dict


def _subtract(parameters, gradients, rate):
    return MLPParameters(parameters.hidden_weights-rate*gradients.hidden_weights,
        parameters.hidden_bias-rate*gradients.hidden_bias,
        parameters.output_weights-rate*gradients.output_weights,
        parameters.output_bias-rate*gradients.output_bias)


def mlp_train(samples, targets, *, hidden_width=2, initial=None, learning_rate=.05,
              hidden_slope=1., output_slope=1., mode='online', max_epochs=10000,
              tolerance=.00025, seed=20261012, shuffle=True):
    """Fresh full-dataset mean loss after every epoch; online or batch gradients.

    Online gradients average a single sample; batch gradients average all samples.
    Bias vectors are trained. Stops at the requested loss tolerance or epoch cap.
    Seeded initialization and online shuffling are deterministic. Invalid numerical
    states raise ValueError rather than emitting a misleading loss history.
    """
    samples, targets = matrix(samples), matrix(targets,'targets')
    integer(hidden_width,'hidden_width'); integer(max_epochs,'max_epochs'); integer(seed,'seed',0)
    positive(learning_rate,'learning_rate'); positive(tolerance,'tolerance')
    positive(hidden_slope,'hidden_slope'); positive(output_slope,'output_slope')
    if mode not in ('online','batch') or not isinstance(shuffle,(bool,np.bool_)):
        raise ValueError('Invalid MLP mode/shuffle')
    if len(targets) != len(samples): raise ValueError('One target row per sample required')
    rng = np.random.default_rng(seed)
    if initial is None:
        # Separate RNG initialization makes online order independent of whether
        # identical starting weights are supplied explicitly.
        parameters = mlp_initialize(samples.shape[1],hidden_width=hidden_width,n_outputs=targets.shape[1],seed=seed)
    else:
        checked = _parameters(initial)
        if checked.hidden_weights.shape[0] != hidden_width: raise ValueError('Initial hidden width must match hidden_width')
        parameters = MLPParameters(*(v.copy() for v in (checked.hidden_weights,checked.hidden_bias,checked.output_weights,checked.output_bias)))
    starting = parameters
    loss, _ = mlp_loss_and_gradients(samples,targets,parameters,hidden_slope=hidden_slope,output_slope=output_slope)
    losses = [loss]; updates = 0
    controls = dict(hidden_width=hidden_width, learning_rate=learning_rate, hidden_slope=hidden_slope,
        output_slope=output_slope, mode=mode, max_epochs=max_epochs, tolerance=tolerance,
        seed=seed, shuffle=bool(shuffle), initial=starting, loss='mean half-squared error per sample')
    if loss <= tolerance:
        return MLPResult(parameters,mlp_forward(samples,parameters,hidden_slope=hidden_slope,output_slope=output_slope),np.array(losses),0,0,'converged',controls)
    for epoch in range(1,max_epochs+1):
        if mode == 'batch':
            _,gradient = mlp_loss_and_gradients(samples,targets,parameters,hidden_slope=hidden_slope,output_slope=output_slope)
            parameters = _subtract(parameters,gradient,learning_rate); updates += 1
        else:
            order = rng.permutation(len(samples)) if shuffle else np.arange(len(samples))
            for i in order:
                _,gradient = mlp_loss_and_gradients(samples[i:i+1],targets[i:i+1],parameters,hidden_slope=hidden_slope,output_slope=output_slope)
                parameters = _subtract(parameters,gradient,learning_rate); updates += 1
        loss, _ = mlp_loss_and_gradients(samples,targets,parameters,hidden_slope=hidden_slope,output_slope=output_slope)
        losses.append(loss)
        if loss <= tolerance: break
    predictions = mlp_forward(samples,parameters,hidden_slope=hidden_slope,output_slope=output_slope)
    status = 'converged' if loss <= tolerance else 'max_epochs'
    return MLPResult(parameters,predictions,np.array(losses),epoch,updates,status,controls)
