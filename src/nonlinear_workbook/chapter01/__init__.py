"""Independent Chapter 1 implementations; see docs/chapter01.md."""
from .maps import (collatz_step, logistic_step, logistic_derivative, logistic_map,
    bernoulli_step, tent_step, bungalow_step, gauss_step, sine_step, sine_derivative,
    circle_step, polynomial_random_step, lcg_step, quadratic_step)
from .numerics import (iterate, IterationResult, dft, radix2_fft, newton, NewtonResult,
    logistic_sweep, period_doubling, feigenbaum_ratios)
from .diagnostics import (stability, scalar_lyapunov, invariant_density, DensityResult,
    moments, autocovariance, monte_carlo, MonteCarloResult, rotation_number,
    symbolic_partition, inverse_symbolic)
from .maps import (henon_step, henon_jacobian, lozi_step, lozi_jacobian,
    standard_step, standard_jacobian, ikeda_step, ikeda_jacobian,
    coupled_logistic_step, coupled_logistic_jacobian, rtn_step, rtn_jacobian)
from .numerics import vector_newton, complex_newton_basins, BasinResult
from .diagnostics import (fixed_point_classification, known_lyapunov_spectrum,
    KnownSpectrumResult, correlation_integral, CorrelationIntegralResult,
    box_counts, BoxCountResult, fit_scaling, ScalingFit, boundedness_scan,
    BoundednessResult)
