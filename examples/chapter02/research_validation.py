"""Bounded deterministic sweeps; prints JSON, selects no optimal parameters."""
import json
import platform
import numpy as np
from nonlinear_workbook import (logistic_map, kantz_curve, wolf_lyapunov,
    local_lyapunov_spectrum, direct_lyapunov, hurst_rs, lz_complexity)

SEED = 20261010

def main():
    rng = np.random.default_rng(SEED)
    x = logistic_map(1600)
    report = dict(seed=SEED, python=platform.python_version(), numpy=np.__version__,
                  expected_logistic_exponent=float(np.log(2)), parameters={},
                  wolf=[], local_spectrum=[], kantz=[], hurst=[], lz=[])
    for dimension in (1, 2):
        for theiler in (0, 20):
            for radius in (.005, .02):
                params = dict(dimension=dimension, delay=1, theiler=theiler,
                              min_distance=1e-6, max_distance=radius, max_angle=.3, max_steps=4)
                try:
                    result = wolf_lyapunov(x, **params)
                    report['wolf'].append(dict(parameters=params, exponent=result.exponent,
                        segments=len(result.segments), replacements=result.replacements,
                        attempts=result.replacement_attempts, elapsed_time=result.elapsed_time,
                        termination=result.termination_reason))
                except ValueError as error:
                    report['wolf'].append(dict(parameters=params, error=str(error)))
            for neighbors in (8, 24):
                params = dict(dimension=dimension, delay=1, theiler=theiler,
                              neighbors=neighbors, qr_burn_in=20)
                try:
                    result = local_lyapunov_spectrum(x, **params)
                    report['local_spectrum'].append(dict(parameters=params,
                        exponents=result.exponents.tolist(), usable_steps=result.usable_steps,
                        termination=result.termination_reason,
                        max_condition=float(np.max(result.condition_numbers)),
                        ranks=result.fit_ranks.tolist(),
                        running_estimates=result.running_estimates.tolist()))
                except ValueError as error:
                    report['local_spectrum'].append(dict(parameters=params, error=str(error)))
    # Same seeded perturbation across parameter choices; curves expose saturation.
    perturbation = rng.normal(size=len(x))
    for noise in (0., .001, .01):
        for dimension in (1, 2):
            for theiler in (0, 20):
                params = dict(dimension=dimension, delay=1, theiler=theiler, epsilon=.01, horizon=12)
                lags, curve, references = kantz_curve(x + noise * perturbation, **params)
                windows = [(0, 3), (2, 5), (7, 11)]
                report['kantz'].append(dict(parameters=params, noise_std=noise,
                    lags=lags.tolist(), curve=curve.tolist(), references=references,
                    fits=[dict(start=a, stop=b, slope=float(np.polyfit(lags[a:b], curve[a:b], 1)[0]))
                          for a,b in windows]))
    for seed in (SEED, SEED + 1, SEED + 2):
        white = np.random.default_rng(seed).normal(size=32768)
        ar = np.empty_like(white)
        ar[0] = white[0]
        for t in range(1, len(ar)): ar[t] = .9 * ar[t-1] + white[t]
        for label, values in [('iid_gaussian', white), ('ar1_phi_0.9', ar[1000:]), ('logistic', x)]:
            h, scales, rs = hurst_rs(values)
            report['hurst'].append(dict(seed=seed, series=label, H=h,
                scales=scales.tolist(), mean_rs=rs.tolist()))
    sequences = [('constant', '0'*100), ('periodic', '01'*50),
                 ('book_parse', '00011010010001'),
                 ('random', np.random.default_rng(42).integers(0, 2, 4000)),
                 ('logistic_partition', (logistic_map(4000) > .5).astype(int))]
    for label, sequence in sequences:
        count, normalized = lz_complexity(sequence)
        report['lz'].append(dict(series=label, count=count, normalized=normalized,
                                length=len(sequence)))
    report['direct'] = [dict(epsilon=e, exponent=direct_lyapunov(x, epsilon=e, theiler=20)[0])
                        for e in (None, .005)]
    report['parameters'] = dict(logistic_samples=len(x), logistic_r=4., logistic_x0=.618,
        logistic_burn_in=1000, sample_interval=1., ar1_phi=.9, ar1_burn_in=1000,
        random_lz_seed=42)
    print(json.dumps(report, indent=2, allow_nan=False))

if __name__ == '__main__': main()
