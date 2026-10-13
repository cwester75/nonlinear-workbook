"""Run with PYTHONPATH=src; all random examples use seed 20261010."""
import json
import numpy as np
from nonlinear_workbook import correlation, logistic_map, lyapunov_map, kantz_curve, hurst_rs, lz_complexity

x = logistic_map(100000)
small = x[:2500]
lags, curve, references = kantz_curve(small, epsilon=0.005, horizon=5)
rng = np.random.default_rng(20261010)
noise = rng.normal(size=65536)
print(json.dumps({
    "seed": 20261010,
    "pearson_affine": correlation(noise, 2 * noise + 3),
    "logistic_known_map_lyapunov": lyapunov_map(x),
    "logistic_expected_lyapunov": float(np.log(2)),
    "kantz_lags": lags.tolist(), "kantz_curve": curve.tolist(),
    "kantz_references": references,
    "kantz_first_three_lag_slope": float(np.polyfit(lags[:3], curve[:3], 1)[0]),
    "iid_gaussian_rs_hurst": hurst_rs(noise)[0],
    "lz_alternating": lz_complexity("01" * 1000),
    "lz_random_binary": lz_complexity(rng.integers(0, 2, 4000).tolist()),
}, indent=2))
