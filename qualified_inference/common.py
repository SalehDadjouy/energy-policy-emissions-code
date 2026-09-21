"""Isolated fixed-order design-based inference; no production dependencies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import MutableMapping
import warnings

import numpy as np
import scipy
from scipy.linalg import solve_discrete_lyapunov, toeplitz
from scipy.stats import norm, t
import statsmodels
from statsmodels.tsa.arima.model import ARIMA


TOL = 1e-10


class InferenceFailure(ValueError):
    def __init__(self, code, detail, diagnostics=None):
        super().__init__(detail)
        self.code = code
        self.diagnostics = diagnostics or {}


@dataclass(frozen=True)
class Specification:
    order: tuple[int, int, int] = (2, 0, 0)
    trend: str = 'c'
    enforce_stationarity: bool = True
    enforce_invertibility: bool = True
    concentrate_scale: bool = False
    missing: str = 'raise'
    estimator: str = 'statespace'
    optimizer: str = 'lbfgs'
    maxiter: int = 500


SPEC = Specification()


@dataclass(frozen=True)
class FittedAR2:
    raw_path: tuple[float, ...]
    raw_mean: float
    raw_scale: float
    phi: tuple[float, float]
    innovation_variance: float
    fitted_mean: float
    state_radius: float
    cache_key: str
    warning_messages: tuple[str, ...]
    optimizer_diagnostics: str
    specification: Specification
    optimizer_settings: str = '{}'


@dataclass(frozen=True)
class Interval:
    status: str
    se: float | None = None
    lower: float | None = None
    upper: float | None = None
    critical: float | None = None
    reason: str | None = None
    detail: str | None = None


def vector(value, name):
    try:
        x = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InferenceFailure('invalid_input', f'{name} must be numeric.') from exc
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise InferenceFailure('invalid_input', f'{name} must be a finite vector.')
    return x


def fit_key(z, spec=SPEC):
    x = vector(z, 'instrument')
    versions = (np.__version__, scipy.__version__, statsmodels.__version__)
    header = json.dumps({'specification': asdict(spec), 'versions': versions,
                         'length': len(x)}, sort_keys=True).encode()
    return hashlib.sha256(header + x.astype('<f8').tobytes()).hexdigest()


def _json_scalar(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def validate_fit(result):
    retvals = getattr(result, 'mle_retvals', {})
    if not retvals.get('converged', False):
        raise InferenceFailure('nonconvergence', 'AR(2) maximum likelihood did not converge.')
    names = list(result.param_names)
    values = np.asarray(result.params, dtype=float)
    if len(names) != len(values) or len(set(names)) != len(names):
        raise InferenceFailure('invalid_parameters', 'Missing or duplicate parameter names.')
    params = dict(zip(names, values))
    if not {'const', 'ar.L1', 'ar.L2', 'sigma2'} <= params.keys():
        raise InferenceFailure('invalid_parameters', 'Named AR(2) parameters are required.')
    if not np.all(np.isfinite(values)):
        raise InferenceFailure('invalid_parameters', 'Fitted parameters are not finite.')
    q = float(params['sigma2'])
    if q <= 0:
        raise InferenceFailure('invalid_variance', 'Innovation variance must be positive.')
    state_q = np.asarray(result.model.ssm['state_cov'], dtype=float)
    if state_q.size != 1 or not np.isfinite(state_q).all():
        raise InferenceFailure('variance_mismatch', 'Unexpected innovation covariance shape.')
    if abs(q - float(state_q.ravel()[0])) > TOL * q:
        raise InferenceFailure('variance_mismatch', 'Named variance differs from state covariance.')
    phi = (float(params['ar.L1']), float(params['ar.L2']))
    radius = float(max(abs(np.linalg.eigvals([[phi[0], phi[1]], [1., 0.]]))))
    if not math.isfinite(radius) or radius >= 1.:
        raise InferenceFailure('nonstationary', 'Fitted AR(2) state is not stationary.')
    return phi, q, float(params['const']), radius


def fit_instrument(z, *, cache: MutableMapping | None = None, spec=SPEC):
    x = vector(z, 'full instrument')
    if len(x) <= 2:
        raise InferenceFailure('invalid_input', 'AR(2) requires observations beyond its two lags.')
    mean, scale = float(x.mean()), float(x.std(ddof=0))
    if not math.isfinite(mean) or not math.isfinite(scale) or scale <= 0:
        raise InferenceFailure('invalid_input', 'Instrument mean/scale is not usable.')
    key = fit_key(x, spec)
    if cache is not None and key in cache:
        return cache[key]
    if spec.order != (2, 0, 0):
        raise InferenceFailure('unsupported_model', 'This protocol supports only AR(2).')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            result = ARIMA(
                (x - mean) / scale, order=spec.order, trend=spec.trend,
                enforce_stationarity=spec.enforce_stationarity,
                enforce_invertibility=spec.enforce_invertibility,
                concentrate_scale=spec.concentrate_scale, missing=spec.missing,
            ).fit(method=spec.estimator, method_kwargs={
                'method': spec.optimizer, 'maxiter': spec.maxiter, 'disp': 0,
            })
        except Exception as exc:
            raise InferenceFailure('fit_exception', f'{type(exc).__name__}: {exc}', {
                'warnings': [f'{w.category.__name__}: {w.message}' for w in caught]
            }) from exc
    messages = tuple(f'{w.category.__name__}: {w.message}' for w in caught)
    try:
        phi, q, mu, radius = validate_fit(result)
    except InferenceFailure as exc:
        exc.diagnostics['warnings'] = messages
        exc.diagnostics['optimizer'] = getattr(result, 'mle_retvals', {})
        raise
    fitted = FittedAR2(tuple(map(float, x)), mean, scale, phi, q, mu, radius,
                      key, messages, json.dumps(result.mle_retvals, default=_json_scalar,
                                                sort_keys=True), spec,
                      json.dumps(getattr(result, 'mle_settings', {}),
                                 default=_json_scalar, sort_keys=True))
    if cache is not None:
        cache[key] = fitted
    return fitted


def check_covariance(matrix, *, reference_norm, centered=False):
    a = np.asarray(matrix, float)
    if (a.ndim != 2 or not a.size or a.shape[0] != a.shape[1] or not np.all(np.isfinite(a))
            or not math.isfinite(reference_norm) or reference_norm <= 0):
        raise InferenceFailure('invalid_covariance', 'Covariance dimensions or values are invalid.')
    tolerance = TOL * reference_norm
    if np.linalg.norm(a - a.T) > tolerance:
        raise InferenceFailure('invalid_covariance', 'Covariance is not symmetric within tolerance.')
    try:
        smallest = float(np.linalg.eigvalsh((a + a.T) / 2).min())
    except np.linalg.LinAlgError as exc:
        raise InferenceFailure('invalid_covariance', 'Covariance eigenvalue check failed.') from exc
    if smallest < -tolerance:
        raise InferenceFailure('invalid_covariance', 'Covariance has a negative eigenvalue beyond roundoff.')
    if centered and np.linalg.norm(a @ np.ones(len(a))) > tolerance:
        raise InferenceFailure('invalid_covariance', 'Covariance is not centered within tolerance.')


def stationary_covariance(fit, length):
    if not isinstance(length, (int, np.integer)) or length < 1:
        raise InferenceFailure('invalid_input', 'Covariance length must be a positive integer.')
    phi, q = fit.phi, fit.innovation_variance
    A = np.array([[phi[0], phi[1]], [1., 0.]])
    if max(abs(np.linalg.eigvals(A))) >= 1 or q <= 0 or not math.isfinite(q):
        raise InferenceFailure('nonstationary', 'No positive stationary AR(2) covariance.')
    try:
        state = solve_discrete_lyapunov(A, np.diag([q, 0.]))
    except np.linalg.LinAlgError as exc:
        raise InferenceFailure('invalid_covariance', 'Stationary covariance solution failed.') from exc
    check_covariance(state, reference_norm=float(np.linalg.norm(state)))
    gamma = np.array([(np.linalg.matrix_power(A, k) @ state)[0, 0]
                      for k in range(length)]) * fit.raw_scale**2
    raw = toeplitz(gamma)
    raw_norm = float(np.linalg.norm(raw))
    check_covariance(raw, reference_norm=raw_norm)
    C = np.eye(length) - np.ones((length, length)) / length
    omega = C @ raw @ C
    check_covariance(omega, reference_norm=raw_norm, centered=True)
    return raw, omega


def score_variance(residual, omega, raw_norm):
    r = vector(residual, 'treatment-adjusted residual')
    omega = np.asarray(omega, float)
    check_covariance(omega, reference_norm=raw_norm, centered=True)
    if omega.shape != (len(r), len(r)):
        raise InferenceFailure('invalid_input', 'Residual and covariance lengths differ.')
    r = r - r.mean()
    value = float(r @ omega @ r)
    bound = TOL * raw_norm * float(r @ r)
    if not math.isfinite(value) or value < -bound:
        raise InferenceFailure('invalid_score', 'Score variance is invalid beyond roundoff.')
    return max(value, 0.)


def critical_value(length):
    if length <= 2:
        raise InferenceFailure('invalid_input', 'Interval reference requires more than two periods.')
    return float(t.ppf(.975, length - 2) if length <= 10 else norm.ppf(.975))


def interval_from_fit(y, w, z, *, tau_hat, pi_hat, fit):
    """Consume frozen point estimates; never fit weights or alter point results."""
    try:
        y, w, z = vector(y, 'outcome'), vector(w, 'treatment'), vector(z, 'post instrument')
        if len(y) != len(w) or len(y) != len(z):
            raise InferenceFailure('invalid_input', 'Post-learning vector lengths differ.')
        if not np.array_equal(z, np.asarray(fit.raw_path[-len(z):])):
            raise InferenceFailure('wrong_instrument', 'Fitted instrument does not match the post window.')
        if not math.isfinite(tau_hat) or not math.isfinite(pi_hat):
            raise InferenceFailure('invalid_point', 'Point-estimation inputs are not finite.')
        zc = z - z.mean()
        denominator = abs(float(zc @ w))
        other = abs(pi_hat) * float(zc @ zc)
        if not math.isfinite(denominator) or denominator <= 0:
            raise InferenceFailure('undefined_ratio', 'The observed ratio denominator is zero or nonfinite.')
        if abs(denominator - other) > TOL * max(denominator, other):
            raise InferenceFailure('point_mismatch', 'Supplied slope does not match the frozen denominator.')
        raw, omega = stationary_covariance(fit, len(z))
        variance = score_variance(y - tau_hat*w, omega, float(np.linalg.norm(raw)))
        se = math.sqrt(variance) / denominator
        crit = critical_value(len(z))
        lower, upper = tau_hat - crit*se, tau_hat + crit*se
        if not np.all(np.isfinite([se, lower, upper])):
            raise InferenceFailure('nonfinite_interval', 'A finite interval could not be calculated.')
        return Interval('ok', se, lower, upper, crit)
    except InferenceFailure as exc:
        return Interval('unavailable', reason=exc.code, detail=str(exc))


def attach_interval(record, interval):
    """Keep the point/comparator record intact even when ARIMA is unavailable."""
    fields = {'arima_' + key: value for key, value in asdict(interval).items()}
    if fields.keys() & record.keys():
        raise ValueError('Refusing to overwrite existing ARIMA result fields.')
    return {**record, **fields}
