"""Package-relative bindings for the independently verified inference routines."""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, toeplitz
from scipy.optimize import fmin_l_bfgs_b
from statsmodels.tsa.arima.model import ARIMA
from sim.estimators import slope_ratio
from sim.inference import centered_slope_ratio_moments
from . import common

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCES = HERE / 'sources'
PROVENANCE = json.loads((HERE / 'provenance.json').read_text())

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def verify_sources():
    for name, expected in PROVENANCE['source_files'].items():
        if digest(SOURCES / name) != expected:
            raise RuntimeError('Qualified source changed: ' + name)
    if digest(HERE / 'common.py') != PROVENANCE['common_sha256']:
        raise RuntimeError('Qualified covariance implementation changed')
    return PROVENANCE

def definitions(name, selected, namespace):
    verify_sources()
    path = SOURCES / name
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in selected]
    if {n.name for n in nodes} != set(selected) or len(nodes) != len(selected):
        raise RuntimeError('Incomplete scientific definition inventory')
    context = dict(namespace, __file__=str(path), __name__=__name__)
    import __future__
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec',
                 flags=__future__.annotations.compiler_flag, dont_inherit=True), context)
    return context

from dataclasses import dataclass
_types = definitions('adapter.py', ['ReportingResult', 'ReportingInferenceUnavailable'], {'dataclass': dataclass})
ReportingResult = _types['ReportingResult']
ReportingInferenceUnavailable = _types['ReportingInferenceUnavailable']

_direct = definitions('adapter.py', ['reporting_arima_result'], {
    'np': np, 'math': math, 'verify_lock': verify_sources, '_public_imports': lambda: slope_ratio,
    '_load_common': lambda: common, 'ReportingResult': ReportingResult,
    'ReportingInferenceUnavailable': ReportingInferenceUnavailable})
reporting_arima_result = _direct['reporting_arima_result']

def interval_for_points(y, w, z, z_process, tau, pi):
    """Use the existing point arithmetic and the observed denominator unchanged."""
    verify_sources()
    try:
        fit = common.fit_instrument(np.asarray(z_process, dtype=float))
        interval = common.interval_from_fit(y, w, z, tau_hat=float(tau), pi_hat=float(pi), fit=fit)
    except common.InferenceFailure as exc:
        raise ReportingInferenceUnavailable(f'{exc.code}: {exc}') from exc
    if interval.status != 'ok':
        raise ReportingInferenceUnavailable(f'{interval.reason}: {interval.detail}')
    return ReportingResult(float(tau), float(pi), interval.se, interval.critical,
                           interval.lower, interval.upper, fit.cache_key)

def principal_arima_result(y, w, z, *, z_process, ar_order=2, varz_mode='window'):
    if ar_order != 2 or varz_mode != 'window':
        raise ReportingInferenceUnavailable('Only the frozen AR(2) post-window specification is supported')
    point = centered_slope_ratio_moments(y, w, z)
    return interval_for_points(y, w, z, z_process, point.tau, point.pi)

def orientation_reporter():
    prior = definitions('orientation_likelihood_diagnostic.py',
        ['plain', 'radius', 'covariance', 'gaussian_loglike', 'model_for'],
        {'np': np, 'math': math, 'cho_factor': cho_factor, 'cho_solve': cho_solve,
         'toeplitz': toeplitz, 'ARIMA': ARIMA, 'COMMON': common})
    helper = definitions('verify_orientation_gradient_stopping.py',
        ['analytic_jacobian', 'central', 'comparison'], {'np': np})
    protocol_identity = HERE / 'orientation_protocol_identity'
    def source_identity(path):
        if Path(path) == protocol_identity:
            return PROVENANCE['orientation_protocol_fingerprint']
        return digest(path)
    prior.update(digest=source_identity, STEPS=(1e-4, 1e-5, 1e-6))
    gradient = definitions('orientation_gradient_candidate.py',
        ['retry_key', 'objective_and_gradient', 'corrected_fit', 'ConditionalFitter'], {
            'np': np, 'math': math, 'hashlib': hashlib, 'json': json, 'Path': Path,
            'warnings': warnings, 'fmin_l_bfgs_b': fmin_l_bfgs_b, 'COMMON': common,
            'prior': SimpleNamespace(**prior), 'PROTOCOL': protocol_identity,
            'GRADIENT_RULE': 'analytic_AR2_J_transpose_times_constrained_complex_step_score_v1',
            'OPTIONS': dict(m=10, factr=1e7, pgtol=1e-5, maxfun=15000, maxiter=500, maxls=20, approx_grad=False),
            **{name: helper[name] for name in ('analytic_jacobian', 'central', 'comparison')}})
    wrapper = definitions('orientation_point_reporting.py', ['OrientationReporter'], {
        'np': np, 'math': math, 'COMMON': common, 'ReportingResult': ReportingResult,
        'ConditionalFitter': gradient['ConditionalFitter'],
        'centered_slope_ratio_moments': centered_slope_ratio_moments})
    return wrapper['OrientationReporter']()

_orientation = orientation_reporter()
orientation_arima_result = _orientation

def install_target_checks(module):
    """Retain the qualified separation of structural and archived comparisons."""
    from .target_controls import defer_archive_failures
    verify_sources()
    path = SOURCES / 'target_anchor_compat.py'
    functions = defer_archive_failures(path.read_text(), {
        'np': np, 'pd': pd, 'Any': object, 'PUBLIC': ROOT / 'validation_data',
        'ARIMA_METHOD': 'ARIMA-Z'}, str(path))
    compat = json.loads((SOURCES / 'target_anchor_compat_protocol.json').read_text())
    module.structural_checks = lambda protocol, raw, reps, runtime: functions['compatibility_checks'](
        module, compat, protocol, raw, reps=reps, runtime=runtime)
