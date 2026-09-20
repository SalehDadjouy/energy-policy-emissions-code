"""Approved likelihood diagnosis only; never exports a production fit."""
import argparse
from dataclasses import asdict
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import warnings

import numpy as np
from scipy.linalg import cho_factor, cho_solve, toeplitz
from statsmodels.tsa.arima.model import ARIMA

from adapter import HERE, PUBLIC, COMMON_PATH, digest, _load_common
from orientation_bfgs_qualification import verify as verify_previous, INPUT, ROOT as PREVIOUS
from reproduction_policy import complete_reproduction

COMMON = _load_common()
ROOT = HERE / 'candidate_outputs/shock_orientation/likelihood_diagnostic_001'
LOCK = ROOT / 'lock.json'
CASES = (('finite_length', 500), ('finite_length', 0), ('longer_length', 0))
INTERIOR = ((0., 0.), (.5, 0.), (-.5, 0.), (.4, .2), (.4, -.2))
STEPS = (1e-4, 1e-5, 1e-6)


def plain(x):
    if isinstance(x, np.ndarray):
        return plain(x.tolist())
    if isinstance(x, np.generic):
        return plain(x.item())
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    if callable(x):
        return getattr(x, '__name__', type(x).__name__)
    return x


def write(path, obj):
    with path.open('x') as f:
        json.dump(plain(obj), f, indent=2, sort_keys=True, allow_nan=False)


def grid_coordinates():
    return np.unique(np.r_[np.linspace(-.95, .95, 41),
                           [s * (1 - 10.**(-j)) for s in (-1, 1) for j in range(2, 7)]])


def phi_from_k(k1, k2):
    return np.array([k1 * (1-k2), k2])


def radius(phi):
    return float(max(abs(np.linalg.eigvals([[phi[0], phi[1]], [1., 0.]]))))


def covariance(phi, n):
    p1, p2 = phi
    if radius(phi) >= 1:
        raise ValueError('Nonstationary coordinates')
    g = np.empty(n)
    g[0] = (1-p2) / ((1+p2) * ((1-p2)**2 - p1**2))
    if n > 1:
        g[1] = p1 * g[0] / (1-p2)
    for k in range(2, n):
        g[k] = p1*g[k-1] + p2*g[k-2]
    return toeplitz(g)


def profile(x, phi):
    rmat = covariance(phi, len(x))
    factor = cho_factor(rmat, lower=True)
    one = np.ones(len(x))
    mu = float((one @ cho_solve(factor, x)) / (one @ cho_solve(factor, one)))
    residual = x - mu
    q = float(residual @ cho_solve(factor, residual) / len(x))
    if q <= 0 or not np.isfinite(q):
        raise ValueError('Invalid profiled variance')
    ld = float(2*np.log(np.diag(factor[0])).sum())
    ll = -.5*(len(x)*(np.log(2*np.pi)+1+np.log(q))+ld)
    return dict(phi=list(phi), mean=mu, variance=q, loglike=float(ll),
                condition=float(np.linalg.cond(rmat)), radius=radius(phi))


def gaussian_loglike(x, params):
    mu, p1, p2, q = params
    factor = cho_factor(q * covariance((p1, p2), len(x)), lower=True)
    r = x - mu
    return float(-.5*(len(x)*np.log(2*np.pi) +
                       2*np.log(np.diag(factor[0])).sum() + r @ cho_solve(factor, r)))


def model_for(x):
    s = COMMON.SPEC
    return ARIMA(x, order=s.order, trend=s.trend,
                 enforce_stationarity=s.enforce_stationarity,
                 enforce_invertibility=s.enforce_invertibility,
                 concentrate_scale=s.concentrate_scale, missing=s.missing)


def compare_point(x, model, phi, compare_common=False):
    row = {'phi': list(phi)}
    try:
        row.update(profile(x, phi))
        p = [row['mean'], *phi, row['variance']]
        row['state_space_loglike'] = float(model.loglike(p, transformed=True))
        row['difference'] = row['loglike'] - row['state_space_loglike']
        row['tolerance'] = 1e-8*max(1., abs(row['state_space_loglike']))
        row['agreement'] = bool(np.isfinite(row['difference']) and abs(row['difference']) <= row['tolerance'])
        row['suspect'] = row['condition'] > 1e12
        row['status'] = 'ok' if row['agreement'] and not row['suspect'] else 'flagged'
        if compare_common:
            fit = COMMON.FittedAR2(tuple(x), 0., 1., tuple(phi), 1., 0., radius(phi), '', (), '{}', COMMON.SPEC)
            raw, _ = COMMON.stationary_covariance(fit, len(x))
            rmat = covariance(phi, len(x))
            row['covariance_difference'] = float(np.max(abs(raw-rmat)))
            row['covariance_agreement'] = bool(np.allclose(raw, rmat, atol=1e-10, rtol=1e-10))
    except Exception as exc:
        row.update(status='unavailable', error=f'{type(exc).__name__}: {exc}')
    return row


def expected_optimizer(config, rep, solver):
    if solver == 'bfgs':
        if rep != 500 or config != 'finite_length':
            return None
        records = json.loads((PREVIOUS / 'screen_001.json').read_text())['records']
        rec = next(r for r in records if r['config'] == config and r['rep'] == rep)
        return rec['diagnostics']['fallback']['diagnostics']['optimizer']
    records = json.loads((INPUT.parent / 'run_001.json').read_text())['records']
    rec = next(r for r in records if r['config'] == config and r['rep'] == rep)
    if rec['status'] != 'ok':
        return rec['diagnostics']['optimizer']
    qualified = json.loads((PREVIOUS / 'screen_001.json').read_text())['records']
    saved = next(r for r in qualified if r['config'] == config and r['rep'] == rep)
    return json.loads(saved['fit']['optimizer_diagnostics'])


def replay(x, config, rep, solver):
    model = model_for(x)
    trace = []
    def capture(v):
        trace.append(np.array(v, dtype=float, copy=True))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = model.fit(method=COMMON.SPEC.estimator, method_kwargs={
            'method': solver, 'maxiter': COMMON.SPEC.maxiter, 'disp': 0, 'callback': capture})
    start = np.asarray(result.mle_settings['start_params']).copy()
    end = np.asarray(result.mlefit.params).copy()
    expected = expected_optimizer(config, rep, solver)
    recorded = plain(result.mle_retvals)
    rec = dict(solver=solver, optimizer=recorded, settings=plain(result.mle_settings),
               warnings=[f'{w.category.__name__}: {w.message}' for w in caught],
               start_unconstrained=start, end_unconstrained=end,
               start_constrained=model.transform_params(start), end_constrained=result.params,
               trajectory_unconstrained=trace,
               archived_optimizer_exact=None if expected is None else recorded == expected,
               archived_optimizer=expected)
    return rec


def derivatives(x, u):
    model = model_for(x)
    n = len(x)
    result = {'unconstrained': u}
    try:
        cs = -model.score(u, transformed=False, method='approx', approx_complex_step=True)/n
        jac = model.transform_jacobian(u)
        checks = []
        for scale in STEPS:
            grad = np.empty(len(u))
            for j in range(len(u)):
                step = scale*max(1., abs(u[j]))
                hi, lo = u.copy(), u.copy()
                hi[j] += step
                lo[j] -= step
                grad[j] = -(model.loglike(hi, transformed=False)-model.loglike(lo, transformed=False))/(2*step*n)
            delta = float(np.max(abs(cs-grad)))
            checks.append(dict(step=scale, gradient=grad, max_difference=delta,
                               agreement=bool(delta <= 1e-5*max(1., float(np.max(abs(cs)))))))
        result.update(status='ok', complex_step_gradient=cs, jacobian=jac, checks=checks,
                      agreement=all(c['agreement'] for c in checks))
    except Exception as exc:
        result.update(status='unavailable', error=f'{type(exc).__name__}: {exc}')
    return result


def locate(x, u):
    model = model_for(x)
    params = model.transform_params(u)
    row = dict(unconstrained=u, constrained=params, k1=float(params[1]/(1-params[2])), k2=float(params[2]))
    row['state_space_loglike'] = float(model.loglike(params, transformed=True))
    try:
        row['gaussian_loglike'] = gaussian_loglike(x, params)
        row['difference'] = row['gaussian_loglike']-row['state_space_loglike']
        row['profile'] = compare_point(x, model, params[1:3])
    except Exception as exc:
        row['error'] = f'{type(exc).__name__}: {exc}'
    return row


def verify():
    verify_previous()
    lock = json.loads(LOCK.read_text())
    for p, expected in lock['sha256'].items():
        if digest(Path(p)) != expected:
            raise RuntimeError(f'Frozen dependency changed: {p}')
    if subprocess.check_output([sys.executable, '-B', '-m', 'pip', 'freeze'], text=True) != lock['pip_freeze']:
        raise RuntimeError('Dependency inventory changed')


def freeze():
    verify_previous()
    paths = [Path(__file__), HERE/'test_orientation_likelihood.py', HERE/'Orientation_Likelihood_Diagnostic_Protocol.txt',
             HERE/'reproduction_policy.py', HERE/'test_reproduction_policy.py', COMMON_PATH,
             PREVIOUS/'lock.json', PREVIOUS/'screen_001.json', PREVIOUS/'screen_002.json']
    import statsmodels.tsa.statespace.sarimax
    import statsmodels.tsa.statespace.tools
    import statsmodels.tsa.statespace.kalman_filter
    import statsmodels.tsa.statespace.initialization
    import statsmodels.tsa.arima.model
    import scipy.optimize._lbfgsb_py
    import scipy.optimize._optimize
    modules = [statsmodels.tsa.statespace.sarimax, statsmodels.tsa.statespace.tools,
               statsmodels.tsa.statespace.kalman_filter, statsmodels.tsa.statespace.initialization,
               statsmodels.tsa.arima.model, scipy.optimize._lbfgsb_py, scipy.optimize._optimize]
    paths += [Path(inspect.getfile(m)) for m in modules]
    # Hash compiled likelihood and optimizer implementations as well as Python sources.
    for directory in (Path(inspect.getfile(modules[0])).parent, Path(inspect.getfile(modules[-1])).parent):
        paths.extend(sorted(directory.glob('*.so')))
    ROOT.mkdir(exist_ok=False)
    write(LOCK, dict(approval='User approved implementation and execution of the isolated likelihood diagnostic only.',
        protocol='orientation_likelihood_diagnostic_v1', sha256={str(p): digest(p) for p in paths},
        prior_lock=str(PREVIOUS/'lock.json'), cases=CASES, interior=INTERIOR,
        grid_rule='41 evenly spaced -0.95 to 0.95 union +/- (1-10^-j), j=2..6; Cartesian product',
        derivative_steps=STEPS, diagnostic_ll_rtol=1e-8, derivative_flag=1e-5, suspect_condition=1e12,
        randomization='None; uses frozen instrument paths from original seeds.', specification=asdict(COMMON.SPEC),
        pip_freeze=subprocess.check_output([sys.executable, '-B', '-m', 'pip', 'freeze'], text=True),
        full_simulation_authorized=False))
    print('Frozen before diagnostic numerical evaluations', flush=True)


def run(name):
    verify()
    out = ROOT/name
    out.mkdir(exist_ok=False)
    with np.load(INPUT) as paths:
        prechecks = {}
        for config, rep in CASES:
            z = paths[config][rep]
            x = (z-z.mean())/z.std(ddof=0)
            m = model_for(x)
            points = [compare_point(x, m, phi, compare_common=True) for phi in INTERIOR]
            prechecks[config, rep] = points
        all_interior_pass = all(r.get('agreement') and r.get('covariance_agreement')
                               for points in prechecks.values() for r in points)
        records = []
        for config, rep in CASES:
            z = paths[config][rep]
            x = (z-z.mean())/z.std(ddof=0)
            model = model_for(x)
            rec = dict(config=config, rep=rep, n=len(x), raw_mean=float(z.mean()), raw_scale=float(z.std(ddof=0)),
                       standardized=x, parameter_names=model.param_names,
                       initialization=model.ssm.initialization.initialization_type,
                       initialization_blocks={str(k): v.initialization_type for k,v in model.ssm.initialization.blocks.items()},
                       loglikelihood_burn=model.loglikelihood_burn)
            rec['interior'] = prechecks[config, rep]
            rec['likelihood_gate'] = bool(model.param_names == ['const', 'ar.L1', 'ar.L2', 'sigma2'] and
                model.loglikelihood_burn == 0 and all(r.get('agreement') and r.get('covariance_agreement') for r in rec['interior']))
            # Preserve failure evidence, but do not interpret a surface of a different likelihood.
            if all_interior_pass and rec['likelihood_gate']:
                rec['solvers'] = []
                for solver in ('lbfgs', 'bfgs'):
                    replayed = replay(x, config, rep, solver)
                    if replayed['archived_optimizer_exact'] is not False:
                        replayed['start'] = locate(x, replayed['start_unconstrained'])
                        replayed['end'] = locate(x, replayed['end_unconstrained'])
                        replayed['trajectory'] = [locate(x, u) for u in replayed['trajectory_unconstrained']]
                        replayed['derivatives_start'] = derivatives(x, replayed['start_unconstrained'])
                        replayed['derivatives_end'] = derivatives(x, replayed['end_unconstrained'])
                    rec['solvers'].append(replayed)
                if rep == 500:
                    grid = []
                    for k1 in grid_coordinates():
                        for k2 in grid_coordinates():
                            row = compare_point(x, model, phi_from_k(k1,k2))
                            row.update(k1=float(k1), k2=float(k2))
                            grid.append(row)
                    write(out/'grid.json', grid)
                    rec['grid_count'] = len(grid)
                    valid = [r for r in grid if r['status'] == 'ok']
                    rec['grid_maximum_reliable'] = max(valid, key=lambda r:r['loglike']) if valid else None
                    rec['grid_status_counts'] = {s:sum(r['status']==s for r in grid) for s in ('ok','flagged','unavailable')}
            records.append(rec)
            print(f'{name} {config} rep {rep}: likelihood gate={rec["likelihood_gate"]}', flush=True)
    verify()
    write(out/'results.json', records)


def comparisons():
    a, b = ROOT/'run_001', ROOT/'run_002'
    files_a = sorted(p.name for p in a.glob('*.json'))
    files_b = sorted(p.name for p in b.glob('*.json'))
    matches = {p:digest(a/p)==digest(b/p) for p in files_a if (b/p).exists()}
    return dict(pass_=files_a==files_b and all(matches.values()), files=matches)


def archive_comparison():
    results = json.loads((ROOT/'run_001/results.json').read_text())
    flags = [s['archived_optimizer_exact'] for r in results for s in r.get('solvers',[]) if s['archived_optimizer_exact'] is not None]
    return {'pass':len(flags)==4 and all(flags), 'checks':len(flags), 'scope':'Original L-BFGS on three cases and failed BFGS on replication 500'}


def execute():
    verify()
    tests = subprocess.run([sys.executable, '-B', '-m', 'unittest', '-v',
        'test_reproduction_policy', 'test_orientation_likelihood'], cwd=HERE, capture_output=True, text=True)
    (ROOT/'tests.txt').open('x').write(tests.stdout+tests.stderr)
    if tests.returncode:
        raise RuntimeError(tests.stdout+tests.stderr)
    def child(name):
        result = subprocess.run([sys.executable, '-B', str(Path(__file__)), name], cwd=HERE,
                                capture_output=True, text=True)
        (ROOT/f'{name}.log').open('x').write(result.stdout+result.stderr)
        print(result.stdout, flush=True)
        if result.returncode:
            raise RuntimeError(result.stderr)
    child('run_001')
    def pair():
        r = comparisons()
        r['pass'] = r.pop('pass_')
        return r
    result = complete_reproduction(verify, lambda:child('run_002'), pair, archive_comparison)
    write(ROOT/'verification.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze','execute','run_001','run_002','verify'])
    stage = parser.parse_args().stage
    if stage == 'freeze':
        freeze()
    elif stage == 'execute':
        execute()
    elif stage == 'verify':
        verify()
        print('Frozen dependencies unchanged')
    else:
        run(stage)
