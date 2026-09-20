"""Verify derivatives and passively recover stopping evidence; no correction."""
import argparse
import inspect
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import warnings

import numpy as np
import scipy.optimize as optimize
import statsmodels.base.optimizer
import statsmodels.tools.numdiff

import orientation_likelihood_diagnostic as prior
from reproduction_policy import complete_reproduction

HERE = prior.HERE
ROOT = HERE/'candidate_outputs/shock_orientation/gradient_stopping_verification_001'
LOCK = ROOT/'lock.json'
CASES = prior.CASES
STEPS = prior.STEPS
BASELINE = prior.ROOT/'run_001/results.json'
OPTIONS = dict(approx_grad=True, epsilon=1e-5, m=10, factr=1e7, pgtol=1e-5,
               maxfun=15000, maxls=20, maxiter=500)
write, plain, digest = prior.write, prior.plain, prior.digest


def transform(u):
    r1=u[1]/np.sqrt(1+u[1]**2)
    r2=u[2]/np.sqrt(1+u[2]**2)
    return np.array([u[0],-(r1+r2*r1),-r2,u[3]**2])


def analytic_jacobian(u):
    r1=u[1]/np.sqrt(1+u[1]**2)
    r2=u[2]/np.sqrt(1+u[2]**2)
    j=np.zeros((4,4))
    j[0,0]=1
    j[1,1]=-(1+r2)/(1+u[1]**2)**1.5
    j[1,2]=-r1/(1+u[2]**2)**1.5
    j[2,2]=-1/(1+u[2]**2)**1.5
    j[3,3]=2*u[3]
    return j


def central(function, point, step):
    cols=[]
    for k in range(len(point)):
        h=step*max(1.,abs(point[k]))
        hi,lo=point.copy(),point.copy()
        hi[k]+=h
        lo[k]-=h
        cols.append((function(hi)-function(lo))/(2*h))
    return np.stack(cols,axis=-1)


def comparison(value, reference, tolerance):
    difference=np.asarray(value)-np.asarray(reference)
    limit=tolerance*max(1.,float(np.max(abs(np.asarray(reference)))))
    delta=float(np.max(abs(difference)))
    return dict(difference=difference,max_difference=delta,limit=limit,
                agreement=bool(np.isfinite(delta) and delta<=limit))


def check_point(x, u, stored):
    model=prior.model_for(x)
    theta=model.transform_params(u)
    direct_theta=transform(u)
    ll=float(model.loglike(theta,transformed=True))
    direct_ll=prior.gaussian_loglike(x,direct_theta)
    result=dict(unconstrained=u,theta=theta,direct_theta=direct_theta,
        transform=comparison(direct_theta,theta,1e-12),state_space_loglike=ll,direct_loglike=direct_ll,
        likelihood=comparison(direct_ll,ll,1e-8),initialization=model.ssm.initialization.initialization_type,
        likelihood_burn=model.loglikelihood_burn)
    if not result['likelihood']['agreement'] or not result['transform']['agreement']:
        result['status']='likelihood_or_transform_mismatch'
        return result
    n=len(x)
    jac=analytic_jacobian(u)
    installed_jac=model.transform_jacobian(u)
    installed=-prior.model_for(x).score(u,transformed=False,method='approx',approx_complex_step=True)/n
    constrained=-prior.model_for(x).score(theta,transformed=True,method='approx',approx_complex_step=True)/n
    chain=jac.T@constrained
    installed_chain=installed_jac@constrained
    result.update(status='evaluated',analytic_jacobian=jac,installed_jacobian=installed_jac,
        jacobian=comparison(installed_jac,jac,1e-6),installed_gradient=installed,
        constrained_gradient=constrained,chain_rule_gradient=chain,
        installed_product_reconstruction=comparison(installed_chain,installed,1e-12),
        archived_gradient_exact=plain(installed)==stored['complex_step_gradient'],steps=[])
    for h in STEPS:
        record={'step':h}
        try:
            nj=central(lambda p:prior.model_for(x).transform_params(p),u,h)
            gu=central(lambda p:-prior.gaussian_loglike(x,transform(p))/n,u,h)
            gt=central(lambda p:-prior.gaussian_loglike(x,p)/n,theta,h)
            record.update(status='ok',numerical_jacobian=nj,direct_unconstrained_gradient=gu,
                direct_constrained_gradient=gt,jacobian=comparison(nj,jac,1e-6),
                constrained=comparison(constrained,gt,1e-5),
                installed=comparison(installed,gu,1e-5),chain_rule=comparison(chain,gu,1e-5))
        except Exception as exc:
            record.update(status='invalid_perturbation',error=f'{type(exc).__name__}: {exc}')
        result['steps'].append(record)
    return result


class PassiveRecorder:
    """Return the exact solver output and objective values without reevaluation."""
    def __init__(self, original):
        self.original=original
        self.calls=[]
        self.returned=None
        self.options=None

    def __call__(self,*args,**kwargs):
        bound=inspect.signature(self.original).bind(*args,**kwargs)
        bound.apply_defaults()
        self.options={k:plain(bound.arguments[k]) for k in (*OPTIONS,'bounds','x0')}
        function=bound.arguments['func']
        def observe(point,*extra):
            before=np.asarray(point).copy()
            value=function(point,*extra)
            self.calls.append(dict(point=before,value=float(value)))
            return value
        bound.arguments['func']=observe
        output=self.original(*bound.args,**bound.kwargs)
        self.returned=plain(output)
        return output


def unchanged_replay(x, baseline):
    model=prior.model_for(x)
    original=optimize.fmin_l_bfgs_b
    recorder=PassiveRecorder(original)
    trace=[]
    def capture(u):
        trace.append(np.asarray(u).copy())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with patch.object(optimize,'fmin_l_bfgs_b',recorder):
            fitted=model.fit(method=prior.COMMON.SPEC.estimator,method_kwargs={
                'method':'lbfgs','maxiter':500,'disp':0,'callback':capture})
    if optimize.fmin_l_bfgs_b is not original:
        raise RuntimeError('Instrumentation not restored')
    current=dict(optimizer=plain(fitted.mle_retvals),start_unconstrained=plain(fitted.mle_settings['start_params']),
                 end_unconstrained=plain(fitted.mlefit.params),end_constrained=plain(fitted.params),
                 trajectory_unconstrained=plain(trace))
    matches={k:current[k]==baseline[k] for k in current}
    optmatches={k:recorder.options[k]==v for k,v in OPTIONS.items()}
    optmatches['bounds']=recorder.options['bounds']==[[None,None]]*4
    matches['objective_call_count']=len(recorder.calls)==current['optimizer']['fcalls']
    raw=recorder.returned[2]
    matches['scipy_gradient']=raw['grad']==current['optimizer']['gopt']
    matches['scipy_iterations']=raw['nit']==current['optimizer']['iterations']
    matches['scipy_objective']=recorder.returned[1]==current['optimizer']['fopt']
    for row in recorder.calls:
        row['matches_accepted_iteration']=[i+1 for i,p in enumerate(trace) if np.array_equal(row['point'],p)]
        row['role']='matches_accepted_iterate' if row['matches_accepted_iteration'] else 'unclassified_objective_evaluation'
    rec=dict(**current,scipy_return=recorder.returned,effective_options=recorder.options,
             options_match=optmatches,baseline_matches=matches,faithful=all(matches.values()) and all(optmatches.values()),
             objective_calls=recorder.calls,warnings=[f'{w.category.__name__}: {w.message}' for w in caught])
    if rec['faithful']:
        u=np.array(current['end_unconstrained'])
        m=prior.model_for(x)
        f=lambda p:-m.loglike(p,transformed=False)/len(x)
        f0=f(u)
        nominal=[]
        represented=[]
        for k in range(len(u)):
            hi=u.copy()
            hi[k]+=1e-5
            change=f(hi)-f0
            nominal.append(change/1e-5)
            represented.append(change/(hi[k]-u[k]))
        rec['endpoint_forward_difference']=dict(nominal_epsilon=1e-5,nominal_gradient=nominal,
            represented_step_gradient=represented,
            nominal_vs_recorded=comparison(nominal,current['optimizer']['gopt'],1e-5),
            represented_vs_recorded=comparison(represented,current['optimizer']['gopt'],1e-5))
    return rec


def verify():
    prior.verify()
    locked=json.loads(LOCK.read_text())
    for p,h in locked['sha256'].items():
        if digest(Path(p))!=h:
            raise RuntimeError(f'Frozen dependency changed: {p}')


def freeze():
    prior.verify()
    if ROOT.exists():
        raise RuntimeError('Refusing to overwrite diagnostic')
    # Controller-only tests precede the freeze; numerical checks follow it.
    checked=subprocess.run([sys.executable,'-B','-m','unittest','-v','test_reproduction_policy'],
                           cwd=HERE,text=True,capture_output=True)
    if checked.returncode:
        raise RuntimeError(checked.stdout+checked.stderr)
    paths=[Path(__file__),HERE/'test_gradient_stopping.py',BASELINE,prior.ROOT/'run_002/results.json',
        prior.ROOT/'verification.json',prior.LOCK,HERE/'Orientation_Gradient_and_Stopping_Verification_Protocol.txt',
        Path(inspect.getfile(statsmodels.tools.numdiff)),Path(inspect.getfile(statsmodels.base.optimizer))]
    ROOT.mkdir(exist_ok=False)
    write(LOCK,dict(approval='User approved gradient and stopping verification only.',
        sha256={str(p):digest(p) for p in paths},cases=CASES,steps=STEPS,options=OPTIONS,
        prior_lock=str(prior.LOCK),new_random_draws=0,new_optimizer_methods=0,
        analytic_jacobian_tolerance=1e-6,gradient_tolerance=1e-5,likelihood_tolerance=1e-8,
        standalone_transformation_tolerance=1e-12,controller_tests=checked.stdout+checked.stderr,
        production_changes_authorized=False))
    print('Verification frozen before numerical checks',flush=True)


def run(name):
    verify()
    out=ROOT/name
    out.mkdir(exist_ok=False)
    previous=json.loads(BASELINE.read_text())
    records=[]
    with np.load(prior.INPUT) as paths:
        for config,rep in CASES:
            baseline=next(r for r in previous if r['config']==config and r['rep']==rep)
            z=paths[config][rep]
            x=(z-z.mean())/z.std(ddof=0)
            rec={'config':config,'rep':rep,'points':[]}
            for solver in baseline['solvers']:
                for position in ('start','end'):
                    u=np.array(solver[f'{position}_unconstrained'])
                    checked=check_point(x,u,solver[f'derivatives_{position}'])
                    checked.update(solver=solver['solver'],position=position)
                    rec['points'].append(checked)
            lb=next(r for r in baseline['solvers'] if r['solver']=='lbfgs')
            rec['replay']=unchanged_replay(x,lb)
            records.append(rec)
            print(f'{name} {config} {rep}: faithful replay={rec["replay"]["faithful"]}; '
                  f'task={rec["replay"]["scipy_return"][2]["task"]}',flush=True)
    verify()
    write(out/'results.json',records)


def execute():
    verify()
    tests=subprocess.run([sys.executable,'-B','-m','unittest','-v','test_gradient_stopping'],
                         cwd=HERE,text=True,capture_output=True)
    (ROOT/'tests.txt').open('x').write(tests.stdout+tests.stderr)
    if tests.returncode:
        raise RuntimeError(tests.stdout+tests.stderr)
    def child(name):
        result=subprocess.run([sys.executable,'-B',str(Path(__file__)),name],cwd=HERE,text=True,capture_output=True)
        (ROOT/f'{name}.log').open('x').write(result.stdout+result.stderr)
        print(result.stdout,flush=True)
        if result.returncode:
            raise RuntimeError(result.stderr)
    child('run_001')
    def pair():
        a,b=(ROOT/f'run_{i:03}/results.json' for i in (1,2))
        same=digest(a)==digest(b)
        return {'pass':same,'results_byte_identical':same}
    def archive():
        records=json.loads((ROOT/'run_001/results.json').read_text())
        replays=[r['replay']['faithful'] for r in records]
        gradients=[p.get('archived_gradient_exact',False) for r in records for p in r['points']]
        return {'pass':len(replays)==3 and all(replays) and len(gradients)==12 and all(gradients),
                'replays':replays,'archived_gradient_matches':gradients}
    result=complete_reproduction(verify,lambda:child('run_002'),pair,archive)
    write(ROOT/'verification.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['freeze','execute','run_001','run_002','verify'])
    stage=parser.parse_args().stage
    if stage=='freeze':
        freeze()
    elif stage=='execute':
        execute()
    elif stage=='verify':
        verify()
        print('All frozen dependencies unchanged')
    else:
        run(stage)
