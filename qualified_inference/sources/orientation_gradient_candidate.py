"""Isolated conditional gradient retry for the unchanged stationary likelihood."""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import warnings

import numpy as np
from scipy.optimize import fmin_l_bfgs_b

import orientation_likelihood_diagnostic as prior
from verify_orientation_gradient_stopping import analytic_jacobian, central, comparison

COMMON=prior.COMMON
PROTOCOL=prior.HERE/'Orientation_Conditional_Gradient_Correction_Protocol.txt'
GRADIENT_RULE='analytic_AR2_J_transpose_times_constrained_complex_step_score_v1'
OPTIONS=dict(m=10,factr=1e7,pgtol=1e-5,maxfun=15000,maxiter=500,maxls=20,approx_grad=False)


def retry_key(z):
    header='|'.join([COMMON.fit_key(z),prior.digest(Path(__file__)),prior.digest(PROTOCOL),GRADIENT_RULE])
    return 'conditional_gradient_v1:'+hashlib.sha256(header.encode()).hexdigest()


def objective_and_gradient(model,u):
    n=model.nobs
    theta=model.transform_params(u)
    value=-float(model.loglike(u,transformed=False))/n
    constrained=model.score(theta,transformed=True,method='approx',approx_complex_step=True)
    gradient=-(analytic_jacobian(u).T@constrained)/n
    return value,gradient


def corrected_fit(z, original_optimizer):
    z=COMMON.vector(z,'full instrument')
    mean,scale=float(z.mean()),float(z.std(ddof=0))
    if len(z)<=2 or not math.isfinite(scale) or scale<=0:
        raise COMMON.InferenceFailure('invalid_input','Invalid full instrument path')
    x=(z-mean)/scale
    model=prior.model_for(x)
    record={'gradient_rule':GRADIENT_RULE,'options':OPTIONS,'original_optimizer':original_optimizer}
    trace=[]
    def capture(u):
        trace.append(np.asarray(u).copy())
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            theta_start=model.start_params
            start=model.untransform_params(theta_start)
            record.update(start_constrained=theta_start,start_unconstrained=start)
            # The combined callback returns the unchanged likelihood and its checked derivative.
            u,fopt,ret=fmin_l_bfgs_b(lambda p:objective_and_gradient(model,p),start,
                bounds=[(None,None)]*len(start),callback=capture,**OPTIONS)
        record.update(warnings=[f'{w.category.__name__}: {w.message}' for w in caught],
                      scipy_return=ret,end_unconstrained=u,fopt=float(fopt),trajectory=trace)
        if ret['warnflag']!=0:
            raise COMMON.InferenceFailure('corrected_nonconvergence','Corrected-gradient retry did not converge')
        theta=model.transform_params(u)
        with warnings.catch_warnings(record=True) as post_warnings:
            warnings.simplefilter('always')
            filtered=model.filter(theta,transformed=True)
        record['warnings'].extend(f'{w.category.__name__}: {w.message}' for w in post_warnings)
        # These are the actual external solver diagnostics, not a claim of another fit.
        filtered.mle_retvals=dict(converged=True,warnflag=int(ret['warnflag']),fopt=float(fopt),
                                  gopt=ret['grad'],fcalls=int(ret['funcalls']),iterations=int(ret['nit']),task=ret['task'])
        phi,q,mu,radius=COMMON.validate_fit(filtered)
        record['end_constrained']=theta
        ll=float(model.loglike(theta,transformed=True))
        direct=prior.gaussian_loglike(x,theta)
        record['likelihood_agreement']=comparison(direct,ll,1e-8)
        record['objective_normalization']=comparison(fopt,-ll/len(z),COMMON.TOL)
        record['loglike']=ll
        old_ll=-len(z)*float(original_optimizer['fopt'])
        if not math.isfinite(old_ll):
            raise COMMON.InferenceFailure('missing_original_likelihood','Original endpoint likelihood is not finite')
        record['original_loglike']=old_ll
        record['likelihood_not_lower']=bool(ll>=old_ll-1e-8*max(1.,abs(old_ll)))
        supplied=objective_and_gradient(model,u)[1]
        record['gradient']=supplied
        record['gradient_norm']=float(np.max(abs(supplied)))
        record['derivative_checks']=[]
        for step in prior.STEPS:
            independent=central(lambda p:-prior.gaussian_loglike(x,model.transform_params(p))/len(x),u,step)
            record['derivative_checks'].append(dict(step=step,independent=independent,
                **comparison(supplied,independent,1e-5)))
        record['same_likelihood_model']=bool(model.loglikelihood_burn==0 and
            model.ssm.initialization.initialization_type=='stationary' and
            model.param_names==['const','ar.L1','ar.L2','sigma2'])
        checks=[record['likelihood_agreement']['agreement'],record['objective_normalization']['agreement'],record['likelihood_not_lower'],
                record['same_likelihood_model'],*[d['agreement'] for d in record['derivative_checks']]]
        if not all(checks):
            raise COMMON.InferenceFailure('corrected_acceptance_failed','Corrected fit failed a predefined acceptance check')
        fitted=COMMON.FittedAR2(tuple(map(float,z)),mean,scale,phi,q,mu,radius,retry_key(z),
            tuple(record['warnings']),json.dumps(prior.plain(filtered.mle_retvals),sort_keys=True),COMMON.SPEC,
            json.dumps({'options':OPTIONS,'gradient_rule':GRADIENT_RULE,'route':'standalone_scipy_lbfgsb'},sort_keys=True))
        # Qualification only supports the two frozen path lengths and their original post windows.
        post={15:10,90:60}.get(len(z))
        if post is None:
            raise COMMON.InferenceFailure('unexpected_path_length','No approved post window for this path')
        COMMON.stationary_covariance(fitted,post)
        record['accepted']=True
        return fitted,prior.plain(record)
    except COMMON.InferenceFailure as exc:
        record.update(accepted=False,reason=exc.code)
        exc.diagnostics=prior.plain(record)
        raise
    except Exception as exc:
        record.update(accepted=False,error=f'{type(exc).__name__}: {exc}')
        raise COMMON.InferenceFailure('corrected_exception',str(exc),prior.plain(record)) from exc


class ConditionalFitter:
    def __init__(self):
        self.original_cache={}
        self.retry_cache={}
        self.original_failure_to_retry={}

    def fit(self,z):
        base=COMMON.fit_key(z)
        if base in self.original_failure_to_retry:
            key=self.original_failure_to_retry[base]
            return self.retry_cache[key]
        try:
            fitted=COMMON.fit_instrument(z,cache=self.original_cache)
            return fitted,{'retry_used':False}
        except COMMON.InferenceFailure as exc:
            if exc.code!='nonconvergence':
                raise
            original={'code':exc.code,'diagnostics':exc.diagnostics}
        fitted,details=corrected_fit(z,original['diagnostics']['optimizer'])
        key=fitted.cache_key
        if key==base or not key.startswith('conditional_gradient_v1:'):
            raise COMMON.InferenceFailure('cache_collision','Corrected fit has an original cache key')
        value=(fitted,{'retry_used':True,'original_failure':original,'retry':details})
        self.retry_cache[key]=value
        self.original_failure_to_retry[base]=key
        return value
