"""Orientation-specific reporting with the original point-estimate arithmetic."""
import math
import sys

import numpy as np

from adapter import PUBLIC, ReportingResult
from orientation_gradient_candidate import COMMON, ConditionalFitter

if str(PUBLIC) not in sys.path:
    sys.path.insert(0, str(PUBLIC))
from sim.inference import centered_slope_ratio_moments


class OrientationReporter:
    """Keep point computation separate from the qualified uncertainty model."""

    def __init__(self, observer=None):
        self.fitter = ConditionalFitter()
        self.observer = observer

    def __call__(self, y, w, z, *, z_process, ar_order=2, varz_mode='window'):
        if ar_order != 2 or varz_mode != 'window':
            raise ValueError('Reporting requires AR(2) and the observed post window')
        point = centered_slope_ratio_moments(y, w, z)
        fit, provenance = self.fitter.fit(z_process)
        interval = COMMON.interval_from_fit(y, w, z, tau_hat=point.tau,
                                            pi_hat=point.pi, fit=fit)
        if interval.status != 'ok':
            raise COMMON.InferenceFailure(interval.reason, interval.detail)
        result = ReportingResult(float(point.tau), float(point.pi), interval.se,
                                 interval.critical, interval.lower, interval.upper,
                                 fit.cache_key)
        if not all(math.isfinite(v) for v in (result.tau, result.pi, result.se,
                                              result.critical, result.lower, result.upper)):
            raise COMMON.InferenceFailure('nonfinite_reporting', 'Invalid reporting result')
        if self.observer is not None:
            self.observer(np.asarray(y), np.asarray(w), np.asarray(z),
                          np.asarray(z_process), fit, provenance, result)
        return result
