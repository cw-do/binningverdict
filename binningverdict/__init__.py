"""
binningverdict
==============

A quantitative log-vs-linear binning decision system for SANS data.

The package implements the closed-form ratio R_MSE comparing the
reconstruction mean-squared error of logarithmic and linear binning schemes
at fixed bin count, together with a Fisher-information branch for
parameter-estimation analysis.

Public API
----------
analyze_binning : workflow entry point
R_MSE           : closed-form MSE ratio
R_Fisher_finiteN: Fisher ratio at finite N (parametric)
derivative      : numerical I'(Q) estimator

References
----------
[Lead Author] et al. (2026). Logarithmic versus linear binning in small-angle
neutron scattering: a unified decision system. Comput. Phys. Commun. (in press).
"""

from .core import (
    analyze_binning,
    R_MSE,
    R_Fisher_finiteN,
    derivative,
)

__version__ = "1.0.0"
__all__ = [
    "analyze_binning",
    "R_MSE",
    "R_Fisher_finiteN",
    "derivative",
]
