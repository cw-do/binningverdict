"""
binningverdict
==============

A quantitative logarithmic-versus-linear binning decision system for
small-angle scattering data.

Given a measured profile (Q, I, sigma), and optionally the instrumental
resolution column dQ, the package returns a verdict of 'log', 'linear' or
'tied' together with the closed-form reconstruction-MSE ratio R_MSE, a
resolution-aware ratio, an optional Fisher-information ratio, and a set of
diagnostics.

Public API
----------
analyze_binning        workflow entry point
R_MSE                  closed-form MSE ratio
resolution_ratio       resolution-aware ratio and its diagnostics
optimal_bin_widths     optimal linear width and logarithmic step
R_Fisher_finiteN       Fisher ratio at finite N (parametric)
background_diagnostic  locate the background-dominated tail
poisson_sigma2         Poisson proxy for either detector geometry
variance_density       convert reported uncertainties to a variance density
derivative             numerical dI/dQ

Reference
---------
C. Do, L. Ding, C.-H. Tung and W.-R. Chen, "binningverdict: a unified
decision system for logarithmic versus linear binning of small-angle neutron
scattering data", Comput. Phys. Commun. (submitted, 2026).
"""

from .core import (
    analyze_binning,
    R_MSE,
    resolution_ratio,
    optimal_bin_widths,
    R_Fisher_finiteN,
    background_diagnostic,
    poisson_sigma2,
    variance_density,
    derivative,
)

__version__ = "1.1.0"
__all__ = [
    "analyze_binning",
    "R_MSE",
    "resolution_ratio",
    "optimal_bin_widths",
    "R_Fisher_finiteN",
    "background_diagnostic",
    "poisson_sigma2",
    "variance_density",
    "derivative",
]
