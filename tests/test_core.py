"""Basic correctness tests for binningverdict."""

import numpy as np
import pytest
from binningverdict import R_MSE, R_Fisher_finiteN, analyze_binning


# ---------------------------------------------------------------------------
# R_MSE: known asymptotic results
# ---------------------------------------------------------------------------
def test_powerlaw_favors_log():
    """Power-law I ~ Q^(-alpha) should give R_MSE < 1 for any alpha > 0."""
    Q = np.logspace(-3, np.log10(0.5), 5000)
    for alpha in [1.0, 2.0, 3.0, 4.0]:
        I = Q ** (-alpha)
        R = R_MSE(Q, I, err=np.sqrt(I))
        assert R < 1.0, f"alpha={alpha}: R_MSE={R:.3f} should be < 1"


def test_guinier_favors_linear_at_wide_window():
    """Guinier with wide window should give R_MSE > 1."""
    Rg = 30.0
    Q = np.logspace(-3, np.log10(0.5), 5000)
    I = np.exp(-Q ** 2 * Rg ** 2 / 3)
    R = R_MSE(Q, I, err=np.sqrt(I))
    assert R > 1.0, f"Guinier wide window: R_MSE={R:.3f} should be > 1"


def test_powerlaw_scale_invariance():
    """Power-law R_MSE should be independent of the Q-window."""
    alpha = 3.0
    R_values = []
    for Qmin, Qmax in [(0.001, 0.1), (0.005, 0.3), (0.01, 1.0)]:
        Q = np.logspace(np.log10(Qmin), np.log10(Qmax), 5000)
        I = Q ** (-alpha)
        R_values.append(R_MSE(Q, I, err=np.sqrt(I)))
    # All within 1%
    for R in R_values:
        assert abs(R - R_values[0]) / R_values[0] < 0.01


# ---------------------------------------------------------------------------
# Fisher: asymptotic equivalence
# ---------------------------------------------------------------------------
def test_fisher_asymptotic_equivalence():
    """F_log/F_L -> 1 as N grows, for any model."""
    Rg = 30.0
    I_func  = lambda Q: np.exp(-Q ** 2 * Rg ** 2 / 3)
    dI_func = lambda Q: -2 * Q * Rg / 3 * I_func(Q)
    for N in [500, 1000, 2000]:
        R_F = R_Fisher_finiteN(0.005, 0.3, N, I_func, dI_func)
        assert abs(R_F - 1.0) < 0.01, f"N={N}: F ratio {R_F:.4f} not within 1% of unity"


# ---------------------------------------------------------------------------
# Workflow: end-to-end behavior
# ---------------------------------------------------------------------------
def test_workflow_returns_required_keys():
    Q = np.logspace(-3, np.log10(0.5), 200)
    I = np.exp(-Q ** 2 * 30 ** 2 / 3)
    res = analyze_binning(Q, I, err=np.sqrt(I))
    for key in ['R_MSE', 'R_MSE_subsampled', 'converged',
                'R_Fisher_finite', 'verdict', 'rationale']:
        assert key in res


def test_workflow_verdict_consistency_at_dense_input():
    """At dense input both schemes should give the same verdict for Guinier."""
    Q_log = np.logspace(np.log10(0.005), np.log10(0.3), 500)
    Q_lin = np.linspace(0.005, 0.3, 500)
    Rg = 30.0
    I_log = np.exp(-Q_log ** 2 * Rg ** 2 / 3)
    I_lin = np.exp(-Q_lin ** 2 * Rg ** 2 / 3)
    res_log = analyze_binning(Q_log, I_log, err=np.sqrt(I_log))
    res_lin = analyze_binning(Q_lin, I_lin, err=np.sqrt(I_lin))
    assert res_log['verdict'] == res_lin['verdict']
