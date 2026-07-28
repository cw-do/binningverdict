"""Correctness tests for binningverdict."""

import numpy as np
import pytest

from binningverdict import (
    analyze_binning, R_MSE, R_Fisher_finiteN, resolution_ratio,
    optimal_bin_widths, background_diagnostic, poisson_sigma2,
)

WINDOW = (1e-3, 0.5)


def grid(n=20000, qa=WINDOW[0], qb=WINDOW[1]):
    return np.logspace(np.log10(qa), np.log10(qb), n)


def guinier(Q, Rg=30.0):
    return np.exp(-Q ** 2 * Rg ** 2 / 3)


# ---------------------------------------------------------------------------
# reconstruction branch
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('geometry', ['1D', '2D'])
@pytest.mark.parametrize('alpha', [1.0, 2.0, 3.0, 4.0])
def test_power_law_favours_log(alpha, geometry):
    """R_MSE < 1 for a power law at every alpha, in both conventions."""
    Q = grid()
    assert R_MSE(Q, Q ** (-alpha), geometry=geometry) < 1.0


@pytest.mark.parametrize('geometry', ['1D', '2D'])
def test_guinier_favours_linear(geometry):
    Q = grid()
    assert R_MSE(Q, guinier(Q), geometry=geometry) > 1.0


@pytest.mark.parametrize('geometry', ['1D', '2D'])
def test_power_law_window_independent(geometry):
    """Scale invariance: the power-law ratio must not depend on the window."""
    vals = []
    for qa, qb in [(1e-3, 0.1), (5e-3, 0.3), (1e-2, 1.0)]:
        Q = grid(qa=qa, qb=qb)
        vals.append(R_MSE(Q, Q ** -3.0, geometry=geometry))
    assert max(abs(v / vals[0] - 1) for v in vals) < 0.01


def test_geometry_changes_only_the_proxy():
    """With measured errors supplied, geometry must not affect R_MSE."""
    Q = grid(400)
    I = guinier(Q)
    err = np.sqrt(I)
    assert R_MSE(Q, I, err=err, geometry='1D') == pytest.approx(
        R_MSE(Q, I, err=err, geometry='2D'))


def test_poisson_proxy_forms():
    Q = grid(50)
    I = guinier(Q)
    assert np.allclose(poisson_sigma2(Q, I, '1D'), I)
    assert np.allclose(poisson_sigma2(Q, I, '2D'), I / Q)


def test_R_MSE_invariant_under_Q_rescaling():
    Q = grid(2000)
    I = guinier(Q)
    err = np.sqrt(I)
    assert R_MSE(Q, I, err=err) == pytest.approx(R_MSE(10 * Q, I, err=err))


# ---------------------------------------------------------------------------
# resolution branch
# ---------------------------------------------------------------------------
def _synthetic_with_resolution(n=300):
    Q = grid(n, 0.005, 0.3)
    I = guinier(Q)
    err = np.sqrt(np.maximum(I, 1e-12) / Q) * np.sqrt(np.gradient(Q))
    dQ = np.sqrt((0.001) ** 2 + (0.05 * Q) ** 2 + np.gradient(Q) ** 2 / 12)
    return Q, I, err, dQ


def test_resolution_moves_toward_unity_without_crossing():
    Q, I, err, dQ = _synthetic_with_resolution()
    out = resolution_ratio(Q, I, err=err, dQ=dQ)
    R, R_res = out['R_MSE'], out['R_res']
    assert out['g'] >= 0.0
    assert np.sign(R_res - 1.0) == np.sign(R - 1.0)
    assert abs(R_res - 1.0) <= abs(R - 1.0) + 1e-12


def test_resolution_identity():
    """R_res must satisfy (R + g)/(1 + g) exactly."""
    Q, I, err, dQ = _synthetic_with_resolution()
    o = resolution_ratio(Q, I, err=err, dQ=dQ)
    assert o['R_res'] == pytest.approx((o['R_MSE'] + o['g']) / (1 + o['g']))


def test_g_invariant_under_Q_rescaling():
    """g is absolute, so it must not depend on the units of Q."""
    Q, I, err, dQ = _synthetic_with_resolution()
    a = resolution_ratio(Q, I, err=err, dQ=dQ)['g']
    b = resolution_ratio(10 * Q, I, err=err, dQ=10 * dQ)['g']
    assert a == pytest.approx(b, rel=1e-6)


def test_no_resolution_returns_none():
    Q = grid(200)
    I = guinier(Q)
    out = resolution_ratio(Q, I, err=np.sqrt(I))
    assert out['R_res'] is None and out['g'] is None


# ---------------------------------------------------------------------------
# Fisher branch
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('geometry', ['1D', '2D'])
def test_fisher_asymptotic_equivalence(geometry):
    Rg = 30.0
    I = lambda q: np.exp(-q ** 2 * Rg ** 2 / 3)
    dI = lambda q: -2 * q * Rg / 3 * I(q)
    for N in (500, 1000, 2000):
        r = R_Fisher_finiteN(0.005, 0.3, N, I, dI, geometry=geometry)
        assert abs(r - 1.0) < 0.01


def test_solid_angle_weight_accelerates_convergence():
    """The power-law ratio is closer to unity with w(Q) = Q than with w = 1."""
    I = lambda q: q ** -3.0
    dI = lambda q: -np.log(q) * q ** -3.0
    for N in (20, 50, 100):
        r1 = R_Fisher_finiteN(0.005, 0.3, N, I, dI, geometry='1D')
        r2 = R_Fisher_finiteN(0.005, 0.3, N, I, dI, geometry='2D')
        assert abs(r2 - 1.0) < abs(r1 - 1.0)


# ---------------------------------------------------------------------------
# background diagnostic
# ---------------------------------------------------------------------------
def test_background_diagnostic_flags_flat_tail():
    Q = grid(400, 0.005, 0.5)
    I = Q ** -4.0 + 1.0                      # Porod decay onto a plateau
    err = np.where(Q < 0.15, np.sqrt(I), np.full_like(Q, 5.0))
    bg = background_diagnostic(Q, I, err)
    assert bg['Q_trim'] is not None
    assert bg['n_trimmed'] > 0


def test_background_diagnostic_quiet_on_clean_data():
    Q = grid(400, 0.005, 0.3)
    I = guinier(Q)
    bg = background_diagnostic(Q, I, np.sqrt(I))
    assert bg['fraction_flagged'] < 0.2


# ---------------------------------------------------------------------------
# workflow
# ---------------------------------------------------------------------------
def test_workflow_returns_required_keys():
    Q, I, err, dQ = _synthetic_with_resolution()
    res = analyze_binning(Q, I, err=err, dQ=dQ)
    for key in ('verdict', 'R_MSE', 'R_MSE_subsampled', 'converged',
                'geometry', 'R_res', 'g', 'rho', 'B_fraction',
                'R_Fisher_finite', 'background', 'h_linear', 'delta_log',
                'rationale'):
        assert key in res


def test_optimal_widths_positive_and_scale_correctly():
    Q, I, err, _ = _synthetic_with_resolution()
    a = optimal_bin_widths(Q, I, err=err)
    b = optimal_bin_widths(10 * Q, I, err=err)
    assert a['h_linear'] > 0 and a['delta_log'] > 0
    assert b['h_linear'] / a['h_linear'] == pytest.approx(10.0, rel=1e-6)
    assert b['delta_log'] == pytest.approx(a['delta_log'], rel=1e-6)


def test_verdict_consistency_at_dense_input():
    Q_log = np.logspace(np.log10(0.005), np.log10(0.3), 500)
    Q_lin = np.linspace(0.005, 0.3, 500)
    a = analyze_binning(Q_log, guinier(Q_log), err=np.sqrt(guinier(Q_log)))
    b = analyze_binning(Q_lin, guinier(Q_lin), err=np.sqrt(guinier(Q_lin)))
    assert a['verdict'] == b['verdict']


def test_bad_geometry_raises():
    Q = grid(50)
    with pytest.raises(ValueError):
        R_MSE(Q, guinier(Q), geometry='3D')
