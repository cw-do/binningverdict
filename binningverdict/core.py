"""
binningverdict.core
===================

Core functions for the log-vs-linear binning decision system for SANS data.

The reconstruction-MSE ratio R_MSE is computed from four moments of the
measured intensity profile (Q, I, sigma):

    R_MSE = (J_1^sigma / sigma_bar^2)^(2/3) * (J_2 / beta_bar)^(1/3)

where the moments are:

    sigma_bar^2  = (1/L) integral sigma^2(Q) dQ
    J_1^sigma    = (1/L) integral sigma^2(Q)/Q dQ
    beta_bar     = (1/L) integral [I'(Q)]^2 dQ
    J_2          = (1/L) integral Q^2 [I'(Q)]^2 dQ

Verdict rule (default tolerance = 0.10):
    R_MSE < 1 - tol  -> log-binning
    R_MSE > 1 + tol  -> linear-binning
    otherwise        -> tied
"""

import numpy as np

_trapezoid = getattr(np, "trapezoid", np.trapz)


# ---------------------------------------------------------------------------
# Core scalar quantities
# ---------------------------------------------------------------------------
def _moments(Q, I, sigma2, I_prime):
    """Compute the four Q-moments entering R_MSE."""
    L = Q[-1] - Q[0]
    sigma_bar_sq = _trapezoid(sigma2,             Q) / L
    J1_sigma     = _trapezoid(sigma2 / Q,         Q) / L
    beta_bar     = _trapezoid(I_prime ** 2,       Q) / L
    J2           = _trapezoid(Q**2 * I_prime**2,  Q) / L
    return sigma_bar_sq, J1_sigma, beta_bar, J2


def derivative(Q, I):
    """
    Numerical derivative dI/dQ on a possibly non-uniform grid.

    Uses numpy.gradient (second-order central differences for interior
    points, first-order at the boundaries). For very noisy data, consider
    smoothing I before calling this routine.
    """
    return np.gradient(I, Q)


def R_MSE(Q, I, err=None, I_prime=None):
    """
    Compute the MSE ratio R_MSE = E*_log / E*_L.

    Parameters
    ----------
    Q : array-like
        Momentum-transfer values (1/A), monotonically increasing.
    I : array-like
        Measured intensities at Q (cm^-1 or arbitrary).
    err : array-like, optional
        1-sigma uncertainties on I. If None, Poisson scaling
        (sigma^2 = I) is assumed.
    I_prime : array-like, optional
        Pre-computed dI/dQ. If None, computed by `derivative(Q, I)`.

    Returns
    -------
    R : float
        R_MSE > 1 favors linear binning;
        R_MSE < 1 favors logarithmic binning.
    """
    Q = np.asarray(Q, dtype=float)
    I = np.asarray(I, dtype=float)
    if I_prime is None:
        I_prime = derivative(Q, I)
    sigma2 = I if err is None else np.asarray(err, dtype=float) ** 2
    sb, J1, bb, J2 = _moments(Q, I, sigma2, I_prime)
    return (J1 / sb) ** (2.0 / 3.0) * (J2 / bb) ** (1.0 / 3.0)


# ---------------------------------------------------------------------------
# Fisher information branch (parametric model required)
# ---------------------------------------------------------------------------
def _linear_bins(Qmin, Qmax, N):
    edges = np.linspace(Qmin, Qmax, N + 1)
    return 0.5 * (edges[:-1] + edges[1:]), np.diff(edges)


def _log_bins(Qmin, Qmax, N):
    edges = np.logspace(np.log10(Qmin), np.log10(Qmax), N + 1)
    return np.sqrt(edges[:-1] * edges[1:]), np.diff(edges)


def R_Fisher_finiteN(Qmin, Qmax, N, I_func, dI_func):
    """
    Finite-N Fisher information ratio F_log / F_L for a parametric model.

    Parameters
    ----------
    Qmin, Qmax : float
        Window bounds.
    N : int
        Number of bins in each scheme.
    I_func, dI_func : callable
        Functions evaluating I(Q) and dI/dtheta(Q) at array Q.

    Returns
    -------
    R : float
        Fisher ratio. By the asymptotic-equivalence theorem, R -> 1 as N grows.
    """
    cL, hL = _linear_bins(Qmin, Qmax, N)
    cG, hG = _log_bins(Qmin, Qmax, N)
    F_L = float(np.sum(hL * dI_func(cL) ** 2 / I_func(cL)))
    F_G = float(np.sum(hG * dI_func(cG) ** 2 / I_func(cG)))
    return F_G / F_L


# ---------------------------------------------------------------------------
# Top-level workflow
# ---------------------------------------------------------------------------
def analyze_binning(Q, I, err=None,
                    model=None, dmodel=None, theta=None, N_future=None,
                    tolerance=0.10, verbose=False):
    """
    Apply the log-vs-linear binning decision workflow.

    Parameters
    ----------
    Q, I : array-like
        Measured profile.
    err : array-like, optional
        1-sigma uncertainties; Poisson assumed if absent.
    model, dmodel : callable, optional
        I(Q;theta) and dI/dtheta(Q;theta). If supplied, the Fisher branch is
        evaluated as well.
    theta : float, optional
        Best estimate of theta to evaluate the Fisher integrand at.
    N_future : int, optional
        Anticipated bin count for the Fisher ratio. Defaults to len(Q).
    tolerance : float, optional
        Half-width of the "tied" verdict band around R_MSE = 1.
    verbose : bool, optional
        Print result to stdout if True.

    Returns
    -------
    result : dict
        Keys: R_MSE, R_MSE_subsampled, converged, R_Fisher_finite, verdict,
        rationale.
    """
    Q = np.asarray(Q, dtype=float)
    I = np.asarray(I, dtype=float)
    if err is not None:
        err = np.asarray(err, dtype=float)

    R_mse = R_MSE(Q, I, err=err)

    # Self-consistency check on a sub-sampled grid
    sub = slice(None, None, 2)
    R_mse_sub = R_MSE(Q[sub], I[sub], err=err[sub] if err is not None else None)
    converged = abs(R_mse - R_mse_sub) / R_mse <= tolerance

    # Optional Fisher branch
    R_F = None
    if model is not None and dmodel is not None:
        Qmin, Qmax = float(Q.min()), float(Q.max())
        N = N_future if N_future is not None else len(Q)
        I_fn  = (lambda q: model(q, theta))  if theta is not None else model
        dI_fn = (lambda q: dmodel(q, theta)) if theta is not None else dmodel
        R_F = R_Fisher_finiteN(Qmin, Qmax, N, I_fn, dI_fn)

    # Verdict
    if abs(R_mse - 1.0) < tolerance:
        verdict = 'tied'
        rationale = (f"R_MSE = {R_mse:.3f} is within {int(tolerance*100)}% of "
                     "unity; the two schemes give essentially equivalent "
                     "reconstruction MSE.")
    elif R_mse < 1.0 - tolerance:
        verdict = 'log'
        rationale = (f"R_MSE = {R_mse:.3f} < 1: log-spaced binning gives "
                     "smaller reconstruction MSE at fixed bin count.")
    else:
        verdict = 'linear'
        rationale = (f"R_MSE = {R_mse:.3f} > 1: linear-spaced binning gives "
                     "smaller reconstruction MSE at fixed bin count.")

    if R_F is not None and N_future is not None:
        if abs(R_F - 1.0) > tolerance:
            rationale += (f" Fisher ratio at N = {N_future} is {R_F:.3f}, "
                          f"favoring {'log' if R_F > 1 else 'linear'} binning "
                          "for parameter estimation.")
        else:
            rationale += (f" Fisher ratio at N = {N_future} is {R_F:.3f}; "
                          "parameter precision is scheme-independent.")

    if not converged:
        rationale += (f" WARNING: subsampled R_MSE = {R_mse_sub:.3f} differs "
                      f"from the full-N estimate by more than {int(tolerance*100)}%. "
                      "The verdict is not yet stable. Recompute with a denser "
                      "input grid (a less aggressively rebinned profile) and "
                      "check whether the verdict stabilizes.")

    result = {
        'R_MSE':            float(R_mse),
        'R_MSE_subsampled': float(R_mse_sub),
        'converged':        bool(converged),
        'R_Fisher_finite':  float(R_F) if R_F is not None else None,
        'verdict':          verdict,
        'rationale':        rationale,
    }

    if verbose:
        print(f"R_MSE   = {R_mse:.4f}")
        if R_F is not None:
            print(f"R_F     = {R_F:.4f}  (N = {N_future})")
        print(f"Verdict = {verdict}")
        print(f"  {rationale}")

    return result
