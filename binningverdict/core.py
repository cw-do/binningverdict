"""
binningverdict.core
===================

Core routines for the logarithmic-versus-linear binning decision system for
small-angle scattering data.

Reconstruction branch
---------------------
The pointwise error density of a binned estimator with local bin width h(Q) is

    eps(Q; h) = sigma^2(Q)/h(Q) + [ B(Q) + h(Q)^2/12 ] * [I'(Q)]^2

The first term is counting noise, the h^2/12 term is the within-bin aliasing
distortion, and B(Q) is the scheme-independent part of the instrumental
Q-variance (collimation, detector element, wavelength spread).  In the
notation of Mildner & Carpenter, J. Appl. Cryst. 17, 249 (1984),

    sigma_Q^2(Q) = B(Q) + h^2/12
    B(Q)         = B_geom + (1/12) (dlambda/lambda)^2 Q^2

Comparing the two canonical schemes at matched bin count gives the closed form

    R_MSE = (J1_sigma / sigmabar^2)^(2/3) * (J2 / betabar)^(1/3)

with

    sigmabar^2 = <sigma^2>      J1_sigma = <sigma^2 / Q>
    betabar    = <(I')^2>       J2       = <Q^2 (I')^2>

and <.> the window-averaged integral.  R_MSE < 1 favours logarithmic binning,
R_MSE > 1 favours linear binning.

Including B(Q) adds the same constant G = <B (I')^2> to the optimum of both
schemes, so that

    R_res = (E*_log + G)/(E*_L + G) = (R_MSE + g)/(1 + g),   g = G / E*_L

and therefore  R_res - 1 = (R_MSE - 1)/(1 + g).  Since g >= 0 the resolution
term moves the ratio toward unity but can never move it across unity, so the
direction of the verdict is invariant.

Normalisation note
------------------
R_MSE is a ratio in which the normalisation of sigma^2 cancels, so reported
uncertainties may be used directly.  The quantities g, R_res and the optimal
bin widths are absolute, and they require sigma^2 to be a variance *density*
defined by  Var(bin of width h) = sigma^2(Q)/h.  A reported uncertainty err_i
belongs to a bin of width h_in,i on the delivered grid, so the density is

    sigma^2_density(Q_i) = err_i^2 * h_in,i

That conversion is applied automatically inside the resolution branch and is
what makes g and h* invariant under a change of the units of Q.

Poisson conventions
-------------------
When no uncertainties are supplied a Poisson proxy is used, and the correct
proxy depends on the detector geometry.

    geometry='1D'  stepped scan (Bonse-Hart), counts ~ h        : sigma^2 ~ I
    geometry='2D'  annular average of an area detector,
                   counts ~ Q h                                 : sigma^2 ~ I/Q

The same geometric factor enters the Fisher branch as the weight w(Q).

Verdict rule (default tolerance 0.10)
    R_MSE < 1 - tol  -> 'log'
    R_MSE > 1 + tol  -> 'linear'
    otherwise        -> 'tied'
"""

from __future__ import annotations

import numpy as np

# numpy >= 2.0 renamed trapz to trapezoid and removed the old name;
# evaluate lazily so that neither branch touches a missing attribute.
_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

#: prefactor of  E* = K a^(2/3) b^(1/3)  for  E(x) = a/x + (b/12) x^2
_K = (3.0 / 2.0 ** (2.0 / 3.0)) * 12.0 ** (-1.0 / 3.0)

_GEOMETRY = {
    "1d": "1D", "1D": "1D", "stepped": "1D", "scan": "1D", "usans": "1D",
    "2d": "2D", "2D": "2D", "annular": "2D", "area": "2D", "pinhole": "2D",
}

__all__ = [
    "derivative", "poisson_sigma2", "variance_density",
    "R_MSE", "optimal_bin_widths", "resolution_ratio",
    "R_Fisher_finiteN", "background_diagnostic", "analyze_binning",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _resolve_geometry(geometry):
    try:
        return _GEOMETRY[str(geometry)]
    except KeyError:
        raise ValueError("geometry must be '1D' or '2D', got "
                         f"{geometry!r}") from None


def _as_sorted_arrays(Q, I, err=None, dQ=None):
    Q = np.asarray(Q, dtype=float)
    I = np.asarray(I, dtype=float)
    if Q.ndim != 1 or I.shape != Q.shape:
        raise ValueError("Q and I must be one-dimensional arrays of equal length")
    if Q.size < 4:
        raise ValueError("at least four points are required")
    order = np.argsort(Q)
    out = [Q[order], I[order]]
    for extra in (err, dQ):
        out.append(None if extra is None
                   else np.asarray(extra, dtype=float)[order])
    return tuple(out)


def derivative(Q, I):
    """dI/dQ on a possibly non-uniform grid (second-order central differences)."""
    return np.gradient(np.asarray(I, dtype=float), np.asarray(Q, dtype=float))


def poisson_sigma2(Q, I, geometry="2D"):
    """
    Poisson proxy for the noise term.

    geometry='1D' -> sigma^2 ~ I        (counts proportional to h)
    geometry='2D' -> sigma^2 ~ I / Q    (counts proportional to Q h)
    """
    Q = np.asarray(Q, dtype=float)
    I = np.asarray(I, dtype=float)
    g = _resolve_geometry(geometry)
    return np.maximum(I if g == "1D" else I / Q, 1e-300)


def variance_density(Q, err):
    """
    Convert reported uncertainties to a variance density.

    Var(bin of width h) = sigma^2(Q)/h, so sigma^2(Q) = err^2 * h_in(Q) with
    h_in the local spacing of the delivered grid.
    """
    Q = np.asarray(Q, dtype=float)
    err = np.asarray(err, dtype=float)
    return err ** 2 * np.gradient(Q)


def _moments(Q, sigma2, I_prime):
    """The four window-averaged moments entering R_MSE."""
    L = Q[-1] - Q[0]
    return (_trapezoid(sigma2, Q) / L,
            _trapezoid(sigma2 / Q, Q) / L,
            _trapezoid(I_prime ** 2, Q) / L,
            _trapezoid(Q ** 2 * I_prime ** 2, Q) / L)


# ---------------------------------------------------------------------------
# reconstruction branch
# ---------------------------------------------------------------------------
def R_MSE(Q, I, err=None, I_prime=None, geometry="2D"):
    """
    Closed-form ratio R_MSE = E*_log / E*_L.

    Parameters
    ----------
    Q, I : array-like
        Momentum transfer (1/A) and intensity.
    err : array-like, optional
        One-sigma uncertainties.  A Poisson proxy set by `geometry` is used
        if omitted.
    I_prime : array-like, optional
        Pre-computed dI/dQ.
    geometry : {'2D', '1D'}
        Detector geometry.  Used only for the Poisson proxy.

    Returns
    -------
    float
        R_MSE < 1 favours log binning, R_MSE > 1 favours linear binning.
    """
    Q, I, err, _ = _as_sorted_arrays(Q, I, err)
    if I_prime is None:
        I_prime = derivative(Q, I)
    sigma2 = poisson_sigma2(Q, I, geometry) if err is None else err ** 2
    sb, J1, bb, J2 = _moments(Q, sigma2, I_prime)
    return float((J1 / sb) ** (2.0 / 3.0) * (J2 / bb) ** (1.0 / 3.0))


def optimal_bin_widths(Q, I, err=None, I_prime=None, geometry="2D"):
    """
    Optimal widths of the two schemes, in physical units of Q.

    Uses the variance density, which is what gives h* the units of Q.

    Returns
    -------
    dict
        h_linear   optimal constant width of the linear scheme
        delta_log  optimal logarithmic step, h_log(Q) = Q * delta_log
        N_linear, N_log   the corresponding bin counts
    """
    Q, I, err, _ = _as_sorted_arrays(Q, I, err)
    if I_prime is None:
        I_prime = derivative(Q, I)
    sigma2 = (poisson_sigma2(Q, I, geometry) * np.gradient(Q) if err is None
              else variance_density(Q, err))
    sb, J1, bb, J2 = _moments(Q, sigma2, I_prime)
    h_lin = float((6.0 * sb / bb) ** (1.0 / 3.0))
    d_log = float((6.0 * J1 / J2) ** (1.0 / 3.0))
    return {"h_linear": h_lin, "delta_log": d_log,
            "N_linear": float((Q[-1] - Q[0]) / h_lin),
            "N_log": float(np.log(Q[-1] / Q[0]) / d_log)}


def resolution_ratio(Q, I, err=None, dQ=None, B=None,
                     I_prime=None, geometry="2D"):
    """
    Resolution-aware ratio R_res = (R_MSE + g)/(1 + g).

    Parameters
    ----------
    dQ : array-like, optional
        Total instrumental Q-resolution, the fourth column of a standard
        reduced file.  The binning contribution of the delivered grid is
        removed internally as B = dQ^2 - h_in^2/12, leaving only the
        scheme-independent part.
    B : array-like, optional
        Scheme-independent Q-variance supplied directly.  Overrides `dQ`.

    Returns
    -------
    dict
        R_MSE       plain ratio
        rho         <B (I')^2> / <(h_in^2/12)(I')^2>, comparing the resolution
                    and binning contributions to the expected squared
                    variation at the grid the profile was delivered on
        B_fraction  mean share of dQ^2 that is collimation and wavelength
                    rather than binning
        R_res       callable, R_res(g) = (R_MSE + g)/(1 + g)
        All of R_res, rho and B_fraction are None if neither dQ nor B is given.

    Notes
    -----
    rho is the quantity that can be read straight off a delivered profile.  A
    value well above one means the instrument, rather than the binning, sets
    the definition in Q for the data in hand.  It is a property of that grid,
    so refining the grid at fixed measurement raises rho as h_in^-2.

    The weight g = G/E*_L is NOT returned as a number, and deliberately so.
    Writing E*_L = (h*_L)^2 betabar / 4 at the optimum gives
    g = 4 <B (I')^2> / ((h*_L)^2 betabar), in which no input bin width
    appears: g is a property of the profile and of the measurement, not of
    the delivered grid.  Evaluating it therefore needs h*_L and hence the
    noise density, which reported uncertainties do not supply on their own.
    Combining the two expressions gives g = (rho/3)(h_in/h*_L)^2, in which
    the factors of h_in cancel identically, so that relation is a consistency
    check rather than a route from rho to g.

    None of this affects the statement that matters.  Since
    R_res(g) - 1 = (R_MSE - 1)/(1 + g) for every g >= 0, the resolution term
    moves the ratio toward unity but never across it, whatever g happens to
    be.  R_res is therefore returned as a callable so that the user may
    inspect the whole family.
    """
    Q, I, err, dQ = _as_sorted_arrays(Q, I, err, dQ)
    if I_prime is None:
        I_prime = derivative(Q, I)
    h_in = np.gradient(Q)
    L = Q[-1] - Q[0]

    R = R_MSE(Q, I, err=err, I_prime=I_prime, geometry=geometry)

    if B is None:
        if dQ is None:
            return {"R_MSE": R, "R_res": None, "g": None,
                    "rho": None, "B_fraction": None}
        B = np.maximum(dQ ** 2 - h_in ** 2 / 12.0, 0.0)
        B_fraction = float(np.mean(1.0 - (h_in ** 2 / 12.0) / dQ ** 2))
    else:
        B = np.asarray(B, dtype=float)
        B_fraction = None

    # rho compares the resolution and binning contributions to the expected
    # squared variation at the grid the profile was delivered on.  It is
    # dimensionless, invariant under a change of the units of Q, and readable
    # straight off the data.  It is a property of that grid: refining the
    # grid at fixed measurement raises rho as h_in^-2.
    rho = float(_trapezoid(B * I_prime ** 2, Q)
                / _trapezoid((h_in ** 2 / 12.0) * I_prime ** 2, Q))
    return {"R_MSE": R, "rho": rho, "B_fraction": B_fraction,
            "R_res": lambda g: (R + g) / (1.0 + g)}


# ---------------------------------------------------------------------------
# Fisher branch
# ---------------------------------------------------------------------------
def _linear_bins(Qmin, Qmax, N):
    e = np.linspace(Qmin, Qmax, N + 1)
    return 0.5 * (e[:-1] + e[1:]), np.diff(e)


def _log_bins(Qmin, Qmax, N):
    e = np.logspace(np.log10(Qmin), np.log10(Qmax), N + 1)
    return np.sqrt(e[:-1] * e[1:]), np.diff(e)


def R_Fisher_finiteN(Qmin, Qmax, N, I_func, dI_func, geometry="2D"):
    """
    Finite-N Fisher ratio F_log / F_L for a parametric model.

    The per-bin contribution is w(Q_k) [dI/dtheta]^2 / I * h_k, with the
    geometric weight w(Q) = Q for an annular average and w(Q) = 1 for a
    stepped scan.  The weight belongs to the integrand rather than to the
    partition, so the sum remains a Riemann sum and the ratio tends to unity
    as N grows.
    """
    g = _resolve_geometry(geometry)
    w = (lambda q: q) if g == "2D" else (lambda q: np.ones_like(q))
    cL, hL = _linear_bins(Qmin, Qmax, N)
    cG, hG = _log_bins(Qmin, Qmax, N)
    F_L = float(np.sum(w(cL) * hL * dI_func(cL) ** 2 / I_func(cL)))
    F_G = float(np.sum(w(cG) * hG * dI_func(cG) ** 2 / I_func(cG)))
    return F_G / F_L


# ---------------------------------------------------------------------------
# background diagnostic
# ---------------------------------------------------------------------------
def background_diagnostic(Q, I, err, window=9, slope_threshold=0.25):
    """
    Locate the background-dominated part of a profile.

    In a signal-dominated region the uncertainty tracks the intensity and the
    local slope d ln(sigma^2)/d ln(I) is of order unity.  Where the profile
    has fallen into incoherent or instrumental background the uncertainty
    decouples from the intensity and the slope approaches zero.  Such a region
    carries no structural signal, yet it still enters all four moments of
    R_MSE and, because those moments are Q-weighted, it can dominate them.

    Returns
    -------
    dict
        slope             local slope at each point, NaN where undefined
        flagged           boolean mask, slope below `slope_threshold`
        Q_trim            suggested upper cut, or None
        n_trimmed         number of points beyond Q_trim
        fraction_flagged  overall share of flagged points
    """
    Q, I, err, _ = _as_sorted_arrays(Q, I, err)
    n = Q.size
    w = max(5, int(window) | 1)
    half = w // 2
    logI, logS = np.log(I), np.log(err ** 2)
    slope = np.full(n, np.nan)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        x, y = logI[lo:hi], logS[lo:hi]
        if np.ptp(x) < 1e-10:
            continue
        slope[i] = np.polyfit(x, y, 1)[0]

    # A window over which the intensity itself is flat leaves the slope
    # undefined.  That is deliberately left unflagged, since a genuine
    # plateau (a polymer melt, say) is not the same thing as background.
    flagged = np.nan_to_num(slope, nan=1.0) < slope_threshold
    Q_trim, n_trim = None, 0
    if flagged[-1]:
        i = n - 1
        while i >= 0 and flagged[i]:
            i -= 1
        if i >= 3:
            Q_trim, n_trim = float(Q[i]), int(n - 1 - i)
    return {"slope": slope, "flagged": flagged, "Q_trim": Q_trim,
            "n_trimmed": n_trim, "fraction_flagged": float(np.mean(flagged))}


# ---------------------------------------------------------------------------
# workflow
# ---------------------------------------------------------------------------
def _verdict(R, tol):
    if abs(R - 1.0) < tol:
        return "tied"
    return "log" if R < 1.0 else "linear"


def analyze_binning(Q, I, err=None, dQ=None, geometry="2D",
                    model=None, dmodel=None, theta=None, N_future=None,
                    tolerance=0.10, check_background=True, verbose=False):
    """
    Apply the log-versus-linear binning decision workflow.

    Parameters
    ----------
    Q, I : array-like
        Measured profile.
    err : array-like, optional
        One-sigma uncertainties.  A Poisson proxy set by `geometry` is used
        if omitted.
    dQ : array-like, optional
        Instrumental Q-resolution.  When supplied the resolution-aware ratio
        is reported alongside the plain one.
    geometry : {'2D', '1D'}
        '2D' for the annular average of an area detector, '1D' for a stepped
        scan.  Affects the Poisson proxy and the Fisher weight.
    model, dmodel, theta, N_future
        Optional parametric model I(Q; theta) and dI/dtheta(Q; theta) for the
        Fisher branch, evaluated at `theta` for `N_future` bins.
    tolerance : float
        Half-width of the tied band around unity.
    check_background : bool
        Run the background diagnostic when uncertainties are available.
    verbose : bool
        Print a summary.

    Returns
    -------
    dict
        verdict, R_MSE, R_MSE_subsampled, converged, geometry, R_res, g, rho,
        B_fraction, R_Fisher_finite, background, h_linear, delta_log, rationale
    """
    geo = _resolve_geometry(geometry)
    Q, I, err, dQ = _as_sorted_arrays(Q, I, err, dQ)
    I_prime = derivative(Q, I)

    R = R_MSE(Q, I, err=err, I_prime=I_prime, geometry=geo)

    sub = slice(None, None, 2)
    R_sub = R_MSE(Q[sub], I[sub],
                  err=None if err is None else err[sub], geometry=geo)
    converged = bool(abs(R - R_sub) / R <= tolerance)

    res = resolution_ratio(Q, I, err=err, dQ=dQ, I_prime=I_prime, geometry=geo)
    widths = optimal_bin_widths(Q, I, err=err, I_prime=I_prime, geometry=geo)

    R_F, N = None, None
    if model is not None and dmodel is not None:
        N = int(N_future) if N_future is not None else Q.size
        I_fn = (lambda q: model(q, theta)) if theta is not None else model
        d_fn = (lambda q: dmodel(q, theta)) if theta is not None else dmodel
        R_F = R_Fisher_finiteN(float(Q[0]), float(Q[-1]), N, I_fn, d_fn, geo)

    bg = None
    if check_background and err is not None:
        bg = background_diagnostic(Q, I, err)

    verdict = _verdict(R, tolerance)
    pct = int(round(tolerance * 100))
    if verdict == "tied":
        rationale = (f"R_MSE = {R:.3f} lies within {pct}% of unity. The two "
                     "schemes give essentially equivalent reconstruction MSE.")
    elif verdict == "log":
        rationale = (f"R_MSE = {R:.3f} < 1. Log-spaced binning gives the "
                     "smaller reconstruction MSE at fixed bin count.")
    else:
        rationale = (f"R_MSE = {R:.3f} > 1. Linear-spaced binning gives the "
                     "smaller reconstruction MSE at fixed bin count.")

    if res["rho"] is not None:
        rationale += (f" At the delivered grid the instrumental resolution "
                      f"contributes {res['rho']:.0f} times as much as the "
                      "binning to the expected squared variation, and "
                      f"{res['B_fraction'] * 100:.0f}% of the reported dQ^2 is "
                      "collimation and wavelength rather than binning. Because "
                      "the resolution term is scheme-independent it moves the "
                      "ratio toward unity without changing the direction of "
                      "the verdict, for any weight g; use result['R_res'](g) "
                      "to evaluate (R_MSE + g)/(1 + g).")

    if R_F is not None:
        if abs(R_F - 1.0) > tolerance:
            rationale += (f" Fisher ratio at N = {N} is {R_F:.3f}, favouring "
                          f"{'log' if R_F > 1 else 'linear'} binning for "
                          "parameter estimation.")
        else:
            rationale += (f" Fisher ratio at N = {N} is {R_F:.3f}, so parameter "
                          "precision is scheme-independent.")

    if bg is not None and bg["Q_trim"] is not None:
        rationale += (f" WARNING: the uncertainty stops tracking the intensity "
                      f"above Q = {bg['Q_trim']:.4g} 1/A ({bg['n_trimmed']} "
                      "points). That region is background dominated. Consider "
                      "trimming it and re-running, since a flat tail can "
                      "dominate the moments.")

    if not converged:
        rationale += (f" WARNING: the subsampled estimate R_MSE = {R_sub:.3f} "
                      f"differs from the full-grid estimate by more than "
                      f"{pct}%. The verdict is not yet stable. Supply a less "
                      "aggressively rebinned profile and check that it settles.")

    result = {
        "verdict": verdict,
        "R_MSE": float(R),
        "R_MSE_subsampled": float(R_sub),
        "converged": converged,
        "geometry": geo,
        "R_res": res["R_res"],
        "rho": res["rho"],
        "B_fraction": res["B_fraction"],
        "R_Fisher_finite": None if R_F is None else float(R_F),
        "background": bg,
        "h_linear": widths["h_linear"],
        "delta_log": widths["delta_log"],
        "rationale": rationale,
    }

    if verbose:
        print(f"geometry          = {geo}")
        print(f"R_MSE             = {R:.4f}")
        print(f"R_MSE_subsampled  = {R_sub:.4f}   (converged: {converged})")
        if res["rho"] is not None:
            print(f"rho               = {res['rho']:.4g}   "
                  f"(B is {res['B_fraction'] * 100:.0f}% of dQ^2)")
            print(f"R_res(g=1)        = {res['R_res'](1.0):.4f}   "
                  "[R_res(g) available for any g >= 0]")
        if R_F is not None:
            print(f"R_Fisher_finite   = {R_F:.4f}")
        print(f"optimal h_linear  = {widths['h_linear']:.4g} 1/A "
              f"(N ~ {widths['N_linear']:.0f})")
        print(f"optimal delta_log = {widths['delta_log']:.4g} "
              f"(N ~ {widths['N_log']:.0f})")
        print(f"verdict           = {verdict}")
        print(f"  {rationale}")

    return result
