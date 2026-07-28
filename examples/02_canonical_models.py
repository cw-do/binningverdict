"""
Example 2: canonical SANS models under both Poisson conventions.

Reproduces the benchmark tables of the accompanying manuscript.  The proxy
for the counting noise depends on the detector geometry:

    geometry='1D'   stepped scan, counts ~ h        ->  sigma^2 ~ I
    geometry='2D'   annular average, counts ~ Q h   ->  sigma^2 ~ I/Q

Every power-law entry stays below unity and every non-power-law entry stays
above unity in both conventions.  The annular weighting roughly doubles the
non-power-law ratios and compresses the power-law ones toward unity.
"""

import numpy as np
from scipy.special import j1
from binningverdict import R_MSE


def power_law(Q, alpha):
    return Q ** (-alpha)


def guinier(Q, Rg):
    return np.exp(-Q ** 2 * Rg ** 2 / 3)


def sphere(Q, R):
    x = Q * R
    F = np.where(x < 1e-4, 1 - x ** 2 / 10,
                 3 * (np.sin(x) - x * np.cos(x)) / x ** 3)
    return F ** 2 + 1e-14


def cylinder(Q, R, L, n_angle=160):
    alpha = np.linspace(1e-4, np.pi / 2 - 1e-4, n_angle)
    P = np.zeros_like(Q)
    for a in alpha:
        qrs, qlc = Q * R * np.sin(a), Q * L * np.cos(a) / 2
        A_rad = np.where(qrs < 1e-6, 1.0, 2 * j1(qrs) / qrs)
        A_len = np.where(qlc < 1e-6, 1.0, np.sin(qlc) / qlc)
        P += (A_rad * A_len) ** 2 * np.sin(a)
    return P * (alpha[1] - alpha[0]) + 1e-14


def lorentzian(Q, Q_star, gamma):
    """Q_star and gamma in 1/A."""
    return 1.0 / ((Q - Q_star) ** 2 + gamma ** 2)


def teubner_strey(Q, a, b, c):
    """a dimensionless, b in A^2, c in A^4."""
    return 1.0 / (a + b * Q ** 2 + c * Q ** 4)


def verdict(R, tol=0.10):
    if abs(R - 1) < tol:
        return 'tied'
    return 'log' if R < 1 else 'linear'


Q = np.logspace(-3, np.log10(0.5), 20000)

cases = [
    ('Power law alpha=1',                     power_law(Q, 1.0)),
    ('Power law alpha=2',                     power_law(Q, 2.0)),
    ('Power law alpha=3',                     power_law(Q, 3.0)),
    ('Power law alpha=4',                     power_law(Q, 4.0)),
    ('Guinier Rg=30 A',                       guinier(Q, 30)),
    ('Guinier Rg=100 A',                      guinier(Q, 100)),
    ('Sphere R=30 A',                         sphere(Q, 30)),
    ('Sphere R=100 A',                        sphere(Q, 100)),
    ('Cylinder R=20 A, L=200 A',              cylinder(Q, 20, 200)),
    ('Cylinder R=50 A, L=1000 A',             cylinder(Q, 50, 1000)),
    ('Lorentzian Q*=0.05, G=0.01 1/A',        lorentzian(Q, 0.05, 0.01)),
    ('Teubner-Strey a=1, b=-100, c=1e5',      teubner_strey(Q, 1.0, -100, 1e5)),
]

print(f"reference window  Q = [{Q[0]:.0e}, {Q[-1]:.1f}] 1/A,  {Q.size} points")
print()
head = f"{'model':<34}{'R (1D)':>9}{'verdict':>9}{'R (2D)':>9}{'verdict':>9}"
print(head)
print('-' * len(head))
for name, I in cases:
    r1 = R_MSE(Q, I, geometry='1D')
    r2 = R_MSE(Q, I, geometry='2D')
    print(f"{name:<34}{r1:>9.3f}{verdict(r1):>9}{r2:>9.3f}{verdict(r2):>9}")

print()
print("Power-law scale invariance (R must not depend on the window):")
for qa, qb in [(1e-3, 0.1), (5e-3, 0.3), (1e-2, 1.0)]:
    Qw = np.logspace(np.log10(qa), np.log10(qb), 20000)
    r1 = R_MSE(Qw, power_law(Qw, 2.0), geometry='1D')
    r2 = R_MSE(Qw, power_law(Qw, 2.0), geometry='2D')
    print(f"  [{qa:g}, {qb:g}]   R(1D) = {r1:.4f}   R(2D) = {r2:.4f}")
