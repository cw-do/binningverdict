"""
Example 2: Run binningverdict on canonical SANS models.

Reproduces the benchmark tables of the accompanying paper.
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
    F = np.where(x < 1e-4, 1 - x**2/10, 3*(np.sin(x) - x*np.cos(x))/x**3)
    return F ** 2

def cylinder(Q, R, L, n_angle=80):
    alpha = np.linspace(1e-4, np.pi/2 - 1e-4, n_angle)
    P = np.zeros_like(Q)
    for a in alpha:
        qrs = Q * R * np.sin(a)
        qlc = Q * L * np.cos(a) / 2
        A_rad = np.where(qrs < 1e-6, 1.0, 2*j1(qrs)/qrs)
        A_len = np.where(qlc < 1e-6, 1.0, np.sin(qlc)/qlc)
        P += (A_rad * A_len) ** 2 * np.sin(a)
    return P * (alpha[1] - alpha[0])

def lorentzian(Q, Q_star, gamma):
    return 1.0 / ((Q - Q_star) ** 2 + gamma ** 2)

def teubner_strey(Q, a, b, c):
    return 1.0 / (a + b * Q ** 2 + c * Q ** 4)


Q = np.logspace(-3, np.log10(0.5), 5000)

print(f"{'Model':<32}{'R_MSE':>10}{'verdict':>10}")
print("-" * 52)

cases = [
    ("Power law alpha=2",            power_law(Q, 2.0)),
    ("Power law alpha=3",            power_law(Q, 3.0)),
    ("Guinier Rg=30 A",              guinier(Q, 30)),
    ("Sphere R=30 A",                sphere(Q, 30)),
    ("Cylinder R=20, L=200 A",       cylinder(Q, 20, 200)),
    ("Lorentzian Q*=0.05",           lorentzian(Q, 0.05, 0.01)),
    ("Teubner-Strey",                teubner_strey(Q, 1.0, -100, 1e5)),
]

for name, I in cases:
    R = R_MSE(Q, I, err=np.sqrt(np.maximum(I, 1e-30)))
    if R < 0.9:
        verdict = "log"
    elif R > 1.1:
        verdict = "linear"
    else:
        verdict = "tied"
    print(f"{name:<32}{R:>10.3f}{verdict:>10}")
