"""
Example 3: the Fisher-information branch.

A parametric model I(Q; theta) and its derivative dI/dtheta are supplied so
that the finite-N ratio F_log / F_L can be evaluated.  The geometric weight
w(Q) belongs to the integrand rather than to the partition, so the sum stays
a Riemann sum and the ratio tends to unity as N grows in both geometries.

    geometry='1D'  ->  w(Q) = 1
    geometry='2D'  ->  w(Q) = Q
"""

import numpy as np
from binningverdict import analyze_binning, R_Fisher_finiteN

Rg = 30.0
Q = np.logspace(np.log10(0.005), np.log10(0.3), 200)
I = np.exp(-Q ** 2 * Rg ** 2 / 3)
err = np.sqrt(I)


def model(Q, theta):
    return np.exp(-Q ** 2 * theta ** 2 / 3)


def dmodel(Q, theta):
    return -2 * Q ** 2 * theta / 3 * model(Q, theta)


analyze_binning(Q, I, err=err, model=model, dmodel=dmodel, theta=Rg,
                N_future=200, geometry='2D', verbose=True)

# Convergence of the ratio for a Guinier profile and for a power law, which
# is the slowest-converging case.  The solid-angle weight accelerates it.
def I_pl(q):
    return q ** -3.0


def dI_pl(q):
    return -np.log(q) * q ** -3.0


print()
print(f"{'N':>6}{'Guinier 1D':>13}{'Guinier 2D':>13}"
      f"{'PowerLaw 1D':>14}{'PowerLaw 2D':>14}")
for N in [20, 50, 100, 200, 500, 1000]:
    row = []
    for I_fn, dI_fn in ((lambda q: model(q, Rg), lambda q: dmodel(q, Rg)),
                        (I_pl, dI_pl)):
        for geo in ('1D', '2D'):
            row.append(R_Fisher_finiteN(0.005, 0.3, N, I_fn, dI_fn, geometry=geo))
    print(f"{N:>6d}{row[0]:>13.4f}{row[1]:>13.4f}{row[2]:>14.4f}{row[3]:>14.4f}")
