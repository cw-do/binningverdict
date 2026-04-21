"""
Example 3: Fisher information branch (parametric model required).

Demonstrates how a structural-model derivative dI/dtheta is supplied
to obtain the Fisher ratio R_F together with R_MSE.
"""

import numpy as np
from binningverdict import analyze_binning


# Synthetic Guinier profile
Rg = 30.0
Q = np.logspace(np.log10(0.005), np.log10(0.3), 200)
I = np.exp(-Q ** 2 * Rg ** 2 / 3)
err = np.sqrt(I)

# Parametric model and its derivative w.r.t. theta = R_g
def model(Q, theta):
    return np.exp(-Q ** 2 * theta ** 2 / 3)

def dmodel(Q, theta):
    return -2 * Q ** 2 * theta / 3 * model(Q, theta)

result = analyze_binning(
    Q, I, err=err,
    model=model, dmodel=dmodel, theta=Rg,
    N_future=200, verbose=True,
)
