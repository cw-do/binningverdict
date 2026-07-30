"""
Example 1: basic usage on a measured EQ-SANS profile.

Uses ``data/SDS_5mgmL_0p1M_20C_Iq.dat``, a four-column reduced file whose
last column is the instrumental Q-resolution.  Supplying that column makes
the workflow report the resolution-aware ratio as well as the plain one.
"""

import numpy as np
from binningverdict import analyze_binning

data = np.loadtxt('data/SDS_5mgmL_0p1M_20C_Iq.dat', comments='#')
Q, I, err, dQ = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

keep = (I > 0) & (err > 0)
Q, I, err, dQ = Q[keep], I[keep], err[keep], dQ[keep]

result = analyze_binning(Q, I, err=err, dQ=dQ, geometry='2D', verbose=True)

print()
print('--- selected fields -------------------------------------------------')
for key in ('verdict', 'R_MSE', 'rho', 'B_fraction',
            'converged', 'h_linear', 'delta_log'):
    print(f'  {key:<12} {result[key]}')

# R_res is returned as a callable, because the weight g is a property of the
# profile and of the measurement rather than of the delivered grid, and
# evaluating it numerically needs the noise density.  The invariance
#     R_res(g) - 1 = (R_MSE - 1)/(1 + g)
# holds for every g >= 0, so the direction of the verdict never changes.
print()
print('  resolution-aware ratio over the whole family:')
print(f"  {'g':>10}{'R_res':>10}")
for g in (0.0, 0.3, 1.0, 3.0, 10.0, 100.0):
    print(f'  {g:>10.1f}{result["R_res"](g):>10.3f}')

# Without the resolution column the plain ratio is unchanged and the
# resolution fields come back as None.
plain = analyze_binning(Q, I, err=err)
print()
print(f'  R_MSE with dQ    : {result["R_MSE"]:.4f}')
print(f'  R_MSE without dQ : {plain["R_MSE"]:.4f}   (identical by construction)')
print(f'  R_res  without dQ: {plain["R_res"]}')
