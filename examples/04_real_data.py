"""
Example 4: batch analysis of measured EQ-SANS profiles.

Runs the workflow over every file in ``data/`` and prints a summary table.
Files with four columns carry the instrumental Q-resolution and therefore
also get the resolution-aware ratio; three-column files report the plain
ratio only.
"""

import numpy as np
from pathlib import Path
from binningverdict import analyze_binning

data_dir = Path('data')
files = sorted(data_dir.glob('*.dat')) + sorted(data_dir.glob('*.txt'))
if not files:
    raise SystemExit(f"No I(q) files found in '{data_dir}/'.")


def load(path):
    d = np.loadtxt(path, comments='#')
    Q, I, err = d[:, 0], d[:, 1], d[:, 2]
    dQ = d[:, 3] if d.shape[1] > 3 else None
    keep = (I > 0) & (err > 0)
    return Q[keep], I[keep], err[keep], (None if dQ is None else dQ[keep])


head = (f"{'file':<40}{'N':>5}{'R_MSE':>9}{'verdict':>9}"
        f"{'rho':>8}{'B/dQ2':>8}{'R_res(1)':>10}  flags")
print(head)
print('-' * len(head))

for path in files:
    Q, I, err, dQ = load(path)
    r = analyze_binning(Q, I, err=err, dQ=dQ, geometry='2D')
    flags = []
    if not r['converged']:
        flags.append('not-converged')
    if r['background'] is not None and r['background']['Q_trim'] is not None:
        flags.append(f"trim>{r['background']['Q_trim']:.3f}")
    fmt = lambda v: 'NA' if v is None else f'{v:.2f}'
    Rres1 = 'NA' if r['rho'] is None else f"{r['R_res'](1.0):.3f}"
    Bfrac = 'NA' if r['B_fraction'] is None else f"{r['B_fraction'] * 100:.0f}%"
    print(f"{path.name:<40}{len(Q):>5d}{r['R_MSE']:>9.3f}{r['verdict']:>9}"
          f"{fmt(r['rho']):>8}{Bfrac:>8}{Rres1:>10}"
          f"  {','.join(flags)}")

print()
print("rho compares the resolution and binning contributions to the expected")
print("squared variation at the grid each profile was delivered on. Values")
print("well above one mean the instrument, not the binning, sets the Q")
print("definition. R_res(1) is the resolution-aware ratio evaluated at g = 1,")
print("shown only as a representative point: the weight g is a property of")
print("the profile and of the measurement, not of the delivered grid, so")
print("R_res is returned as a callable. Whatever g may be, the identity")
print("R_res(g) - 1 = (R_MSE - 1)/(1 + g) keeps the sign of R_MSE - 1, so the")
print("direction of the verdict is invariant.")
