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
        f"{'R_res':>9}{'g':>10}{'rho':>8}  flags")
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
    fmt = lambda v: 'NA' if v is None else f'{v:.3f}'
    print(f"{path.name:<40}{len(Q):>5d}{r['R_MSE']:>9.3f}{r['verdict']:>9}"
          f"{fmt(r['R_res']):>9}{fmt(r['g']):>10}{fmt(r['rho']):>8}"
          f"  {','.join(flags)}")

print()
print("R_res is the resolution-aware ratio (R + g)/(1 + g).  Because the same")
print("constant enters both schemes it can only move the ratio toward unity,")
print("never across it, so the direction of the verdict is invariant.")
print("rho compares the resolution and binning bias terms at the delivered")
print("grid; values well above one mean the instrument, not the binning,")
print("sets the Q definition.")
