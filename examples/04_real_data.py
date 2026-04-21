"""
Example 4: Apply binningverdict to a batch of measured EQ-SANS profiles.
"""

import numpy as np
from pathlib import Path
from binningverdict import analyze_binning


# Replace path with your own folder of (Q, I, sigma) .dat files
data_dir = Path('eqsans_profiles')

if not data_dir.exists():
    raise SystemExit(f"Place .dat files in '{data_dir}/' before running.")

print(f"{'file':<40}{'N_in':>6}{'R_MSE':>10}{'verdict':>10}")
print("-" * 70)

for f in sorted(data_dir.glob('*.dat')) + sorted(data_dir.glob('*.txt')):
    data = np.loadtxt(f, comments='#')
    Q, I, err = data[:, 0], data[:, 1], data[:, 2]
    pos = (I > 0) & (err > 0)
    Q, I, err = Q[pos], I[pos], err[pos]
    res = analyze_binning(Q, I, err=err)
    flag = '*' if not res['converged'] else ' '
    print(f"{f.name:<40}{len(Q):>6d}{res['R_MSE']:>10.3f}"
          f"{res['verdict']:>10}{flag}")

print("\n* = subsample-vs-full disagreement >10% (run on denser input)")
