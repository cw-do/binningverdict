"""
Example 5: locating and trimming a background-dominated tail.

Where a profile has fallen into incoherent or instrumental background the
uncertainty stops tracking the intensity, so the local slope

    d ln(sigma^2) / d ln(I)

falls toward zero.  That region carries no structural signal but still enters
all four moments of R_MSE, and because those moments are Q-weighted it can
dominate them.  The recommended practice is to trim the profile to the range
used for model fitting, then confirm that the verdict is unchanged.
"""

import numpy as np
from binningverdict import analyze_binning, background_diagnostic

data = np.loadtxt('data/SDS_5mgmL_0p1M_20C_Iq.dat', comments='#')
Q, I, err, dQ = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
keep = (I > 0) & (err > 0)
Q, I, err, dQ = Q[keep], I[keep], err[keep], dQ[keep]

bg = background_diagnostic(Q, I, err)
print(f"points flagged      : {bg['fraction_flagged'] * 100:.0f}%")
print(f"suggested Q_trim    : {bg['Q_trim']}")
print(f"points beyond Q_trim: {bg['n_trimmed']}")
print()
print(f"{'Q (1/A)':>10}{'I':>12}{'sigma^2':>12}{'slope':>9}  flagged")
for i in range(0, len(Q), max(1, len(Q) // 15)):
    print(f"{Q[i]:>10.4f}{I[i]:>12.4g}{err[i] ** 2:>12.4g}"
          f"{bg['slope'][i]:>9.2f}  {'yes' if bg['flagged'][i] else ''}")

print()
full = analyze_binning(Q, I, err=err, dQ=dQ)
print(f"untrimmed : R_MSE = {full['R_MSE']:.3f}  ->  {full['verdict']}")

if bg['Q_trim'] is not None:
    m = Q <= bg['Q_trim']
    trimmed = analyze_binning(Q[m], I[m], err=err[m], dQ=dQ[m])
    print(f"trimmed   : R_MSE = {trimmed['R_MSE']:.3f}  ->  {trimmed['verdict']}"
          f"   ({m.sum()} of {len(Q)} points kept)")
    if trimmed['verdict'] == full['verdict']:
        print()
        print("The two agree, so the verdict is not being driven by the")
        print("background region and either range may be reported.")
    else:
        print()
        print("The two disagree. Prefer the trimmed result, since the flat")
        print("tail carries no structural information.")
