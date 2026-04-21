"""
Example 1: Basic usage of binningverdict.

Loads a measured (Q, I, sigma) profile and runs the workflow.
"""

import numpy as np
from binningverdict import analyze_binning


# Replace with your own .dat file (Q, I, sigma columns)
data = np.loadtxt('your_measurement.dat', comments='#')
Q   = data[:, 0]
I   = data[:, 1]
err = data[:, 2]

# Filter positive points
pos = (I > 0) & (err > 0)
Q, I, err = Q[pos], I[pos], err[pos]

result = analyze_binning(Q, I, err=err, verbose=True)
