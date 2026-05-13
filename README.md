# binningverdict

A quantitative log-vs-linear binning decision system for small-angle
neutron scattering (SANS) data, accompanying the submitted manuscript:

> *Logarithmic versus linear binning in small-angle neutron scattering: a
> unified decision system synthesizing four decades of information-theoretic
> and statistical analyses*.

The manuscript being submitted is included in this repository as
[`manuscript.pdf`](manuscript.pdf).

Given a measured intensity profile `(Q, I, sigma)`, the workflow returns a
verdict of `'log'`, `'linear'`, or `'tied'` based on a closed-form ratio
`R_MSE` that compares the reconstruction mean-squared error of the two
schemes at fixed bin count. Optionally, a Fisher-information ratio is also
reported when a parametric model is supplied.

## Installation

```bash
git clone https://github.com/cw-do/binningverdict
cd binningverdict
pip install -e .
```

Requires only Python ≥ 3.8 and NumPy.

## Quick start

```python
import numpy as np
from binningverdict import analyze_binning

# Example: load a measured (Q, I, sigma) profile
data = np.loadtxt('your_measurement.dat', comments='#')
Q, I, err = data[:, 0], data[:, 1], data[:, 2]

result = analyze_binning(Q, I, err=err, verbose=True)
# R_MSE   = 2.07
# Verdict = linear
#   R_MSE = 2.070 > 1: linear-spaced binning gives smaller
#   reconstruction MSE at fixed bin count.
```

The returned dictionary contains:

| key                  | meaning                                              |
|----------------------|------------------------------------------------------|
| `R_MSE`              | closed-form MSE ratio                                |
| `R_MSE_subsampled`   | same, evaluated on every other point (sanity check)  |
| `converged`          | True if the two estimates agree within tolerance     |
| `R_Fisher_finite`    | finite-N Fisher ratio (only if a model was given)    |
| `verdict`            | `'log'` / `'linear'` / `'tied'`                      |
| `rationale`          | human-readable explanation                           |

## Examples

The `examples/` folder contains:

- `01_basic_usage.py` — minimal worked example on one EQ-SANS profile
- `02_canonical_models.py` — Power law, Guinier, sphere, cylinder,
  Lorentzian, Teubner-Strey
- `03_fisher_branch.py` — using the parametric model for Fisher analysis
- `04_real_data.py` — case study on five measured EQ-SANS profiles

## Tests

```bash
pip install pytest
pytest tests/
```

## Citation

If you use `binningverdict` in published work, please cite:

```bibtex
@article{binningverdict2026,
  author  = {Do, Changwoo and Ding, Lijie and Tung, Chi-Huan and Chen, Wei-Ren},
  title   = {Logarithmic versus linear binning in small-angle neutron
             scattering: a unified decision system},
  journal = {TBD},
  year    = {2026}
}
```

## License

MIT.
