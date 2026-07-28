# binningverdict

A quantitative logarithmic-versus-linear binning decision system for
small-angle scattering data, accompanying the submitted manuscript:

> *binningverdict: a unified decision system for logarithmic versus linear
> binning of small-angle neutron scattering data.*

The manuscript is included in this repository as
[`manuscript.pdf`](manuscript.pdf).

Given a measured profile `(Q, I, sigma)` the workflow returns a verdict of
`'log'`, `'linear'` or `'tied'`, based on a closed-form ratio `R_MSE` that
compares the reconstruction mean-squared error of the two schemes at fixed
bin count. If the instrumental resolution column `dQ` is supplied, a
resolution-aware ratio is reported as well. A Fisher-information ratio is
available when a parametric model is given.

## What is new in 1.1

Version 1.1 implements the corrections and additions made during peer review.

| change | why |
|---|---|
| `geometry='2D'` / `'1D'` | The counts in an annular bin scale as `Q h`, not `h`, so the Poisson proxy for an area detector is `sigma^2 ~ I/Q`. A stepped scan such as Bonse–Hart keeps `sigma^2 ~ I`. |
| `dQ=` argument, `resolution_ratio` | The full Q-variance is `B(Q) + h^2/12`, not `h^2/12` alone. `B` enters both schemes identically, so it can only move the ratio toward unity, never across it. |
| Solid-angle weight in the Fisher branch | `w(Q) = Q` for an annular average. The theorem is unchanged and convergence improves. |
| `background_diagnostic` | Locates the background-dominated tail, where `sigma` stops tracking `I` and the moments can be dominated by a region carrying no signal. |
| `optimal_bin_widths` | Reports the optimal linear width and logarithmic step in physical units. |
| NumPy 2.x fix | `np.trapz` was removed in NumPy 2.0; the fallback is now evaluated lazily. |

Note that `geometry` defaults to `'2D'`. It affects the result **only** when
uncertainties are not supplied, since measured uncertainties already carry the
solid-angle weighting.

## Installation

```bash
git clone https://github.com/cw-do/binningverdict
cd binningverdict
pip install -e .
```

Requires Python ≥ 3.8 and NumPy. SciPy is needed only by
`examples/02_canonical_models.py`.

## Quick start

```python
import numpy as np
from binningverdict import analyze_binning

data = np.loadtxt('SDS_5mgmL_0p1M_20C_Iq.dat', comments='#')
Q, I, err, dQ = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

result = analyze_binning(Q, I, err=err, dQ=dQ, verbose=True)
```

```text
geometry          = 2D
R_MSE             = 2.5656
R_MSE_subsampled  = 3.4553   (converged: False)
R_res             = 1.3399   (g = 3.606, rho = 4.73)
optimal h_linear  = 0.00312 1/A (N ~ 140)
optimal delta_log = 0.03526 (N ~ 103)
verdict           = linear
```

### Returned fields

| key | meaning |
|---|---|
| `verdict` | `'log'`, `'linear'` or `'tied'` |
| `R_MSE` | closed-form MSE ratio |
| `R_MSE_subsampled` | same on every other point, the self-consistency check |
| `converged` | `True` if the two agree within tolerance |
| `geometry` | `'2D'` or `'1D'` |
| `R_res` | resolution-aware ratio `(R + g)/(1 + g)`, or `None` |
| `g` | weight of the resolution term, `G / E*_L` |
| `rho` | resolution bias divided by binning bias at the delivered grid |
| `B_fraction` | share of `dQ^2` that is collimation and wavelength |
| `R_Fisher_finite` | finite-N Fisher ratio, if a model was supplied |
| `background` | diagnostic dictionary, or `None` |
| `h_linear`, `delta_log` | optimal widths in physical units |
| `rationale` | human-readable explanation and warnings |

## Reading the resolution output

`R_res` follows from adding the same constant `G = <B (I')^2>` to the
optimum of both schemes:

```
R_res = (E*_log + G) / (E*_L + G) = (R_MSE + g) / (1 + g)
R_res - 1 = (R_MSE - 1) / (1 + g)
```

Since `g >= 0`, the resolution term moves the ratio toward unity but can
never move it across unity. The direction of the verdict is an invariant.

For real SANS data `g` is often large, because the instrumental resolution
greatly exceeds the statistically optimal bin width. `R_res` then sits close
to unity, which is the correct statement that a measurement whose Q-variance
is entirely instrumental is indifferent to the binning scheme. That is also
why `R_MSE`, and not `R_res`, is the useful decision statistic: the
resolution is fixed by the instrument, whereas the partition is the thing the
analyst actually controls.

`rho` is easier to read. It compares the two bias terms at the grid actually
delivered, so `rho = 5` means the instrument contributes five times more to
the Q definition than the binning does.

### Normalisation

`R_MSE` is a ratio in which the normalisation of `sigma^2` cancels, so
reported uncertainties are used directly. `g`, `R_res` and the optimal widths
are absolute, and internally use the variance *density*
`sigma^2 = err^2 * h_in`, which follows from `Var(bin of width h) =
sigma^2(Q)/h`. Without that conversion `g` would depend on whether `Q` is
given in `1/A` or `1/nm`.

## Command-line use

```bash
python binningverdict.py your_measurement_Iq.dat
python binningverdict.py *_Iq.dat
python binningverdict.py --geometry 1D rocking_scan.txt
python binningverdict.py --no-resolution your_measurement_Iq.dat
```

Input files are ASCII tables with at least three numeric columns:

```text
Q   intensity   intensity_error   [Q_error]
```

The fourth column, when present, is used as the instrumental resolution.
Batch mode writes `binningverdict_report.txt` as a tab-delimited table.

## Examples

Run from inside `examples/`:

| script | what it shows |
|---|---|
| `01_basic_usage.py` | one EQ-SANS profile with its `dQ` column |
| `02_canonical_models.py` | canonical models under both conventions |
| `03_fisher_branch.py` | Fisher branch and both geometric weights |
| `04_real_data.py` | batch over `examples/data/` |
| `05_background_trimming.py` | locating and trimming a background tail |

`examples/data/` contains three measured EQ-SANS profiles, one of which
(`SDS_5mgmL_0p1M_20C_Iq.dat`) carries a fourth resolution column.

## Tests

```bash
pip install pytest
pytest tests/
```

## Citation

```bibtex
@article{binningverdict2026,
  author  = {Do, Changwoo and Ding, Lijie and Tung, Chi-Huan and Chen, Wei-Ren},
  title   = {binningverdict: a unified decision system for logarithmic versus
             linear binning of small-angle neutron scattering data},
  journal = {Comput. Phys. Commun.},
  year    = {2026},
  note    = {submitted}
}
```

## License

MIT.
