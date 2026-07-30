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
| `rho`, and `R_res` as a callable | `rho` compares the two contributions at the delivered grid and is readable off the data. The weight `g` is a property of the profile and of the measurement, not of the grid, so no single number is claimed for it. |
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
rho               = 4.73   (B is 79% of dQ^2)
R_res(g=1)        = 1.7828   [R_res(g) available for any g >= 0]
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
| `R_res` | callable, `R_res(g) = (R_MSE + g)/(1 + g)`, or `None` |
| `rho` | resolution divided by binning contribution at the delivered grid |
| `B_fraction` | share of `dQ^2` that is collimation and wavelength |
| `R_Fisher_finite` | finite-N Fisher ratio, if a model was supplied |
| `background` | diagnostic dictionary, or `None` |
| `h_linear`, `delta_log` | optimal widths in physical units |
| `rationale` | human-readable explanation and warnings |

## Reading the resolution output

`R_res` follows from adding the same constant `G = <B (I')^2>` to the
optimum of both schemes:

```
R_res(g) = (E*_log + G) / (E*_L + G) = (R_MSE + g) / (1 + g),   g = G / E*_L
R_res(g) - 1 = (R_MSE - 1) / (1 + g)
```

Since `g >= 0`, the resolution term moves the ratio toward unity but can
never move it across unity. The direction of the verdict is an invariant, and
that statement needs no numerical value of `g`.

### Why `g` is not returned as a number

Writing `E*_L = (h*_L)^2 betabar / 4` at the optimum gives

```
g = 4 <B (I')^2> / ((h*_L)^2 betabar)
```

in which no input bin width appears. `g` is a property of the profile and of
the measurement, not of the grid the profile was delivered on. Evaluating it
therefore requires `h*_L`, and hence the noise density of the measurement,
which reported uncertainties do not supply on their own: they are variances
of the intensity in the bins of the delivered grid, and for a merged or
otherwise post-processed reduction the corresponding widths are not always
recoverable. `R_res` is therefore returned as a callable so the whole family
can be inspected.

### What `rho` measures

`rho` is defined at the delivered grid and is readable straight off the data:

```
rho = <B (I')^2> / <(h_in^2/12) (I')^2>
```

`rho = 5` means the instrument contributes five times as much as the binning
to the expected squared variation of `I` for the data in hand. It is
dimensionless and invariant under a change of the units of `Q`. Because it
carries `h_in^2` in its denominator it is a property of that grid, not a
proxy for `g`: refining the grid at fixed measurement raises `rho` as
`h_in^-2` while leaving `g` unchanged. The exact relation
`g = (rho/3)(h_in/h*_L)^2` has the two factors of `h_in` cancel identically,
so it is a consistency check rather than a route from `rho` to `g`.

### Normalisation of R_MSE

`R_MSE` is a ratio, so reported uncertainties are used directly and no
assumption about counting statistics is needed. The optimal widths returned
by `optimal_bin_widths` are absolute and internally use the variance
*density* `sigma^2 = err^2 * h_in`, which follows from
`Var(bin of width h) = sigma^2(Q)/h`.

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

The first example also prints `R_res(g)` over a range of `g`, which is the
intended way to read the resolution result.

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
