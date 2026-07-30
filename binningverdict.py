#!/usr/bin/env python
"""
Standalone command-line runner for binningverdict.

Usage
-----
    python binningverdict.py sample_Iq.dat
    python binningverdict.py *_Iq.dat
    python binningverdict.py --geometry 1D scan.txt
    python binningverdict.py --no-resolution sample_Iq.dat

Input files are plain-text I(q) tables with at least three columns:

    Q   intensity   intensity_error   [Q_error]

The optional fourth column is the instrumental Q-resolution.  When present it
is used to report rho, which compares the resolution and binning
contributions at the delivered grid, together with the share of dQ^2 that is
collimation rather than binning and the resolution-aware ratio evaluated at
g = 1 as a representative point.  Pass --no-resolution to ignore the column.
Header and comment lines are skipped automatically.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
from pathlib import Path
from typing import Iterable

import numpy as np

REPORT_FILE = "binningverdict_report.txt"
REPORT_FIELDS = (
    "file",
    "verdict",
    "R_MSE",
    "R_MSE_subsampled",
    "converged",
    "rho",
    "B_fraction",
    "R_res_at_g1",
    "geometry",
    "Q_trim",
    "rationale",
)


def _load_core_module():
    """Load the package core without being shadowed by this script name."""
    core_path = Path(__file__).resolve().parent / "binningverdict" / "core.py"
    spec = importlib.util.spec_from_file_location("_binningverdict_core", core_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load binningverdict core from {core_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_core = _load_core_module()
analyze_binning = _core.analyze_binning
R_MSE = _core.R_MSE
resolution_ratio = _core.resolution_ratio
optimal_bin_widths = _core.optimal_bin_widths
R_Fisher_finiteN = _core.R_Fisher_finiteN
background_diagnostic = _core.background_diagnostic
derivative = _core.derivative

__version__ = "1.1.0"
__all__ = [
    "analyze_binning",
    "R_MSE",
    "resolution_ratio",
    "optimal_bin_widths",
    "R_Fisher_finiteN",
    "background_diagnostic",
    "derivative",
]


def _numeric_columns(line: str):
    """Return numeric columns from a data line, or None for a header."""
    line = line.split("#", 1)[0].strip()
    if not line:
        return None
    parts = line.replace(",", " ").split()
    if len(parts) < 3:
        return None
    try:
        return [float(part) for part in parts]
    except ValueError:
        return None


def load_iq_file(filename):
    """
    Load Q, I, intensity error and, when present, the Q-resolution.

    Returns
    -------
    (Q, I, err, dQ) with dQ set to None if the file has only three columns.
    """
    path = Path(filename)
    rows, widths = [], []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            cols = _numeric_columns(line)
            if cols is not None and len(cols) >= 3:
                rows.append(cols[:4] if len(cols) >= 4 else cols[:3])
                widths.append(len(rows[-1]))

    if not rows:
        raise ValueError("no rows with at least three numeric columns were found")

    ncol = 4 if min(widths) >= 4 else 3
    data = np.asarray([r[:ncol] for r in rows], dtype=float)
    Q, I, err = data[:, 0], data[:, 1], data[:, 2]
    dQ = data[:, 3] if ncol == 4 else None

    valid = np.isfinite(Q) & np.isfinite(I) & np.isfinite(err)
    valid &= (Q > 0) & (I > 0) & (err > 0)
    if dQ is not None:
        valid &= np.isfinite(dQ) & (dQ > 0)
    Q, I, err = Q[valid], I[valid], err[valid]
    if dQ is not None:
        dQ = dQ[valid]

    if len(Q) < 4:
        raise ValueError("fewer than four valid positive data points remain")

    order = np.argsort(Q)
    Q, I, err = Q[order], I[order], err[order]
    if dQ is not None:
        dQ = dQ[order]

    Q_unique, idx = np.unique(Q, return_index=True)
    Q, I, err = Q_unique, I[idx], err[idx]
    if dQ is not None:
        dQ = dQ[idx]
    if len(Q) < 4:
        raise ValueError("fewer than four unique Q values remain")

    return Q, I, err, dQ


def expand_inputs(patterns: Iterable[str]):
    """Expand shell-style globs while preserving explicitly supplied files."""
    files = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        files.extend(Path(m) for m in matches) if matches else files.append(Path(pattern))
    seen, unique = set(), []
    for path in files:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def analyze_file(path: Path, geometry="2D", use_resolution=True):
    """Analyze one file and return report-ready values."""
    try:
        Q, I, err, dQ = load_iq_file(path)
        result = analyze_binning(Q, I, err=err,
                                 dQ=dQ if use_resolution else None,
                                 geometry=geometry)
        bg = result["background"]
        return {
            "file": str(path),
            "verdict": result["verdict"],
            "R_MSE": result["R_MSE"],
            "R_MSE_subsampled": result["R_MSE_subsampled"],
            "converged": result["converged"],
            "rho": result["rho"],
            "B_fraction": result["B_fraction"],
            "R_res_at_g1": (None if result["rho"] is None
                            else result["R_res"](1.0)),
            "geometry": result["geometry"],
            "Q_trim": None if bg is None else bg["Q_trim"],
            "rationale": result["rationale"],
        }
    except Exception as exc:
        return {k: (str(path) if k == "file"
                    else "ERROR" if k == "verdict"
                    else False if k == "converged"
                    else str(exc) if k == "rationale"
                    else None)
                for k in REPORT_FIELDS}


def format_value(value):
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")


def print_single_result(row):
    for field in REPORT_FIELDS:
        print(f"{field:<18}= {format_value(row[field])}")


def write_report(rows, output):
    with Path(output).open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\t".join(REPORT_FIELDS) + "\n")
        for row in rows:
            handle.write("\t".join(format_value(row[f]) for f in REPORT_FIELDS))
            handle.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run binningverdict on one or more I(q) ASCII data files.")
    parser.add_argument("files", nargs="+",
                        help="Input .dat/.txt files or globs such as *_Iq.dat.")
    parser.add_argument("-o", "--output", default=REPORT_FILE,
                        help=f"Batch report path (default: {REPORT_FILE}).")
    parser.add_argument("-g", "--geometry", default="2D", choices=["1D", "2D"],
                        help="Detector geometry: 2D annular average (default) "
                             "or 1D stepped scan.")
    parser.add_argument("--no-resolution", action="store_true",
                        help="Ignore the fourth dQ column if present.")
    args = parser.parse_args(argv)

    files = expand_inputs(args.files)
    if not files:
        parser.error("no input files matched")

    rows = [analyze_file(p, args.geometry, not args.no_resolution)
            for p in files]

    if len(rows) == 1:
        print_single_result(rows[0])
    else:
        write_report(rows, args.output)
        print(f"Wrote {args.output} with {len(rows)} rows.")
        print(f"{'file':<44}{'verdict':>9}{'R_MSE':>9}{'rho':>9}")
        for r in rows:
            print(f"{Path(r['file']).name:<44}{r['verdict']:>9}"
                  f"{format_value(r['R_MSE']):>9}{format_value(r['rho']):>9}")

    return 1 if any(r["verdict"] == "ERROR" for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
