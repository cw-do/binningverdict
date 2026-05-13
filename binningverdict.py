#!/usr/bin/env python
"""
Standalone command-line runner for binningverdict.

Usage
-----
    python binningverdict.py sample_Iq.dat
    python binningverdict.py *_Iq.dat

Input files are plain-text I(q) tables with at least three columns:
Q, intensity, intensity_error. A fourth Q-error column is allowed and ignored.
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
    "R_Fisher_finite",
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
R_Fisher_finiteN = _core.R_Fisher_finiteN
derivative = _core.derivative

__version__ = "1.0.0"
__all__ = [
    "analyze_binning",
    "R_MSE",
    "R_Fisher_finiteN",
    "derivative",
]


def _numeric_columns(line: str) -> list[float] | None:
    """Return numeric columns from a data line, or None for a header/comment."""
    line = line.split("#", 1)[0].strip()
    if not line:
        return None

    line = line.replace(",", " ")
    parts = line.split()
    if len(parts) < 3:
        return None

    try:
        return [float(part) for part in parts]
    except ValueError:
        return None


def load_iq_file(filename: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Q, I, and intensity error from a text I(q) data file.

    The file may contain one or more header lines. At least three numeric
    columns are required; a fourth q-error column is accepted but ignored.
    """
    path = Path(filename)
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            cols = _numeric_columns(line)
            if cols is not None and len(cols) >= 3:
                rows.append(cols[:3])

    if not rows:
        raise ValueError("no rows with at least three numeric columns were found")

    data = np.asarray(rows, dtype=float)
    Q, I, err = data[:, 0], data[:, 1], data[:, 2]

    valid = np.isfinite(Q) & np.isfinite(I) & np.isfinite(err)
    valid &= (Q > 0) & (I > 0) & (err > 0)
    Q, I, err = Q[valid], I[valid], err[valid]

    if len(Q) < 4:
        raise ValueError("fewer than four valid positive data points remain")

    order = np.argsort(Q)
    Q, I, err = Q[order], I[order], err[order]

    unique_Q, unique_idx = np.unique(Q, return_index=True)
    Q, I, err = unique_Q, I[unique_idx], err[unique_idx]
    if len(Q) < 4:
        raise ValueError("fewer than four unique Q values remain")

    return Q, I, err


def expand_inputs(patterns: Iterable[str]) -> list[Path]:
    """Expand shell-style globs while preserving explicitly supplied files."""
    files: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            files.extend(Path(match) for match in matches)
        else:
            files.append(Path(pattern))

    seen = set()
    unique_files = []
    for path in files:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            seen.add(key)
            unique_files.append(path)
    return unique_files


def analyze_file(path: Path) -> dict[str, object]:
    """Analyze one file and return report-ready values."""
    try:
        Q, I, err = load_iq_file(path)
        result = analyze_binning(Q, I, err=err)
        return {
            "file": str(path),
            "verdict": result["verdict"],
            "R_MSE": result["R_MSE"],
            "R_MSE_subsampled": result["R_MSE_subsampled"],
            "converged": result["converged"],
            "R_Fisher_finite": result["R_Fisher_finite"],
            "rationale": result["rationale"],
        }
    except Exception as exc:
        return {
            "file": str(path),
            "verdict": "ERROR",
            "R_MSE": None,
            "R_MSE_subsampled": None,
            "converged": False,
            "R_Fisher_finite": None,
            "rationale": str(exc),
        }


def format_value(value: object) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")


def print_single_result(row: dict[str, object]) -> None:
    print(f"file              = {row['file']}")
    print(f"verdict           = {row['verdict']}")
    print(f"R_MSE             = {format_value(row['R_MSE'])}")
    print(f"R_MSE_subsampled  = {format_value(row['R_MSE_subsampled'])}")
    print(f"converged         = {format_value(row['converged'])}")
    print(f"R_Fisher_finite   = {format_value(row['R_Fisher_finite'])}")
    print(f"rationale         = {row['rationale']}")


def write_report(rows: list[dict[str, object]], output: str | Path) -> None:
    with Path(output).open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\t".join(REPORT_FIELDS) + "\n")
        for row in rows:
            handle.write("\t".join(format_value(row[field]) for field in REPORT_FIELDS))
            handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run binningverdict on one or more I(q) ASCII data files."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Input .dat/.txt files or glob patterns such as *_Iq.dat.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=REPORT_FILE,
        help=f"Batch report path (default: {REPORT_FILE}).",
    )
    args = parser.parse_args(argv)

    files = expand_inputs(args.files)
    if not files:
        parser.error("no input files matched")

    rows = [analyze_file(path) for path in files]

    if len(rows) == 1:
        print_single_result(rows[0])
    else:
        write_report(rows, args.output)
        print(f"Wrote {args.output} with {len(rows)} rows.")

    return 1 if any(row["verdict"] == "ERROR" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
