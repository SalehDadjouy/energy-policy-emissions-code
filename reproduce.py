#!/usr/bin/env python3
"""Run the complete paper reproduction workflow."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(*args: str) -> None:
    command = [sys.executable, *args]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def reset(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def run_sensitivities(base: Path, *, quick: bool) -> None:
    reset(base)
    reps = "2" if quick else "1000"
    run(
        "analysis/state_panel_and_confounding_sensitivity.py",
        "--mode",
        "sample",
        "--reps",
        reps,
        "--seed",
        "20261024",
        "--out",
        str(base / "state_panel"),
    )
    run(
        "analysis/state_panel_and_confounding_sensitivity.py",
        "--mode",
        "grid",
        "--reps",
        reps,
        "--seed",
        "20261024",
        "--out",
        str(base / "confounding_grid"),
    )
    if quick:
        run(
            "analysis/finite_window_weight_learning.py",
            "--stage",
            "all",
            "--population-draws",
            "20",
            "--reps",
            "2",
            "--out",
            str(base / "finite_window_weight_learning"),
        )
        run(
            "analysis/target_anchor_sensitivity.py",
            "--reps",
            "2",
            "--out",
            str(base / "target_anchor"),
        )
        orientation_reps = "2"
    else:
        run(
            "analysis/finite_window_weight_learning.py",
            "--stage",
            "all",
            "--out",
            str(base / "finite_window_weight_learning"),
        )
        run(
            "analysis/target_anchor_sensitivity.py",
            "--execute-full",
            "--out",
            str(base / "target_anchor"),
        )
        orientation_reps = "1000"
    run(
        "analysis/shock_orientation_sensitivity.py",
        "--reps",
        orientation_reps,
        "--seed",
        "42",
        "--out",
        str(base / "shock_orientation"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce every empirical and simulation result reported in the paper.")
    parser.add_argument("--quick", action="store_true", help="Run a structural smoke test with fewer Monte Carlo draws.")
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Skip the prespecified sensitivity analyses and reproduce only the principal empirical and simulation results.",
    )
    args = parser.parse_args()

    if args.quick:
        base = ROOT / "outputs" / "quick"
        run("empirical.py", "--n-sims", "500", "--out", str(base / "empirical"))
        run("run_simulation.py", "--reps", "10", "--out", str(base / "simulation"))
        run("audit_denominator_tails.py", "--out-dir", str(base / "simulation"))
        run("make_exposure_figure.py", "--out", str(base / "figures"))
        if not args.core_only:
            run_sensitivities(base / "sensitivity", quick=True)
        print(f"PASS: quick workflow completed; outputs are in {base}")
        return

    run("empirical.py", "--n-sims", "80000", "--out", str(ROOT / "outputs" / "empirical"))
    run("run_simulation.py", "--reps", "1000", "--seed", "20261024", "--out", str(ROOT / "outputs" / "simulation"))
    run("audit_denominator_tails.py", "--out-dir", str(ROOT / "outputs" / "simulation"))
    run("make_exposure_figure.py", "--out", str(ROOT / "outputs" / "figures"))
    if args.core_only:
        run("validate_results.py")
    else:
        run_sensitivities(ROOT / "outputs" / "sensitivity", quick=False)
        run("validate_results.py", "--sensitivity-outputs", str(ROOT / "outputs" / "sensitivity"))


if __name__ == "__main__":
    main()
