"""Prepare a run directory without executing an analysis."""
from pathlib import Path
import shutil

from .reporting import ROOT, PROVENANCE, digest

def stage_objective(output):
    """Copy the verified bank once; never replace an existing run directory."""
    output = Path(output).resolve()
    bank = ROOT / "data/finite_window_population_objective.npz"
    if digest(bank) != PROVENANCE["objective_bank_sha256"]:
        raise RuntimeError("Frozen objective bank changed")
    if output == ROOT or ROOT in output.parents:
        raise ValueError("Execution outputs must be outside the candidate package")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(bank, output / "population_objective.npz")
    if digest(output / "population_objective.npz") != PROVENANCE["objective_bank_sha256"]:
        raise RuntimeError("Objective bank copy mismatch")
    return ["analysis/finite_window_weight_learning.py", "--stage", "evaluate",
            "--out", str(output)]
