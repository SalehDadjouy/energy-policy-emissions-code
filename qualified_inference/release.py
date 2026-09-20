"""Integrity, environment, and output controls for the replication workflow."""
from __future__ import annotations
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")

def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def load(path):
    return json.loads(Path(path).read_text())

def write_new(path, record):
    with Path(path).open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")

def file_inventory(root):
    root = Path(root)
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*"))
            if p.is_file() and not any(part in (".git", ".venv", "__pycache__") for part in p.relative_to(root).parts)
            and p.name != "PACKAGE_LOCK.json"}

def verify_package(root=ROOT):
    expected = load(root / "PACKAGE_LOCK.json")["files"]
    actual = file_inventory(root)
    if actual != expected:
        changed = sorted(p for p in set(actual) | set(expected) if actual.get(p) != expected.get(p))
        raise RuntimeError("Package integrity mismatch: " + ", ".join(changed))
    return sha(root / "PACKAGE_LOCK.json")

def environment():
    import numpy as np
    import scipy
    pins = {}
    for line in (ROOT / "requirements-reproduction.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            name, version = line.strip().split("==")
            installed = importlib.metadata.version(name)
            if installed != version:
                raise RuntimeError(f"Dependency mismatch: {name} {installed}; required {version}")
            pins[name] = installed
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Qualification requires Python 3.11")
    buffers = []
    for library in (np, scipy):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Install .pyyaml. for better output")
            library.show_config()
        buffers.append(stream.getvalue())
    from matplotlib import font_manager
    font = Path(font_manager.findfont(font_manager.FontProperties(
        family=["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"])))
    return {"python": platform.python_version(), "implementation": platform.python_implementation(),
            "compiler": platform.python_compiler(), "system": platform.system(),
            "release": platform.release(), "machine": platform.machine(), "packages": pins,
            "numerical_libraries": buffers,
            "thread_environment": {k: os.environ.get(k) for k in THREADS},
            "figure_font": {"name": font.name, "sha256": sha(font)},
            "python_hash_seed": "0", "source_date_epoch": "0"}

def require_environment(expected, actual):
    if expected != actual:
        raise RuntimeError("Environment differs from the frozen qualification environment")

def output_destination(path, root=ROOT):
    path, root = Path(path).resolve(), Path(root).resolve()
    if path == root or root in path.parents or path in root.parents:
        raise ValueError("Output must be outside and must not contain the package")
    return path

def commands(output, root=ROOT):
    routes = load(root / "WORKFLOW.json")["routes"]
    return [(r, [sys.executable, "-B", str(root / "qualified_inference/runner.py"),
                 str(output), r["script"],
                 *[a.replace("{output}", str(output)) for a in r["args"]]]) for r in routes]

def check_frame(path, contract):
    import pandas as pd
    frame = pd.read_csv(path)
    if len(frame) != contract["rows"]:
        raise RuntimeError(f"{path.name}: row count {len(frame)} != {contract['rows']}")
    missing = set(contract["columns"]) - set(frame.columns)
    if missing:
        raise RuntimeError(f"{path.name}: missing columns {sorted(missing)}")
    keys = contract.get("keys", [])
    if keys and (frame[keys].isna().any().any() or frame.duplicated(keys).any()):
        raise RuntimeError(f"{path.name}: missing or duplicate keys")
    for name in ("covered", "contains_target", "excluded"):
        if name in frame and not frame[name].isin([0, 1]).all():
            raise RuntimeError(f"{path.name}: invalid {name} indicator")
    if "groups" in contract:
        columns = contract["group_keys"]
        observed = frame.groupby(columns, dropna=False).size().reset_index(name="n")
        expected = pd.DataFrame(contract["groups"])[[*columns, "n"]]
        observed = observed.sort_values(columns).reset_index(drop=True)
        expected = expected.sort_values(columns).reset_index(drop=True)
        if not observed.equals(expected.astype(observed.dtypes.to_dict())):
            raise RuntimeError(f"{path.name}: replication-cell inventory differs")

def check_outputs(output, route, root=ROOT):
    result = {}
    for relative in route["outputs"]:
        path = output / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError("Missing or empty output: " + relative)
        if relative in route.get("csv_contracts", {}):
            check_frame(path, route["csv_contracts"][relative])
        result[relative] = sha(path)
    if route["name"] == "principal":
        if any(load(output / "simulation/manifest.json")["skipped"].values()):
            raise RuntimeError("Principal simulation skipped replications")
    if route["name"] == "shock_orientation":
        import pandas as pd
        cells = pd.read_csv(output / "sensitivity/shock_orientation/cell_status.csv")
        raw = pd.read_csv(output / "sensitivity/shock_orientation/raw.csv")
        key = ["config", "T0", "T1", "rep", "branch", "arm", "severity"]
        eligible = cells.loc[cells.excluded.eq(0), key].sort_values(key).reset_index(drop=True)
        counts = raw.groupby(key).size().reset_index(name="n")
        if not counts.n.eq(6).all() or not counts[key].sort_values(key).reset_index(drop=True).equals(eligible):
            raise RuntimeError("Orientation rows do not match the specified exclusion decisions")
    return result

def archived_comparison(output, route, root=ROOT):
    import validate_results
    comparisons = []
    for relative, details in route.get("references", {}).items():
        try:
            validate_results.compare_reference(output / relative, root / details["path"], keys=details["keys"])
            comparisons.append({"file": relative, "status": "pass"})
        except AssertionError as exc:
            comparisons.append({"file": relative, "status": "mismatch", "detail": str(exc)})
    if route["name"] == "target_anchor":
        checks = load(output / "sensitivity/target_anchor/invariance_checks.json")
        for method, detail in checks["legacy_reference"].items():
            if detail.get("status") == "mismatch":
                comparisons.append({"file": "target_anchor_baseline:" + method, **detail})
    return {"status": "mismatch" if any(c["status"] != "pass" for c in comparisons) else "pass",
            "comparisons": comparisons}

def stage_inputs(route, output, root=ROOT):
    if route["name"] == "finite_window_weight_learning":
        from .workflow import stage_objective
        stage_objective(output / "sensitivity/finite_window_weight_learning")

def execute(output, expected_environment, *, resume=False, root=ROOT,
            launcher=subprocess.run, integrity=check_outputs, archive=archived_comparison):
    identity = verify_package(root)
    env = environment()
    require_environment(expected_environment, env)
    output = output_destination(output, root)
    context = {"package_lock_sha256": identity, "environment": env}
    if resume:
        if load(output / "run_context.json") != context:
            raise RuntimeError("Resume context differs")
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_new(output / "run_context.json", context)
        (output / "control").mkdir()
        (output / "logs").mkdir()
        (output / "cache").mkdir()
        (output / "cache/tmp").mkdir()
    child_env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0",
                     SOURCE_DATE_EPOCH="0", MPLCONFIGDIR=str(output / "cache/matplotlib"),
                     TMPDIR=str(output / "cache/tmp"))
    reports = []
    for route, command in commands(output, root):
        verify_package(root)
        name = route["name"]
        receipt = output / "control" / (name + ".json")
        if receipt.exists():
            previous = load(receipt)
            if integrity(output, route, root) != previous["output_hashes"]:
                raise RuntimeError("Completed outputs changed: " + name)
            reports.append(previous)
            continue
        started = output / "control" / (name + ".started.json")
        if started.exists():
            raise RuntimeError("Incomplete stage preserved; investigate before resuming: " + name)
        write_new(started, {"name": name, "command": command})
        stage_inputs(route, output, root)
        log = output / "logs" / (name + ".txt")
        with log.open("x") as stream:
            process = launcher(command, cwd=root, env=child_env, stdout=stream, stderr=subprocess.STDOUT)
        verify_package(root)
        if process.returncode:
            raise RuntimeError(f"Execution failed in {name}; log and partial outputs preserved")
        hashes = integrity(output, route, root)
        try:
            archived = archive(output, route, root)
        except Exception as exc:
            archived = {"status": "error", "detail": f"{type(exc).__name__}: {exc}"}
        record = {"name": name, "execution_integrity": "pass", "output_hashes": hashes,
                  "archive_compatibility": archived}
        write_new(receipt, record)
        reports.append(record)
    verify_package(root)
    status = {"execution_integrity": "pass",
              "archive_compatibility": "mismatch" if any(r["archive_compatibility"]["status"] != "pass"
                                                       for r in reports) else "pass",
              "stages": reports}
    final = output / "run_result.json"
    if final.exists():
        if load(final) != status:
            raise RuntimeError("Existing completion record differs")
    else:
        write_new(final, status)
    return status

def reproduction_permitted(primary):
    return primary.get("execution_integrity") == "pass"

def release_acceptable(comparison):
    return (comparison.get("same_environment_reproduction") == "pass"
            and set(comparison.get("archive_compatibility", {})) == {"primary", "clean_clone"}
            and all(s == "pass" for s in comparison["archive_compatibility"].values()))

def compare_runs(first, second, root=ROOT):
    first, second = Path(first), Path(second)
    left, right = load(first / "run_context.json"), load(second / "run_context.json")
    if left != right or left["package_lock_sha256"] != verify_package(root):
        raise RuntimeError("Runs do not share the frozen package and environment")
    for base in (first, second):
        status = load(base / "run_result.json")
        if not reproduction_permitted(status):
            raise RuntimeError("Execution integrity has not passed")
        for route, _ in commands(base, root):
            receipt = load(base / "control" / (route["name"] + ".json"))
            if check_outputs(base, route, root) != receipt["output_hashes"]:
                raise RuntimeError("Output hash mismatch before comparison")
    differences = []
    checked = []
    for route in load(root / "WORKFLOW.json")["routes"]:
        for name in route["reproduction_files"]:
            checked.append(name)
            if sha(first / name) != sha(second / name):
                differences.append(name)
    return {"same_environment_reproduction": "pass" if not differences else "mismatch",
            "comparison_rule": "byte-identical declared scientific files",
            "files_checked": checked, "different_files": differences,
            "archive_compatibility": {label: load(base / "run_result.json")["archive_compatibility"]
                                      for label, base in (("primary", first), ("clean_clone", second))}}
