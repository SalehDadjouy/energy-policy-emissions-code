#!/usr/bin/env python3
"""Preflight, explicit execution, and independent comparison of the frozen workflow."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main():
    import json
    from qualified_inference import release
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--structural-only", action="store_true")
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--compare", nargs=2, type=Path, metavar=("PRIMARY", "CLEAN_CLONE"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--environment-lock", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.structural_only:
        subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-v"],
                       cwd=ROOT, check=True)
        print("PASS: structural tests only; no numerical reproduction claimed.")
    elif args.execute:
        if args.output is None or args.environment_lock is None:
            parser.error("--execute requires --output and a frozen --environment-lock")
        result = release.execute(args.output, release.load(args.environment_lock), resume=args.resume)
        print(json.dumps(result, indent=2))
    elif args.compare:
        result = release.compare_runs(*args.compare)
        result["release_acceptable"] = release.release_acceptable(result)
        print(json.dumps(result, indent=2))
        if not result["release_acceptable"]:
            raise SystemExit(1)
    else:
        if args.output or args.environment_lock or args.resume:
            parser.error("Execution arguments require --execute")
        print(json.dumps({"mode": "preflight_only", "package_lock_sha256": release.verify_package(),
                          "environment": release.environment(),
                          "workflow": release.load(ROOT / "WORKFLOW.json")}, indent=2))

if __name__ == "__main__":
    main()
