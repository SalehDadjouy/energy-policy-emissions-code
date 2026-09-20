"""Derive target-contrast inventory from source without fitting or simulating."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
from typing import Any

import numpy as np
import pandas as pd
from qualified_inference import release

ROOT = release.ROOT

class TargetContractTests(unittest.TestCase):
    def source_fixture(self, reps=2):
        protocol = release.load(ROOT / "analysis/protocols/target_anchor_sensitivity_v1.json")
        tree = ast.parse((ROOT / "analysis/target_anchor_sensitivity.py").read_text())
        names = {"target_map", "midpoint_role", "build_weight_contrasts"}
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        self.assertEqual(len(functions), len(names))
        namespace = {"Any": Any, "np": np, "pd": pd,
                     "ANCHOR_MODES": tuple(protocol["anchor_modes"]),
                     "BRANCHES": tuple(protocol["paired_execution"]["branches"])}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "target-source-fixture", "exec"), namespace)
        weights = {(mode, target["role"], rep, branch, estimator): np.array([-1., 0., 1.])
                   for mode in protocol["anchor_modes"] for target in protocol["empirical_targets"]
                   for rep in range(reps) for branch in protocol["paired_execution"]["branches"]
                   for estimator in ("TSLS", "SIV")}
        return protocol, namespace["build_weight_contrasts"](protocol, weights, reps=reps)

    def contract(self):
        route = next(r for r in release.load(ROOT / "WORKFLOW.json")["routes"] if r["name"] == "target_anchor")
        return route["csv_contracts"]["sensitivity/target_anchor/weight_target_contrasts.csv"]

    def test_source_uses_two_endpoints_not_three_targets(self):
        protocol, frame = self.source_fixture()
        self.assertEqual(len(frame), 64)
        self.assertEqual(frame.target_role.nunique(), 2)
        self.assertNotIn("restricted_midpoint", set(frame.target_role))
        expected = len(frame) // 2 * protocol["paired_execution"]["reps"]
        self.assertEqual(self.contract()["rows"], expected)

    def test_contract_groups_match_source_fixture(self):
        protocol, frame = self.source_fixture()
        contract = self.contract()
        expected = frame.groupby(contract["group_keys"]).size().reset_index(name="n")
        expected["n"] = protocol["paired_execution"]["reps"]
        actual = pd.DataFrame(contract["groups"])
        keys = contract["group_keys"]
        pd.testing.assert_frame_equal(expected.sort_values(keys).reset_index(drop=True),
                                      actual[expected.columns].sort_values(keys).reset_index(drop=True))

    def test_all_target_contrasts_match_archived_group_inventory(self):
        summary = pd.read_csv(ROOT / "reference_outputs/sensitivity/target_anchor/weight_target_contrast_summary.csv")
        contract = self.contract()
        self.assertEqual(int(summary.n.sum()), contract["rows"])
        keys = contract["group_keys"]
        pd.testing.assert_frame_equal(summary[[*keys, "n"]].sort_values(keys).reset_index(drop=True),
                                      pd.DataFrame(contract["groups"])[[*keys, "n"]].sort_values(keys).reset_index(drop=True))

    def test_old_count_and_missing_endpoint_rejected(self):
        _, frame = self.source_fixture()
        contract = json.loads(json.dumps(self.contract()))
        contract["rows"] = len(frame)
        for group in contract["groups"]:
            group["n"] = 2
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "contrasts.csv"
            frame.to_csv(path, index=False)
            release.check_frame(path, contract)
            with self.assertRaisesRegex(RuntimeError, "row count"):
                release.check_frame(path, {**contract, "rows": 96})
            altered = frame.copy()
            altered["target_role"] = "restricted_tsls"
            altered.to_csv(path, index=False)
            with self.assertRaises(RuntimeError):
                release.check_frame(path, contract)

