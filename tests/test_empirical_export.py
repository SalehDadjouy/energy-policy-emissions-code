"""Check empirical export precision without fitting or simulating a model."""
import ast
import hashlib
from io import StringIO
from pathlib import Path
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIELDS = {"tsls_se_arima", "siv_se_arima", "tsls_p_arima", "siv_p_arima"}
BASELINE = "946e09e89b73f258da1e369d625e34d359323c7d1fd8124aca8512c9047d0a1d"

def expressions():
    tree = ast.parse((ROOT / "empirical.py").read_text())
    fields = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for i, (key, value) in enumerate(zip(node.keys, node.values)):
                if isinstance(key, ast.Constant) and key.value in FIELDS:
                    if key.value in fields:
                        raise AssertionError("Duplicate export field")
                    fields[key.value] = value
                    if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name) or value.func.id != "float" or len(value.args) != 1:
                        raise AssertionError("Export must retain computed precision")
                    node.values[i] = value.args[0]
    return tree, fields

class EmpiricalExportTests(unittest.TestCase):
    def test_only_four_export_wrappers_changed(self):
        tree, fields = expressions()
        self.assertEqual(set(fields), FIELDS)
        digest = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
        self.assertEqual(digest, BASELINE)

    def test_computed_values_survive_export_expressions(self):
        _, fields = expressions()
        namespace = {"se_arima_t": 3.451046738048249, "se_arima_r": 4.001902346426548,
                     "est_t": {"tau": -0.7}, "est_r": {"tau": -2.2}, "df_arima": 8,
                     "p_two_sided_t": lambda value, df: 0.84527065958536 if value > -0.3 else 0.5948912119586363}
        values = {key: eval(compile(ast.Expression(body=value), "export-expression", "eval"), namespace)
                  for key, value in fields.items()}
        self.assertEqual(values["tsls_se_arima"], namespace["se_arima_t"])
        self.assertEqual(values["siv_se_arima"], namespace["se_arima_r"])
        self.assertEqual(values["tsls_p_arima"], 0.84527065958536)
        self.assertEqual(values["siv_p_arima"], 0.5948912119586363)
        frame = pd.DataFrame([values])
        restored = pd.read_csv(StringIO(frame.to_csv(index=False)), float_precision="round_trip")
        pd.testing.assert_frame_equal(frame, restored, check_exact=True)

    def test_simulation_code_does_not_call_empirical_export(self):
        callers = []
        for path in ROOT.rglob("*.py"):
            if path.relative_to(ROOT).parts[0] in ("tests", ".git", ".venv"):
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Call):
                    name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else None
                    if name == "run_sample":
                        callers.append(str(path.relative_to(ROOT)))
        self.assertEqual(callers, ["empirical.py"])
