"""Fast, data-free checks for the canonical v3 split contract."""
from __future__ import annotations

import unittest

import pandas as pd

from scripts.cv_runner import fold_manifest, make_fold_split


class V3SplitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = pd.DataFrame([
            {"Drug_ID": f"D{drug}", "Cell_Line_ID": f"C{cell}", "Y": float(drug + cell)}
            for drug in range(12) for cell in range(3)
        ])

    def test_drug_cold_entities_are_disjoint(self) -> None:
        for fold in range(6):
            parts = make_fold_split(self.frame, 6, "drug_cold", seed=42, fold_idx=fold)
            drugs = {name: set(part["Drug_ID"]) for name, part in parts.items()}
            self.assertFalse(drugs["train"] & drugs["valid"])
            self.assertFalse(drugs["train"] & drugs["test"])
            self.assertFalse(drugs["valid"] & drugs["test"])

    def test_manifest_changes_when_fold_data_changes(self) -> None:
        parts = make_fold_split(self.frame, 6, "drug_cold", seed=42, fold_idx=0)
        first = fold_manifest("graphdrp", "drug_cold", 0, parts)
        altered = {name: value.copy() for name, value in parts.items()}
        altered["train"].loc[altered["train"].index[0], "Y"] += 1.0
        second = fold_manifest("graphdrp", "drug_cold", 0, altered)
        self.assertNotEqual(
            first["parts"]["train"]["data_fingerprint"],
            second["parts"]["train"]["data_fingerprint"],
        )


if __name__ == "__main__":
    unittest.main()
