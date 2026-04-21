import unittest

from src.app.offline_ingestion.nodes.normalize_node import make_normalize_node
from src.core.utils.drug_name_normalizer import build_alias_norms, normalize_drug_name


class DrugNameNormalizerTests(unittest.TestCase):
    def test_normalize_collapses_space_variants(self):
        self.assertEqual(normalize_drug_name("乌鸡白凤丸"), normalize_drug_name("乌鸡 白凤丸"))

    def test_normalize_handles_common_punctuation(self):
        self.assertEqual(normalize_drug_name("乌鸡（白凤丸）"), normalize_drug_name("乌鸡(白凤丸)"))

    def test_build_alias_norms_deduplicates_normalized_variants(self):
        alias_norms = build_alias_norms(["乌鸡白凤丸", "乌鸡 白凤丸"])
        self.assertEqual(alias_norms, [normalize_drug_name("乌鸡白凤丸")])

    def test_normalize_node_adds_canonical_and_alias_fields(self):
        node = make_normalize_node()
        state = node(
            {
                "parsed_records": [
                    {
                        "drug_name": "乌鸡 白凤丸",
                        "raw_label": "药品",
                        "ingredients": [],
                        "effects": [],
                        "symptoms": [],
                        "diseases": [],
                        "syndromes": [],
                        "populations": [],
                        "adverse_reactions": [],
                    }
                ]
            }
        )
        record = state["parsed_records"][0]
        self.assertEqual(record["canonical_name"], "乌鸡 白凤丸")
        self.assertEqual(record["normalized_name"], normalize_drug_name("乌鸡白凤丸"))
        self.assertTrue(record["node_id"].startswith("drug_"))
        self.assertIn(normalize_drug_name("乌鸡 白凤丸"), record["alias_norms"])


if __name__ == "__main__":
    unittest.main()
