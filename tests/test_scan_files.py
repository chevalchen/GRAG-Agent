import os
import tempfile
import unittest

from src.app.offline_ingestion.tools.scan_files import scan_recipe_files


class ScanFilesTests(unittest.TestCase):
    def test_scan_prefers_dishes_dir_and_filters_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            recipe_dir = d
            os.makedirs(os.path.join(recipe_dir, "dishes", "ok"), exist_ok=True)
            os.makedirs(os.path.join(recipe_dir, "dishes", ".github"), exist_ok=True)
            os.makedirs(os.path.join(recipe_dir, "other"), exist_ok=True)

            with open(os.path.join(recipe_dir, "dishes", "ok", "a.md"), "w", encoding="utf-8") as f:
                f.write("# A\n")
            with open(os.path.join(recipe_dir, "dishes", ".github", "b.md"), "w", encoding="utf-8") as f:
                f.write("# B\n")
            with open(os.path.join(recipe_dir, "other", "c.md"), "w", encoding="utf-8") as f:
                f.write("# C\n")

            files = scan_recipe_files(recipe_dir, excluded_directories=[".github"])
            rels = [rel for _, rel in files]
            self.assertEqual(rels, [os.path.join("dishes", "ok", "a.md")])

    def test_scan_falls_back_to_root_when_no_dishes(self):
        with tempfile.TemporaryDirectory() as d:
            recipe_dir = d
            os.makedirs(os.path.join(recipe_dir, "x"), exist_ok=True)
            with open(os.path.join(recipe_dir, "x", "a.md"), "w", encoding="utf-8") as f:
                f.write("# A\n")
            files = scan_recipe_files(recipe_dir, excluded_directories=[])
            self.assertEqual(len(files), 1)
            self.assertTrue(files[0][1].endswith(os.path.join("x", "a.md")))

