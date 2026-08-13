from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from project_paths import DATA_ROOT, task_dir


class PublicRepositoryTests(unittest.TestCase):
    def test_data_root_is_configurable(self):
        self.assertIsInstance(DATA_ROOT, Path)

    def test_task_aliases_are_stable(self):
        self.assertEqual(task_dir("B_classification").name, "B分类")
        self.assertEqual(task_dir("B_recommendation").name, "B推荐")


if __name__ == "__main__":
    unittest.main()

