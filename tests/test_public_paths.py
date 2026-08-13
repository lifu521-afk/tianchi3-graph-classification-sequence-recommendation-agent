from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from project_paths import DATA_ROOT, task_dir
from agent.agent import load_config

    


class PublicRepositoryTests(unittest.TestCase):
    def test_data_root_is_configurable(self):
        self.assertIsInstance(DATA_ROOT, Path)

    def test_task_aliases_are_stable(self):
        self.assertEqual(task_dir("B_classification").name, "B分类")
        self.assertEqual(task_dir("B_recommendation").name, "B推荐")


    def test_public_agent_registry_has_runnable_scripts(self):
        config = load_config()
        self.assertTrue(config["experiments"])
        root = Path(__file__).resolve().parents[1]
        for item in config["experiments"]:
            self.assertTrue((root / item["script"]).exists())


if __name__ == "__main__":
    unittest.main()
