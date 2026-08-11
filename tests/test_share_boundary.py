import unittest

from tools.check_share_boundary import classify_paths


class ShareBoundaryTests(unittest.TestCase):
    def test_rejects_local_documents_and_private_tree(self):
        paths = [
            "AGENTS.md",
            "notes/HANDOFF.md",
            "Docs/Claude.MD",
            "Non-Sharable/archive/session.txt",
            r"Non_Sharable\data\capture.ktf",
        ]
        self.assertEqual({path for path, _ in classify_paths(paths)}, set(paths))

    def test_allows_public_product_paths(self):
        paths = [
            "README.md",
            "main.py",
            "tests/test_mosaic_ui.py",
            "docs/agent-workflow.png",
        ]
        self.assertEqual(classify_paths(paths), [])


if __name__ == "__main__":
    unittest.main()
