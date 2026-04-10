from __future__ import annotations

from pathlib import Path
import unittest


class ContractFilesTests(unittest.TestCase):
    def test_openapi_contract_contains_generate_endpoint(self) -> None:
        contract_path = Path("assistant/api/openapi.yaml")
        content = contract_path.read_text(encoding="utf-8")
        self.assertIn("/generate", content)
        self.assertIn("code:", content)

    def test_readme_mentions_api_launch(self) -> None:
        readme = Path("README.md").read_text(encoding="utf-8")
        self.assertIn("python -m assistant.api", readme)
        self.assertIn("assistant.localscript.self_check", readme)


if __name__ == "__main__":
    unittest.main()
