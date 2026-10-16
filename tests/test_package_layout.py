"""Regression checks for the installable localrag package layout."""

from pathlib import Path
import subprocess
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class PackageLayoutTests(unittest.TestCase):
    def test_core_and_ui_are_inside_localrag_package(self) -> None:
        expected = (
            REPO_ROOT / "localrag" / "__init__.py",
            REPO_ROOT / "localrag" / "core" / "application.py",
            REPO_ROOT / "localrag" / "ui" / "main.py",
        )
        for path in expected:
            with self.subTest(path=path):
                self.assertTrue(path.is_file(), f"Expected package file missing: {path}")

    def test_package_imports_without_optional_site_packages(self) -> None:
        code = (
            "import localrag; "
            "from localrag.core import network; "
            "from pathlib import Path; "
            "print(Path(localrag.__file__).resolve())"
        )
        result = subprocess.run(
            [sys.executable, "-B", "-S", "-c", code],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"Package import failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        actual_path = Path(result.stdout.strip()).resolve()
        expected_path = (REPO_ROOT / "localrag" / "__init__.py").resolve()
        self.assertEqual(actual_path, expected_path)


if __name__ == "__main__":
    unittest.main()
