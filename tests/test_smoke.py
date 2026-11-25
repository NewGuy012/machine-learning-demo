"""Minimal smoke and integrity tests (stdlib unittest, zero extra deps)."""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
import tomllib
import unittest
import warnings
from pathlib import Path

from scripts import build_site

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.py"))
IMPORT_TO_PKG = {"sklearn": "scikit-learn"}
SCRIPT_META_RE = re.compile(r"(?ms)^# /// script\n(.*?)^# ///$")


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _third_party_imports(source: str) -> set[str]:
    tree = ast.parse(source)
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return {IMPORT_TO_PKG.get(r, r) for r in roots if r not in sys.stdlib_module_names}


class SmokeTests(unittest.TestCase):
    def test_notebooks_execute_cleanly(self) -> None:
        self.assertTrue(NOTEBOOKS)
        for nb in NOTEBOOKS:
            with self.subTest(notebook=nb.name):
                mod = _load_module(nb)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    outputs, _ = mod.app.run()
                self.assertEqual([str(w.message) for w in caught], [])
                self.assertIn("Reference", outputs[-1].text)
                self.assertIn("marimo-table", outputs[-1].text)

    def test_pep723_dependencies_match_imports(self) -> None:
        for nb in NOTEBOOKS:
            with self.subTest(notebook=nb.name):
                src = nb.read_text(encoding="utf-8")
                match = SCRIPT_META_RE.search(src)
                self.assertIsNotNone(match)
                body = "\n".join(line.removeprefix("# ") for line in match.group(1).splitlines())
                declared = set(tomllib.loads(body)["dependencies"])
                self.assertEqual(declared, _third_party_imports(src))

    def test_build_site_catalog_and_escaping(self) -> None:
        build_site.validate_catalog(build_site.NOTEBOOK_CATALOG)
        card = build_site.render_card(
            {
                "slug": "01_supervised_learning",
                "title": '<script>alert("x")</script>',
                "subtitle": "A & B",
                "description": 'Quote "test"',
                "tags": ["<b>tag</b>"],
            }
        )
        self.assertNotIn("<script>", card)
        self.assertIn("&lt;script&gt;", card)
        self.assertIn("&lt;b&gt;tag&lt;/b&gt;", card)
        with self.assertRaises(ValueError):
            build_site.validate_slug("../evil")


if __name__ == "__main__":
    unittest.main()
