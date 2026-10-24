#!/usr/bin/env python3
"""Tests for scripts/license_check.py: the repository passes, and each place
the licence is stated, broken on its own, is reported."""
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import license_check as lc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FILES = ["Cargo.toml", "LICENSE-Proprietary", "README.md", "README_JP.md", "deny.toml"]

def copy_tree() -> Path:
    d = Path(tempfile.mkdtemp(prefix="license-check-"))
    for f in FILES:
        shutil.copy(ROOT / f, d / f)
    return d


def problems(root: Path) -> list:
    *p, _ = lc.check(root)
    return p


def edit(path: Path, old: str, new: str):
    text = path.read_text(encoding="utf-8")
    assert old in text, f"{old!r} not in {path.name}"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


class LicenseCheck(unittest.TestCase):
    def test_the_repository_agrees(self):
        *p, compared = lc.check(ROOT)
        self.assertEqual(p, [])
        self.assertGreaterEqual(compared, 20)

    def test_a_copy_agrees(self):
        self.assertEqual(problems(copy_tree()), [])

    def test_an_open_licence_in_the_manifest(self):
        d = copy_tree()
        edit(d / "Cargo.toml", 'license = "LicenseRef-Proprietary"', 'license = "Apache-2.0"')
        self.assertTrue(any("Cargo.toml: license" in x for x in problems(d)))

    def test_a_manifest_without_publish_false(self):
        d = copy_tree()
        edit(d / "Cargo.toml", "publish = false\n", "")
        self.assertIn("Cargo.toml: publish = false is missing", problems(d))

    def test_the_include_list_without_the_licence_text(self):
        d = copy_tree()
        edit(d / "Cargo.toml", '"/LICENSE-Proprietary"', '"/LICENSE-APACHE"')
        p = problems(d)
        self.assertIn("Cargo.toml: include does not list /LICENSE-Proprietary", p)
        self.assertIn("Cargo.toml: include still lists /LICENSE-APACHE", p)

    def test_a_missing_licence_text(self):
        d = copy_tree()
        (d / "LICENSE-Proprietary").unlink()
        self.assertIn("LICENSE-Proprietary: missing", problems(d))

    def test_leftover_previous_licence_files(self):
        for old in ("LICENSE-APACHE", "LICENSE-MIT", "NOTICE"):
            d = copy_tree()
            (d / old).write_text("old\n", encoding="utf-8")
            self.assertIn(f"{old}: still present", problems(d))

    def test_a_readme_linking_a_previous_licence_file(self):
        d = copy_tree()
        edit(d / "README_JP.md", "[LICENSE-Proprietary](LICENSE-Proprietary)",
             "[LICENSE-Proprietary](LICENSE-Proprietary) または [Apache](LICENSE-APACHE)")
        self.assertIn("README_JP.md: links LICENSE-APACHE", problems(d))

    def test_a_readme_licence_section_without_the_licence_text(self):
        d = copy_tree()
        edit(d / "README.md", "[LICENSE-Proprietary](LICENSE-Proprietary)", "LICENSE-Proprietary")
        self.assertIn("README.md: the licence section does not link LICENSE-Proprietary", problems(d))

    def test_a_stale_licence_line_outside_the_section(self):
        # the line under the introduction, before the contents
        d = copy_tree()
        edit(d / "README.md", "License: LicenseRef-Proprietary", "License: Apache-2.0")
        self.assertTrue(any("README.md:" in x and "licence line" in x for x in problems(d)), problems(d))

    def test_a_stale_licence_line_in_the_japanese_readme(self):
        d = copy_tree()
        edit(d / "README_JP.md", "License: LicenseRef-Proprietary", "License: MIT OR Apache-2.0")
        self.assertTrue(any("README_JP.md:" in x and "licence line" in x for x in problems(d)), problems(d))

    def test_an_open_licence_badge(self):
        d = copy_tree()
        edit(d / "README.md", "License: LicenseRef-Proprietary",
             "[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE-Proprietary)\n\nLicense: LicenseRef-Proprietary")
        self.assertTrue(any("licence badge" in x for x in problems(d)), problems(d))

    def test_a_deny_comment_calling_the_crate_open(self):
        for words in ("MIT OR Apache-2.0", "Apache-2.0", "AGPL-3.0"):
            d = copy_tree()
            edit(d / "deny.toml", "[licenses]", f"[licenses]\n# this crate is {words}")
            self.assertTrue(any("describes this crate as an open licence" in x for x in problems(d)), words)

    def test_an_empty_tree_compares_but_fails(self):
        d = Path(tempfile.mkdtemp(prefix="license-check-empty-"))
        p = problems(d)
        self.assertIn("Cargo.toml: missing", p)
        self.assertIn("LICENSE-Proprietary: missing", p)


if __name__ == "__main__":
    unittest.main()
