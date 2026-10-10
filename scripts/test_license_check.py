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
FILES = ["Cargo.toml", "LICENSE-APACHE", "NOTICE", "README.md", "README_JP.md", "deny.toml"]


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
        self.assertGreaterEqual(compared, 16)

    def test_a_copy_agrees(self):
        self.assertEqual(problems(copy_tree()), [])

    def test_the_old_dual_licence_in_the_manifest(self):
        d = copy_tree()
        edit(d / "Cargo.toml", 'license = "Apache-2.0"', 'license = "MIT OR Apache-2.0"')
        self.assertTrue(any("Cargo.toml: license" in x for x in problems(d)))

    def test_a_leftover_mit_licence_file(self):
        d = copy_tree()
        (d / "LICENSE-MIT").write_text("MIT\n", encoding="utf-8")
        self.assertIn("LICENSE-MIT: still present", problems(d))

    def test_a_missing_notice(self):
        d = copy_tree()
        (d / "NOTICE").unlink()
        self.assertIn("NOTICE: missing", problems(d))

    def test_a_notice_without_the_attribution(self):
        d = copy_tree()
        edit(d / "NOTICE", "This product includes ALICE-Analytics", "This product contains ALICE-Analytics")
        self.assertIn("NOTICE: the attribution sentence is missing", problems(d))

    def test_a_readme_linking_the_mit_licence(self):
        d = copy_tree()
        edit(d / "README_JP.md", "(LICENSE-APACHE) (0.4.0 から)", "(LICENSE-APACHE) または [MIT](LICENSE-MIT)")
        self.assertIn("README_JP.md: links LICENSE-MIT", problems(d))

    def test_a_readme_licence_section_without_notice(self):
        d = copy_tree()
        text = (d / "README.md").read_text(encoding="utf-8").replace("[NOTICE](NOTICE)", "NOTICE")
        (d / "README.md").write_text(text, encoding="utf-8")
        self.assertIn("README.md: the licence section does not link NOTICE", problems(d))

    def test_a_stale_licence_line_outside_the_section(self):
        # the line under the introduction, before the contents, as it read before 0.4.0
        d = copy_tree()
        edit(d / "README.md", "License: Apache-2.0 (0.3.0 and earlier: MIT OR Apache-2.0)",
             "License: MIT OR Apache-2.0")
        self.assertTrue(any("README.md:" in x and "licence line" in x for x in problems(d)), problems(d))

    def test_a_stale_licence_line_in_the_japanese_readme(self):
        d = copy_tree()
        edit(d / "README_JP.md", "License: Apache-2.0 (0.3.0 以前は MIT OR Apache-2.0)",
             "License: MIT OR Apache-2.0")
        self.assertTrue(any("README_JP.md:" in x and "licence line" in x for x in problems(d)), problems(d))

    def test_a_dual_licence_badge(self):
        d = copy_tree()
        edit(d / "README.md", "License: Apache-2.0",
             "[![license](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-APACHE)\n\nLicense: Apache-2.0")
        self.assertTrue(any("licence badge" in x for x in problems(d)), problems(d))

    def test_a_deny_comment_calling_the_crate_mit(self):
        d = copy_tree()
        edit(d / "deny.toml", "[licenses]", "[licenses]\n# this crate is MIT OR Apache-2.0")
        self.assertTrue(any("describes this crate as MIT" in x for x in problems(d)))

    def test_an_empty_tree_compares_but_fails(self):
        d = Path(tempfile.mkdtemp(prefix="license-check-empty-"))
        p = problems(d)
        self.assertIn("Cargo.toml: missing", p)
        self.assertIn("NOTICE: missing", p)


if __name__ == "__main__":
    unittest.main()
