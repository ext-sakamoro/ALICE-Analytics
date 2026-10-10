#!/usr/bin/env python3
"""Tests for scripts/deprecation_pin.py.

Each case builds a small tree in a temporary directory, breaks exactly one
thing, and asserts that the check fails. The first case runs the check
against this repository.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deprecation_pin as dp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

MODULE = """//! module doc

// the module's own items refer to each other; the deprecation is for callers
#![allow(deprecated)]

/// A generator
#[deprecated(
    since = "0.4.0",
    note = "not private"
)]
#[derive(Clone)]
pub struct Gen {
    state: u64,
}

#[deprecated(since = "0.4.0", note = "not private")]
pub const BITS: usize = 64;

impl Gen {
    pub fn new() -> Self {
        Self { state: 1 }
    }
}
"""

TEST = """// `privacy` is deprecated and still pinned here until 0.5.0 removes it
#![allow(deprecated)]
#[test]
fn t() {}
"""


def run(files: dict[str, str]) -> tuple[int, str]:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        for rel, text in files.items():
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = dp.run(root)
        return code, err.getvalue()


def good() -> dict[str, str]:
    return {"src/privacy.rs": MODULE, "tests/panic_contract.rs": TEST}


class DeprecationPin(unittest.TestCase):
    def test_the_repository_is_green(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = dp.run(ROOT)
        self.assertEqual(code, 0, err.getvalue())

    def test_the_good_tree_is_green(self):
        code, err = run(good())
        self.assertEqual(code, 0, err)

    def test_counts_both_items(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "src").mkdir()
            (root / "src/privacy.rs").write_text(MODULE, encoding="utf-8")
            compared, problems = dp.check_module(root)
        self.assertEqual((compared, problems), (2, []))

    def test_a_multi_line_attribute_removed_is_red(self):
        files = good()
        files["src/privacy.rs"] = MODULE.replace(
            '#[deprecated(\n    since = "0.4.0",\n    note = "not private"\n)]\n', ""
        )
        code, err = run(files)
        self.assertEqual(code, 1)
        self.assertIn("`Gen` has no #[deprecated]", err)

    def test_a_single_line_attribute_removed_is_red(self):
        files = good()
        files["src/privacy.rs"] = MODULE.replace(
            '#[deprecated(since = "0.4.0", note = "not private")]\n', ""
        )
        code, err = run(files)
        self.assertEqual(code, 1)
        self.assertIn("`BITS` has no #[deprecated]", err)

    def test_an_attribute_separated_by_a_blank_line_does_not_count(self):
        files = good()
        files["src/privacy.rs"] = MODULE.replace(
            '#[deprecated(since = "0.4.0", note = "not private")]\n',
            '#[deprecated(since = "0.4.0", note = "not private")]\n\n',
        )
        code, err = run(files)
        self.assertEqual(code, 1)
        self.assertIn("`BITS`", err)

    def test_allow_outside_the_allowed_files_is_red(self):
        files = good()
        files["src/lib.rs"] = "// re-export of the deprecated module\n#[allow(deprecated)]\npub use x::y;\n"
        code, err = run(files)
        self.assertEqual(code, 1)
        self.assertIn("src/lib.rs:2: allow(deprecated) outside", err)

    def test_allow_in_a_list_is_found(self):
        files = good()
        files["examples/e.rs"] = "#![allow(dead_code, deprecated)]\nfn main() {}\n"
        code, err = run(files)
        self.assertEqual(code, 1)
        self.assertIn("examples/e.rs:1", err)

    def test_allow_without_a_reason_is_red(self):
        files = good()
        files["tests/panic_contract.rs"] = TEST.split("\n", 1)[1]
        code, err = run(files)
        self.assertEqual(code, 1)
        self.assertIn("needs a `// <reason>` line", err)

    def test_a_doc_comment_is_not_a_reason(self):
        files = good()
        files["tests/panic_contract.rs"] = "//! the crate's tests of the module\n" + TEST.split("\n", 1)[1]
        code, err = run(files)
        self.assertEqual(code, 1)

    def test_target_is_skipped(self):
        files = good()
        files["target/debug/build/x.rs"] = "#[allow(deprecated)]\n"
        code, err = run(files)
        self.assertEqual(code, 0, err)

    def test_comparing_nothing_is_red(self):
        files = good()
        files["src/privacy.rs"] = "//! emptied\n"
        code, err = run(files)
        self.assertEqual(code, 1)
        self.assertIn("no pub item compared", err)

    def test_a_missing_module_is_red(self):
        code, err = run({"tests/panic_contract.rs": TEST})
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
