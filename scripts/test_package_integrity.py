#!/usr/bin/env python3
"""Tests for scripts/package_integrity.py (run: python3 scripts/test_package_integrity.py).

The registry and cargo are injected: `versions_of` stands for the crates.io index and the
runner for cargo, so each case fixes exactly what is published and what cargo reports.
"""

from __future__ import annotations

import io
import os
import sys
import tarfile
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import package_integrity as pi  # noqa: E402

SHA = "72cd19faf37760e228ba6155fd3bdb5a0a85bcfa"


class Requirements(unittest.TestCase):
    def test_range_and_caret_forms(self) -> None:
        self.assertTrue(pi.satisfies((0, 11, 0), ">=0.11, <0.12"))
        self.assertFalse(pi.satisfies((0, 10, 0), ">=0.11, <0.12"))
        self.assertFalse(pi.satisfies((0, 12, 0), ">=0.11, <0.12"))
        self.assertTrue(pi.satisfies((0, 10, 3), "0.10"))
        self.assertFalse(pi.satisfies((0, 11, 0), "0.10"))
        self.assertTrue(pi.satisfies((7, 4, 0), "7"))
        self.assertFalse(pi.satisfies((8, 0, 0), ">=7, <8"))
        self.assertTrue(pi.satisfies((0, 0, 3), "^0.0.3"))
        self.assertFalse(pi.satisfies((0, 0, 4), "^0.0.3"))
        self.assertTrue(pi.satisfies((1, 2, 9), "~1.2"))
        self.assertFalse(pi.satisfies((1, 3, 0), "~1.2"))
        self.assertTrue(pi.satisfies((0, 5, 1), "=0.5.1"))
        self.assertFalse(pi.satisfies((0, 5, 2), "=0.5.1"))

    def test_unsupported_syntax_is_a_problem(self) -> None:
        for bad in ("0.11-beta", ">= ", "0.11,", "latest"):
            with self.assertRaises(pi.Problem, msg=bad):
                pi.comparators(bad)


class Fixture(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.root = base / "ALICE-Crate"
        (self.root / ".github").mkdir(parents=True)
        (self.root / ".github" / "alice-siblings").write_text(f"ALICE-Zip {SHA}\n")
        (self.root / "Cargo.toml").write_text(
            '[package]\nname = "alice-crate"\nversion = "0.6.0"\n\n[dependencies]\n'
            'alice-zip = { path = "../ALICE-Zip", version = ">=0.11, <0.12", optional = true }\n'
        )
        (base / "ALICE-Zip").mkdir()
        (base / "ALICE-Zip" / "Cargo.toml").write_text('[package]\nname = "alice-zip"\nversion = "0.11.0"\n')
        self.target = base / "target"
        self.calls: list[list[str]] = []
        self.published = {"alice-zip": [(0, 10, 0), (0, 9, 0)]}

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def versions_of(self, name: str) -> list:
        return self.published[name]

    def write_crate(self) -> None:
        out = self.target / "package"
        out.mkdir(parents=True, exist_ok=True)
        with tarfile.open(out / "alice-crate-0.6.0.crate", "w:gz") as tar:
            data = b'[package]\nname = "alice-crate"\nversion = "0.6.0"\n'
            info = tarfile.TarInfo("alice-crate-0.6.0/Cargo.toml")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

    def runner(self, package_error: str | None = None, build_error: str | None = None):
        def run(cmd: list[str], cwd: Path) -> tuple[int, str]:
            self.calls.append(cmd)
            if cmd[1] == "package":
                if package_error:
                    return 101, package_error
                self.write_crate()
                return 0, "Packaged"
            if build_error:
                return 101, build_error
            if "--locked" not in cmd:
                (cwd / "Cargo.lock").write_text('[[package]]\nname = "alice-det-math"\n')
            return 0, "Finished"

        return run

    def main(self, *argv: str, runner=None) -> tuple[int, str]:
        buf = io.StringIO()
        os.environ["CARGO_TARGET_DIR"] = str(self.target)
        with redirect_stdout(buf):
            rc = pi.main(["--root", str(self.root), *argv], versions_of=self.versions_of, runner=runner or self.runner())
        return rc, buf.getvalue()


class PackageIntegrity(Fixture):
    def test_unpublished_sibling_is_blocked_patched_and_green(self) -> None:
        rc, out = self.main("--expect-in-lock", "alice-det-math")
        self.assertEqual(rc, 0, out)
        self.assertIn("blocked: alice-zip >=0.11, <0.12 unpublished (crates.io max 0.10.0)", out)
        self.assertIn("blocked 1", out)
        self.assertIn("packages inspected 1", out)
        self.assertTrue(all(any("patch.crates-io.alice-zip.path=" in a for a in c) for c in self.calls), self.calls)

    def test_non_sibling_packaging_error_stays_red(self) -> None:
        rc, out = self.main(runner=self.runner(package_error="error: failed to read `src/missing.rs`"))
        self.assertEqual(rc, 1, out)
        self.assertIn("cargo package failed", out)
        rc, out = self.main(runner=self.runner(build_error="error[E0432]: unresolved import"))
        self.assertEqual(rc, 1, out)

    def test_sibling_on_the_registry_cannot_hide_a_failure(self) -> None:
        self.published["alice-zip"] = [(0, 11, 0), (0, 10, 0)]
        failure = "failed to select a version for the requirement `alice-zip`"
        rc, out = self.main(runner=self.runner(package_error=failure))
        self.assertEqual(rc, 1, out)
        self.assertIn("blocked 0", out)
        self.assertFalse(any("patch.crates-io" in a for c in self.calls for a in c), self.calls)

    def test_publish_gate_is_red_while_anything_is_blocked(self) -> None:
        rc, out = self.main("--publish-gate")
        self.assertEqual(rc, 1, out)
        self.assertIn("publish gate: FAIL", out)
        self.assertEqual(self.calls, [])
        self.published["alice-zip"] = [(0, 11, 0)]
        rc, out = self.main("--publish-gate")
        self.assertEqual(rc, 0, out)
        self.assertIn("publish gate: OK", out)

    def test_no_crate_produced_fails(self) -> None:
        def run(cmd: list[str], cwd: Path) -> tuple[int, str]:
            return 0, "Packaged"

        rc, out = self.main(runner=run)
        self.assertEqual(rc, 1, out)
        self.assertIn("inspected nothing", out)

    def test_sibling_version_must_satisfy_the_requirement(self) -> None:
        (self.root.parent / "ALICE-Zip" / "Cargo.toml").write_text('[package]\nname = "alice-zip"\nversion = "0.10.0"\n')
        rc, out = self.main()
        self.assertEqual(rc, 1, out)
        self.assertIn("does not satisfy", out)

    def test_unlisted_sibling_path_fails(self) -> None:
        (self.root / ".github" / "alice-siblings").write_text(f"ALICE-DB {SHA}\n")
        rc, out = self.main()
        self.assertEqual(rc, 1, out)
        self.assertIn("not in .github/alice-siblings", out)

    def test_expected_lock_entry_missing_fails(self) -> None:
        rc, out = self.main("--expect-in-lock", "alice-not-there")
        self.assertEqual(rc, 1, out)
        self.assertIn("did not resolve alice-not-there", out)

    def test_unreadable_index_fails(self) -> None:
        def down(name: str) -> list:
            raise OSError("network down")

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = pi.main(["--root", str(self.root)], versions_of=down, runner=self.runner())
        self.assertEqual(rc, 1, buf.getvalue())
        self.assertIn("index unreadable", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
