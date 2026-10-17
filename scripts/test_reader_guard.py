#!/usr/bin/env python3
"""Tests of scripts/reader_guard.py on throwaway trees."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reader_guard  # noqa: E402

# one planted line per pattern, each must be reported under that pattern
PLANTS = {
    "serde Deserialize": "#[derive(Deserialize)] pub struct S;",
    "serde_json reader": "let v = serde_json::from_str(s);",
    "binary codec": "let v = bincode::deserialize(b);",
    "from_str / from_slice / from_reader": "let v = toml::from_str(s);",
    "decode function": "pub fn from_json(s: &str) -> Snapshot { todo!() }",
    "str parse": "let x = s.parse::<f64>();",
    "file input": "let t = std::fs::read_to_string(p);",
    "network input": "let s = std::net::TcpStream::connect(a);",
    "C ABI": 'pub extern "C" fn f() {}',
    "Python binding": "#[pyfunction] fn f() {}",
    "wasm binding": "#[wasm_bindgen] pub fn f() {}",
    "type erasure": "fn f(x: &dyn Any) {}",
    "bytes to number": "let x = f64::from_le_bytes(b);",
    "reader trait": "use std::io::Read;",
    "conversion from bytes or text": "impl TryFrom<&[u8]> for Snapshot { type Error = (); }",
}

# reader shapes the structural half must report even with neutral names
SHAPES = {
    "try_from": "impl TryFrom<&[u8]> for ZzA {\n    type Error = ();\n    fn try_from(_b: &[u8]) -> Result<Self, ()> { Err(()) }\n}\n",
    "unpack_snapshot": "pub fn unpack_snapshot(bytes: &[u8]) -> Vec<f64> {\n    bytes.chunks(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()\n}\n",
    "restore": "pub fn restore(src: &mut impl std::io::Read) -> u8 {\n    let mut b = [0u8; 1];\n    let _ = src.read_exact(&mut b);\n    b[0]\n}\n",
    "generic_reader": "pub fn ingest<R: Read>(r: R) -> u8 { 0 }\n",
    "owned_bytes": "pub fn ingest_owned(v: Vec<u8>) -> u8 { 0 }\n",
    "text": "pub fn ingest_text(s: String) -> u8 { 0 }\n",
}


class ReaderGuardTest(unittest.TestCase):
    def tree(self, files: dict[str, str]) -> Path:
        d = Path(tempfile.mkdtemp())
        (d / "src").mkdir()
        for name, text in files.items():
            p = d / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
        return d

    def test_clean_tree_passes(self):
        d = self.tree({"src/export.rs": "pub fn snapshot_to_json(s: &S) -> String { String::new() }\n"})
        self.assertEqual(reader_guard.check(d), ([], 1))

    def test_every_pattern_is_found(self):
        self.assertEqual(set(PLANTS), set(reader_guard.PATTERNS), "a pattern has no plant")
        for name, line in PLANTS.items():
            problems, _ = reader_guard.check(self.tree({"src/lib.rs": line + "\n"}))
            self.assertTrue(any(f": {name}:" in p for p in problems), (name, problems))

    def test_test_modules_comments_and_strings_are_not_production(self):
        text = ('#[cfg(test)]\nmod tests {\n    fn from_json() {}\n}\n'
                '// let v = serde_json::from_str(s);\n'
                'pub const DOC: &str = "fn from_bytes( File::open";\n')
        self.assertEqual(reader_guard.check(self.tree({"src/lib.rs": text}))[0], [])

    def test_reader_shapes_with_neutral_names_are_found(self):
        for name, text in SHAPES.items():
            problems, _ = reader_guard.check(self.tree({"src/stats.rs": text}))
            self.assertTrue(any("src/stats.rs::" in p and "not in scripts/reader_allowlist.toml" in p
                                for p in problems), (name, problems))

    def test_allowlisted_fn_passes_and_stale_or_short_entries_fail(self):
        d = self.tree({"src/sketch.rs": "pub fn hash_bytes(data: &[u8]) -> u64 { 0 }\n"
                                        "pub fn hash_bytes(data: &[u8]) -> u64 { 1 }\n",
                       "scripts/reader_allowlist.toml":
                           '[allow]\n"src/sketch.rs::hash_bytes" = "hashes caller bytes, never decodes them"\n'
                           '"src/sketch.rs::hash_bytes#2" = "hashes caller bytes, never decodes them"\n'})
        self.assertEqual(reader_guard.check(d)[0], [])
        d = self.tree({"src/sketch.rs": "pub fn other() {}\n",
                       "scripts/reader_allowlist.toml": '[allow]\n"src/sketch.rs::hash_bytes" = "short"\n'})
        problems, _ = reader_guard.check(d)
        self.assertTrue(any("stale" in p for p in problems), problems)

    def test_build_script_and_proc_macro_fail(self):
        d = self.tree({"src/lib.rs": "\n", "build.rs": "fn main() {}\n",
                       "Cargo.toml": '[package]\nname = "x"\n[lib]\nproc-macro = true\n'})
        problems, _ = reader_guard.check(d)
        self.assertTrue(any(p.startswith("build.rs") for p in problems), problems)
        self.assertTrue(any("proc-macro" in p for p in problems), problems)

    def test_no_source_file_fails(self):
        d = Path(tempfile.mkdtemp())
        (d / "src").mkdir()
        self.assertEqual(reader_guard.main(["--root", str(d)]), 1)


if __name__ == "__main__":
    unittest.main()
