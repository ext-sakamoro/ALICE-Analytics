#!/usr/bin/env python3
"""The crate reads external bytes only in the decided readers below, and has no
type-erasing entry point (ALICE-Law G7 n/a, G8 pass).

The decided readers (READERS) are the sketches' `decode` (src/sketch.rs) and
the two little-endian helpers it reads fields with: the byte form of a sketch
carries ANALYTICS_SEMANTICS_ID and `decode` refuses any other identifier, which
is G8's handshake (tests/sketch_bytes.rs
`every_sketch_decode_checks_the_meaning_identifier`; scripts/api_boundary.toml
lists each decode as that handshake). Any other reader fails here, and so does a
READERS entry that matches no line (stale).

alice-analytics writes text for monitoring (src/export.rs: JSON and the
Prometheus exposition format) and reads nothing back: no decoder, no file or
network input, no FFI, Python or wasm export, no `dyn Any`, no build-time
generated value. G8 asks a durable format to carry the meaning identifier for
its reader; with no reader in the crate the exports are not a durable format
of this crate. G7 asks every type-erasing boundary that receives another
crate's Law-bearing value to check the producer's identifier; with no such
boundary the list is empty.

This scanner is the text layer; the structural layer is scripts/api_boundary.py,
which enumerates every fn, impl and re-export of the public API that takes or
returns bytes, text or a reader from rustdoc JSON. This text layer reads the
production code of src/ (strings,
chars, comments and `#[cfg(test)]` modules removed, by the same reader as
scripts/entropy_guard.py) and fails on any reader or boundary pattern below,
on a build.rs or a proc-macro crate, and when it read no file (compared
nothing). Adding a reader is a decision that has to change the G7 / G8 cells
of alice-law-conformance.toml (and this list), not something that can land
silently.

usage: scripts/reader_guard.py [--root DIR]
"""
from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from entropy_guard import production_lines  # noqa: E402

# (file, pattern name, line text) -> why this reader is decided; each must match at least one line
READERS = {
    ("src/sketch.rs", "decode function", "pub fn decode(bytes: &[u8]) -> Result<Self, SketchBytesError> {"):
        "a sketch's decode: refuses bytes stamped with another ANALYTICS_SEMANTICS_ID (G8 handshake)",
    ("src/sketch.rs", "bytes to number", "u32::from_le_bytes(b)"):
        "get_u32: reads a field of a sketch's byte form, after decode checked the identifier",
    ("src/sketch.rs", "bytes to number", "u64::from_le_bytes(b)"):
        "get_u64: reads a field of a sketch's byte form, after decode checked the identifier",
}

# name -> regex over production code (string literals are reduced to "")
PATTERNS = {
    "serde Deserialize": r"\bDeserialize\b",
    "serde_json reader": r"\bserde_json\s*::\s*from_\w+",
    "binary codec": r"\b(?:bincode|bitcode|postcard|ciborium|rmp_serde|prost)\s*::",
    "from_str / from_slice / from_reader": r"\bfrom_(?:str|slice|reader)\s*\(",
    "decode function": r"\bfn\s+(?:from_bytes|from_json|decode\w*|deserialize\w*|parse\w*|read\w*|load\w*|import\w*)\b",
    "str parse": r"\.parse\s*::\s*<|\bstr\s*::\s*parse\b",
    "file input": r"\b(?:fs\s*::\s*read\w*|File\s*::\s*open|OpenOptions|read_to_string|read_to_end|BufReader)\b",
    "network input": r"\b(?:TcpStream|TcpListener|UdpSocket|recv_from)\b",
    "C ABI": r"\bextern\s*\"\"|#\[\s*no_mangle\s*\]|#\[\s*unsafe\s*\(\s*no_mangle",
    "Python binding": r"\bpyo3\b|#\[\s*py(?:function|class|methods|module)\b",
    "wasm binding": r"\bwasm_bindgen\b",
    "type erasure": r"\bdyn\s+Any\b|\bdowncast(?:_ref|_mut)?\b",
    "bytes to number": r"\b(?:[iuf](?:8|16|32|64|128)|[iu]size)\s*::\s*from_(?:le|be|ne)_bytes\b",
    "reader trait": r"\bread_exact\b|\b(?:std\s*::\s*)?io\s*::\s*(?:Buf)?Read\b",
    "conversion from bytes or text": r"\bimpl\b[^{;]*\b(?:TryFrom|From)\s*<\s*(?:&\s*(?:'\w+\s+)?(?:mut\s+)?(?:\[\s*u8|str)|Vec\s*<\s*u8|\[\s*u8\s*;|String|Box\s*<\s*\[\s*u8)|\bimpl\b[^{;]*\bFromStr\b",
}




def check(root: Path, readers: dict | None = None) -> tuple[list[str], int]:
    readers = READERS if readers is None else readers
    problems: list[str] = []
    files = sorted((root / "src").rglob("*.rs")) if (root / "src").exists() else []
    rxs = {name: re.compile(rx) for name, rx in PATTERNS.items()}
    used: set[tuple[str, str, str]] = set()
    for path in files:
        rel = path.relative_to(root).as_posix()
        for n, code in production_lines(path.read_text(encoding="utf-8")):
            for name, rx in rxs.items():
                if rx.search(code):
                    key = (rel, name, code.strip())
                    if key in readers:
                        used.add(key)
                    else:
                        problems.append(f"{rel}:{n}: {name}: `{code.strip()}`")
    for key in sorted(set(readers) - used):
        problems.append(f"READERS entry matches no line (stale): {key}")
    if (root / "build.rs").exists():
        problems.append("build.rs: a build script can compute values at build time (G7)")
    cargo = root / "Cargo.toml"
    if cargo.exists():
        lib = tomllib.loads(cargo.read_text(encoding="utf-8")).get("lib", {})
        if lib.get("proc-macro"):
            problems.append("Cargo.toml: a proc-macro crate computes values at build time (G7)")
    return problems, len(files)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    root = Path(ap.parse_args(argv).root)
    problems, files = check(root)
    for p in problems:
        print(f"reader_guard: {p}", file=sys.stderr)
    if files == 0:
        print("reader_guard: no src file read (compared nothing)", file=sys.stderr)
        return 1
    if problems:
        print("reader_guard: a reader or type-erasing entry point outside READERS appeared (or a "
              "READERS entry is stale); G7 of alice-law-conformance.toml is n/a and G8 pass only "
              "while the readers are exactly the decided ones", file=sys.stderr)
        return 1
    print(f"reader_guard: {files} files, {len(PATTERNS)} patterns, {len(READERS)} decided readers, "
          "no other reader or boundary")
    return 0


if __name__ == "__main__":
    sys.exit(main())
