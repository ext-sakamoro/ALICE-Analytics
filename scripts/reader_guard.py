#!/usr/bin/env python3
"""The crate has no reader of external bytes and no type-erasing entry point
(ALICE-Law G7 and G8, both n/a while this holds).

alice-analytics writes text for monitoring (src/export.rs: JSON and the
Prometheus exposition format) and reads nothing back: no decoder, no file or
network input, no FFI, Python or wasm export, no `dyn Any`, no build-time
generated value. G8 asks a durable format to carry the meaning identifier for
its reader; with no reader in the crate the exports are not a durable format
of this crate. G7 asks every type-erasing boundary that receives another
crate's Law-bearing value to check the producer's identifier; with no such
boundary the list is empty.

This scanner keeps that true, in two halves. The structural half lists every
fn of the production code whose parameters take bytes, text or a reader
(`&[u8]`, `[u8; N]`, `Vec<u8>`, `&str`, `String`, `impl Read`, a `Read` bound)
and requires each in scripts/reader_allowlist.toml (`<file>::<fn>`, `#2` for a
second fn of the same name in the file) with the reason it is not a
reader of external Law-bearing bytes (a new one, or a stale entry, fails). The
pattern half reads the production code of src/ (strings,
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

# Structural half: every fn (pub or not) of the production code whose
# parameters take bytes, text or a reader, keyed `<file>::<fn>`, must be in
# scripts/reader_allowlist.toml with the reason it is not a reader of external
# Law-bearing bytes; a new such fn fails, a stale entry fails
SIG_RE = re.compile(r"\bfn\s+(\w+)\s*(<(?:[^<>]|<[^<>]*>)*>)?\s*\(([^)]*)\)([^{;]*)", re.S)
BYTE_PARAM = re.compile(
    r"&\s*(?:'\w+\s+)?(?:mut\s+)?\[\s*u8\s*[\];]|\bVec\s*<\s*u8\s*>|\[\s*u8\s*;|&\s*(?:'\w+\s+)?str\b"
    r"|\bString\b|\bimpl\s+(?:(?:std\s*::\s*)?io\s*::\s*)?(?:Buf)?Read\b|\bBytes(?:Mut)?\b|\bCow\s*<[^>]*\b(?:str|\[\s*u8)")
READ_BOUND = re.compile(r"\b(?:Buf)?Read\b")



def check(root: Path) -> tuple[list[str], int]:
    problems: list[str] = []
    files = sorted((root / "src").rglob("*.rs")) if (root / "src").exists() else []
    rxs = {name: re.compile(rx) for name, rx in PATTERNS.items()}
    for path in files:
        rel = path.relative_to(root).as_posix()
        for n, code in production_lines(path.read_text(encoding="utf-8")):
            for name, rx in rxs.items():
                if rx.search(code):
                    problems.append(f"{rel}:{n}: {name}: `{code.strip()}`")
    allow_path = root / "scripts" / "reader_allowlist.toml"
    allowed = tomllib.loads(allow_path.read_text(encoding="utf-8")).get("allow", {}) if allow_path.exists() else {}
    seen: set[str] = set()
    for path in files:
        rel = path.relative_to(root).as_posix()
        text = "\n".join(code for _, code in production_lines(path.read_text(encoding="utf-8")))
        for m in SIG_RE.finditer(text):
            name, generics, params, tail = m.group(1), m.group(2) or "", m.group(3), m.group(4)
            if BYTE_PARAM.search(params) or (READ_BOUND.search(generics + tail) and READ_BOUND.search(generics + tail + params)):
                key = f"{rel}::{name}"
                n = 2
                while key in seen:  # a second fn of the same name in the file
                    key, n = f"{rel}::{name}#{n}", n + 1
                seen.add(key)
                if key not in allowed:
                    problems.append(f"{key}: takes bytes, text or a reader ({' '.join(params.split())}); "
                                    f"not in scripts/reader_allowlist.toml")
    for key, reason in sorted(allowed.items()):
        if key not in seen:
            problems.append(f"scripts/reader_allowlist.toml: {key} matches no fn (stale entry)")
        elif len(reason.strip()) < 12:
            problems.append(f"scripts/reader_allowlist.toml: {key}: reason shorter than 12 characters")
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
        print("reader_guard: a reader or type-erasing entry point appeared; G7 / G8 of "
              "alice-law-conformance.toml are n/a only while there is none", file=sys.stderr)
        return 1
    print(f"reader_guard: {files} files, {len(PATTERNS)} patterns, no reader or boundary")
    return 0


if __name__ == "__main__":
    sys.exit(main())
