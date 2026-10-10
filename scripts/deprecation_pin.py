#!/usr/bin/env python3
"""Pin the deprecation of the `privacy` module.

`privacy` is not differentially private and is removed in 0.5.0. Until then:

1. every top-level `pub` item of src/privacy.rs carries `#[deprecated(...)]`
   (in its attribute block, multi-line attributes included);
2. `allow(deprecated)` appears only in ALLOWED (the module itself and the
   tests that still pin its values), each directly under a `//` comment that
   gives the reason (at least MIN_REASON characters).

Comparing nothing (no pub item found) is a failure, so a renamed or emptied
module cannot pass by accident.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

MODULE = "src/privacy.rs"
ALLOWED = {
    "src/privacy.rs",
    "tests/determinism_golden.rs",
    "tests/panic_contract.rs",
    "tests/analytic_oracle.rs",
}
MIN_REASON = 12
SKIP_DIRS = {"target", ".git"}

PUB_ITEM_RE = re.compile(r"^pub\s+(?:struct|enum|const|static|fn|type|trait|union)\s+(\w+)")
ALLOW_RE = re.compile(r"allow\s*\([^)]*\bdeprecated\b")


def attribute_block(lines: list[str], i: int) -> list[str]:
    """The doc comments and attributes directly above line i (stops at a blank
    line or at code at column 0 that is not an attribute or comment)."""
    block = []
    j = i - 1
    while j >= 0:
        s = lines[j]
        if not s.strip():
            break
        if s.startswith(("#", "//", " ", "\t", ")")):
            block.append(s)
            j -= 1
            continue
        break
    return block


def check_module(root: Path) -> tuple[int, list[str]]:
    path = root / MODULE
    if not path.is_file():
        return 0, [f"{MODULE}: not found"]
    lines = path.read_text(encoding="utf-8").split("\n")
    compared, problems = 0, []
    for i, line in enumerate(lines):
        m = PUB_ITEM_RE.match(line)
        if not m:
            continue
        compared += 1
        if not any("#[deprecated" in b for b in attribute_block(lines, i)):
            problems.append(f"{MODULE}:{i + 1}: pub item `{m.group(1)}` has no #[deprecated]")
    return compared, problems


def check_allows(root: Path) -> tuple[int, list[str]]:
    found, problems = 0, []
    for path in sorted(root.rglob("*.rs")):
        rel = path.relative_to(root).as_posix()
        if SKIP_DIRS.intersection(path.relative_to(root).parts):
            continue
        lines = path.read_text(encoding="utf-8").split("\n")
        for i, line in enumerate(lines):
            if not ALLOW_RE.search(line):
                continue
            found += 1
            if rel not in ALLOWED:
                problems.append(f"{rel}:{i + 1}: allow(deprecated) outside the allowed files")
                continue
            prev = lines[i - 1].strip() if i > 0 else ""
            reason = prev[2:].strip() if prev.startswith("//") and not prev.startswith(("///", "//!")) else ""
            if len(reason) < MIN_REASON:
                problems.append(
                    f"{rel}:{i + 1}: allow(deprecated) needs a `// <reason>` line directly above"
                    f" ({MIN_REASON}+ characters)"
                )
    return found, problems


def run(root: Path) -> int:
    compared, problems = check_module(root)
    allows, more = check_allows(root)
    problems += more
    for p in problems:
        print(f"error: {p}", file=sys.stderr)
    if compared == 0:
        print(f"error: no pub item compared in {MODULE}", file=sys.stderr)
        return 1
    print(
        f"deprecation-pin: {compared} pub items in {MODULE}, {allows} allow(deprecated),"
        f" {len(problems)} problems"
    )
    return 1 if problems else 0


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    return run(Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    sys.exit(main())
