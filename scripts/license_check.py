#!/usr/bin/env python3
"""The licence is stated the same way everywhere it is stated.

From 0.4.0 ALICE-Analytics is Apache-2.0 only. This checks that:

- `Cargo.toml` says `license = "Apache-2.0"`;
- `LICENSE-APACHE` and `NOTICE` exist and no `LICENSE-MIT` is left;
- `NOTICE` carries the attribution sentence;
- both READMEs link `LICENSE-APACHE` and `NOTICE` in their licence section and
  link no `LICENSE-MIT`;
- no comment in `deny.toml` describes this crate as MIT (the `MIT` entry in
  its allow list is for dependencies and stays).

A run that compares nothing fails.

    python3 scripts/license_check.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPDX = "Apache-2.0"
ATTRIBUTION = "This product includes ALICE-Analytics, Copyright 2025-2026 Moroya Sakamoto"
READMES = {"README.md": "## License", "README_JP.md": "## ライセンス"}


def table(text: str, name: str) -> str:
    m = re.search(r"^\[" + re.escape(name) + r"\]\s*$(.*?)(?=^\[|\Z)", text, re.M | re.S)
    return m.group(1) if m else ""


def string_field(section: str, key: str):
    m = re.search(r'^\s*' + re.escape(key) + r'\s*=\s*"([^"]*)"', section, re.M)
    return m.group(1) if m else None


def section(text: str, heading: str) -> str:
    i = text.find("\n" + heading + "\n")
    if i < 0:
        return ""
    rest = text[i + len(heading) + 2:]
    j = rest.find("\n## ")
    return rest if j < 0 else rest[:j]


def check(root: Path) -> list:
    """Problems found under `root` (empty when everything agrees), and the
    number of comparisons made, as the last element."""
    problems = []
    compared = 0

    def expect(ok: bool, message: str):
        nonlocal compared
        compared += 1
        if not ok:
            problems.append(message)

    manifest = root / "Cargo.toml"
    if manifest.is_file():
        lic = string_field(table(manifest.read_text(encoding="utf-8"), "package"), "license")
        expect(lic == SPDX, f"Cargo.toml: license is {lic!r}, not {SPDX!r}")
    else:
        expect(False, "Cargo.toml: missing")

    expect((root / "LICENSE-APACHE").is_file(), "LICENSE-APACHE: missing")
    expect(not (root / "LICENSE-MIT").exists(), "LICENSE-MIT: still present")
    notice = root / "NOTICE"
    expect(notice.is_file(), "NOTICE: missing")
    if notice.is_file():
        expect(ATTRIBUTION in " ".join(notice.read_text(encoding="utf-8").split()),
               "NOTICE: the attribution sentence is missing")

    for rel, heading in READMES.items():
        path = root / rel
        if not path.is_file():
            expect(False, f"{rel}: missing")
            continue
        text = path.read_text(encoding="utf-8")
        lic = section(text, heading)
        expect(bool(lic), f"{rel}: no `{heading}` section")
        expect("(LICENSE-APACHE)" in lic, f"{rel}: the licence section does not link LICENSE-APACHE")
        expect("(NOTICE)" in lic, f"{rel}: the licence section does not link NOTICE")
        expect("LICENSE-MIT" not in text, f"{rel}: links LICENSE-MIT")

    deny = root / "deny.toml"
    if deny.is_file():
        for n, line in enumerate(deny.read_text(encoding="utf-8").splitlines(), 1):
            comment = line.split("#", 1)[1] if "#" in line else ""
            if re.search(r"\b(this|our) (crate|repository)\b.*\bMIT\b|\bMIT\b.*\b(this|our) (crate|repository)\b",
                         comment, re.I):
                problems.append(f"deny.toml:{n}: a comment describes this crate as MIT")
        compared += 1

    return problems + [compared]


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    *problems, compared = check(ROOT)
    for p in problems:
        print(f"error: {p}", file=sys.stderr)
    if compared == 0:
        print("error: compared nothing", file=sys.stderr)
        return 1
    print(f"license_check: {compared} comparisons, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
