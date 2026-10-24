#!/usr/bin/env python3
"""The licence is stated the same way everywhere it is stated.

ALICE-Analytics is LicenseRef-Proprietary (versions published before keep
the licence they were published under). This checks that:

- `Cargo.toml` says `license = "LicenseRef-Proprietary"` and `publish = false`,
  and its `include` list carries `/LICENSE-Proprietary` and none of the
  previous licence files;
- `LICENSE-Proprietary` exists and no `LICENSE-APACHE`, `LICENSE-MIT` or
  `NOTICE` is left;
- both READMEs link `LICENSE-Proprietary` in their licence section and link
  none of the previous licence files, and every `License:` line anywhere in
  them states LicenseRef-Proprietary first and no licence badge names an open
  licence;
- no comment in `deny.toml` describes this crate as MIT, Apache or AGPL (the
  entries in its allow list are for dependencies and stay).

A run that compares nothing fails.

    python3 scripts/license_check.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPDX = "LicenseRef-Proprietary"
TEXT = "LICENSE-Proprietary"
OLD_FILES = ("LICENSE-APACHE", "LICENSE-MIT", "NOTICE")
OPEN = re.compile(r"MIT|Apache|AGPL|GPL", re.I)
READMES = {"README.md": "## License", "README_JP.md": "## ライセンス"}
# a line stating the licence, e.g. `License: LicenseRef-Proprietary`
LICENSE_LINE = re.compile(r"^\s*(?:\*\*)?(?:License|Licence|ライセンス)(?:\*\*)?\s*[:：]")
# a shields.io licence badge, e.g. `img.shields.io/badge/license-MIT-green`
LICENSE_BADGE = re.compile(r"img\.shields\.io/badge/licen[cs]e-([^)\s\"]+)", re.I)


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
        package = table(manifest.read_text(encoding="utf-8"), "package")
        lic = string_field(package, "license")
        expect(lic == SPDX, f"Cargo.toml: license is {lic!r}, not {SPDX!r}")
        expect(re.search(r"^\s*publish\s*=\s*false\s*$", package, re.M) is not None,
               "Cargo.toml: publish = false is missing")
        m = re.search(r"^\s*include\s*=\s*\[(.*?)\]", package, re.M | re.S)
        listed = set(re.findall(r'"/?([^"]+)"', m.group(1))) if m else set()
        expect(TEXT in listed, f"Cargo.toml: include does not list /{TEXT}")
        for old in OLD_FILES:
            expect(old not in listed, f"Cargo.toml: include still lists /{old}")
    else:
        expect(False, "Cargo.toml: missing")

    expect((root / TEXT).is_file(), f"{TEXT}: missing")
    for old in OLD_FILES:
        expect(not (root / old).exists(), f"{old}: still present")

    for rel, heading in READMES.items():
        path = root / rel
        if not path.is_file():
            expect(False, f"{rel}: missing")
            continue
        text = path.read_text(encoding="utf-8")
        lic = section(text, heading)
        expect(bool(lic), f"{rel}: no `{heading}` section")
        expect(f"({TEXT})" in lic, f"{rel}: the licence section does not link {TEXT}")
        for old in OLD_FILES:
            expect(f"({old})" not in text, f"{rel}: links {old}")
        # every licence line and badge in the whole file, not only the section
        for n, line in enumerate(text.splitlines(), 1):
            if LICENSE_LINE.match(line):
                expect(line.strip().startswith(f"License: {SPDX}"),
                       f"{rel}:{n}: the licence line does not start with `License: {SPDX}`: {line.strip()!r}")
            for badge in LICENSE_BADGE.findall(line):
                expect(not OPEN.search(badge), f"{rel}:{n}: licence badge says {badge!r}")

    deny = root / "deny.toml"
    if deny.is_file():
        for n, line in enumerate(deny.read_text(encoding="utf-8").splitlines(), 1):
            comment = line.split("#", 1)[1] if "#" in line else ""
            if re.search(r"\b(this|our) (crate|repository)\b.*\b(MIT|Apache|AGPL)\b"
                         r"|\b(MIT|Apache|AGPL)\b.*\b(this|our) (crate|repository)\b", comment, re.I):
                problems.append(f"deny.toml:{n}: a comment describes this crate as an open licence")
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
