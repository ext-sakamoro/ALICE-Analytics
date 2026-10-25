#!/usr/bin/env python3
"""Each private sibling gets its own deploy key, and nothing else gets one.

A sibling named in .github/alice-siblings-private ("<Repo>" per line, "#" comments) is
cloned by .github/actions/alice-siblings over ssh with the key in env ALICE_DEPKEY_<REPO>
(the name after "ALICE-", upper case, "-" as "_"). This check reads every workflow and
fails when:

- a line of the private list is malformed or repeated, or names a repository that
  .github/alice-siblings does not list;
- a step that uses the action does not pass, in its own `env:`, exactly
  `ALICE_DEPKEY_<REPO>: ${{ secrets.ALICE_DEPKEY_<REPO> }}` for each private sibling
  (a missing key, a key for a repository that is not private, another value);
- ALICE_DEPKEY_ appears anywhere else in a workflow (a job or workflow `env:`, a `run:`,
  another step), so a key reaches no other step;
- the private list names a repository while no step uses the action;
- no workflow file is read.

With no private sibling and no step that uses the action the pass says
"N workflows read, 0 action steps, 0 private siblings".

    python3 scripts/alice_siblings_keys_check.py [repo root]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

LIST_FILE = ".github/alice-siblings"
PRIVATE_FILE = ".github/alice-siblings-private"
WORKFLOWS = ".github/workflows"
NAME_RE = re.compile(r"ALICE-[A-Za-z0-9-]+")
LINE_RE = re.compile(r"(ALICE-[A-Za-z0-9-]+) ([0-9a-f]{40})")
USES_RE = re.compile(r"^(\s*)(?:-\s+)?uses:\s*['\"]?\./(?:[\w.-]+/)?\.github/actions/alice-siblings['\"]?\s*$")
STEP_RE = re.compile(r"^(\s*)-\s")
ENV_LINE_RE = re.compile(r"^\s*([A-Za-z0-9_]+):\s*(.*?)\s*$")


def key_var(repo: str) -> str:
    return "ALICE_DEPKEY_" + repo.removeprefix("ALICE-").upper().replace("-", "_")


def read_names(path: Path, pattern: re.Pattern[str], what: str, problems: list[str]) -> list[str]:
    names: list[str] = []
    if not path.is_file():
        return names
    for no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = pattern.fullmatch(line)
        if not m:
            problems.append(f"{what}:{no}: not {'<ALICE-Repo>' if pattern is NAME_RE else '<ALICE-Repo> <40-hex SHA>'}: {line!r}")
            continue
        name = m.group(1) if m.groups() else m.group(0)
        if name in names:
            problems.append(f"{what}:{no}: {name} is listed twice")
        names.append(name)
    return names


def action_steps(lines: list[str]) -> list[tuple[int, int]]:
    """(first, end) line ranges of the steps that use the action"""
    out: list[tuple[int, int]] = []
    for i, line in enumerate(lines):
        if not USES_RE.match(line):
            continue
        # the step starts at the nearest "- " at or above this line
        start = i
        while start >= 0 and not STEP_RE.match(lines[start]):
            start -= 1
        if start < 0:
            continue
        indent = len(STEP_RE.match(lines[start]).group(1))
        end = start + 1
        while end < len(lines):
            s = lines[end]
            if s.strip() and not s.lstrip().startswith("#") and len(s) - len(s.lstrip()) <= indent:
                break
            end += 1
        out.append((start, end))
    return out


def step_env(lines: list[str], start: int, end: int) -> dict[str, str]:
    env: dict[str, str] = {}
    i = start
    while i < end:
        m = re.match(r"^(\s*)(-\s+)?env:\s*$", lines[i])
        if m:
            # the env keys are indented deeper than "env:" itself
            indent = len(m.group(1)) + len(m.group(2) or "")
            i += 1
            while i < end:
                s = lines[i]
                if s.strip() and len(s) - len(s.lstrip()) <= indent:
                    break
                e = ENV_LINE_RE.match(s)
                if e and not s.lstrip().startswith("#"):
                    env[e.group(1)] = e.group(2)
                i += 1
            continue
        i += 1
    return env


def check(root: Path) -> list[str]:
    problems: list[str] = []
    listed = read_names(root / LIST_FILE, LINE_RE, LIST_FILE, problems)
    private = read_names(root / PRIVATE_FILE, NAME_RE, PRIVATE_FILE, problems)
    for name in private:
        if name not in listed:
            problems.append(f"{PRIVATE_FILE}: {name} is not in {LIST_FILE}")
    want = {key_var(n): f"${{{{ secrets.{key_var(n)} }}}}" for n in private}
    files = sorted(p for p in (root / WORKFLOWS).glob("*.y*ml") if p.is_file()) if (root / WORKFLOWS).is_dir() else []
    users = 0
    for wf in files:
        rel = wf.relative_to(root).as_posix()
        lines = wf.read_text(encoding="utf-8").splitlines()
        allowed: set[int] = set()
        for start, end in action_steps(lines):
            users += 1
            env = step_env(lines, start, end)
            got = {k: v for k, v in env.items() if k.startswith("ALICE_DEPKEY_")}
            if got != want:
                problems.append(f"{rel}:{start + 1}: the step passes {got or 'no key'}, "
                                f"{PRIVATE_FILE} needs {want or 'no key'}")
            for i in range(start, end):
                e = ENV_LINE_RE.match(lines[i])
                if e and want.get(e.group(1)) == e.group(2):
                    allowed.add(i)
        for i, line in enumerate(lines):
            if "ALICE_DEPKEY_" in line and i not in allowed:
                problems.append(f"{rel}:{i + 1}: ALICE_DEPKEY_ outside the env of a step that uses the action")
    if not files:
        problems.append(f"no workflow read under {WORKFLOWS} (the check compared nothing)")
    if private and users == 0:
        problems.append(f"{PRIVATE_FILE} lists {', '.join(private)} but no step uses the action")
    if not problems:
        print(f"alice siblings keys: {len(files)} workflows read, {users} action steps, "
              f"{len(private)} private siblings ({', '.join(private) or 'none'})")
    return problems


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parent.parent
    problems = check(root)
    for p in problems:
        print(f"ERROR: {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
