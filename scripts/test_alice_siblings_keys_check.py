#!/usr/bin/env python3
"""Tests of scripts/alice_siblings_keys_check.py and of the private-sibling path of
.github/actions/alice-siblings.

The action's script is read out of action.yml and run with bash against local
repositories (ALICE_SIBLINGS_GIT_BASE / ALICE_SIBLINGS_PRIVATE_GIT_BASE). With
ALICE_SIBLINGS_NET_TEST=1 it also connects to github.com over ssh with a throwaway key:
the published host key gets as far as the key exchange (the server then refuses the
key), and another host key stops before it ("Host key verification failed").

    python3 scripts/test_alice_siblings_keys_check.py
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alice_siblings_keys_check as kc  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ACTION = ROOT / ".github/actions/alice-siblings/action.yml"
PUBLISHED = "github.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl"
SHA = "0123456789abcdef0123456789abcdef01234567"


def action_script() -> str:
    """the bash of the action's single run step (the block under `run: |`)"""
    lines = ACTION.read_text(encoding="utf-8").splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip() == "run: |") + 1
    indent = len(lines[start]) - len(lines[start].lstrip())
    body = []
    for l in lines[start:]:
        if l.strip() and len(l) - len(l.lstrip()) < indent:
            break
        body.append(l[indent:])
    return "\n".join(body) + "\n"


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(cwd), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


class KeysCheck(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "repo"
        (self.root / ".github/workflows").mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write(self, rel: str, text: str) -> None:
        (self.root / rel).write_text(text, encoding="utf-8")

    def wf(self, env: str = "", extra: str = "") -> str:
        step_env = f"        env:\n{env}" if env else ""
        return ("jobs:\n  build:\n    runs-on: ubuntu-latest\n    steps:\n"
                "      - uses: actions/checkout@v5\n"
                "      - uses: ./.github/actions/alice-siblings\n"
                f"{step_env}"
                "      - run: cargo test\n" + extra)

    def setup_private(self, wf: str | None = None, private: str = "ALICE-Bamboo\n") -> None:
        self.write(".github/alice-siblings", f"ALICE-Bamboo {SHA}\nALICE-Zip {SHA}\n")
        self.write(".github/alice-siblings-private", private)
        self.write(".github/workflows/ci.yml", wf if wf is not None else
                   self.wf("          ALICE_DEPKEY_BAMBOO: ${{ secrets.ALICE_DEPKEY_BAMBOO }}\n"))

    def problems(self) -> list[str]:
        return kc.check(self.root)

    def test_private_key_on_the_step_passes(self) -> None:
        self.setup_private()
        self.assertEqual(self.problems(), [])

    def test_no_private_and_no_key_passes(self) -> None:
        self.write(".github/alice-siblings", f"ALICE-Zip {SHA}\n")
        self.write(".github/workflows/ci.yml", self.wf())
        self.assertEqual(self.problems(), [])

    def test_no_workflow_fails(self) -> None:
        shutil.rmtree(self.root / ".github/workflows")
        self.assertTrue(any("no workflow read" in p for p in self.problems()))

    def test_missing_key_fails(self) -> None:
        self.setup_private(self.wf())
        self.assertTrue(any("passes no key" in p for p in self.problems()))

    def test_key_for_a_public_sibling_fails(self) -> None:
        self.setup_private(self.wf("          ALICE_DEPKEY_BAMBOO: ${{ secrets.ALICE_DEPKEY_BAMBOO }}\n"
                                   "          ALICE_DEPKEY_ZIP: ${{ secrets.ALICE_DEPKEY_ZIP }}\n"))
        self.assertTrue(any("ALICE_DEPKEY_ZIP" in p for p in self.problems()))

    def test_key_without_a_private_list_fails(self) -> None:
        self.write(".github/alice-siblings", f"ALICE-Zip {SHA}\n")
        self.write(".github/workflows/ci.yml", self.wf("          ALICE_DEPKEY_ZIP: ${{ secrets.ALICE_DEPKEY_ZIP }}\n"))
        self.assertTrue(self.problems())

    def test_other_value_fails(self) -> None:
        self.setup_private(self.wf("          ALICE_DEPKEY_BAMBOO: ${{ secrets.OTHER }}\n"))
        self.assertTrue(self.problems())

    def test_key_in_a_job_env_fails(self) -> None:
        wf = self.wf("          ALICE_DEPKEY_BAMBOO: ${{ secrets.ALICE_DEPKEY_BAMBOO }}\n").replace(
            "    runs-on: ubuntu-latest\n",
            "    runs-on: ubuntu-latest\n    env:\n      ALICE_DEPKEY_BAMBOO: ${{ secrets.ALICE_DEPKEY_BAMBOO }}\n")
        self.setup_private(wf)
        self.assertTrue(any("outside the env" in p for p in self.problems()))

    def test_key_in_another_step_fails(self) -> None:
        self.setup_private(self.wf("          ALICE_DEPKEY_BAMBOO: ${{ secrets.ALICE_DEPKEY_BAMBOO }}\n",
                                   "      - run: echo \"$ALICE_DEPKEY_BAMBOO\"\n"))
        self.assertTrue(any("outside the env" in p for p in self.problems()))

    def test_private_not_listed_fails(self) -> None:
        self.setup_private(private="ALICE-Bamboo\nALICE-Print\n")
        self.assertTrue(any("ALICE-Print is not in" in p for p in self.problems()))

    def test_private_without_any_action_step_fails(self) -> None:
        self.setup_private("jobs:\n  lint:\n    steps:\n      - run: true\n")
        self.assertTrue(any("no step uses the action" in p for p in self.problems()))

    def test_malformed_and_repeated_private_lines_fail(self) -> None:
        self.setup_private(private="ALICE-Bamboo\nALICE-Bamboo\n")
        self.assertTrue(any("listed twice" in p for p in self.problems()))
        self.setup_private(private="bamboo\n")
        self.assertTrue(any("not <ALICE-Repo>" in p for p in self.problems()))

    def test_every_step_that_uses_the_action_is_read(self) -> None:
        good = "          ALICE_DEPKEY_BAMBOO: ${{ secrets.ALICE_DEPKEY_BAMBOO }}\n"
        wf = self.wf(good) + "  second:\n    steps:\n      - uses: ./ALICE-X/.github/actions/alice-siblings\n"
        self.setup_private(wf)
        self.assertTrue(any("passes no key" in p for p in self.problems()))

    def test_this_repository_passes(self) -> None:
        self.assertEqual(kc.check(ROOT), [])


@unittest.skipUnless(shutil.which("bash") and shutil.which("git"), "needs bash and git")
class Action(unittest.TestCase):
    """the action's script against local repositories"""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.src = self.tmp / "src"
        self.ws = self.tmp / "ws"
        (self.ws / "app/.github").mkdir(parents=True)
        self.rt = self.tmp / "rt"
        self.rt.mkdir()
        self.shas: dict[str, str] = {}
        for repo, bare_name in (("ALICE-Zip", "ALICE-Zip"), ("ALICE-Bamboo", "ALICE-Bamboo.git")):
            d = self.src / bare_name
            d.mkdir(parents=True)
            git(d, "init", "-q")
            (d / "f").write_text(repo)
            git(d, "add", "f")
            git(d, "commit", "-qm", "x")
            self.shas[repo] = git(d, "rev-parse", "HEAD")
        self.script = self.tmp / "action.sh"
        self.script.write_text(action_script(), encoding="utf-8")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_action(self, siblings: list[str], private: list[str] | None, keys: dict[str, str]) -> subprocess.CompletedProcess:
        gh = self.ws / "app/.github"
        (gh / "alice-siblings").write_text("".join(f"{r} {self.shas.get(r, SHA)}\n" for r in siblings))
        p = gh / "alice-siblings-private"
        if private is None:
            p.unlink(missing_ok=True)
        else:
            p.write_text("".join(f"{r}\n" for r in private))
        for r in ("ALICE-Zip", "ALICE-Bamboo"):
            shutil.rmtree(self.ws / r, ignore_errors=True)
        env = {k: v for k, v in os.environ.items() if not k.startswith("ALICE_DEPKEY_")}
        env.update(GITHUB_WORKSPACE=str(self.ws), REPO_DIR="app", RUNNER_TEMP=str(self.rt),
                   ALICE_SIBLINGS_GIT_BASE=str(self.src), ALICE_SIBLINGS_PRIVATE_GIT_BASE=str(self.src), **keys)
        return subprocess.run(["bash", str(self.script)], env=env, capture_output=True, text=True)

    def assertFails(self, r: subprocess.CompletedProcess, text: str) -> None:
        self.assertNotEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn(text, r.stdout + r.stderr)

    def test_public_only(self) -> None:
        r = self.run_action(["ALICE-Zip"], None, {})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("ALICE-Zip sibling at", r.stdout)

    def test_private_with_its_key_and_the_key_is_removed(self) -> None:
        r = self.run_action(["ALICE-Zip", "ALICE-Bamboo"], ["ALICE-Bamboo"], {"ALICE_DEPKEY_BAMBOO": "k"})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(git(self.ws / "ALICE-Bamboo", "rev-parse", "HEAD"), self.shas["ALICE-Bamboo"])
        self.assertFalse((self.rt / "alice-siblings-keys").exists(), "the key directory is left behind")
        self.assertNotIn("sshCommand", git(self.ws / "ALICE-Bamboo", "config", "--list"))

    def test_key_removed_when_the_step_fails(self) -> None:
        r = self.run_action(["ALICE-Bamboo", "ALICE-Missing"], ["ALICE-Bamboo"], {"ALICE_DEPKEY_BAMBOO": "k"})
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse((self.rt / "alice-siblings-keys").exists(), "the key directory is left behind")

    def test_private_without_its_key(self) -> None:
        self.assertFails(self.run_action(["ALICE-Bamboo"], ["ALICE-Bamboo"], {}), "ALICE_DEPKEY_BAMBOO is empty")

    def test_key_for_a_public_sibling(self) -> None:
        self.assertFails(self.run_action(["ALICE-Zip"], None, {"ALICE_DEPKEY_ZIP": "k"}),
                         "ALICE_DEPKEY_ZIP is set but names no repository")

    def test_private_not_listed(self) -> None:
        self.assertFails(self.run_action(["ALICE-Zip"], ["ALICE-Bamboo"], {"ALICE_DEPKEY_BAMBOO": "k"}),
                         "ALICE-Bamboo is in .github/alice-siblings-private but not in")

    def test_malformed_private_line(self) -> None:
        self.assertFails(self.run_action(["ALICE-Zip"], ["bamboo"], {}), "not '<ALICE-Repo>'")

    def test_empty_list(self) -> None:
        self.assertFails(self.run_action([], None, {}), "lists no repository")


class HostKey(unittest.TestCase):
    def test_the_published_host_key_is_pinned_strictly(self) -> None:
        text = action_script()
        self.assertIn(f"echo '{PUBLISHED}'", text)
        self.assertIn("-o StrictHostKeyChecking=yes", text)
        self.assertIn("-o UserKnownHostsFile=", text)
        self.assertIn("-F /dev/null", text)
        self.assertNotRegex(text, r"(?i)StrictHostKeyChecking=(no|accept-new|off)")
        self.assertNotRegex(text, r"git\s+config[^\n]*(sshCommand|insteadOf)")

    @unittest.skipUnless(os.environ.get("ALICE_SIBLINGS_NET_TEST") == "1", "ALICE_SIBLINGS_NET_TEST=1 runs it")
    def test_host_key_against_github(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        try:
            subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(tmp / "k")], check=True)
            key = (tmp / "k").read_text()
            script = action_script()

            def clone(text: str) -> subprocess.CompletedProcess:
                ws = tmp / "ws"
                shutil.rmtree(ws, ignore_errors=True)
                (ws / "app/.github").mkdir(parents=True)
                (ws / "app/.github/alice-siblings").write_text(f"ALICE-Bamboo {SHA}\n")
                (ws / "app/.github/alice-siblings-private").write_text("ALICE-Bamboo\n")
                (tmp / "a.sh").write_text(text)
                (tmp / "rt").mkdir(exist_ok=True)
                env = {k: v for k, v in os.environ.items() if not k.startswith("ALICE_")}
                env.update(GITHUB_WORKSPACE=str(ws), REPO_DIR="app", RUNNER_TEMP=str(tmp / "rt"),
                           ALICE_DEPKEY_BAMBOO=key, GIT_TERMINAL_PROMPT="0")
                return subprocess.run(["bash", str(tmp / "a.sh")], env=env, capture_output=True, text=True, timeout=120)

            ok = clone(script)
            self.assertNotEqual(ok.returncode, 0)
            self.assertIn("Permission denied (publickey)", ok.stderr, ok.stderr)
            # another valid ed25519 host key (the throwaway key's public half)
            other = (tmp / "k.pub").read_text().split()[1]
            wrong = clone(script.replace(PUBLISHED, f"github.com ssh-ed25519 {other}"))
            self.assertNotEqual(wrong.returncode, 0)
            self.assertIn("Host key verification failed", wrong.stderr, wrong.stderr)
            self.assertNotIn("Permission denied", wrong.stderr)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=1)
