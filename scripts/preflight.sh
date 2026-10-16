#!/usr/bin/env bash
# Local reproduction of the CI gates before `git push`: every command below is
# the one .github/workflows/ci.yml, security-audit.yml or semver-checks.yml runs, with the same
# arguments. A step this script does not cover is a step that can only fail
# remotely, so a step added to a workflow is added here in the same commit.
#
# Not reproduced here: the four-OS test matrix (this runs the host only), the
# packaged-crate build, the informational coverage job, and the fuzz runs
# (fuzz.yml; the targets are built when nightly + cargo-fuzz exist).
#
# usage: scripts/preflight.sh [--quick]
#   (none)   every gate: static checks, clippy, no_std builds, feature
#            powerset, docs, the full test suites (including the no_std lib
#            unit tests), the example, MSRV, and the security jobs (cargo audit / deny / machete) and the semver gate
#   --quick  static checks, clippy, no_std builds, docs and `cargo test --lib`;
#            skips the full suites, the example, the feature powerset, MSRV
#            and the security jobs
set -euo pipefail
cd "$(dirname "$0")/.."

quick=0
case "${1:-}" in
  --quick) quick=1 ;;
  "") ;;
  *) echo "usage: scripts/preflight.sh [--quick]" >&2; exit 2 ;;
esac
MSRV=1.87

step() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
need() { command -v "$1" >/dev/null 2>&1 || { echo "missing tool: $1 ($2)" >&2; exit 1; }; }
# `cargo clippy` reuses fresh `cargo check` artifacts and then lints nothing;
# touching the crate root invalidates only this crate's fingerprints.
relint() { touch src/lib.rs; }
add_target() { rustup target list --installed | grep -qx "$1" || rustup target add "$1"; }
PEDANTIC=(-W clippy::pedantic -D warnings)

need actionlint "brew install actionlint"
need python3 "python 3.9+"

step "ci.yml / actionlint: workflow YAML"
actionlint .github/workflows/*.yml

step "ci.yml / actionlint: workflow-timeouts (every job has timeout-minutes)"
python3 scripts/test_workflow_timeouts.py
python3 scripts/workflow_timeouts.py

step "ci.yml / actionlint: workflow-concurrency (branch runs cancel, main never)"
python3 scripts/test_workflow_concurrency.py
python3 scripts/workflow_concurrency.py

step "ci.yml / fmt: cargo fmt --check"
cargo fmt --all -- --check

step "ci.yml / docs-lint: tests + public documents / CHANGELOG structure"
python3 scripts/test_docs_lint.py
python3 scripts/docs_lint.py --check

step "ci.yml / docs-lint: deprecation pin of the privacy module"
python3 scripts/test_deprecation_pin.py
python3 scripts/deprecation_pin.py

step "ci.yml / docs-lint: entropy guard (no OS entropy / wall clock in production code)"
python3 scripts/test_entropy_guard.py
python3 scripts/entropy_guard.py

step "ci.yml / docs-lint: no built binary is tracked"
python3 scripts/test_tracked_binaries.py
python3 scripts/tracked_binaries.py

step "ci.yml / docs-lint: run_tests.sh (zero-test guard) tests"
python3 scripts/test_run_tests.py

step "ci.yml / clippy: clippy canary data (generator tests, generated from clippy.toml, no drift)"
python3 scripts/test_gen_clippy_canary_calls.py
python3 scripts/gen_clippy_canary_calls.py --check

step "ci.yml / clippy: clippy canary tests (parsers, accounting, one end-to-end run)"
python3 scripts/test_clippy_canary.py

step "ci.yml / clippy: libm ban list (tests)"
python3 scripts/test_libm_ban_check.py

step "ci.yml / docs-lint + doc: licence agrees and its files are packaged"
python3 scripts/test_license_check.py
python3 scripts/license_check.py
list=$(cargo package --list --allow-dirty)
for f in LICENSE-APACHE NOTICE; do printf '%s\n' "$list" | grep -qx "$f" || { echo "the package lacks $f"; exit 1; }; done

step "security-audit.yml / stub-guard"
scripts/stub_guard.sh

step "ci.yml / clippy: default features, all features, no default features (pedantic)"
relint
cargo clippy --all-targets -- "${PEDANTIC[@]}"
relint
cargo clippy --all-targets --all-features -- "${PEDANTIC[@]}"
relint
cargo clippy --all-targets --no-default-features -- "${PEDANTIC[@]}"

step "ci.yml / no-std: host checks, thumbv7em-none-eabihf build + clippy"
cargo check --lib --no-default-features
cargo check --lib --no-default-features --features law
add_target thumbv7em-none-eabihf
cargo build --lib --no-default-features --target thumbv7em-none-eabihf
cargo build --lib --no-default-features --features law --target thumbv7em-none-eabihf
relint
cargo clippy --lib --no-default-features --features law --target thumbv7em-none-eabihf -- "${PEDANTIC[@]}"

step "ci.yml / doc: rustdoc -D warnings (default + all features)"
RUSTDOCFLAGS="-Dwarnings" cargo doc --lib --no-deps
RUSTDOCFLAGS="-Dwarnings" cargo doc --lib --no-deps --all-features

step "fuzz.yml / build every fuzz target (nightly; the runs need the runner)"
if rustup toolchain list | grep -q '^nightly' && cargo +nightly fuzz --version >/dev/null 2>&1; then
  (cd fuzz && cargo +nightly fuzz build)
else
  echo "skip: nightly / cargo-fuzz not installed" >&2
fi

step "public-text.yml / public-text: Public text (ALICE-Law checker at the pin; tree, then the commits)"
python3 scripts/test_public_text.py
scripts/public_text.sh

if [[ $quick -eq 1 ]]; then
  step "cargo test --lib (quick)"
  cargo test --lib
  echo; echo "preflight --quick OK (full test suites, example, feature powerset, MSRV and security jobs skipped)"; exit 0
fi

step "ci.yml / clippy: clippy canary (every planted banned call is flagged)"
python3 scripts/clippy_canary.py

step "ci.yml / clippy: libm ban list (complete for the resolved libm, and enforced)"
python3 scripts/libm_ban_check.py

step "ci.yml / test: every feature, default features, no_std lib, example (golden files must run tests)"
scripts/run_tests.sh determinism_golden,float_edge_golden -- cargo test --all-features
scripts/run_tests.sh determinism_golden,float_edge_golden -- cargo test
# the no_std lane: the only one that runs the libm rounding path on a host
cargo test --lib --no-default-features
cargo test --lib --no-default-features --features law
cargo run --example residual_summary --features law

step "release-profile.yml: golden tests under the release profile (each file must run tests)"
scripts/run_tests.sh determinism_golden,float_edge_golden -- cargo test --release --test determinism_golden --test float_edge_golden

step "ci.yml / feature-powerset: cargo hack (depth 2)"
need cargo-hack "cargo install cargo-hack --locked"
cargo hack check --lib --feature-powerset --depth 2

step "ci.yml / msrv: rust-version = $MSRV"
if rustup toolchain list | grep -q "^$MSRV"; then
  cargo +"$MSRV" check --lib
  cargo +"$MSRV" check --lib --all-features
else
  echo "toolchain $MSRV not installed (rustup toolchain install $MSRV --profile minimal)" >&2
  exit 1
fi

step "security-audit.yml: cargo audit / cargo deny / cargo machete"
need cargo-audit "cargo install cargo-audit --locked"
need cargo-deny "cargo install cargo-deny --locked"
need cargo-machete "cargo install cargo-machete --locked"
cargo audit --db "${CARGO_TARGET_DIR:-target}/advisory-db" --deny yanked
cargo deny --all-features check all
cargo machete

step "semver-checks.yml: semver gate tests + cargo semver-checks against the latest crates.io release"
need cargo-semver-checks "cargo install cargo-semver-checks --version 0.50.0 --locked"
python3 scripts/test_semver_gate.py
python3 scripts/semver_gate.py --config .github/semver-gate.toml --target-dir "${CARGO_TARGET_DIR:-target}/semver"

echo; echo "preflight OK"
