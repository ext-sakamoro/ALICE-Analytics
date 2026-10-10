# ALICE-Analytics

[日本語](README_JP.md)

Probabilistic data structures and streaming statistics with stated error
bounds: HyperLogLog cardinality, DDSketch quantiles with a relative-error
guarantee, Count-Min frequencies and heavy hitters, streaming moments /
covariance / regression, windowed aggregates and anomaly detectors. The core is `no_std` with fixed-size, stack-allocated
sketches. With the `law` feature it also summarises how a set of points deviates
from a fitted `alice_zip::law::SignalLaw`.

It is not a time-series database, a metrics backend or a dashboard: it keeps
summaries in memory and leaves storage and transport to the caller. Sketches
answer approximate queries; when exact quantiles or exact distinct counts are
needed, keep the raw values.

License: MIT OR Apache-2.0

## Contents

- [Installation](#installation)
- [Example](#example)
- [Residual distribution of a law](#residual-distribution-of-a-law)
- [Features](#features)
- [Modules](#modules)
- [Error bounds](#error-bounds)
- [Determinism](#determinism)
- [Minimum supported Rust version](#minimum-supported-rust-version)
- [Building and testing](#building-and-testing)
- [Related crates](#related-crates)
- [License](#license)

## Installation

```sh
cargo add alice-analytics
# with the residual summary of alice-zip laws
cargo add alice-analytics --features law
```

## Example

```rust
use alice_analytics::sketch::{DDSketch, HyperLogLog};

// quantiles with a relative-error guarantee of 1 %
let mut latency = DDSketch::new(0.01);
for ms in 1..=1000 {
    latency.insert(f64::from(ms));
}
let p99 = latency.quantile(0.99);
assert!((p99 - 990.0).abs() <= 0.01 * 990.0);

// distinct count in 16 KiB of registers
let mut users = HyperLogLog::new();
for id in 0..10_000_u64 {
    users.insert(&id);
}
let n = users.cardinality();
assert!((n - 10_000.0).abs() < 0.05 * 10_000.0);
```

## Residual distribution of a law

`law::residual_summary(&law, &points)` (feature `law`) streams the residuals
`y − f(x)` of `points` about an `alice_zip::law::SignalLaw` into a
`DDSketch2048` and returns a `ResidualSummary`:

| Field | Meaning |
|-------|---------|
| `count` | points summarised |
| `out_of_range` | points whose `x` is outside the law's valid range (or not finite); the law is not evaluated there, nothing is extrapolated |
| `non_finite` | points in range whose `y − f(x)` is NaN or infinite; counted, not summarised |
| `distribution` | `None` when nothing was summarised; otherwise `mean`, `min`, `max`, `max_abs` (exact) and `p50`, `p90`, `p99` (sketched) |
| `relative_accuracy` | `α`: each reported quantile `q̂` of a residual `q` satisfies `\|q̂ − q\| ≤ α·\|q\|` |
| `accurate_range` | residual magnitudes for which that bound holds; zero is always exact |
| `outside_accuracy` | non-zero residuals outside `accurate_range` (0 means the bound holds for every quantile) |

The quantile of rank `⌈q·n⌉` is reported, the same order statistic the
sketch defines. Residuals are divided by a power-of-two scale before they are
inserted (by default the largest power of two not above the law's own residual
RMS), which moves `accurate_range` to the size of the residuals without
rounding. `law::ResidualSketch` is the streaming form: `push` one point at a
time, read `quantile(q)` for any `q`, choose `α` and the scale.

```rust,ignore
use alice_analytics::law::residual_summary;

let s = residual_summary(&law, &new_points);
println!("{} summarised, {} outside the valid range", s.count, s.out_of_range);
if let Some(d) = s.distribution {
    println!("p50 {:+e}  p99 {:+e}  max |r| {:e}", d.p50, d.p99, d.max_abs);
}
```

A complete program is in `examples/residual_summary.rs`
(`cargo run --example residual_summary --features law`).

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `std` | yes | standard-library float functions, entropy-seeded privacy constructors, JSON / Prometheus export |
| `law` | no | `law` module: residual summary of an `alice_zip::law::SignalLaw` (adds `alice-zip` without its `std` feature, and `alloc`) |
| `simd` | no | no effect; kept so existing feature lists still build |

Without `std` the privacy mechanisms take an explicit seed, and `sqrt` /
`ceil` / `floor` / `round` / `mul_add` come from `libm`; IEEE 754 requires all
five to be correctly rounded, so they return the same bits as the inherent
methods. The transcendentals go through `alice-det-math` in both builds, so a
`no_std` build and a `std` build agree bit for bit (see
[Determinism](#determinism)). CI builds the library for
`thumbv7em-none-eabihf` with and without `law`.

## Modules

| Module | Contents |
|--------|----------|
| `sketch` | `HyperLogLog10/12/14/16`, `DDSketch128` … `DDSketch2048`, `CountMinSketch1024x5` / `2048x7` / `4096x5`, `HeavyHitters5/10/20`, `FnvHasher`, `Mergeable` |
| `stats` | `StreamingStats` (mean / variance / skewness / kurtosis), `CovarianceMatrix`, `quantile_sorted`, `percentile_rank`, `iqr` |
| `window` | `TumblingWindow`, `SlidingWindow`, `HierarchicalRollup` |
| `streaming_ops` | `SimpleMovingAverage`, `ExponentialMovingAverage`, `ChangeRate`, `LinearRegression`, `LinearRegressionFull` |
| `anomaly` | `StreamingMedian`, `MadDetector`, `ZScoreDetector`, `EwmaDetector`, `CompositeDetector` |
| `privacy` | **deprecated, not differentially private** (predictable noise source; removed in 0.4.0): use `alice_crypto::dp` |
| `pipeline` | `MetricPipeline`, `MetricRegistry`, `MetricSnapshot`, `RingBuffer` |
| `export` | JSON and Prometheus text for `MetricSnapshot` (`std`) |
| `law` | `residual_summary`, `ResidualSketch`, `ResidualSummary` (feature `law`) |

## Error bounds

| Structure | Bound | Checked in |
|-----------|-------|------------|
| `DDSketch` | `\|q̂ − q\| ≤ α·\|q\|` for magnitudes inside `accurate_range()`, negative values included | `tests/analytic_oracle.rs` |
| `HyperLogLog` | standard error `1.04/√m` (`m` registers) | `tests/analytic_oracle.rs` |
| `CountMinSketch` | estimate ≥ true count, overestimate ≤ `ε·N`, `ε = e/w` | `tests/analytic_oracle.rs` |
| `law::ResidualSketch` | the `DDSketch` bound on residuals, exact `mean` / `min` / `max` | `tests/law_residual.rs` |

The expected values in these tests are closed forms or two-pass references
written in the test files, not outputs of the functions under test.

A **non-finite sample** (`NaN`, `±inf`) is not a magnitude: `DDSketch::insert`
counts it in `non_finite()` and leaves `count`, `sum`, `min`, `max` and every
bin alone, so the ranks behind a quantile are the ranks of the values that
reached a bin. This is the same classification the `law` module applies to a
non-finite residual. A **finite** magnitude outside `accurate_range()` is
counted and filed in the nearest edge bin: it keeps its rank, but not the `α`
bound.

## Determinism

The same inputs produce the **same bits** on every supported target. This
matters because sketches are merged across machines, telemetry is replayed,
and an audit may re-derive a quantile from the raw events: a last-ulp
difference between two hosts turns an estimate into two estimates.

| Tier | What | Why it is bit-exact |
|------|------|---------------------|
| IEEE basic operations | `+ − × ÷`, `sqrt`, `mul_add`, `ceil` / `floor` / `round` | IEEE 754 requires correct rounding, so every target agrees. `mul_add` is a fused multiply-add with a single rounding; a target without an FMA instruction uses the correctly rounded software `fma` |
| Transcendentals | `ln`, `exp`, `powf` | Routed through [`alice-det-math`](https://crates.io/crates/alice-det-math), whose kernels are built from the operations above in a fixed evaluation order. The platform `libm` is **not** used: its last ulp differs between macOS, glibc, MSVC and wasm |
| Integer powers | `γⁿ` for the `DDSketch` bin layout | Binary exponentiation with a fixed association order, rather than `powi`, whose multiplication tree is unspecified |
| Pseudo-random | `XorShift64` and everything seeded from it | Integer state; a seeded constructor replays exactly. The entropy-seeded constructors (`std` only) are by design not reproducible |

Enforcement is mechanical, in two layers:

* `clippy.toml` lists the inherent `f32` / `f64` transcendentals under
  `disallowed-methods`, and CI runs
  `cargo clippy --all-targets --all-features -- -D warnings`, so reintroducing
  `x.ln()` is a compile error rather than a silent divergence. This is the
  layer that catches a regression on the machine that writes it.
* `tests/determinism_golden.rs` serialises the outputs of seven scenarios
  (one per module with float arithmetic) bit for bit and compares a SHA-256
  against a recorded constant. Each scenario also asserts that it serialised a
  non-zero number of bytes, so a scenario that stops exercising its module
  fails instead of passing on the hash of an empty buffer. CI runs the file on
  macOS `aarch64`, Linux `x86_64`, Linux `aarch64` and Windows `x86_64`.

### Naming the arithmetic

Bit-exactness across machines is only half of what a stored estimate needs.
The other half is being able to say *which* arithmetic produced it, because a
quantile computed under one implementation of `ln` and one computed under
another are two numbers, however close they look.

`alice_analytics::SEMANTICS_ID` is that name: the 32-byte identifier
`alice-det-math` publishes for its own numeric behaviour, re-exported here.
Record it next to any estimate that is stored, transmitted or merged; two
results are comparable as numbers only if the identifier agrees. It is pinned
as hex by `golden_semantics_id`, so an arithmetic change cannot reach a release
unnoticed.

With the `law` feature, `law::ResidualSummary::law_id` goes one step further
and names both halves at once: it is the summarised law's own identifier taken
under `SEMANTICS_ID`, so it changes when `f(x)` changes *and* when the
arithmetic changes. It covers what `f(x)` reads (the domain and the
coefficients) and not how the law was obtained, so two laws fitted from
different measurements that evaluate identically share one identifier.

<!-- claim-test: golden_sketch -->
<!-- claim-test: golden_stats -->
<!-- claim-test: golden_window -->
<!-- claim-test: golden_anomaly -->
<!-- claim-test: golden_privacy -->
<!-- claim-test: golden_streaming_ops -->
<!-- claim-test: golden_law -->
<!-- claim-test: golden_semantics_id -->
<!-- claim-test: golden_law_id -->

**Determinism is not correctness.** A golden hash pins whatever the code does
today, including a mistake; the error bounds above are checked separately
against closed forms. The two are independent and both are required.

**Outside the guarantee**: a target that does not honour IEEE 754 for the
basic operations (32-bit x86 built for x87 without SSE2), any build with
fast-math style flags, and the entropy-seeded privacy constructors. Results
across a *version* change of this crate or of `alice-det-math` are also not
pinned — only results across platforms at a given version. What `SEMANTICS_ID`
adds is not a promise that the arithmetic never changes, but the ability to
tell that it did.

## Minimum supported Rust version

`rust-version = "1.87"` (measured: 1.86 does not build the library). CI checks
the library on 1.87 with default and with all features; the development
toolchain is pinned in `rust-toolchain.toml`.

## Building and testing

```sh
cargo test --all-features
cargo test --all-features --test determinism_golden  # cross-platform bit equality
cargo test --all-features --test panic_contract      # degenerate-input contracts
cargo build --lib --no-default-features --features law --target thumbv7em-none-eabihf
scripts/preflight.sh          # every CI gate that runs locally
scripts/preflight.sh --quick  # static checks, clippy, builds, docs, cargo test --lib
```

## Related crates

- [`alice-det-math`](https://crates.io/crates/alice-det-math) — the
  cross-platform bit-exact `ln` / `exp` / `powf` every float path here goes
  through
- [`alice-zip`](https://crates.io/crates/alice-zip) — `law::SignalLaw`: a
  fitted law with its valid range, residual statistics, provenance and oracle
  cases, whose residual distribution the `law` feature summarises

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
