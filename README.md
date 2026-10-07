# ALICE-Analytics

[日本語](README_JP.md)

Probabilistic data structures and streaming statistics with stated error
bounds: HyperLogLog cardinality, DDSketch quantiles with a relative-error
guarantee, Count-Min frequencies and heavy hitters, streaming moments /
covariance / regression, windowed aggregates, anomaly detectors and local
differential privacy. The core is `no_std` with fixed-size, stack-allocated
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

Without `std` the float functions come from `libm` (results may differ from the
platform library in the last ulp), and the privacy mechanisms take an explicit
seed. CI builds the library for `thumbv7em-none-eabihf` with and without `law`.

## Modules

| Module | Contents |
|--------|----------|
| `sketch` | `HyperLogLog10/12/14/16`, `DDSketch128` … `DDSketch2048`, `CountMinSketch1024x5` / `2048x7` / `4096x5`, `HeavyHitters5/10/20`, `FnvHasher`, `Mergeable` |
| `stats` | `StreamingStats` (mean / variance / skewness / kurtosis), `CovarianceMatrix`, `quantile_sorted`, `percentile_rank`, `iqr` |
| `window` | `TumblingWindow`, `SlidingWindow`, `HierarchicalRollup` |
| `streaming_ops` | `SimpleMovingAverage`, `ExponentialMovingAverage`, `ChangeRate`, `LinearRegression`, `LinearRegressionFull` |
| `anomaly` | `StreamingMedian`, `MadDetector`, `ZScoreDetector`, `EwmaDetector`, `CompositeDetector` |
| `privacy` | `LaplaceNoise`, `RandomizedResponse`, `Rappor`, `PrivacyBudget`, `PrivateAggregator`, `XorShift64` |
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

## Minimum supported Rust version

`rust-version = "1.87"` (measured: 1.86 does not build the library). CI checks
the library on 1.87 with default and with all features; the development
toolchain is pinned in `rust-toolchain.toml`.

## Building and testing

```sh
cargo test --all-features
cargo build --lib --no-default-features --features law --target thumbv7em-none-eabihf
scripts/preflight.sh          # every CI gate that runs locally
scripts/preflight.sh --quick  # static checks, clippy, builds, docs, cargo test --lib
```

## Related crates

- [`alice-zip`](https://crates.io/crates/alice-zip) — `law::SignalLaw`: a
  fitted law with its valid range, residual statistics, provenance and oracle
  cases, whose residual distribution the `law` feature summarises

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
