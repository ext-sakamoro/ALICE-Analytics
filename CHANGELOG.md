# Changelog

All notable changes to ALICE-Analytics will be documented in this file.

## [Unreleased]

### Fixed
- **`no_std` build が一度も通っていなかった** (`--no-default-features` で 38 error: f64 `mul_add` / `sqrt` / `ln` / `exp` / `powf` / `ceil` / `floor` / `round` + `std::f64::consts::LN_2`、6 module) — `src/math.rs` の `FloatExt` trait (`libm` 委譲、`std` 時は不使用) と `core::f64::consts` で修正、`export` の `MetricSnapshot` import を `std` gate host / bare-metal `thumbv7em-none-eabihf` / `x86_64` cross で build + clippy を確認

### Changed
- `libm = "0.2"` を依存に追加 (`no_std` build のみ使用、std build の依存 0 → 1 だが std 時は未使用) `no_std` の `sqrt` / `exp` 等は platform libm と最終 ulp で異なりうる (`src/math.rs` doc)
- CI: `no_std` job (host + thumbv7em + clippy) / `feature-powerset` (cargo-hack depth 2) / doc `-D warnings` を追加、`rust-toolchain.toml` (1.98.1 + thumbv7em target) 新設

## [0.1.0] - 2026-02-23

### Added
- `HyperLogLog` — cardinality estimation (10/12/14/16-bit precision variants)
- `DDSketch` — relative-error quantile estimation (128/256/512/1024/2048-bin variants)
- `CountMinSketch` — frequency estimation with configurable width and depth
- `HeavyHitters` — approximate top-K tracking (5/10/20 variants)
- `LaplaceNoise` / `RandomizedResponse` / `Rappor` — local differential privacy
- `PrivacyBudget` / `PrivateAggregator` — privacy budget tracking
- `MadDetector` / `ZScoreDetector` / `EwmaDetector` / `CompositeDetector` — streaming anomaly detection
- `MetricPipeline` — event-driven metric aggregation with ring buffer
- `MetricRegistry` — named metric registration and lookup
- `Mergeable` trait — all sketches support distributed merge
- `no_std` compatible core
- 44 tests (37 unit + 7 doc-test)
