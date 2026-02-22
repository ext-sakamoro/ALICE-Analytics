# Changelog

All notable changes to ALICE-Analytics will be documented in this file.

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
