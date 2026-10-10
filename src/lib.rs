//! ALICE-Analytics — High-Performance Telemetry & Statistical Estimation
//!
//! Probabilistic data structures for streaming analytics with mathematical
//! error guarantees and minimal memory footprint.
//!
//! # Modules
//!
//! - [`sketch`]: `HyperLogLog` / `DDSketch` / Count-Min / Heavy Hitters
//! - [`stats`]: Percentile rank, IQR, covariance, Welford streaming stats
//! - [`window`]: Tumbling / sliding / hierarchical window aggregates
//! - [`anomaly`]: MAD / Z-Score / EWMA / composite anomaly detectors
//! - [`privacy`]: **deprecated, not private** (see its Security section); use `alice_crypto::dp`
//! - [`pipeline`]: Lock-free metric aggregation pipeline
//! - [`export`]: JSON / Prometheus export of metric snapshots
//! - [`streaming_ops`]: Streaming aggregation operators
//! - `law` (feature `law`): residual distribution of a fitted
//!   `alice_zip::law::SignalLaw`, summarised with a `DDSketch`
//!
//! # Example
//!
//! ```rust
//! use alice_analytics::sketch::{DDSketch, HyperLogLog};
//!
//! // quantiles with a relative-error guarantee of 1 %
//! let mut latency = DDSketch::new(0.01);
//! for ms in 1..=1000 {
//!     latency.insert(f64::from(ms));
//! }
//! let p99 = latency.quantile(0.99);
//! assert!((p99 - 990.0).abs() <= 0.01 * 990.0);
//!
//! // distinct count in 16 KiB of registers
//! let mut users = HyperLogLog::new();
//! for id in 0..10_000_u64 {
//!     users.insert(&id);
//! }
//! let n = users.cardinality();
//! assert!((n - 10_000.0).abs() < 0.05 * 10_000.0);
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

pub(crate) mod math;

pub mod anomaly;
pub mod export;
pub mod pipeline;
pub mod privacy;
pub mod sketch;
pub mod stats;
pub mod streaming_ops;
pub mod window;

#[cfg(feature = "law")]
pub mod law;

/// Identifier of the arithmetic every estimate in this crate is computed with
///
/// Quantiles, cardinalities, anomaly scores and privacy noise all read
/// transcendentals (`ln` / `exp` / `powf`), and a transcendental is only
/// reproducible if its implementation is pinned too: the platform `libm`
/// differs in the last ulp between macOS, glibc, MSVC and wasm, and one ulp of
/// `ln` moves a `DDSketch` bin boundary, which moves the reported quantile.
/// This crate therefore routes every transcendental through
/// [`alice_det_math`], and re-exports that crate's identifier of its own
/// numeric behaviour here.
///
/// Record it next to any estimate that is stored, transmitted or merged. Two
/// results are comparable as numbers only if they were produced under the same
/// identifier; when it differs, the two were computed by different arithmetic
/// and agreement between them is not guaranteed at the bit level, however
/// close the values look.
///
/// Pinned as hex by `tests/determinism_golden.rs`, so a change in the
/// arithmetic cannot reach a release unnoticed. With the `law` feature,
/// `law::ResidualSummary::law_id` mixes this identifier with the law it
/// summarises, giving a single value that names both (not linked here: the
/// module is behind that feature, so the link would not resolve in a build
/// without it).
pub use alice_det_math::SEMANTICS_ID;
