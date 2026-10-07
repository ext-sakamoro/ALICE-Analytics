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
//! - [`privacy`]: Laplace noise / Randomized Response / RAPPOR / privacy budget
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
