//! ALICE-Analytics — High-Performance Telemetry & Statistical Estimation
//!
//! Probabilistic data structures for streaming analytics with mathematical
//! error guarantees and minimal memory footprint.
//!
//! # Modules
//!
//! - [`sketch`]: HyperLogLog / DDSketch / Count-Min / Heavy Hitters
//! - [`stats`]: Percentile rank, IQR, covariance, Welford streaming stats
//! - [`window`]: Tumbling / sliding / hierarchical window aggregates
//! - [`anomaly`]: MAD / Z-Score / EWMA / composite anomaly detectors
//! - [`privacy`]: Laplace noise / Randomized Response / RAPPOR / privacy budget
//! - [`pipeline`]: Lock-free metric aggregation pipeline
//! - [`export`]: JSON / Prometheus export of metric snapshots
//! - [`streaming_ops`]: Streaming aggregation operators

#![cfg_attr(not(feature = "std"), no_std)]

pub mod anomaly;
pub mod export;
pub mod pipeline;
pub mod privacy;
pub mod sketch;
pub mod stats;
pub mod streaming_ops;
pub mod window;
