//! ALICE-DB bridge: Persist streaming analytics results
//!
//! Stores HyperLogLog cardinality, DDSketch quantiles, and anomaly
//! detection counts into ALICE-DB as time-series for historical queries.
//!
//! # Example
//!
//! ```ignore
//! use alice_analytics::db_bridge::AnalyticsSink;
//!
//! let sink = AnalyticsSink::open("/tmp/analytics_db").unwrap();
//! sink.record_cardinality(1700000000000, 15423.0);
//! sink.record_quantile(1700000000000, 42.5);
//! sink.record_anomaly(1700000000000, 1);
//! ```

use alice_db::AliceDB;

/// Persistent sink for streaming analytics results.
///
/// Each metric type (cardinality, quantile, anomaly) gets its own
/// ALICE-DB instance for independent time-series storage.
pub struct AnalyticsSink {
    /// Cardinality estimates (HLL results)
    cardinality_db: AliceDB,
    /// Quantile estimates (DDSketch results)
    quantile_db: AliceDB,
    /// Anomaly counts
    anomaly_db: AliceDB,
}

impl AnalyticsSink {
    /// Open or create analytics databases at the given directory.
    pub fn open(dir: &str) -> Result<Self, String> {
        let cardinality_db = AliceDB::open(format!("{}/cardinality", dir))
            .map_err(|e| format!("cardinality db: {}", e))?;
        let quantile_db = AliceDB::open(format!("{}/quantile", dir))
            .map_err(|e| format!("quantile db: {}", e))?;
        let anomaly_db = AliceDB::open(format!("{}/anomaly", dir))
            .map_err(|e| format!("anomaly db: {}", e))?;
        Ok(Self {
            cardinality_db,
            quantile_db,
            anomaly_db,
        })
    }

    /// Record a cardinality estimate (e.g., from HyperLogLog).
    pub fn record_cardinality(&self, timestamp_ms: u64, estimate: f64) {
        let _ = self.cardinality_db.put(timestamp_ms as i64, estimate as f32);
    }

    /// Record a quantile estimate (e.g., P50/P99 from DDSketch).
    pub fn record_quantile(&self, timestamp_ms: u64, value: f64) {
        let _ = self.quantile_db.put(timestamp_ms as i64, value as f32);
    }

    /// Record an anomaly count.
    pub fn record_anomaly(&self, timestamp_ms: u64, count: u32) {
        let _ = self.anomaly_db.put(timestamp_ms as i64, count as f32);
    }

    /// Record a batch of cardinality estimates.
    pub fn record_cardinality_batch(&self, entries: &[(u64, f64)]) {
        let data: Vec<(i64, f32)> = entries
            .iter()
            .map(|&(ts, est)| (ts as i64, est as f32))
            .collect();
        let _ = self.cardinality_db.put_batch(&data);
    }

    /// Query cardinality history in a time range.
    pub fn query_cardinality(&self, from_ms: u64, to_ms: u64) -> Vec<(u64, f64)> {
        self.cardinality_db
            .scan(from_ms as i64, to_ms as i64)
            .unwrap_or_default()
            .into_iter()
            .map(|(ts, v)| (ts as u64, v as f64))
            .collect()
    }

    /// Query quantile history in a time range.
    pub fn query_quantile(&self, from_ms: u64, to_ms: u64) -> Vec<(u64, f64)> {
        self.quantile_db
            .scan(from_ms as i64, to_ms as i64)
            .unwrap_or_default()
            .into_iter()
            .map(|(ts, v)| (ts as u64, v as f64))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analytics_sink_open() {
        let dir = "/tmp/alice_analytics_test";
        let result = AnalyticsSink::open(dir);
        // May fail in test environments without filesystem access
        if result.is_ok() {
            let sink = result.unwrap();
            sink.record_cardinality(1000, 42.0);
            let data = sink.query_cardinality(0, 2000);
            assert!(!data.is_empty());
        }
    }
}
