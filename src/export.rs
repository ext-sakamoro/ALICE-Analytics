//! Data Export — JSON / Prometheus テキスト形式
//!
//! `MetricSnapshot` から JSON 文字列や Prometheus exposition format への変換。
//! 外部依存なし（手動フォーマット）。

#[cfg(feature = "std")]
extern crate alloc;

#[cfg(feature = "std")]
use alloc::format;
#[cfg(feature = "std")]
use alloc::string::String;

use crate::pipeline::MetricSnapshot;

// ============================================================================
// JSON Export
// ============================================================================

/// `MetricSnapshot` を JSON 文字列に変換。
#[cfg(feature = "std")]
#[must_use]
pub fn snapshot_to_json(snapshot: &MetricSnapshot) -> String {
    format!(
        concat!(
            "{{",
            "\"name_hash\":{},",
            "\"counter\":{},",
            "\"gauge\":{},",
            "\"cardinality\":{},",
            "\"p50\":{},",
            "\"p95\":{},",
            "\"p99\":{},",
            "\"mean\":{},",
            "\"min\":{},",
            "\"max\":{},",
            "\"event_count\":{}",
            "}}"
        ),
        snapshot.name_hash,
        format_f64(snapshot.counter),
        format_f64(snapshot.gauge),
        format_f64(snapshot.cardinality),
        format_f64(snapshot.p50),
        format_f64(snapshot.p95),
        format_f64(snapshot.p99),
        format_f64(snapshot.mean),
        format_f64(snapshot.min),
        format_f64(snapshot.max),
        snapshot.event_count,
    )
}

/// 複数の `MetricSnapshot` を JSON 配列に変換。
#[cfg(feature = "std")]
#[must_use]
pub fn snapshots_to_json(snapshots: &[MetricSnapshot]) -> String {
    let mut out = String::from("[");
    for (i, snap) in snapshots.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&snapshot_to_json(snap));
    }
    out.push(']');
    out
}

// ============================================================================
// Prometheus Exposition Format
// ============================================================================

/// `MetricSnapshot` を Prometheus exposition format に変換。
///
/// ```text
/// # TYPE metric_counter counter
/// metric_counter{name_hash="12345"} 100.0
/// # TYPE metric_gauge gauge
/// metric_gauge{name_hash="12345"} 42.0
/// # TYPE metric_p99 gauge
/// metric_p99{name_hash="12345"} 99.5
/// ```
#[cfg(feature = "std")]
#[must_use]
pub fn snapshot_to_prometheus(snapshot: &MetricSnapshot, prefix: &str) -> String {
    use core::fmt::Write;
    let hash = snapshot.name_hash;
    let mut out = String::with_capacity(512);

    // Counter
    let _ = writeln!(out, "# TYPE {prefix}_counter counter");
    let _ = writeln!(
        out,
        "{prefix}_counter{{name_hash=\"{hash}\"}} {}",
        format_f64(snapshot.counter)
    );

    // Gauge
    let _ = writeln!(out, "# TYPE {prefix}_gauge gauge");
    let _ = writeln!(
        out,
        "{prefix}_gauge{{name_hash=\"{hash}\"}} {}",
        format_f64(snapshot.gauge)
    );

    // Cardinality
    let _ = writeln!(out, "# TYPE {prefix}_cardinality gauge");
    let _ = writeln!(
        out,
        "{prefix}_cardinality{{name_hash=\"{hash}\"}} {}",
        format_f64(snapshot.cardinality)
    );

    // Quantiles
    let _ = writeln!(out, "# TYPE {prefix}_latency summary");
    let _ = writeln!(
        out,
        "{prefix}_latency{{name_hash=\"{hash}\",quantile=\"0.5\"}} {}",
        format_f64(snapshot.p50)
    );
    let _ = writeln!(
        out,
        "{prefix}_latency{{name_hash=\"{hash}\",quantile=\"0.95\"}} {}",
        format_f64(snapshot.p95)
    );
    let _ = writeln!(
        out,
        "{prefix}_latency{{name_hash=\"{hash}\",quantile=\"0.99\"}} {}",
        format_f64(snapshot.p99)
    );

    // Event count
    let _ = writeln!(
        out,
        "{prefix}_events_total{{name_hash=\"{hash}\"}} {}",
        snapshot.event_count
    );

    out
}

/// 複数の `MetricSnapshot` を Prometheus 形式に変換。
#[cfg(feature = "std")]
#[must_use]
pub fn snapshots_to_prometheus(snapshots: &[MetricSnapshot], prefix: &str) -> String {
    let mut out = String::new();
    for snap in snapshots {
        out.push_str(&snapshot_to_prometheus(snap, prefix));
        out.push('\n');
    }
    out
}

// ============================================================================
// Helpers
// ============================================================================

/// f64をフォーマット（NaN/Inf対応）。
#[cfg(feature = "std")]
fn format_f64(v: f64) -> String {
    if v.is_nan() {
        "null".to_string()
    } else if v.is_infinite() {
        if v.is_sign_positive() {
            "\"+Inf\"".to_string()
        } else {
            "\"-Inf\"".to_string()
        }
    } else {
        format!("{v}")
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[cfg(feature = "std")]
mod tests {
    use super::*;

    fn make_snapshot() -> MetricSnapshot {
        MetricSnapshot {
            name_hash: 12345,
            counter: 100.0,
            gauge: 42.0,
            cardinality: 50.0,
            p50: 25.0,
            p95: 80.0,
            p99: 99.5,
            mean: 45.0,
            min: 1.0,
            max: 200.0,
            event_count: 1000,
        }
    }

    #[test]
    fn json_single_snapshot() {
        let snap = make_snapshot();
        let json = snapshot_to_json(&snap);
        assert!(json.contains("\"name_hash\":12345"));
        assert!(json.contains("\"counter\":100"));
        assert!(json.contains("\"event_count\":1000"));
        assert!(json.starts_with('{'));
        assert!(json.ends_with('}'));
    }

    #[test]
    fn json_array() {
        let snaps = vec![make_snapshot(), make_snapshot()];
        let json = snapshots_to_json(&snaps);
        assert!(json.starts_with('['));
        assert!(json.ends_with(']'));
        // 2つのオブジェクトが含まれる
        assert_eq!(json.matches("name_hash").count(), 2);
    }

    #[test]
    fn json_empty_array() {
        let json = snapshots_to_json(&[]);
        assert_eq!(json, "[]");
    }

    #[test]
    fn prometheus_single() {
        let snap = make_snapshot();
        let prom = snapshot_to_prometheus(&snap, "alice");
        assert!(prom.contains("# TYPE alice_counter counter"));
        assert!(prom.contains("alice_counter{name_hash=\"12345\"} 100"));
        assert!(prom.contains("# TYPE alice_gauge gauge"));
        assert!(prom.contains("quantile=\"0.99\""));
        assert!(prom.contains("alice_events_total"));
    }

    #[test]
    fn prometheus_multiple() {
        let snaps = vec![make_snapshot(), make_snapshot()];
        let prom = snapshots_to_prometheus(&snaps, "test");
        // 2つ分のメトリクス
        assert_eq!(prom.matches("# TYPE test_counter counter").count(), 2);
    }

    #[test]
    fn json_nan_handling() {
        let mut snap = make_snapshot();
        snap.p99 = f64::NAN;
        let json = snapshot_to_json(&snap);
        assert!(json.contains("null"));
    }

    #[test]
    fn json_inf_handling() {
        let mut snap = make_snapshot();
        snap.max = f64::INFINITY;
        let json = snapshot_to_json(&snap);
        assert!(json.contains("+Inf"));
    }

    #[test]
    fn prometheus_prefix() {
        let snap = make_snapshot();
        let prom = snapshot_to_prometheus(&snap, "my_app");
        assert!(prom.contains("my_app_counter"));
        assert!(prom.contains("my_app_gauge"));
        assert!(prom.contains("my_app_latency"));
    }

    #[test]
    fn format_f64_normal() {
        assert_eq!(format_f64(42.5), "42.5");
        assert_eq!(format_f64(0.0), "0");
    }
}
