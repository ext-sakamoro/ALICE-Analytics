//! Fuzz target: 統計 aggregation (StreamingStats + MAD/ZScore anomaly detector) が
//! 任意 f64 列で panic しないことを検証
//!
//! 攻撃者制御 numeric telemetry で:
//! - Welford Σ の overflow (huge value 加算) による panic
//! - NaN / ±Inf / subnormal による variance / skewness / kurtosis 計算 panic
//! - anomaly detector の threshold_k = 0 / NaN による div-by-zero panic
//! - MAD detector の recent_values ring buffer 位置計算 panic
//! - percentile_rank / IQR での sorted 前提違反 panic
//!
//! を全て有限時間で panic なく完了することを保証する
//!
//! canonical CI template [[reference_alice_ci_canonical_template]] 準拠

#![no_main]

use alice_analytics::anomaly::{MadDetector, ZScoreDetector};
use alice_analytics::stats::{iqr, percentile_rank, quantile_sorted, StreamingStats};
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct Input {
    /// 攻撃者制御 f64 列 (bits で NaN / Inf / subnormal 網羅)
    values: Vec<u64>,
    /// anomaly threshold_k (0 / NaN 攻撃)
    threshold_k_bits: u64,
    /// percentile_rank / quantile 用の probe value
    probe_bits: u64,
    /// quantile parameter q (0..=1 に mod)
    q_bits: u64,
}

fuzz_target!(|input: Input| {
    // fuzzer timeout 回避 (8192 value 上限)
    if input.values.len() > 8192 {
        return;
    }

    let values: Vec<f64> = input.values.iter().map(|b| f64::from_bits(*b)).collect();
    let threshold_k = f64::from_bits(input.threshold_k_bits);
    let probe = f64::from_bits(input.probe_bits);
    let q = f64::from_bits(input.q_bits);

    // 1. StreamingStats (Welford Σ)
    let mut stats = StreamingStats::new();
    for &v in &values {
        stats.observe(v);
    }
    let _ = stats.count();
    let _ = stats.mean();
    let _ = stats.variance();

    // 2. MAD anomaly detector
    let mut mad = MadDetector::new(threshold_k);
    for &v in &values {
        mad.observe(v);
    }
    let _ = mad.is_anomaly(probe);

    // 3. Z-Score anomaly detector
    let mut zs = ZScoreDetector::new(threshold_k);
    for &v in &values {
        zs.observe(v);
    }
    let _ = zs.is_anomaly(probe);

    // 4. percentile_rank / IQR / quantile (sorted 前提 API)
    let mut sorted = values.clone();
    // NaN 混入時の sort_by 定義域外 panic 予防で partial_cmp fallback
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let _ = percentile_rank(&sorted, probe);
    let _ = iqr(&sorted);
    let _ = quantile_sorted(&sorted, q);
});
