//! Fuzz target: 確率的 sketch (HyperLogLog / DDSketch) の insert + merge + query が
//! 任意入力で panic しないことを検証
//!
//! 攻撃者制御 telemetry で:
//! - HyperLogLog insert_bytes の huge slice / empty slice による bucket 計算 panic
//! - HyperLogLog merge (Mergeable trait) の register サイズ不整合 panic (同型なので index OOB は起きない想定、検証)
//! - DDSketch insert の NaN / ±Inf / 0.0 / negative による log() 演算 panic
//! - DDSketch bucket_index の f64 → i32 → usize cast panic (huge value / subnormal)
//! - DDSketch quantile(q) の q = NaN / <0 / >1 による OOB panic
//!
//! を全て有限時間で panic なく完了することを保証する
//!
//! canonical CI template [[reference_alice_ci_canonical_template]] 準拠

#![no_main]

use alice_analytics::sketch::{DDSketch256, HyperLogLog10, Mergeable};
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct Input {
    /// HyperLogLog insert 対象 (byte slice 群、attacker 制御)
    hll_items_a: Vec<Vec<u8>>,
    hll_items_b: Vec<Vec<u8>>,
    /// DDSketch insert 対象 (f64 bits で NaN / Inf / negative 網羅)
    dd_values_a: Vec<u64>,
    dd_values_b: Vec<u64>,
    /// DDSketch alpha (constructor 引数、0 / 1 / negative は API 誤用だが panic 避ける)
    alpha_bits: u64,
    /// quantile probe (q = NaN / <0 / >1 攻撃)
    q_bits_1: u64,
    q_bits_2: u64,
    q_bits_3: u64,
}

fuzz_target!(|input: Input| {
    // fuzzer timeout 回避
    if input.hll_items_a.len() > 1024
        || input.hll_items_b.len() > 1024
        || input.dd_values_a.len() > 1024
        || input.dd_values_b.len() > 1024
    {
        return;
    }
    // 個々の byte slice も上限 (16 KB)
    for slice in input.hll_items_a.iter().chain(input.hll_items_b.iter()) {
        if slice.len() > 16 * 1024 {
            return;
        }
    }

    // 1. HyperLogLog insert + merge + cardinality
    let mut hll_a = HyperLogLog10::new();
    let mut hll_b = HyperLogLog10::new();
    for item in &input.hll_items_a {
        hll_a.insert_bytes(item);
    }
    for item in &input.hll_items_b {
        hll_b.insert_bytes(item);
    }
    // merge (Mergeable trait) は sibling sketch の register max 統合
    hll_a.merge(&hll_b);
    let _ = hll_a.cardinality();

    // 2. DDSketch insert + merge + quantile
    let alpha_raw = f64::from_bits(input.alpha_bits);
    let alpha = if alpha_raw.is_finite() && (1e-6..=0.5).contains(&alpha_raw) {
        alpha_raw
    } else {
        0.01
    };
    let mut dd_a = DDSketch256::new(alpha);
    let mut dd_b = DDSketch256::new(alpha);
    for bits in &input.dd_values_a {
        dd_a.insert(f64::from_bits(*bits));
    }
    for bits in &input.dd_values_b {
        dd_b.insert(f64::from_bits(*bits));
    }
    dd_a.merge(&dd_b);

    // quantile 3 サンプル (NaN / <0 / >1 攻撃を bits で網羅)
    let _ = dd_a.quantile(f64::from_bits(input.q_bits_1));
    let _ = dd_a.quantile(f64::from_bits(input.q_bits_2));
    let _ = dd_a.quantile(f64::from_bits(input.q_bits_3));
});
