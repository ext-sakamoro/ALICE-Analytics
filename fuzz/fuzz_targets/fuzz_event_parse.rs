//! Fuzz target: `MetricPipeline::submit` + `flush` が任意 MetricEvent 列で panic しないことを検証
//!
//! 攻撃者制御 telemetry (name_hash / type / value / timestamp) の pipeline 処理で:
//! - value = NaN / ±Inf / subnormal による DDSketch 内部の log() / log2() 演算 panic
//! - MetricType タグ変換の分岐漏れによる unreachable panic
//! - ring buffer 満杯状態での push/pop 順序 race panic (single-thread なので stateful のみ)
//! - name_hash 集中 (%SLOTS 衝突多発) 時の slot 上書き panic
//!
//! を全て有限時間で panic なく完了することを保証する
//!
//! canonical CI template [[reference_alice_ci_canonical_template]] 準拠

#![no_main]

use alice_analytics::pipeline::{MetricEvent, MetricPipeline, MetricType};
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct RawEvent {
    name_hash: u64,
    /// 0..=3 に mod して MetricType に mapping
    type_tag: u8,
    /// f64 bits (NaN / Inf / subnormal 含む攻撃者制御)
    value_bits: u64,
    timestamp: u64,
}

#[derive(Debug, Arbitrary)]
struct Input {
    /// DDSketch alpha (0 / negative / huge は insert 側の f64 演算 panic 温床)
    alpha_bits: u64,
    events: Vec<RawEvent>,
}

fn to_metric_type(tag: u8) -> MetricType {
    match tag % 4 {
        0 => MetricType::Counter,
        1 => MetricType::Gauge,
        2 => MetricType::Histogram,
        _ => MetricType::Unique,
    }
}

fuzz_target!(|input: Input| {
    // fuzzer timeout 回避 (4096 event 上限)
    if input.events.len() > 4096 {
        return;
    }

    // alpha は [1e-6, 0.5] の実用 range に強制 (0 / negative / huge は API 誤用)
    let alpha_raw = f64::from_bits(input.alpha_bits);
    let alpha = if alpha_raw.is_finite() && (1e-6..=0.5).contains(&alpha_raw) {
        alpha_raw
    } else {
        0.01
    };

    let mut pipeline: MetricPipeline<64, 256> = MetricPipeline::new(alpha);
    for raw in &input.events {
        let event = MetricEvent {
            name_hash: raw.name_hash,
            metric_type: to_metric_type(raw.type_tag),
            value: f64::from_bits(raw.value_bits),
            timestamp: raw.timestamp,
        };
        // submit は queue 満杯で false を返すが panic せず
        let _ = pipeline.submit(event);
    }

    // 全 event を処理 (DDSketch / counter / gauge 内部の演算 panic をここで検知)
    pipeline.flush();

    // slot 探索も panic せず
    for raw in &input.events {
        let _ = pipeline.get_slot(raw.name_hash);
    }
    let _ = pipeline.total_events();
});
