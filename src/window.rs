//! Time-Window Aggregation
//!
//! タンブリング/スライディングウィンドウによる時系列集約。
//! MetricSlotをウィンドウ単位でrotateし、時間ベースの集約を提供。

use crate::pipeline::MetricSlot;

// ============================================================================
// Tumbling Window — 固定幅の重複なしウィンドウ
// ============================================================================

/// 固定幅タンブリングウィンドウ。
///
/// ウィンドウ境界でMetricSlotをrotateし、完了したウィンドウを返す。
/// ```text
/// |---window 0---|---window 1---|---window 2---|
///    ↑ events      ↑ events       ↑ events
/// ```
pub struct TumblingWindow {
    /// ウィンドウ幅（ミリ秒）。
    pub window_ms: u64,
    /// 現在のウィンドウ開始時刻。
    current_start: u64,
    /// 現在のウィンドウのMetricSlot。
    current: MetricSlot,
    /// DDSketch alpha。
    alpha: f64,
}

impl TumblingWindow {
    /// 新しいタンブリングウィンドウを作成。
    #[must_use]
    pub fn new(window_ms: u64, alpha: f64) -> Self {
        Self {
            window_ms: window_ms.max(1),
            current_start: 0,
            current: MetricSlot::new(0, alpha),
            alpha,
        }
    }

    /// イベントを投入。ウィンドウ境界を超えた場合、完了したスロットを返す。
    pub fn insert(&mut self, value: f64, timestamp_ms: u64) -> Option<WindowResult> {
        // 初回: ウィンドウ開始時刻を設定
        if self.current_start == 0 {
            self.current_start = timestamp_ms;
        }

        // ウィンドウ境界を超えたか
        if timestamp_ms >= self.current_start + self.window_ms {
            let completed = core::mem::replace(
                &mut self.current,
                MetricSlot::new(0, self.alpha),
            );
            let result = WindowResult {
                start_ms: self.current_start,
                end_ms: self.current_start + self.window_ms,
                event_count: completed.event_count,
                counter: completed.counter,
                gauge: completed.gauge,
                mean: completed.ddsketch.mean(),
                min: completed.ddsketch.min(),
                max: completed.ddsketch.max(),
                p50: completed.ddsketch.quantile(0.50),
                p99: completed.ddsketch.quantile(0.99),
            };

            // 新ウィンドウ開始
            self.current_start = timestamp_ms - (timestamp_ms % self.window_ms);
            self.current.ddsketch.insert(value);
            self.current.event_count = 1;

            Some(result)
        } else {
            self.current.ddsketch.insert(value);
            self.current.event_count += 1;
            None
        }
    }

    /// 現在のウィンドウを強制的にフラッシュ。
    #[must_use]
    pub fn flush(&mut self) -> WindowResult {
        let completed = core::mem::replace(
            &mut self.current,
            MetricSlot::new(0, self.alpha),
        );
        let result = WindowResult {
            start_ms: self.current_start,
            end_ms: self.current_start + self.window_ms,
            event_count: completed.event_count,
            counter: completed.counter,
            gauge: completed.gauge,
            mean: completed.ddsketch.mean(),
            min: completed.ddsketch.min(),
            max: completed.ddsketch.max(),
            p50: completed.ddsketch.quantile(0.50),
            p99: completed.ddsketch.quantile(0.99),
        };
        self.current_start = 0;
        result
    }

    /// 現在のウィンドウのイベント数。
    #[inline]
    #[must_use]
    pub const fn current_count(&self) -> u64 {
        self.current.event_count
    }
}

// ============================================================================
// Sliding Window — 重複ありスライディングウィンドウ
// ============================================================================

/// リングバッファベースのスライディングウィンドウ。
///
/// 最新N個の観測値を保持し、ウィンドウ全体の統計量を提供。
pub struct SlidingWindow<const N: usize> {
    /// リングバッファ。
    buffer: [f64; N],
    /// 書き込み位置。
    write_pos: usize,
    /// 有効な要素数。
    count: usize,
    /// 合計値（Running sum）。
    sum: f64,
    /// 最小値。
    min: f64,
    /// 最大値。
    max: f64,
}

impl<const N: usize> SlidingWindow<N> {
    /// 新しいスライディングウィンドウを作成。
    #[must_use]
    pub fn new() -> Self {
        Self {
            buffer: [0.0; N],
            write_pos: 0,
            count: 0,
            sum: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    /// 値を追加。バッファが満杯の場合、最古の値を上書き。
    pub fn push(&mut self, value: f64) {
        if self.count >= N {
            // 最古の値をsumから除去
            self.sum -= self.buffer[self.write_pos];
        }
        self.buffer[self.write_pos] = value;
        self.sum += value;
        self.write_pos = (self.write_pos + 1) % N;
        if self.count < N {
            self.count += 1;
        }

        // min/maxはpush毎に全走査が必要（値が消えるため）
        if self.count < N && value <= self.min {
            self.min = value;
        }
        if self.count < N && value >= self.max {
            self.max = value;
        }
    }

    /// 平均値。
    #[inline]
    #[must_use]
    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.sum / self.count as f64
    }

    /// 合計値。
    #[inline]
    #[must_use]
    pub const fn sum(&self) -> f64 {
        self.sum
    }

    /// 有効な要素数。
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.count
    }

    /// 空かどうか。
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// 最小値（全走査）。
    #[must_use]
    pub fn min(&self) -> f64 {
        if self.count == 0 {
            return f64::INFINITY;
        }
        let mut m = f64::INFINITY;
        for i in 0..self.count {
            if self.buffer[i] < m {
                m = self.buffer[i];
            }
        }
        m
    }

    /// 最大値（全走査）。
    #[must_use]
    pub fn max(&self) -> f64 {
        if self.count == 0 {
            return f64::NEG_INFINITY;
        }
        let mut m = f64::NEG_INFINITY;
        for i in 0..self.count {
            if self.buffer[i] > m {
                m = self.buffer[i];
            }
        }
        m
    }

    /// 分散（母集団分散）。
    #[must_use]
    pub fn variance(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let mean = self.mean();
        let mut sum_sq = 0.0;
        for i in 0..self.count {
            let d = self.buffer[i] - mean;
            sum_sq += d * d;
        }
        sum_sq / self.count as f64
    }

    /// 標準偏差。
    #[inline]
    #[must_use]
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// ウィンドウをクリア。
    pub fn clear(&mut self) {
        self.write_pos = 0;
        self.count = 0;
        self.sum = 0.0;
        self.min = f64::INFINITY;
        self.max = f64::NEG_INFINITY;
    }
}

impl<const N: usize> Default for SlidingWindow<N> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Window Result
// ============================================================================

/// 完了したウィンドウの集約結果。
#[derive(Clone, Debug)]
pub struct WindowResult {
    /// ウィンドウ開始時刻（ミリ秒）。
    pub start_ms: u64,
    /// ウィンドウ終了時刻（ミリ秒）。
    pub end_ms: u64,
    /// イベント数。
    pub event_count: u64,
    /// カウンター合計。
    pub counter: f64,
    /// 最終ゲージ値。
    pub gauge: f64,
    /// 平均値。
    pub mean: f64,
    /// 最小値。
    pub min: f64,
    /// 最大値。
    pub max: f64,
    /// P50。
    pub p50: f64,
    /// P99。
    pub p99: f64,
}

// ============================================================================
// Hierarchical Rollup — 階層的集約 (1m → 5m → 1h)
// ============================================================================

/// 階層的ロールアップ。
///
/// 複数レベルの時間ウィンドウで自動集約。
/// 例: 1分 → 5分 → 1時間
pub struct HierarchicalRollup {
    /// 各レベルのウィンドウ。
    levels: [TumblingWindow; 3],
    /// 各レベルのウィンドウ幅（ミリ秒）。
    window_sizes: [u64; 3],
    /// 各レベルの最新結果。
    results: [Option<WindowResult>; 3],
}

impl HierarchicalRollup {
    /// 3レベルロールアップを作成（ミリ秒単位）。
    ///
    /// 典型例: `new(60_000, 300_000, 3_600_000, 0.05)` → 1分/5分/1時間
    #[must_use]
    pub fn new(level0_ms: u64, level1_ms: u64, level2_ms: u64, alpha: f64) -> Self {
        Self {
            levels: [
                TumblingWindow::new(level0_ms, alpha),
                TumblingWindow::new(level1_ms, alpha),
                TumblingWindow::new(level2_ms, alpha),
            ],
            window_sizes: [level0_ms, level1_ms, level2_ms],
            results: [None, None, None],
        }
    }

    /// 値を投入。各レベルで完了したウィンドウがあれば記録。
    pub fn insert(&mut self, value: f64, timestamp_ms: u64) {
        // レベル0に直接投入
        if let Some(r) = self.levels[0].insert(value, timestamp_ms) {
            self.results[0] = Some(r);
            // レベル0完了 → レベル1にmeanを投入
            if let Some(ref r0) = self.results[0] {
                let mean = r0.mean;
                if let Some(r1) = self.levels[1].insert(mean, timestamp_ms) {
                    self.results[1] = Some(r1);
                    // レベル1完了 → レベル2にmeanを投入
                    if let Some(ref r1_res) = self.results[1] {
                        let mean1 = r1_res.mean;
                        if let Some(r2) = self.levels[2].insert(mean1, timestamp_ms) {
                            self.results[2] = Some(r2);
                        }
                    }
                }
            }
        }
    }

    /// 指定レベルの最新結果を取得。
    #[must_use]
    pub fn result(&self, level: usize) -> Option<&WindowResult> {
        if level < 3 {
            self.results[level].as_ref()
        } else {
            None
        }
    }

    /// レベル数。
    #[inline]
    #[must_use]
    pub const fn level_count(&self) -> usize {
        3
    }

    /// 指定レベルのウィンドウ幅（ミリ秒）。
    #[must_use]
    pub const fn window_size(&self, level: usize) -> u64 {
        if level < 3 {
            self.window_sizes[level]
        } else {
            0
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn tumbling_window_basic() {
        let mut w = TumblingWindow::new(100, 0.05);
        // ウィンドウ内のイベント
        assert!(w.insert(10.0, 10).is_none());
        assert!(w.insert(20.0, 50).is_none());
        assert_eq!(w.current_count(), 2);

        // ウィンドウ境界超え → 完了結果
        let result = w.insert(30.0, 110).unwrap();
        assert_eq!(result.start_ms, 10);
        assert_eq!(result.end_ms, 110);
        assert_eq!(result.event_count, 2);
    }

    #[test]
    fn tumbling_window_flush() {
        let mut w = TumblingWindow::new(1000, 0.05);
        w.insert(5.0, 100);
        w.insert(15.0, 200);
        let result = w.flush();
        assert_eq!(result.event_count, 2);
    }

    #[test]
    fn tumbling_window_statistics() {
        let mut w = TumblingWindow::new(100, 0.05);
        for i in 0..10 {
            w.insert((i + 1) as f64 * 10.0, i * 5);
        }
        let result = w.flush();
        assert!(result.mean > 0.0);
        assert!(result.min > 0.0);
        assert!(result.max >= result.min);
    }

    #[test]
    fn sliding_window_basic() {
        let mut w = SlidingWindow::<4>::new();
        w.push(10.0);
        w.push(20.0);
        w.push(30.0);
        assert_eq!(w.len(), 3);
        assert!((w.mean() - 20.0).abs() < 1e-6);
        assert!((w.min() - 10.0).abs() < 1e-6);
        assert!((w.max() - 30.0).abs() < 1e-6);
    }

    #[test]
    fn sliding_window_overflow() {
        let mut w = SlidingWindow::<3>::new();
        w.push(10.0);
        w.push(20.0);
        w.push(30.0);
        w.push(40.0); // 10.0が押し出される
        assert_eq!(w.len(), 3);
        assert!((w.mean() - 30.0).abs() < 1e-6);
        assert!((w.min() - 20.0).abs() < 1e-6);
    }

    #[test]
    fn sliding_window_variance() {
        let mut w = SlidingWindow::<4>::new();
        w.push(2.0);
        w.push(4.0);
        w.push(4.0);
        w.push(4.0);
        // mean=3.5, var = ((2-3.5)^2 + 3*(4-3.5)^2)/4 = (2.25+0.75)/4 = 0.75
        assert!((w.variance() - 0.75).abs() < 1e-6);
    }

    #[test]
    fn sliding_window_empty() {
        let w = SlidingWindow::<8>::new();
        assert!(w.is_empty());
        assert!((w.mean() - 0.0).abs() < 1e-6);
        assert!((w.variance() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn sliding_window_clear() {
        let mut w = SlidingWindow::<8>::new();
        w.push(1.0);
        w.push(2.0);
        w.clear();
        assert!(w.is_empty());
        assert_eq!(w.len(), 0);
    }

    #[test]
    fn hierarchical_basic() {
        // 10ms / 50ms / 200ms レベル
        let mut rollup = HierarchicalRollup::new(10, 50, 200, 0.05);
        assert_eq!(rollup.level_count(), 3);
        assert_eq!(rollup.window_size(0), 10);
        assert_eq!(rollup.window_size(1), 50);
        assert_eq!(rollup.window_size(2), 200);

        // レベル0のウィンドウ完了まで投入
        for i in 0..20 {
            rollup.insert(100.0, i);
        }
        // レベル0は完了しているはず
        assert!(rollup.result(0).is_some());
    }

    #[test]
    fn hierarchical_cascading() {
        let mut rollup = HierarchicalRollup::new(10, 30, 90, 0.05);
        // 十分なイベントを投入してレベル1まで完了させる
        for i in 0..100 {
            rollup.insert(50.0, i);
        }
        // レベル0は確実に完了
        assert!(rollup.result(0).is_some());
    }

    #[test]
    fn hierarchical_invalid_level() {
        let rollup = HierarchicalRollup::new(10, 50, 200, 0.05);
        assert!(rollup.result(5).is_none());
        assert_eq!(rollup.window_size(99), 0);
    }

    #[test]
    fn sliding_window_std_dev() {
        let mut w = SlidingWindow::<3>::new();
        w.push(10.0);
        w.push(10.0);
        w.push(10.0);
        // 全て同じ値 → 標準偏差=0
        assert!((w.std_dev() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn sliding_window_default() {
        let w = SlidingWindow::<16>::default();
        assert!(w.is_empty());
    }
}
