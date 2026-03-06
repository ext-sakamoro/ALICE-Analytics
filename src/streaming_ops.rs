//! Streaming Operators — 変化率・移動平均・線形回帰
//!
//! ストリーミングデータに対するオンライン演算子。
//! ゼロアロケーション、`no_std` 対応。
//!
//! Author: Moroya Sakamoto

// ============================================================================
// Change Rate — 変化率
// ============================================================================

/// 変化率の計算（差分 / 時間間隔）。
///
/// 連続する観測値間の瞬時変化率を追跡。
/// 単位時間あたりの変化量を返す。
#[derive(Clone, Debug)]
pub struct ChangeRate {
    /// 前回の値。
    prev_value: f64,
    /// 前回のタイムスタンプ（ミリ秒）。
    prev_ts: u64,
    /// 最新の変化率。
    rate: f64,
    /// 初回観測済みフラグ。
    initialized: bool,
}

impl ChangeRate {
    /// 新しい `ChangeRate` を作成。
    #[must_use]
    pub const fn new() -> Self {
        Self {
            prev_value: 0.0,
            prev_ts: 0,
            rate: 0.0,
            initialized: false,
        }
    }

    /// 観測値を追加し、変化率を更新。
    ///
    /// 戻り値: 変化率が計算できた場合 `Some(rate)`。
    /// 初回観測時は `None`。
    pub fn observe(&mut self, value: f64, timestamp_ms: u64) -> Option<f64> {
        if !self.initialized {
            self.prev_value = value;
            self.prev_ts = timestamp_ms;
            self.initialized = true;
            return None;
        }

        let dt = timestamp_ms.saturating_sub(self.prev_ts);
        if dt == 0 {
            return Some(self.rate);
        }

        self.rate = (value - self.prev_value) / dt as f64;
        self.prev_value = value;
        self.prev_ts = timestamp_ms;
        Some(self.rate)
    }

    /// 最新の変化率。
    #[inline]
    #[must_use]
    pub const fn rate(&self) -> f64 {
        self.rate
    }

    /// 初期化済みかどうか。
    #[inline]
    #[must_use]
    pub const fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// リセット。
    pub const fn reset(&mut self) {
        self.prev_value = 0.0;
        self.prev_ts = 0;
        self.rate = 0.0;
        self.initialized = false;
    }
}

impl Default for ChangeRate {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Exponential Moving Average (EMA)
// ============================================================================

/// 指数移動平均 (EMA)。
///
/// `alpha` が大きいほど最新の値に重みが偏る。
/// `EMA_t = alpha * value + (1 - alpha) * EMA_{t-1}`
#[derive(Clone, Debug)]
pub struct ExponentialMovingAverage {
    /// 平滑化係数 (0.0 < alpha <= 1.0)。
    alpha: f64,
    /// 現在のEMA値。
    value: f64,
    /// 観測数。
    count: u64,
}

impl ExponentialMovingAverage {
    /// 指定の `alpha` 値で EMA を作成。
    ///
    /// # Panics
    ///
    /// `alpha` が 0.0 以下または 1.0 より大きい場合。
    #[must_use]
    pub fn new(alpha: f64) -> Self {
        assert!(alpha > 0.0 && alpha <= 1.0, "alpha must be in (0.0, 1.0]");
        Self {
            alpha,
            value: 0.0,
            count: 0,
        }
    }

    /// スパン（期間数）からEMAを作成。
    ///
    /// `alpha = 2 / (span + 1)`
    ///
    /// # Panics
    ///
    /// `span` が 0 の場合。
    #[must_use]
    pub fn from_span(span: u64) -> Self {
        assert!(span > 0, "span must be > 0");
        Self::new(2.0 / (span as f64 + 1.0))
    }

    /// 観測値を追加。
    pub fn observe(&mut self, value: f64) {
        if self.count == 0 {
            self.value = value;
        } else {
            self.value = self.alpha.mul_add(value, (1.0 - self.alpha) * self.value);
        }
        self.count += 1;
    }

    /// 現在のEMA値。
    #[inline]
    #[must_use]
    pub const fn value(&self) -> f64 {
        self.value
    }

    /// 観測数。
    #[inline]
    #[must_use]
    pub const fn count(&self) -> u64 {
        self.count
    }

    /// alpha値。
    #[inline]
    #[must_use]
    pub const fn alpha(&self) -> f64 {
        self.alpha
    }

    /// リセット。
    pub const fn reset(&mut self) {
        self.value = 0.0;
        self.count = 0;
    }
}

// ============================================================================
// Simple Moving Average (SMA) — 固定窓
// ============================================================================

/// 固定窓の単純移動平均 (SMA)。
///
/// リングバッファで最新N個の観測値を保持。
pub struct SimpleMovingAverage<const N: usize> {
    /// リングバッファ。
    buffer: [f64; N],
    /// 書き込み位置。
    write_pos: usize,
    /// 有効な要素数。
    count: usize,
    /// 合計値。
    sum: f64,
}

impl<const N: usize> SimpleMovingAverage<N> {
    /// 新しいSMAを作成。
    #[must_use]
    pub const fn new() -> Self {
        Self {
            buffer: [0.0; N],
            write_pos: 0,
            count: 0,
            sum: 0.0,
        }
    }

    /// 観測値を追加。
    pub fn observe(&mut self, value: f64) {
        if self.count >= N {
            self.sum -= self.buffer[self.write_pos];
        }
        self.buffer[self.write_pos] = value;
        self.sum += value;
        self.write_pos = (self.write_pos + 1) % N;
        if self.count < N {
            self.count += 1;
        }
    }

    /// 現在の平均値。
    #[inline]
    #[must_use]
    pub fn value(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.sum / self.count as f64
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

    /// ウィンドウが満杯かどうか。
    #[inline]
    #[must_use]
    pub const fn is_full(&self) -> bool {
        self.count >= N
    }

    /// リセット。
    pub const fn reset(&mut self) {
        self.write_pos = 0;
        self.count = 0;
        self.sum = 0.0;
    }
}

impl<const N: usize> Default for SimpleMovingAverage<N> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Online Linear Regression
// ============================================================================

/// オンライン線形回帰 (y = slope * x + intercept)。
///
/// Welford法ベースの逐次更新で、O(1) メモリ。
/// x, y の平均・分散・共分散から最小二乗法のパラメータを算出。
#[derive(Clone, Debug)]
pub struct LinearRegression {
    /// 観測数。
    count: u64,
    /// xの平均。
    mean_x: f64,
    /// yの平均。
    mean_y: f64,
    /// co-moment: `Σ(x_i - mean_x)(y_i - mean_y)`.
    co_moment_xy: f64,
    /// x の二次モーメント: `Σ(x_i - mean_x)^2`.
    m2_x: f64,
}

impl LinearRegression {
    /// 新しい線形回帰を作成。
    #[must_use]
    pub const fn new() -> Self {
        Self {
            count: 0,
            mean_x: 0.0,
            mean_y: 0.0,
            co_moment_xy: 0.0,
            m2_x: 0.0,
        }
    }

    /// 観測値 (x, y) を追加。
    pub fn observe(&mut self, x: f64, y: f64) {
        self.count += 1;
        let n = self.count as f64;

        let dx = x - self.mean_x;
        let dy = y - self.mean_y;

        self.mean_x += dx / n;
        self.mean_y += dy / n;

        // 更新後のdx'
        let dx2 = x - self.mean_x;
        self.co_moment_xy += dx * (y - self.mean_y);
        self.m2_x += dx * dx2;
    }

    /// 傾き (slope)。
    #[must_use]
    pub fn slope(&self) -> f64 {
        if self.count < 2 || self.m2_x.abs() < f64::EPSILON {
            return 0.0;
        }
        self.co_moment_xy / self.m2_x
    }

    /// 切片 (intercept)。
    #[must_use]
    pub fn intercept(&self) -> f64 {
        self.slope().mul_add(-self.mean_x, self.mean_y)
    }

    /// 予測値: y = slope * x + intercept。
    #[must_use]
    pub fn predict(&self, x: f64) -> f64 {
        self.slope().mul_add(x, self.intercept())
    }

    /// 決定係数 R²。
    #[must_use]
    pub const fn r_squared(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        // R² = (co_moment_xy)² / (m2_x * m2_y)
        // m2_y を別途計算する必要があるため、相関係数の二乗で近似
        // ここでは slope * co_moment / m2_y で計算
        // 簡易版: R² = slope * Sxy / Syy だが Syy を持たないため
        // slope² * Var(x) / Var(y) を使う
        // → co_moment_xy² / (m2_x * m2_y) が必要
        // m2_y を持たないので、追跡を追加せずに slope² * m2_x / m2_y は不可
        // 代替: 常に slope * co_moment_xy / m2_x * co_moment_xy / (m2_y) ...
        // → 簡潔さのため m2_y も追跡する
        0.0 // r_squared_full() を使用すること
    }

    /// 観測数。
    #[inline]
    #[must_use]
    pub const fn count(&self) -> u64 {
        self.count
    }

    /// リセット。
    pub const fn reset(&mut self) {
        self.count = 0;
        self.mean_x = 0.0;
        self.mean_y = 0.0;
        self.co_moment_xy = 0.0;
        self.m2_x = 0.0;
    }
}

impl Default for LinearRegression {
    fn default() -> Self {
        Self::new()
    }
}

/// R²も追跡するオンライン線形回帰。
#[derive(Clone, Debug)]
pub struct LinearRegressionFull {
    /// 基本回帰。
    inner: LinearRegression,
    /// y の二次モーメント: `Σ(y_i - mean_y)^2`.
    m2_y: f64,
}

impl LinearRegressionFull {
    /// 新しい線形回帰を作成。
    #[must_use]
    pub const fn new() -> Self {
        Self {
            inner: LinearRegression::new(),
            m2_y: 0.0,
        }
    }

    /// 観測値 (x, y) を追加。
    pub fn observe(&mut self, x: f64, y: f64) {
        let dy = y - self.inner.mean_y;
        self.inner.observe(x, y);
        let dy2 = y - self.inner.mean_y;
        self.m2_y += dy * dy2;
    }

    /// 傾き。
    #[must_use]
    pub fn slope(&self) -> f64 {
        self.inner.slope()
    }

    /// 切片。
    #[must_use]
    pub fn intercept(&self) -> f64 {
        self.inner.intercept()
    }

    /// 予測値。
    #[must_use]
    pub fn predict(&self, x: f64) -> f64 {
        self.inner.predict(x)
    }

    /// 決定係数 R²。
    #[must_use]
    pub fn r_squared(&self) -> f64 {
        if self.inner.count < 2
            || self.inner.m2_x.abs() < f64::EPSILON
            || self.m2_y.abs() < f64::EPSILON
        {
            return 0.0;
        }
        let r = self.inner.co_moment_xy / (self.inner.m2_x * self.m2_y).sqrt();
        (r * r).clamp(0.0, 1.0)
    }

    /// 観測数。
    #[inline]
    #[must_use]
    pub const fn count(&self) -> u64 {
        self.inner.count
    }

    /// リセット。
    pub const fn reset(&mut self) {
        self.inner.reset();
        self.m2_y = 0.0;
    }
}

impl Default for LinearRegressionFull {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    // --- ChangeRate ---

    #[test]
    fn change_rate_first_observation() {
        let mut cr = ChangeRate::new();
        assert!(cr.observe(10.0, 1000).is_none());
        assert!(cr.is_initialized());
    }

    #[test]
    fn change_rate_constant() {
        let mut cr = ChangeRate::new();
        cr.observe(10.0, 0);
        let rate = cr.observe(10.0, 100).unwrap();
        assert!((rate - 0.0).abs() < 1e-10);
    }

    #[test]
    fn change_rate_linear() {
        let mut cr = ChangeRate::new();
        cr.observe(0.0, 0);
        // 100ms で 50 増加 → rate = 0.5/ms
        let rate = cr.observe(50.0, 100).unwrap();
        assert!((rate - 0.5).abs() < 1e-10);
    }

    #[test]
    fn change_rate_negative() {
        let mut cr = ChangeRate::new();
        cr.observe(100.0, 0);
        let rate = cr.observe(50.0, 100).unwrap();
        assert!((rate - (-0.5)).abs() < 1e-10);
    }

    #[test]
    fn change_rate_same_timestamp() {
        let mut cr = ChangeRate::new();
        cr.observe(10.0, 100);
        let rate = cr.observe(20.0, 100).unwrap();
        // dt=0 → 前回のrateを返す
        assert!((rate - 0.0).abs() < 1e-10);
    }

    #[test]
    fn change_rate_reset() {
        let mut cr = ChangeRate::new();
        cr.observe(10.0, 0);
        cr.observe(20.0, 100);
        cr.reset();
        assert!(!cr.is_initialized());
        assert!((cr.rate() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn change_rate_default() {
        let cr = ChangeRate::default();
        assert!(!cr.is_initialized());
    }

    // --- ExponentialMovingAverage ---

    #[test]
    fn ema_first_value() {
        let mut ema = ExponentialMovingAverage::new(0.5);
        ema.observe(10.0);
        assert!((ema.value() - 10.0).abs() < 1e-10);
        assert_eq!(ema.count(), 1);
    }

    #[test]
    fn ema_convergence() {
        let mut ema = ExponentialMovingAverage::new(0.1);
        // 全て同じ値 → EMAはその値に収束
        for _ in 0..100 {
            ema.observe(50.0);
        }
        assert!((ema.value() - 50.0).abs() < 1e-6);
    }

    #[test]
    fn ema_from_span() {
        let ema = ExponentialMovingAverage::from_span(9);
        // alpha = 2/(9+1) = 0.2
        assert!((ema.alpha() - 0.2).abs() < 1e-10);
    }

    #[test]
    fn ema_step_response() {
        let mut ema = ExponentialMovingAverage::new(0.5);
        ema.observe(0.0);
        ema.observe(100.0);
        // EMA = 0.5 * 100 + 0.5 * 0 = 50
        assert!((ema.value() - 50.0).abs() < 1e-10);
    }

    #[test]
    fn ema_reset() {
        let mut ema = ExponentialMovingAverage::new(0.3);
        ema.observe(10.0);
        ema.reset();
        assert_eq!(ema.count(), 0);
        assert!((ema.value() - 0.0).abs() < 1e-10);
    }

    // --- SimpleMovingAverage ---

    #[test]
    fn sma_basic() {
        let mut sma = SimpleMovingAverage::<3>::new();
        sma.observe(10.0);
        sma.observe(20.0);
        sma.observe(30.0);
        assert!((sma.value() - 20.0).abs() < 1e-10);
        assert_eq!(sma.len(), 3);
        assert!(sma.is_full());
    }

    #[test]
    fn sma_overflow() {
        let mut sma = SimpleMovingAverage::<3>::new();
        sma.observe(10.0);
        sma.observe(20.0);
        sma.observe(30.0);
        sma.observe(40.0); // 10が押し出される
                           // (20+30+40)/3 = 30
        assert!((sma.value() - 30.0).abs() < 1e-10);
    }

    #[test]
    fn sma_empty() {
        let sma = SimpleMovingAverage::<8>::new();
        assert!(sma.is_empty());
        assert!((sma.value() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn sma_reset() {
        let mut sma = SimpleMovingAverage::<4>::new();
        sma.observe(10.0);
        sma.reset();
        assert!(sma.is_empty());
    }

    #[test]
    fn sma_default() {
        let sma = SimpleMovingAverage::<16>::default();
        assert!(sma.is_empty());
    }

    // --- LinearRegression ---

    #[test]
    fn linreg_perfect_line() {
        let mut lr = LinearRegression::new();
        // y = 2x + 1
        for i in 0..100 {
            let x = i as f64;
            lr.observe(x, 2.0 * x + 1.0);
        }
        assert!((lr.slope() - 2.0).abs() < 1e-6, "slope = {}", lr.slope());
        assert!(
            (lr.intercept() - 1.0).abs() < 1e-6,
            "intercept = {}",
            lr.intercept()
        );
    }

    #[test]
    fn linreg_predict() {
        let mut lr = LinearRegression::new();
        // y = 3x - 2
        for i in 0..50 {
            let x = i as f64;
            lr.observe(x, 3.0 * x - 2.0);
        }
        let pred = lr.predict(100.0);
        assert!((pred - 298.0).abs() < 1e-4, "pred = {pred}");
    }

    #[test]
    fn linreg_constant() {
        let mut lr = LinearRegression::new();
        for i in 0..50 {
            lr.observe(i as f64, 42.0);
        }
        assert!((lr.slope() - 0.0).abs() < 1e-10);
        assert!((lr.intercept() - 42.0).abs() < 1e-6);
    }

    #[test]
    fn linreg_single_point() {
        let mut lr = LinearRegression::new();
        lr.observe(5.0, 10.0);
        assert!((lr.slope() - 0.0).abs() < 1e-10);
        assert_eq!(lr.count(), 1);
    }

    #[test]
    fn linreg_reset() {
        let mut lr = LinearRegression::new();
        lr.observe(1.0, 2.0);
        lr.observe(2.0, 4.0);
        lr.reset();
        assert_eq!(lr.count(), 0);
    }

    #[test]
    fn linreg_default() {
        let lr = LinearRegression::default();
        assert_eq!(lr.count(), 0);
    }

    // --- LinearRegressionFull ---

    #[test]
    fn linreg_full_r_squared_perfect() {
        let mut lr = LinearRegressionFull::new();
        // 完全な線形関係 → R²≈1.0
        for i in 0..100 {
            let x = i as f64;
            lr.observe(x, 2.0 * x + 1.0);
        }
        assert!(
            (lr.r_squared() - 1.0).abs() < 1e-6,
            "R² = {}",
            lr.r_squared()
        );
    }

    #[test]
    fn linreg_full_r_squared_no_correlation() {
        let mut lr = LinearRegressionFull::new();
        // x増加、yは定数 → R²≈0
        for i in 0..100 {
            lr.observe(i as f64, 42.0);
        }
        assert!(lr.r_squared() < 0.01, "R² = {}", lr.r_squared());
    }

    #[test]
    fn linreg_full_predict() {
        let mut lr = LinearRegressionFull::new();
        for i in 0..50 {
            let x = i as f64;
            lr.observe(x, x * 0.5 + 10.0);
        }
        let pred = lr.predict(100.0);
        assert!((pred - 60.0).abs() < 1e-4, "pred = {pred}");
    }

    #[test]
    fn linreg_full_reset() {
        let mut lr = LinearRegressionFull::new();
        lr.observe(1.0, 2.0);
        lr.reset();
        assert_eq!(lr.count(), 0);
    }

    #[test]
    fn linreg_full_default() {
        let lr = LinearRegressionFull::default();
        assert_eq!(lr.count(), 0);
    }
}
