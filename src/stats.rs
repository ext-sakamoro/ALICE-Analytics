//! Advanced Statistical Functions
//!
//! パーセンタイルランク、IQR、共分散/相関行列、Welford法オンライン統計。
//! `no_std` 対応、ゼロアロケーション設計。

// ============================================================================
// Percentile Rank
// ============================================================================

/// ソート済み配列中の値のパーセンタイルランク。
///
/// `value` が `sorted_data` 中で何パーセント以下にあるかを返す `[0.0, 100.0]`。
#[inline]
#[must_use]
pub fn percentile_rank(sorted_data: &[f64], value: f64) -> f64 {
    if sorted_data.is_empty() {
        return 0.0;
    }
    let mut count_below = 0usize;
    let mut count_equal = 0usize;
    for &v in sorted_data {
        if v < value {
            count_below += 1;
        } else if (v - value).abs() < f64::EPSILON {
            count_equal += 1;
        }
    }
    // パーセンタイルランク = (below + 0.5 * equal) / N * 100
    let n = sorted_data.len() as f64;
    0.5f64.mul_add(count_equal as f64, count_below as f64) / n * 100.0
}

// ============================================================================
// IQR (Interquartile Range)
// ============================================================================

/// IQR結果。
#[derive(Clone, Copy, Debug)]
pub struct IqrResult {
    /// 第1四分位数 (Q1)。
    pub q1: f64,
    /// 中央値 (Q2)。
    pub median: f64,
    /// 第3四分位数 (Q3)。
    pub q3: f64,
    /// IQR = Q3 - Q1。
    pub iqr: f64,
    /// 下限 = Q1 - 1.5 * IQR。
    pub lower_fence: f64,
    /// 上限 = Q3 + 1.5 * IQR。
    pub upper_fence: f64,
}

/// ソート済み配列のIQRを計算。
///
/// 線形補間でQ1/Q3を算出。
#[must_use]
pub fn iqr(sorted_data: &[f64]) -> Option<IqrResult> {
    let n = sorted_data.len();
    if n < 4 {
        return None;
    }

    let q1 = quantile_sorted(sorted_data, 0.25);
    let median = quantile_sorted(sorted_data, 0.50);
    let q3 = quantile_sorted(sorted_data, 0.75);
    let iqr_val = q3 - q1;

    Some(IqrResult {
        q1,
        median,
        q3,
        iqr: iqr_val,
        lower_fence: 1.5f64.mul_add(-iqr_val, q1),
        upper_fence: 1.5f64.mul_add(iqr_val, q3),
    })
}

/// ソート済み配列の任意パーセンタイル（線形補間）。
#[inline]
#[must_use]
pub fn quantile_sorted(sorted_data: &[f64], q: f64) -> f64 {
    let n = sorted_data.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted_data[0];
    }
    let q = q.clamp(0.0, 1.0);
    let pos = q * (n - 1) as f64;
    let lower = pos.floor() as usize;
    let upper = pos.ceil() as usize;
    if lower == upper {
        return sorted_data[lower];
    }
    let frac = pos - lower as f64;
    sorted_data[lower].mul_add(1.0 - frac, sorted_data[upper] * frac)
}

// ============================================================================
// Covariance & Correlation Matrix
// ============================================================================

/// 共分散行列（最大8変数）。
///
/// Welford法のオンライン計算。
pub struct CovarianceMatrix<const D: usize> {
    /// 観測数。
    count: u64,
    /// 各変数の平均。
    means: [f64; D],
    /// co-moment行列 C[i][j]（上三角 + 対角）。
    co_moments: [[f64; D]; D],
}

impl<const D: usize> CovarianceMatrix<D> {
    /// 新しい共分散行列を初期化。
    #[must_use]
    pub const fn new() -> Self {
        Self {
            count: 0,
            means: [0.0; D],
            co_moments: [[0.0; D]; D],
        }
    }

    /// 観測値ベクトルを追加（Welfordオンラインアルゴリズム）。
    pub fn observe(&mut self, values: &[f64; D]) {
        self.count += 1;
        let n = self.count as f64;

        let mut dx = [0.0; D];
        for (i, d) in dx.iter_mut().enumerate() {
            *d = values[i] - self.means[i];
        }

        // 平均値更新
        for (i, mean) in self.means.iter_mut().enumerate() {
            *mean += dx[i] / n;
        }

        // co-moment更新
        for (i, (&val, &mean_i)) in values.iter().zip(self.means.iter()).enumerate() {
            let dx2_i = val - mean_i;
            for (j, dx_j) in dx.iter().enumerate().skip(i) {
                self.co_moments[i][j] += dx_j * dx2_i;
            }
        }
    }

    /// 共分散 Cov(i, j)。
    #[must_use]
    pub fn covariance(&self, i: usize, j: usize) -> f64 {
        if self.count < 2 || i >= D || j >= D {
            return 0.0;
        }
        let (r, c) = if i <= j { (i, j) } else { (j, i) };
        self.co_moments[r][c] / (self.count - 1) as f64
    }

    /// 相関係数 r(i, j)。
    #[must_use]
    pub fn correlation(&self, i: usize, j: usize) -> f64 {
        if i == j {
            return 1.0;
        }
        let cov = self.covariance(i, j);
        let var_i = self.covariance(i, i);
        let var_j = self.covariance(j, j);
        let denom = (var_i * var_j).sqrt();
        if denom < f64::EPSILON {
            return 0.0;
        }
        (cov / denom).clamp(-1.0, 1.0)
    }

    /// 変数iの分散。
    #[inline]
    #[must_use]
    pub fn variance(&self, i: usize) -> f64 {
        self.covariance(i, i)
    }

    /// 変数iの平均。
    #[inline]
    #[must_use]
    pub const fn mean(&self, i: usize) -> f64 {
        if i < D {
            self.means[i]
        } else {
            0.0
        }
    }

    /// 観測数。
    #[inline]
    #[must_use]
    pub const fn count(&self) -> u64 {
        self.count
    }
}

impl<const D: usize> Default for CovarianceMatrix<D> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Streaming Stats (Welford拡張)
// ============================================================================

/// オンラインストリーミング統計。
///
/// Welford法で mean/variance/skewness/kurtosis をO(1)メモリで追跡。
#[derive(Clone, Debug)]
pub struct StreamingStats {
    count: u64,
    mean: f64,
    m2: f64,
    m3: f64,
    m4: f64,
    min: f64,
    max: f64,
}

impl StreamingStats {
    /// 新しい `StreamingStats` を作成。
    #[must_use]
    pub const fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            m3: 0.0,
            m4: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    /// 観測値を追加。
    pub fn observe(&mut self, value: f64) {
        let n1 = self.count;
        self.count += 1;
        let n = self.count as f64;

        let delta = value - self.mean;
        let delta_n = delta / n;
        let delta_n2 = delta_n * delta_n;
        let term1 = delta * delta_n * n1 as f64;

        self.mean += delta_n;

        // 4次モーメント更新（Pebay 2008）
        self.m4 += (4.0 * delta_n).mul_add(
            -self.m3,
            (term1 * delta_n2).mul_add(n.mul_add(n, -(3.0 * n)) + 3.0, 6.0 * delta_n2 * self.m2),
        );
        self.m3 += (term1 * delta_n).mul_add(n - 2.0, -(3.0 * delta_n * self.m2));
        self.m2 += term1;

        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
    }

    /// 観測数。
    #[inline]
    #[must_use]
    pub const fn count(&self) -> u64 {
        self.count
    }

    /// 平均値。
    #[inline]
    #[must_use]
    pub const fn mean(&self) -> f64 {
        self.mean
    }

    /// 母集団分散。
    #[must_use]
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        self.m2 / self.count as f64
    }

    /// 標本分散。
    #[must_use]
    pub fn sample_variance(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        self.m2 / (self.count - 1) as f64
    }

    /// 標準偏差。
    #[inline]
    #[must_use]
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// 歪度 (Skewness)。
    #[must_use]
    pub fn skewness(&self) -> f64 {
        if self.count < 3 || self.m2 < f64::EPSILON {
            return 0.0;
        }
        let n = self.count as f64;
        n.sqrt() * self.m3 / self.m2.powf(1.5)
    }

    /// 尖度 (Excess Kurtosis)。
    #[must_use]
    pub fn kurtosis(&self) -> f64 {
        if self.count < 4 || self.m2 < f64::EPSILON {
            return 0.0;
        }
        let n = self.count as f64;
        n * self.m4 / (self.m2 * self.m2) - 3.0
    }

    /// 最小値。
    #[inline]
    #[must_use]
    pub const fn min(&self) -> f64 {
        self.min
    }

    /// 最大値。
    #[inline]
    #[must_use]
    pub const fn max(&self) -> f64 {
        self.max
    }

    /// リセット。
    pub const fn reset(&mut self) {
        self.count = 0;
        self.mean = 0.0;
        self.m2 = 0.0;
        self.m3 = 0.0;
        self.m4 = 0.0;
        self.min = f64::INFINITY;
        self.max = f64::NEG_INFINITY;
    }
}

impl Default for StreamingStats {
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

    #[test]
    fn percentile_rank_basic() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let rank = percentile_rank(&data, 3.0);
        // (2 below + 0.5 * 1 equal) / 5 * 100 = 50%
        assert!((rank - 50.0).abs() < 1e-6);
    }

    #[test]
    fn percentile_rank_empty() {
        assert!((percentile_rank(&[], 5.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn percentile_rank_below_all() {
        let data = [10.0, 20.0, 30.0];
        let rank = percentile_rank(&data, 1.0);
        assert!((rank - 0.0).abs() < 1e-6);
    }

    #[test]
    fn percentile_rank_above_all() {
        let data = [10.0, 20.0, 30.0];
        let rank = percentile_rank(&data, 50.0);
        assert!((rank - 100.0).abs() < 1e-6);
    }

    #[test]
    fn iqr_basic() {
        // データ: 1,2,3,4,5,6,7,8
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = iqr(&data).unwrap();
        // Q1=2.75, Q2=4.5, Q3=6.25, IQR=3.5
        assert!((result.q1 - 2.75).abs() < 1e-6);
        assert!((result.median - 4.5).abs() < 1e-6);
        assert!((result.q3 - 6.25).abs() < 1e-6);
        assert!((result.iqr - 3.5).abs() < 1e-6);
    }

    #[test]
    fn iqr_too_small() {
        let data = [1.0, 2.0, 3.0];
        assert!(iqr(&data).is_none());
    }

    #[test]
    fn iqr_fences() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = iqr(&data).unwrap();
        assert!(result.lower_fence < result.q1);
        assert!(result.upper_fence > result.q3);
    }

    #[test]
    fn quantile_sorted_edges() {
        let data = [10.0, 20.0, 30.0];
        assert!((quantile_sorted(&data, 0.0) - 10.0).abs() < 1e-6);
        assert!((quantile_sorted(&data, 1.0) - 30.0).abs() < 1e-6);
        assert!((quantile_sorted(&data, 0.5) - 20.0).abs() < 1e-6);
    }

    #[test]
    fn covariance_matrix_basic() {
        let mut cov = CovarianceMatrix::<2>::new();
        // 完全正相関: x=y
        for i in 0..100 {
            let v = i as f64;
            cov.observe(&[v, v]);
        }
        let r = cov.correlation(0, 1);
        assert!((r - 1.0).abs() < 1e-6);
    }

    #[test]
    fn covariance_matrix_negative_correlation() {
        let mut cov = CovarianceMatrix::<2>::new();
        // 完全負相関: x = -y
        for i in 0..100 {
            let v = i as f64;
            cov.observe(&[v, -v]);
        }
        let r = cov.correlation(0, 1);
        assert!((r - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn covariance_matrix_self_correlation() {
        let cov = CovarianceMatrix::<3>::new();
        // 自己相関は常に1.0
        assert!((cov.correlation(0, 0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn covariance_matrix_variance() {
        let mut cov = CovarianceMatrix::<2>::new();
        // 0,1,2,...,99 → sample variance = 833.25
        for i in 0..100 {
            cov.observe(&[i as f64, 0.0]);
        }
        let var = cov.variance(0);
        // Welford co-moment / (n-1) で標本分散
        // 0..100 の標本分散 = 100*99/12 = 833.333...
        assert!(var > 800.0 && var < 870.0, "var = {var}");
    }

    #[test]
    fn streaming_stats_basic() {
        let mut s = StreamingStats::new();
        for v in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            s.observe(v);
        }
        assert_eq!(s.count(), 8);
        assert!((s.mean() - 5.0).abs() < 1e-6);
        assert!((s.min() - 2.0).abs() < 1e-6);
        assert!((s.max() - 9.0).abs() < 1e-6);
    }

    #[test]
    fn streaming_stats_variance() {
        let mut s = StreamingStats::new();
        for v in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            s.observe(v);
        }
        // 母集団分散 = 4.0 (厳密値)
        assert!((s.variance() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn streaming_stats_empty() {
        let s = StreamingStats::new();
        assert_eq!(s.count(), 0);
        assert!((s.mean() - 0.0).abs() < 1e-6);
        assert!((s.variance() - 0.0).abs() < 1e-6);
        assert!((s.skewness() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn streaming_stats_reset() {
        let mut s = StreamingStats::new();
        s.observe(10.0);
        s.observe(20.0);
        s.reset();
        assert_eq!(s.count(), 0);
    }

    #[test]
    fn streaming_stats_skewness_symmetric() {
        let mut s = StreamingStats::new();
        // 対称分布 → skewness ≈ 0
        for v in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0] {
            s.observe(v);
        }
        assert!(s.skewness().abs() < 0.1);
    }

    #[test]
    fn streaming_stats_kurtosis_uniform() {
        let mut s = StreamingStats::new();
        // 一様分布の尖度は約-1.2
        for i in 0..1000 {
            s.observe(i as f64);
        }
        assert!(s.kurtosis() < 0.0);
    }

    #[test]
    fn streaming_stats_default() {
        let s = StreamingStats::default();
        assert_eq!(s.count(), 0);
    }

    #[test]
    fn covariance_matrix_default() {
        let cov = CovarianceMatrix::<4>::default();
        assert_eq!(cov.count(), 0);
    }
}
