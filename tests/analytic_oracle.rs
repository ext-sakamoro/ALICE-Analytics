//! Analytic oracles — closed-form checks for the statistics in ALICE-Analytics
//! (CLAUDE.md § 解析解突合テスト規律, 2026-09-17).
//!
//! Expected values come from closed forms or f64 two-pass references written
//! in this file, never from the crate function under test.  Default
//! constructors (`StreamingStats::default()`, `HyperLogLog::new()`,
//! `CountMinSketch::new()`) are the paths a consumer takes first.
//!
//! Oracle sources:
//! - moments: two-pass mean / population & sample variance / g₁ skewness /
//!   excess kurtosis on known sets; order independence (Pébay 2008 is exact)
//! - quantile: R-7 linear interpolation; percentile rank (below + ½·equal)/n;
//!   IQR fences Q1 − 1.5·IQR, Q3 + 1.5·IQR
//! - covariance / correlation: y = 2x ⇒ ρ = 1, orthogonal ⇒ 0, symmetry
//! - windows: mean / min / max / variance of the last N values; EMA step
//!   response 1 − (1−α)ᵏ, α = 2/(span+1); regression on an exact line, R² = 1
//! - HyperLogLog: |estimate − n| ≤ 3·1.04/√m · n; DDSketch: relative error ≤ α
//!   at every quantile; Count-Min: estimate ≥ truth, error ≤ ε·N with
//!   ε = e/w; FNV-1a 64 vectors through the MurmurHash3 fmix64 finaliser
//! - anomaly: running median of odd / even sets, MAD = median|x − med|,
//!   EWMA / EWMV recursions in f64
//! - privacy: xorshift64 determinism, Laplace(b) sample mean 0 / variance 2b²,
//!   randomized response unbiasing P(1) = p·t + (1−p)/2

// statistical oracles cast freely and use short algebraic names; the
// crate's pedantic gate is about API code, not the reference arithmetic here
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::float_cmp,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::unreadable_literal,
    clippy::items_after_statements,
    clippy::suboptimal_flops,
    clippy::needless_range_loop
)]

use alice_analytics::anomaly::{EwmaDetector, MadDetector, StreamingMedian};
use alice_analytics::sketch::{CountMinSketch, DDSketch, FnvHasher, HeavyHitters10, HyperLogLog};
use alice_analytics::stats::{
    iqr, percentile_rank, quantile_sorted, CovarianceMatrix, StreamingStats,
};
use alice_analytics::streaming_ops::{
    ChangeRate, ExponentialMovingAverage, LinearRegression, LinearRegressionFull,
    SimpleMovingAverage,
};
use alice_analytics::window::SlidingWindow;

fn lcg(seed: &mut u64) -> f64 {
    *seed = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*seed >> 11) as f64 / (1u64 << 53) as f64
}

fn two_pass(xs: &[f64]) -> (f64, f64, f64, f64, f64) {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let m2 = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>();
    let m3 = xs.iter().map(|x| (x - mean).powi(3)).sum::<f64>();
    let m4 = xs.iter().map(|x| (x - mean).powi(4)).sum::<f64>();
    let var = m2 / n;
    let svar = m2 / (n - 1.0);
    let skew = n.sqrt() * m3 / m2.powf(1.5);
    let kurt = n * m4 / (m2 * m2) - 3.0;
    (mean, var, svar, skew, kurt)
}

// ───────────────────────── moments / quantiles ────────────────────────────

#[test]
fn streaming_moments_equal_the_two_pass_reference_in_any_order() {
    let sets: Vec<Vec<f64>> = vec![
        (0..100).map(f64::from).collect(),
        vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0],
        vec![-5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0], // heavy right tail
        (0..1000)
            .map(|i| ((i * 7919) % 1000) as f64 * 0.001 - 0.3)
            .collect(),
    ];
    for xs in &sets {
        let (mean, var, svar, skew, kurt) = two_pass(xs);
        let mut s = StreamingStats::default();
        for &x in xs {
            s.observe(x);
        }
        assert_eq!(s.count() as usize, xs.len());
        assert!((s.mean() - mean).abs() < 1e-9 * mean.abs().max(1.0), "mean");
        assert!(
            (s.variance() - var).abs() < 1e-9 * var.max(1.0),
            "variance {} vs {var}",
            s.variance()
        );
        assert!(
            (s.sample_variance() - svar).abs() < 1e-9 * svar.max(1.0),
            "sample variance"
        );
        assert!((s.std_dev() - var.sqrt()).abs() < 1e-9 * var.sqrt().max(1.0));
        assert!(
            (s.skewness() - skew).abs() < 1e-7 * skew.abs().max(1.0),
            "skew {} vs {skew}",
            s.skewness()
        );
        assert!(
            (s.kurtosis() - kurt).abs() < 1e-6 * kurt.abs().max(1.0),
            "kurtosis {} vs {kurt}",
            s.kurtosis()
        );
        assert_eq!(s.min(), xs.iter().copied().fold(f64::INFINITY, f64::min));
        assert_eq!(
            s.max(),
            xs.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        );
        // order independence: reversed and interleaved feeds give the same moments
        let mut r = StreamingStats::default();
        for &x in xs.iter().rev() {
            r.observe(x);
        }
        assert!((r.skewness() - s.skewness()).abs() < 1e-7 * skew.abs().max(1.0));
        assert!((r.kurtosis() - s.kurtosis()).abs() < 1e-6 * kurt.abs().max(1.0));
    }
    // symmetric set ⇒ skewness exactly 0, uniform-ish ⇒ negative excess kurtosis
    let mut s = StreamingStats::default();
    for x in [-3.0, -1.0, 0.0, 1.0, 3.0] {
        s.observe(x);
    }
    assert!(s.skewness().abs() < 1e-12);
    assert!(s.kurtosis() < 0.0);
}

#[test]
fn quantiles_percentile_rank_and_iqr_follow_r7_and_the_tukey_fences() {
    let sorted = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    // R-7: position q·(n−1), linear interpolation
    assert_eq!(quantile_sorted(&sorted, 0.0), 1.0);
    assert_eq!(quantile_sorted(&sorted, 1.0), 10.0);
    assert_eq!(quantile_sorted(&sorted, 0.5), 5.5);
    assert_eq!(quantile_sorted(&sorted, 0.25), 3.25);
    assert_eq!(quantile_sorted(&sorted, 0.75), 7.75);
    assert_eq!(quantile_sorted(&sorted, 0.1), 1.9);
    assert_eq!(quantile_sorted(&[42.0], 0.3), 42.0);
    assert_eq!(quantile_sorted(&[], 0.3), 0.0);
    // percentile rank = (below + ½ equal)/n · 100
    let with_ties = [1.0, 2.0, 2.0, 2.0, 5.0, 9.0];
    assert!((percentile_rank(&with_ties, 2.0) - (1.0 + 1.5) / 6.0 * 100.0).abs() < 1e-12);
    assert!((percentile_rank(&with_ties, 9.0) - (5.0 + 0.5) / 6.0 * 100.0).abs() < 1e-12);
    assert!((percentile_rank(&with_ties, 100.0) - 100.0).abs() < 1e-12);
    assert_eq!(percentile_rank(&with_ties, 0.0), 0.0);
    // IQR on 1..=10: Q1 = 3.25, Q3 = 7.75, IQR = 4.5, fences −3.5 / 14.5
    let r = iqr(&sorted).unwrap();
    assert_eq!((r.q1, r.median, r.q3, r.iqr), (3.25, 5.5, 7.75, 4.5));
    assert!((r.lower_fence - (3.25 - 6.75)).abs() < 1e-12);
    assert!((r.upper_fence - (7.75 + 6.75)).abs() < 1e-12);
    assert!(iqr(&[1.0, 2.0, 3.0]).is_none());
}

#[test]
fn covariance_matrix_is_symmetric_and_correlation_is_exact_on_known_relations() {
    let mut cm: CovarianceMatrix<3> = CovarianceMatrix::new();
    let mut seed = 5u64;
    let mut rows = Vec::new();
    for _ in 0..500 {
        let x = lcg(&mut seed) * 10.0;
        let z = lcg(&mut seed) - 0.5; // independent
        rows.push([x, 2.0 * x + 1.0, z]);
    }
    for r in &rows {
        cm.observe(r);
    }
    // two-pass sample covariance reference
    let n = rows.len() as f64;
    let mean = |k: usize| rows.iter().map(|r| r[k]).sum::<f64>() / n;
    let cov = |a: usize, b: usize| {
        let (ma, mb) = (mean(a), mean(b));
        rows.iter().map(|r| (r[a] - ma) * (r[b] - mb)).sum::<f64>() / (n - 1.0)
    };
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (cm.covariance(i, j) - cov(i, j)).abs() < 1e-9 * cov(i, j).abs().max(1.0),
                "cov({i},{j})"
            );
            assert_eq!(cm.covariance(i, j), cm.covariance(j, i), "symmetry");
        }
        assert_eq!(cm.variance(i), cm.covariance(i, i));
        assert_eq!(cm.correlation(i, i), 1.0);
    }
    assert!(
        (cm.correlation(0, 1) - 1.0).abs() < 1e-12,
        "y = 2x + 1 ⇒ ρ = 1"
    );
    assert!(
        cm.correlation(0, 2).abs() < 0.1,
        "independent ⇒ ρ ≈ 0 ({})",
        cm.correlation(0, 2)
    );
    assert!(
        (cm.covariance(0, 1) - 2.0 * cm.variance(0)).abs() < 1e-9 * cm.variance(0),
        "cov(x, 2x+1) = 2 var(x)"
    );
}

// ───────────────────────── windows / streaming ops ────────────────────────

#[test]
fn sliding_window_and_moving_averages_are_the_closed_forms_of_the_last_n() {
    let mut w: SlidingWindow<8> = SlidingWindow::new();
    let mut sma: SimpleMovingAverage<8> = SimpleMovingAverage::new();
    let xs: Vec<f64> = (0..50).map(|i| ((i * 37) % 17) as f64 - 8.0).collect();
    for (k, &x) in xs.iter().enumerate() {
        w.push(x);
        sma.observe(x);
        let last = &xs[k.saturating_sub(7)..=k];
        let n = last.len() as f64;
        let mean = last.iter().sum::<f64>() / n;
        let var = last.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        assert!((w.mean() - mean).abs() < 1e-12, "k={k} mean");
        assert!((sma.value() - mean).abs() < 1e-12, "k={k} sma");
        assert_eq!(
            w.min(),
            last.iter().copied().fold(f64::INFINITY, f64::min),
            "k={k} min"
        );
        assert_eq!(
            w.max(),
            last.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            "k={k} max"
        );
        assert!(
            (w.variance() - var).abs() < 1e-9,
            "k={k} variance {} vs {var}",
            w.variance()
        );
        assert!((w.std_dev() - var.sqrt()).abs() < 1e-9);
    }
    // EMA: constant input converges to it; a unit step from 0 reaches 1 − (1−α)ᵏ
    let alpha = 0.3;
    let mut ema = ExponentialMovingAverage::new(alpha);
    ema.observe(0.0);
    for k in 1..=40 {
        ema.observe(1.0);
        let expected = 1.0 - (1.0 - alpha).powi(k);
        assert!(
            (ema.value() - expected).abs() < 1e-12,
            "step {k}: {} vs {expected}",
            ema.value()
        );
    }
    let ema = ExponentialMovingAverage::from_span(9);
    let mut probe = ema;
    probe.observe(0.0);
    probe.observe(1.0);
    assert!((probe.value() - 2.0 / 10.0).abs() < 1e-12, "α = 2/(span+1)");
    // change rate = Δv / Δt
    let mut cr = ChangeRate::new();
    assert!(cr.observe(10.0, 1000).is_none());
    assert_eq!(cr.observe(16.0, 1003), Some(2.0));
    assert_eq!(
        cr.observe(16.0, 1003),
        Some(2.0),
        "dt = 0 keeps the last rate"
    );
    assert_eq!(cr.observe(6.0, 1008), Some(-2.0));
}

#[test]
fn streaming_regression_recovers_an_exact_line_and_matches_the_normal_equations() {
    let mut lr = LinearRegression::new();
    let mut full = LinearRegressionFull::new();
    for i in 0..200 {
        let x = i as f64 * 0.5 - 20.0;
        let y = -3.0 * x + 7.0;
        lr.observe(x, y);
        full.observe(x, y);
    }
    assert!((lr.slope() + 3.0).abs() < 1e-9 && (lr.intercept() - 7.0).abs() < 1e-9);
    assert!((full.slope() + 3.0).abs() < 1e-9 && (full.intercept() - 7.0).abs() < 1e-9);
    assert!(
        (full.r_squared() - 1.0).abs() < 1e-12,
        "exact line ⇒ R² = 1"
    );
    assert!((lr.predict(100.0) - (-293.0)).abs() < 1e-9);
    // noisy: f64 normal equations
    let mut seed = 9u64;
    let pts: Vec<(f64, f64)> = (0..300)
        .map(|i| {
            (
                i as f64,
                0.7 * i as f64 - 12.0 + (lcg(&mut seed) - 0.5) * 8.0,
            )
        })
        .collect();
    let mut lr = LinearRegressionFull::new();
    for &(x, y) in &pts {
        lr.observe(x, y);
    }
    let n = pts.len() as f64;
    let (sx, sy) = (
        pts.iter().map(|p| p.0).sum::<f64>(),
        pts.iter().map(|p| p.1).sum::<f64>(),
    );
    let sxy: f64 = pts.iter().map(|p| p.0 * p.1).sum();
    let sxx: f64 = pts.iter().map(|p| p.0 * p.0).sum();
    let syy: f64 = pts.iter().map(|p| p.1 * p.1).sum();
    let a = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    let b = (sy - a * sx) / n;
    let r2 = (n * sxy - sx * sy).powi(2) / ((n * sxx - sx * sx) * (n * syy - sy * sy));
    assert!(
        (lr.slope() - a).abs() < 1e-9 * a.abs(),
        "slope {} vs {a}",
        lr.slope()
    );
    assert!((lr.intercept() - b).abs() < 1e-9 * b.abs().max(1.0));
    assert!(
        (lr.r_squared() - r2).abs() < 1e-9,
        "R² {} vs {r2}",
        lr.r_squared()
    );
}

// ───────────────────────── sketches ───────────────────────────────────────

#[test]
fn sketches_stay_inside_their_published_error_bounds() {
    // HyperLogLog (m = 2¹⁴): σ = 1.04/√m ≈ 0.81 %, assert 3σ
    for n in [1_000u64, 20_000, 200_000] {
        let mut hll = HyperLogLog::new();
        for i in 0..n {
            hll.insert(&(i.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xABCD));
        }
        let est = hll.cardinality();
        let sigma = 1.04 / (16384f64).sqrt();
        assert!(
            (est - n as f64).abs() <= 3.0 * sigma * n as f64,
            "HLL n={n}: estimate {est} outside 3σ ({:.2} %)",
            (est - n as f64).abs() / n as f64 * 100.0
        );
    }
    // DDSketch(α): every quantile within relative error α (2048 bins, the
    // published operating range for α ≥ 0.01; the 6-decade stream below needs
    // ln(10⁶)/ln((1+α)/(1−α)) ≈ 345 bins)
    let alpha = 0.02;
    let mut dd = DDSketch::new(alpha);
    let mut seed = 3u64;
    let mut values: Vec<f64> = (0..20_000)
        .map(|_| (lcg(&mut seed) * 12.0).exp() * 0.01)
        .collect();
    for &v in &values {
        dd.insert(v);
    }
    values.sort_by(f64::total_cmp);
    // the guarantee is rank based: the ⌈q·n⌉-th order statistic within α
    let order_stat = |sorted: &[f64], q: f64| {
        sorted[((q * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len()) - 1]
    };
    for q in [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999] {
        let truth = order_stat(&values, q);
        let got = dd.quantile(q);
        assert!(
            ((got - truth) / truth).abs() <= alpha + 1e-9,
            "DDSketch q={q}: {got} vs {truth} ({:.3} %)",
            ((got - truth) / truth).abs() * 100.0
        );
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    assert!(
        ((dd.mean() - mean) / mean).abs() < 1e-9,
        "the mean is exact"
    );
    // negative values: the same bound, with the quantile order preserved
    let mut neg = DDSketch::new(alpha);
    let mut nvals: Vec<f64> = values.iter().map(|v| -v).collect();
    for &v in &nvals {
        neg.insert(v);
    }
    nvals.sort_by(f64::total_cmp);
    for q in [0.01, 0.5, 0.99] {
        let truth = order_stat(&nvals, q);
        let got = neg.quantile(q);
        assert!(
            ((got - truth) / truth).abs() <= alpha + 1e-9,
            "negative q={q}: {got} vs {truth}"
        );
    }
    assert!(
        neg.quantile(0.0) < neg.quantile(1.0),
        "negative quantiles are ordered"
    );
    // Count-Min: never under-estimates; over-estimate ≤ ε·N with ε = e/w at the
    // published confidence 1 − e⁻ᵈ (checked on a Zipf-like stream)
    let mut cm = CountMinSketch::new();
    let mut truth = std::collections::HashMap::new();
    let mut total = 0u64;
    for _ in 0..50_000u64 {
        let item = (lcg(&mut seed).powi(3) * 5000.0) as u64;
        cm.insert(&item);
        *truth.entry(item).or_insert(0u64) += 1;
        total += 1;
    }
    let eps = std::f64::consts::E / 1024.0;
    assert!((cm.error_bound() - eps).abs() < 1e-12);
    assert!((cm.confidence() - (1.0 - (-5f64).exp())).abs() < 1e-12);
    let mut violations = 0;
    for (item, &count) in &truth {
        let est = cm.estimate(item);
        assert!(
            est >= count,
            "Count-Min under-estimated {item}: {est} < {count}"
        );
        if (est - count) as f64 > eps * total as f64 {
            violations += 1;
        }
    }
    assert!(
        (violations as f64) < 0.02 * truth.len() as f64,
        "{violations} of {} items above ε·N (allowed 1 − confidence ≈ 0.7 %)",
        truth.len()
    );
    // heavy hitters: the single most frequent hash is reported first
    let mut hh = HeavyHitters10::new();
    for i in 0..10_000u64 {
        hh.insert_hash(if i % 3 == 0 { 7 } else { 1000 + i });
    }
    assert_eq!(hh.top().next().map(|e| e.hash), Some(7));
    // the crate's hash is FNV-1a 64 followed by the MurmurHash3 fmix64
    // finaliser — reference FNV vectors ("" / "a" / "foobar") pushed through an
    // independent fmix64 written here
    let fmix64 = |mut h: u64| {
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        h ^ (h >> 33)
    };
    for (input, fnv) in [
        (&b""[..], 0xcbf2_9ce4_8422_2325u64),
        (b"a", 0xaf63_dc4c_8601_ec8c),
        (b"foobar", 0x8594_4171_f739_67e8),
    ] {
        assert_eq!(
            FnvHasher::hash_bytes(input),
            fmix64(fnv),
            "fmix64(fnv1a({input:?}))"
        );
    }
}

// ───────────────────────── anomaly detectors ──────────────────────────────

#[test]
fn running_median_mad_and_ewma_detectors_match_their_definitions() {
    let mut m = StreamingMedian::new();
    let xs = [5.0, 1.0, 9.0, 3.0, 7.0, 2.0, 8.0];
    for (k, &x) in xs.iter().enumerate() {
        m.push(x);
        let mut so: Vec<f64> = xs[..=k].to_vec();
        so.sort_by(f64::total_cmp);
        let truth = if so.len() % 2 == 1 {
            so[so.len() / 2]
        } else {
            f64::midpoint(so[so.len() / 2 - 1], so[so.len() / 2])
        };
        assert_eq!(m.median(), truth, "median after {} values", k + 1);
    }
    // MAD: median 5, |x − 5| = {2,2,0,1,1} ⇒ MAD 1 ; 5 + 3·1.4826·1 is the fence
    let base = [3.0, 4.0, 5.0, 6.0, 7.0];
    let mut det = MadDetector::new(3.0);
    for &x in &base {
        det.observe(x);
    }
    assert_eq!(det.median(), 5.0);
    assert_eq!(det.mad(), 1.0);
    assert!(!det.is_anomaly(9.4), "5 + 4.4 < 3·1.4826");
    assert!(det.is_anomaly(9.5), "5 + 4.5 > 3·1.4826");
    assert!(det.is_anomaly(-100.0));
    assert!(
        (det.anomaly_score(9.4478) - 3.0).abs() < 1e-3,
        "score = |x − med| / (1.4826·MAD)"
    );
    // EWMA / EWMV recursion in f64
    let (alpha, k) = (0.2, 3.0);
    let mut e = EwmaDetector::new(alpha, k);
    let (mut ewma, mut var) = (0.0f64, 0.0f64);
    let stream = [10.0, 10.5, 9.5, 10.2, 9.8, 10.1, 30.0, 10.0];
    for (i, &x) in stream.iter().enumerate() {
        if i >= 3 {
            let expected_anomaly = (x - ewma).abs() > k * var.sqrt();
            assert_eq!(e.is_anomaly(x), expected_anomaly, "step {i} ({x})");
            if var > 0.0 {
                assert!((e.anomaly_score(x) - (x - ewma).abs() / var.sqrt()).abs() < 1e-9);
            }
        }
        e.observe(x);
        if i == 0 {
            ewma = x;
        } else {
            let d = x - ewma;
            ewma += alpha * d;
            var = (1.0 - alpha) * (var + alpha * d * d);
        }
    }
    // after the spike the f64 state says exactly what is anomalous now
    for probe in [30.0, 10.05, -20.0] {
        assert_eq!(
            e.is_anomaly(probe),
            (probe - ewma).abs() > k * var.sqrt(),
            "probe {probe}"
        );
    }
}

// ───────────────────────── privacy ────────────────────────────────────────

#[test]
fn privacy_primitives_are_deterministic_and_statistically_calibrated() {
    use alice_analytics::privacy::{LaplaceNoise, RandomizedResponse, XorShift64};
    let mut a = XorShift64::new(0xDEAD_BEEF);
    let mut b = XorShift64::new(0xDEAD_BEEF);
    for _ in 0..1000 {
        let (x, y) = (a.next_f64(), b.next_f64());
        assert_eq!(x.to_bits(), y.to_bits(), "same seed ⇒ same stream");
        assert!((0.0..1.0).contains(&x));
    }
    // Laplace(b): mean 0, variance 2b²  (n = 200 000 ⇒ SE of the mean = b·√2/√n)
    let scale = 2.0f64; // sensitivity 1, ε = 0.5
    let mut lap = LaplaceNoise::with_seed(1.0, 0.5, 42);
    let n = 200_000;
    let (mut s1, mut s2) = (0.0f64, 0.0f64);
    for _ in 0..n {
        let v = lap.sample();
        s1 += v;
        s2 += v * v;
    }
    let mean = s1 / n as f64;
    let var = s2 / n as f64 - mean * mean;
    assert!(
        mean.abs() < 5.0 * scale * 2f64.sqrt() / (n as f64).sqrt(),
        "Laplace mean {mean}"
    );
    assert!(
        ((var - 2.0 * scale * scale) / (2.0 * scale * scale)).abs() < 0.03,
        "Laplace variance {var} vs {}",
        2.0 * scale * scale
    );
    // privatize(v) = v + sample: two seeds in lock-step give the same offset
    let mut l1 = LaplaceNoise::with_seed(1.0, 0.5, 99);
    let mut l2 = LaplaceNoise::with_seed(1.0, 0.5, 99);
    assert_eq!((l1.privatize(10.0) - 10.0).to_bits(), l2.sample().to_bits());
    // randomized response: P(report 1) = p·t + (1 − p)/2, and the estimator inverts it
    let (p, t) = (0.75f64, 0.3f64);
    let mut rr = RandomizedResponse::with_probability(p, 7);
    let mut seed = 11u64;
    let (mut ones, n) = (0u64, 100_000u64);
    for _ in 0..n {
        let truth = lcg(&mut seed) < t;
        if rr.privatize(truth) {
            ones += 1;
        }
    }
    let expected_rate = p * t + (1.0 - p) / 2.0;
    let observed = ones as f64 / n as f64;
    assert!(
        (observed - expected_rate).abs() < 0.01,
        "report rate {observed} vs {expected_rate}"
    );
    let est = RandomizedResponse::estimate_proportion(p, n, ones);
    assert!((est - t).abs() < 0.02, "unbiased estimate {est} vs {t}");
    // exact algebra of the estimator on the expected count
    let exact = RandomizedResponse::estimate_proportion(
        p,
        1_000_000,
        (expected_rate * 1_000_000.0).round() as u64,
    );
    assert!((exact - t).abs() < 1e-5);
}
