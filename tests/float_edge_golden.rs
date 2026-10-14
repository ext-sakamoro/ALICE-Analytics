//! Bit-exact golden values for the IEEE edge cases of the public float
//! functions: signed zero, catastrophic cancellation between near-equal large
//! values, opposite-sign sums to zero, collinear points far from the origin
//! and zero-variance input.
//!
//! `tests/determinism_golden.rs` hashes long fixed scenarios; a hash says *that*
//! something moved but not *which* value. Each test here pins one value with
//! `to_bits()` (never a tolerance), so a dropped sign, a re-associated sum or a
//! `mul_add` whose operands were swapped fails with the exact input named in
//! the test. Every pinned value was checked against the closed form before it
//! was recorded (the expected decimal value is in the comment of each case).
//!
//! The values are the same in debug and `--release`: CI runs this file in both
//! profiles (`.github/workflows/release-profile.yml`).
//!
//! Not pinned on purpose:
//! - `min()` / `max()` of a stream that holds both `-0.0` and `+0.0`: the
//!   result depends on the arrival order (`[-0.0, +0.0]` gives `max = -0.0`,
//!   `[+0.0, -0.0]` gives `min = +0.0`), unlike IEEE 754-2019 `minimum` /
//!   `maximum`, which order `-0.0 < +0.0`. Pinning it would freeze that.
//! - the `privacy` module: deprecated, and its values are already pinned in
//!   `tests/determinism_golden.rs`.

use alice_analytics::anomaly::EwmaDetector;
use alice_analytics::stats::{iqr, quantile_sorted, StreamingStats};
use alice_analytics::streaming_ops::{ExponentialMovingAverage, LinearRegression};
use alice_analytics::window::SlidingWindow;

const NEG_ZERO: u64 = 0x8000_0000_0000_0000;
const POS_ZERO: u64 = 0x0000_0000_0000_0000;

fn stats_of(values: &[f64]) -> StreamingStats {
    let mut s = StreamingStats::new();
    for &v in values {
        s.observe(v);
    }
    s
}

// ---------------------------------------------------------------------------
// StreamingStats (Welford / Pebay)
// ---------------------------------------------------------------------------

#[test]
fn streaming_stats_mean_of_single_negative_zero_is_positive_zero() {
    // input [-0.0]: mean = 0.0 + (-0.0 - 0.0) / 1 and IEEE gives +0.0 + -0.0 = +0.0
    // (the accumulator starts at +0.0, as a sum would)
    let s = stats_of(&[-0.0]);
    assert_eq!(s.mean().to_bits(), POS_ZERO);
}

#[test]
fn streaming_stats_variance_of_near_equal_large_values() {
    // input 1e8 + {0, 1, 2}: population variance 2/3, sample variance exactly 1
    // (a sum-of-squares formula loses every digit here: 3e16 vs 3e16 + 2)
    let s = stats_of(&[1e8, 1e8 + 1.0, 1e8 + 2.0]);
    assert_eq!(s.mean().to_bits(), 0x4197_d784_0400_0000); // 100000001.0
    assert_eq!(s.variance().to_bits(), 0x3fe5_5555_5555_5555); // 0.6666666666666666
    assert_eq!(s.sample_variance().to_bits(), 0x3ff0_0000_0000_0000); // 1.0
}

#[test]
fn streaming_stats_skewness_of_symmetric_large_values_is_zero() {
    // input 1e8 + {0, 1, 2}: symmetric, so the third moment cancels to exactly +0.0
    let s = stats_of(&[1e8, 1e8 + 1.0, 1e8 + 2.0]);
    assert_eq!(s.skewness().to_bits(), POS_ZERO);
}

#[test]
fn streaming_stats_kurtosis_of_four_near_equal_large_values() {
    // input 1e8 + {0, 1, 2, 3}: m2 = 5, m4 = 10.25, excess kurtosis 4 * 10.25 / 25 - 3 = -1.36
    // and variance 1.25, both exact after the 1e8 offset cancels
    let s = stats_of(&[1e8, 1e8 + 1.0, 1e8 + 2.0, 1e8 + 3.0]);
    assert_eq!(s.variance().to_bits(), 0x3ff4_0000_0000_0000); // 1.25
    assert_eq!(s.kurtosis().to_bits(), 0xbff5_c28f_5c28_f5c3); // -1.36
}

#[test]
fn streaming_stats_opposite_sign_values_mean_zero() {
    // input [1e16, -1e16]: the mean cancels to exactly +0.0, variance (1e16)^2 = 1e32
    let s = stats_of(&[1e16, -1e16]);
    assert_eq!(s.mean().to_bits(), POS_ZERO);
    assert_eq!(s.variance().to_bits(), 0x4693_b8b5_b505_6e17); // 1e32
}

#[test]
fn streaming_stats_constant_input_has_zero_moments() {
    // input 0.1 five times (0.1 is not representable): variance, skewness and
    // kurtosis must be exactly +0.0, never a tiny negative variance
    let s = stats_of(&[0.1; 5]);
    assert_eq!(s.mean().to_bits(), 0x3fb9_9999_9999_999a); // 0.1
    assert_eq!(s.variance().to_bits(), POS_ZERO);
    assert_eq!(s.skewness().to_bits(), POS_ZERO);
    assert_eq!(s.kurtosis().to_bits(), POS_ZERO);
}

// ---------------------------------------------------------------------------
// LinearRegression
// ---------------------------------------------------------------------------

#[test]
fn linear_regression_collinear_points_with_large_mean_x() {
    // input y = 2x + 1 at x = 1e8 + {0, 1, 2}: slope 2 and intercept 1 exactly;
    // the intercept is mean_y - slope * mean_x = 200000003 - 200000002, a
    // cancellation the fused `mul_add` must keep exact
    let mut r = LinearRegression::new();
    for x in [1e8, 1e8 + 1.0, 1e8 + 2.0] {
        r.observe(x, 2.0f64.mul_add(x, 1.0));
    }
    assert_eq!(r.slope().to_bits(), 0x4000_0000_0000_0000); // 2.0
    assert_eq!(r.intercept().to_bits(), 0x3ff0_0000_0000_0000); // 1.0
    assert_eq!(r.predict(0.0).to_bits(), 0x3ff0_0000_0000_0000); // 1.0
}

#[test]
fn linear_regression_flat_negative_zero_line_is_positive_zero() {
    // input y = -0.0 at x = 1e8 + {0, 1, 2}: the co-moment cancels to +0.0, so
    // slope +0.0 and intercept +0.0 * -mean_x + +0.0 = +0.0
    let mut r = LinearRegression::new();
    for x in [1e8, 1e8 + 1.0, 1e8 + 2.0] {
        r.observe(x, -0.0);
    }
    assert_eq!(r.slope().to_bits(), POS_ZERO);
    assert_eq!(r.intercept().to_bits(), POS_ZERO);
}

#[test]
fn linear_regression_zero_x_variance_returns_mean_y() {
    // input (5, 1) three times: m2_x = 0, so slope takes the guarded 0.0 and the
    // intercept is mean_y = 1.0 (no division by zero, no NaN)
    let mut r = LinearRegression::new();
    for _ in 0..3 {
        r.observe(5.0, 1.0);
    }
    assert_eq!(r.slope().to_bits(), POS_ZERO);
    assert_eq!(r.intercept().to_bits(), 0x3ff0_0000_0000_0000); // 1.0
}

// ---------------------------------------------------------------------------
// ExponentialMovingAverage / EwmaDetector
// ---------------------------------------------------------------------------

#[test]
fn ema_of_negative_zeros_keeps_negative_zero() {
    // input [-0.0, -0.0], alpha 0.5: the first value is taken as is, then
    // 0.5 * -0.0 + 0.5 * -0.0 = -0.0 + -0.0 = -0.0 (IEEE keeps the sign of a sum of -0.0)
    let mut e = ExponentialMovingAverage::new(0.5);
    e.observe(-0.0);
    assert_eq!(e.value().to_bits(), NEG_ZERO);
    e.observe(-0.0);
    assert_eq!(e.value().to_bits(), NEG_ZERO);
}

#[test]
fn ema_of_opposite_large_values_cancels_to_positive_zero() {
    // input [1e8, -1e8], alpha 0.5: 0.5 * -1e8 + 0.5 * 1e8 = exactly +0.0
    let mut e = ExponentialMovingAverage::new(0.5);
    e.observe(1e8);
    e.observe(-1e8);
    assert_eq!(e.value().to_bits(), POS_ZERO);
}

#[test]
fn ema_asymmetric_alpha_on_near_equal_large_values() {
    // input [1e8, 1e8 + 4], alpha 0.25: 0.25 * (1e8 + 4) + 0.75 * 1e8 = 1e8 + 1 exactly
    // (alpha != 0.5, so swapping the new and the previous value gives 1e8 + 3)
    let mut e = ExponentialMovingAverage::new(0.25);
    e.observe(1e8);
    e.observe(1e8 + 4.0);
    assert_eq!(e.value().to_bits(), 0x4197_d784_0400_0000); // 100000001.0
}

#[test]
fn ewma_detector_of_negative_zeros_is_positive_zero() {
    // input [-0.0, -0.0]: ewma = -0.0 + 0.5 * (-0.0 - -0.0) = -0.0 + +0.0 = +0.0
    // (the detector updates by a difference, unlike ExponentialMovingAverage),
    // deviation estimate +0.0
    let mut d = EwmaDetector::new(0.5, 3.0);
    d.observe(-0.0);
    d.observe(-0.0);
    assert_eq!(d.ewma().to_bits(), POS_ZERO);
    assert_eq!(d.std_dev().to_bits(), POS_ZERO);
}

#[test]
fn ewma_detector_constant_large_input_scores() {
    // input 1e8 + 1 four times: zero spread, so the same value scores +0.0 and
    // any other value scores +inf (the documented zero-variance contract)
    let mut d = EwmaDetector::new(0.5, 3.0);
    for _ in 0..4 {
        d.observe(1e8 + 1.0);
    }
    assert_eq!(d.ewma().to_bits(), 0x4197_d784_0400_0000); // 100000001.0
    assert_eq!(d.std_dev().to_bits(), POS_ZERO);
    assert_eq!(d.anomaly_score(1e8 + 1.0).to_bits(), POS_ZERO);
    assert_eq!(
        d.anomaly_score(1e8 + 2.0).to_bits(),
        f64::INFINITY.to_bits()
    );
}

#[test]
fn ewma_detector_near_equal_large_ramp() {
    // input 1e8 + {0, 1, 2}, alpha 0.5: ewma 1e8 + 1.25, variance
    // 0.5 * (0.5 * 1.5^2 + 0.25) = 0.6875, std sqrt(0.6875) = 0.82915619758885,
    // score of 1e8 + 2 is 0.75 / 0.82915619758885 = 0.9045340337332909
    let mut d = EwmaDetector::new(0.5, 3.0);
    for v in [1e8, 1e8 + 1.0, 1e8 + 2.0] {
        d.observe(v);
    }
    assert_eq!(d.ewma().to_bits(), 0x4197_d784_0500_0000); // 100000001.25
    assert_eq!(d.std_dev().to_bits(), 0x3fea_8872_93fd_6f34); // 0.82915619758885
    assert_eq!(
        d.anomaly_score(1e8 + 2.0).to_bits(),
        0x3fec_f1f1_5ba0_1c39 // 0.9045340337332909
    );
}

// ---------------------------------------------------------------------------
// SlidingWindow
// ---------------------------------------------------------------------------

#[test]
fn sliding_window_variance_of_near_equal_large_values() {
    // input 1e8 + {0, 1, 2, 3}: mean 1e8 + 1.5, variance 1.25, std sqrt(1.25)
    let mut w = SlidingWindow::<4>::new();
    for v in [1e8, 1e8 + 1.0, 1e8 + 2.0, 1e8 + 3.0] {
        w.push(v);
    }
    assert_eq!(w.mean().to_bits(), 0x4197_d784_0600_0000); // 100000001.5
    assert_eq!(w.variance().to_bits(), 0x3ff4_0000_0000_0000); // 1.25
    assert_eq!(w.std_dev().to_bits(), 0x3ff1_e377_9b97_f4a8); // 1.118033988749895
}

#[test]
fn sliding_window_constant_input_has_zero_variance() {
    // input 0.1 four times: the deviations from the mean are exactly zero
    let mut w = SlidingWindow::<4>::new();
    for _ in 0..4 {
        w.push(0.1);
    }
    assert_eq!(w.variance().to_bits(), POS_ZERO);
    assert_eq!(w.std_dev().to_bits(), POS_ZERO);
}

// ---------------------------------------------------------------------------
// quantile_sorted / iqr
// ---------------------------------------------------------------------------

#[test]
fn quantile_between_signed_zeros() {
    // [-0.0, -0.0] at q 0.5: -0.0 * 0.5 + -0.0 * 0.5 = -0.0 (sum of two -0.0)
    // [-0.0, +0.0] at q 0.5: -0.0 * 0.5 + +0.0 * 0.5 = +0.0 (mixed-sign sum is +0.0)
    assert_eq!(quantile_sorted(&[-0.0, -0.0], 0.5).to_bits(), NEG_ZERO);
    assert_eq!(quantile_sorted(&[-0.0, 0.0], 0.5).to_bits(), POS_ZERO);
}

#[test]
fn quantile_interpolates_between_near_equal_large_values() {
    // [1e8, 1e8 + 1] at q 0.5: 1e8 + 0.5, exact (the halfway point is representable)
    assert_eq!(
        quantile_sorted(&[1e8, 1e8 + 1.0], 0.5).to_bits(),
        0x4197_d784_0200_0000 // 100000000.5
    );
}

#[test]
fn iqr_fences_of_near_equal_large_values() {
    // input 1e8 + {0, 1, 2, 3}: q1 1e8 + 0.75, q3 1e8 + 2.25, iqr 1.5,
    // fences q1 - 2.25 = 1e8 - 1.5 and q3 + 2.25 = 1e8 + 4.5
    let r = iqr(&[1e8, 1e8 + 1.0, 1e8 + 2.0, 1e8 + 3.0]).expect("n >= 4");
    assert_eq!(r.q1.to_bits(), 0x4197_d784_0300_0000); // 100000000.75
    assert_eq!(r.q3.to_bits(), 0x4197_d784_0900_0000); // 100000002.25
    assert_eq!(r.iqr.to_bits(), 0x3ff8_0000_0000_0000); // 1.5
    assert_eq!(r.lower_fence.to_bits(), 0x4197_d783_fa00_0000); // 99999998.5
    assert_eq!(r.upper_fence.to_bits(), 0x4197_d784_1200_0000); // 100000004.5
}

#[test]
fn iqr_of_negative_zeros() {
    // input [-0.0; 4]: q1 = q3 = -0.0, iqr = -0.0 - -0.0 = +0.0 (x - x is +0.0),
    // lower fence 1.5 * -(+0.0) + -0.0 = -0.0, upper fence 1.5 * +0.0 + -0.0 = +0.0
    let r = iqr(&[-0.0; 4]).expect("n >= 4");
    assert_eq!(r.q1.to_bits(), NEG_ZERO);
    assert_eq!(r.q3.to_bits(), NEG_ZERO);
    assert_eq!(r.iqr.to_bits(), POS_ZERO);
    assert_eq!(r.lower_fence.to_bits(), NEG_ZERO);
    assert_eq!(r.upper_fence.to_bits(), POS_ZERO);
}
