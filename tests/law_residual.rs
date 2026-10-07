//! Oracles for `law::ResidualSketch` / `law::residual_summary`.
//!
//! Every expected value is written here from a closed form: the laws are
//! built from stored parts so that `f(x)` is exact in f64, the residuals are
//! chosen integers (or dyadic fractions), and the exact quantile is the order
//! statistic `x_(⌈q·n⌉)` of the sorted residuals — the rank the `DDSketch`
//! reports. The bound checked for each reported quantile is the sketch's
//! documented one: `|q̂ − q| ≤ α·|q|` for `|q|` inside the accurate range.
#![cfg(feature = "law")]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::unreadable_literal
)]

use alice_analytics::law::{
    residual_summary, PointClass, ResidualError, ResidualSketch, DEFAULT_RELATIVE_ACCURACY,
};
use alice_zip::law::{Provenance, ResidualStats, SignalLaw, SignalLawParts, ValidRange};

/// Rounding allowance on top of α: a value on a bin edge `γ^i` sits exactly
/// α from the representative, and `ln` / `powf` may round it across.
const ROUND: f64 = 1e-12;

/// `f(x) = c0 + c1·u` with `u = (x − lo) / (hi − lo)`; with `lo = 0`, `hi = 8`
/// the division is by a power of two, so `f` is exact for dyadic `x`
fn line_law(c0: f64, c1: f64, evidence: Vec<(f64, f64)>) -> SignalLaw {
    SignalLaw::from_parts(SignalLawParts {
        coefficients: vec![c0, c1],
        domain: ValidRange { lo: 0.0, hi: 8.0 },
        evidence,
        residual: ResidualStats {
            n: 0,
            rms: 0.0,
            max_abs: 0.0,
        },
        provenance: Provenance::new("synthetic", "closed form"),
        oracles: Vec::new(),
    })
    .expect("valid parts")
}

/// `y = 1 + 2x` on `[0, 8]`, fitted exactly (evidence RMS 0 ⇒ scale 1)
fn exact_line() -> SignalLaw {
    let ev: Vec<(f64, f64)> = (0..=8)
        .map(|i| (f64::from(i), 1.0 + 2.0 * f64::from(i)))
        .collect();
    line_law(1.0, 16.0, ev)
}

fn truth(x: f64) -> f64 {
    1.0 + 2.0 * x
}

/// points at dyadic `x` in `[0, 8]` whose residuals are exactly `rs`
fn points_with(rs: &[f64]) -> Vec<(f64, f64)> {
    rs.iter()
        .enumerate()
        .map(|(i, &r)| {
            let x = f64::from((i % 33) as u32) * 0.25; // 0, 0.25, …, 8
            (x, truth(x) + r)
        })
        .collect()
}

/// order statistic at rank ⌈q·n⌉ (1-based), the rank `DDSketch` reports
fn exact_quantile(rs: &[f64], q: f64) -> f64 {
    let mut s = rs.to_vec();
    s.sort_by(f64::total_cmp);
    let rank = ((q * s.len() as f64).ceil() as usize).max(1);
    s[rank - 1]
}

fn assert_rel(got: f64, want: f64, alpha: f64, what: &str) {
    let bound = alpha.mul_add(want.abs(), ROUND * want.abs());
    assert!(
        (got - want).abs() <= bound,
        "{what}: got {got}, exact {want}, |err| {} > α·|q| = {bound}",
        (got - want).abs()
    );
}

// ---------------------------------------------------------------------------
// quantiles against exact order statistics
// ---------------------------------------------------------------------------

#[test]
fn quantiles_of_uniform_positive_residuals_are_within_alpha() {
    let rs: Vec<f64> = (1..=1000).map(f64::from).collect();
    let law = exact_line();
    let s = residual_summary(&law, &points_with(&rs));
    assert_eq!(s.count, 1000);
    assert_eq!(s.out_of_range, 0);
    assert_eq!(s.non_finite, 0);
    assert_eq!(s.relative_accuracy, DEFAULT_RELATIVE_ACCURACY);
    assert_eq!(s.outside_accuracy, 0);
    let d = s.distribution.expect("non-empty");
    // closed forms: x_(500) = 500, x_(900) = 900, x_(990) = 990
    assert_rel(d.p50, 500.0, 0.01, "p50");
    assert_rel(d.p90, 900.0, 0.01, "p90");
    assert_rel(d.p99, 990.0, 0.01, "p99");
    // min / max / mean are exact, not sketched: Σ1..1000 = 500500
    assert_eq!(d.min, 1.0);
    assert_eq!(d.max, 1000.0);
    assert_eq!(d.max_abs, 1000.0);
    assert_eq!(d.mean, 500.5);
}

#[test]
fn every_percentile_is_within_alpha_for_three_accuracies() {
    let rs: Vec<f64> = (1..=2000).map(|i| f64::from(i) * 0.125).collect();
    let law = exact_line();
    let pts = points_with(&rs);
    for alpha in [0.005, 0.01, 0.05] {
        let mut sk = ResidualSketch::new(&law, alpha).expect("valid α");
        sk.extend(&pts);
        for k in 1..=99 {
            let q = f64::from(k) / 100.0;
            let got = sk.quantile(q).expect("non-empty");
            assert_rel(
                got,
                exact_quantile(&rs, q),
                alpha,
                &format!("α {alpha} q {q}"),
            );
        }
    }
}

#[test]
fn signed_residuals_keep_order_and_exact_zero() {
    // odd residuals −999, −997, …, 999 plus ten exact zeros
    let mut rs: Vec<f64> = (1..=1000).map(|i| f64::from(2 * i - 1001)).collect();
    rs.extend(core::iter::repeat_n(0.0, 10));
    let law = exact_line();
    let mut sk = ResidualSketch::new(&law, 0.01).unwrap();
    sk.extend(&points_with(&rs));
    let mut prev = f64::NEG_INFINITY;
    for k in 1..=99 {
        let q = f64::from(k) / 100.0;
        let want = exact_quantile(&rs, q);
        let got = sk.quantile(q).unwrap();
        assert_rel(got, want, 0.01, &format!("q {q}"));
        assert!(
            got >= prev,
            "quantiles must not decrease: q {q} gave {got} < {prev}"
        );
        prev = got;
        if want == 0.0 {
            assert_eq!(got, 0.0, "a zero residual is reported as exactly 0 (q {q})");
        }
    }
    let d = sk.summary().distribution.unwrap();
    assert_eq!(d.mean, 0.0); // Σ(2i − 1001) = 0
    assert_eq!(d.min, -999.0);
    assert_eq!(d.max, 999.0);
    assert_eq!(d.max_abs, 999.0);
}

#[test]
fn all_negative_residuals() {
    let rs: Vec<f64> = (1..=1000).map(|i| -f64::from(i)).collect();
    let s = residual_summary(&exact_line(), &points_with(&rs));
    let d = s.distribution.unwrap();
    // ascending −1000 … −1: x_(500) = −501, x_(900) = −101, x_(990) = −11
    assert_rel(d.p50, -501.0, 0.01, "p50");
    assert_rel(d.p90, -101.0, 0.01, "p90");
    assert_rel(d.p99, -11.0, 0.01, "p99");
    assert_eq!(d.max, -1.0);
    assert_eq!(d.max_abs, 1000.0);
    assert_eq!(d.mean, -500.5);
}

#[test]
fn residual_is_y_minus_law_not_y() {
    // a law with a slope: residuals 3 everywhere, while y itself spans 1..17
    let rs = vec![3.0; 33];
    let d = residual_summary(&exact_line(), &points_with(&rs))
        .distribution
        .unwrap();
    assert_eq!(d.min, 3.0);
    assert_eq!(d.max, 3.0);
    assert_eq!(d.mean, 3.0);
    assert_rel(d.p50, 3.0, 0.01, "p50");
}

// ---------------------------------------------------------------------------
// valid range: no extrapolation
// ---------------------------------------------------------------------------

#[test]
fn points_outside_the_valid_range_are_counted_and_excluded() {
    let law = exact_line();
    let mut pts = points_with(&[1.0, 2.0, 3.0, 4.0]);
    // a residual of 1e6 at each, if they were (wrongly) extrapolated
    for x in [
        -0.25,
        8.25,
        -1e9,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        pts.push((x, truth(x) + 1e6));
    }
    let s = residual_summary(&law, &pts);
    assert_eq!(s.count, 4);
    assert_eq!(s.out_of_range, 6);
    assert_eq!(s.non_finite, 0);
    let d = s.distribution.unwrap();
    assert_eq!(d.max, 4.0);
    assert_eq!(d.mean, 2.5);
}

#[test]
fn range_edges_are_inside() {
    let law = exact_line();
    let s = residual_summary(&law, &[(0.0, truth(0.0) + 2.0), (8.0, truth(8.0) + 4.0)]);
    assert_eq!(s.count, 2);
    assert_eq!(s.out_of_range, 0);
    assert_eq!(s.distribution.unwrap().mean, 3.0);
}

#[test]
fn push_reports_the_class_of_each_point() {
    let law = exact_line();
    let mut sk = ResidualSketch::new(&law, 0.01).unwrap();
    assert_eq!(sk.push(2.0, 7.5), PointClass::Summarised { residual: 2.5 });
    assert_eq!(sk.push(9.0, 0.0), PointClass::OutOfRange);
    assert_eq!(sk.push(f64::NAN, 0.0), PointClass::OutOfRange);
    assert_eq!(sk.push(2.0, f64::NAN), PointClass::NonFinite);
    assert_eq!(sk.push(2.0, f64::INFINITY), PointClass::NonFinite);
    let s = sk.summary();
    assert_eq!((s.count, s.out_of_range, s.non_finite), (1, 2, 2));
}

// ---------------------------------------------------------------------------
// degenerate input
// ---------------------------------------------------------------------------

#[test]
fn no_points_gives_no_distribution() {
    let law = exact_line();
    let s = residual_summary(&law, &[]);
    assert_eq!((s.count, s.out_of_range, s.non_finite), (0, 0, 0));
    assert!(s.distribution.is_none());
    let sk = ResidualSketch::new(&law, 0.01).unwrap();
    assert_eq!(sk.quantile(0.5), None);
}

#[test]
fn all_points_out_of_range_gives_no_distribution() {
    let s = residual_summary(&exact_line(), &[(-1.0, 0.0), (9.0, 0.0), (f64::NAN, 1.0)]);
    assert_eq!((s.count, s.out_of_range), (0, 3));
    assert!(s.distribution.is_none());
}

#[test]
fn nan_y_is_counted_and_does_not_poison_the_mean() {
    let pts = [
        (1.0, truth(1.0) + 1.0),
        (2.0, f64::NAN),
        (3.0, truth(3.0) + 3.0),
        (4.0, f64::NEG_INFINITY),
    ];
    let s = residual_summary(&exact_line(), &pts);
    assert_eq!((s.count, s.non_finite), (2, 2));
    let d = s.distribution.unwrap();
    assert_eq!(d.mean, 2.0);
    assert!(d.p50.is_finite() && d.p99.is_finite());
}

#[test]
fn quantile_outside_unit_interval_or_nan_is_none() {
    let law = exact_line();
    let mut sk = ResidualSketch::new(&law, 0.01).unwrap();
    sk.extend(&points_with(&[1.0, 2.0]));
    assert_eq!(sk.quantile(-0.01), None);
    assert_eq!(sk.quantile(1.01), None);
    assert_eq!(sk.quantile(f64::NAN), None);
    assert!(sk.quantile(0.0).is_some() && sk.quantile(1.0).is_some());
}

#[test]
fn invalid_accuracy_and_scale_are_rejected() {
    let law = exact_line();
    for a in [0.0, 1.0, -0.1, 1.5, f64::NAN, f64::INFINITY] {
        assert_eq!(
            ResidualSketch::new(&law, a).err(),
            Some(ResidualError::InvalidAccuracy),
            "α = {a}"
        );
    }
    for sc in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::MIN_POSITIVE / 4.0] {
        assert_eq!(
            ResidualSketch::with_scale(&law, 0.01, sc).err(),
            Some(ResidualError::InvalidScale),
            "scale = {sc}"
        );
    }
}

// ---------------------------------------------------------------------------
// the accurate range and the residual scale
// ---------------------------------------------------------------------------

/// `DDSketch2048` (offset `2048/4 = 512`): bins cover `(γ^(i−1), γ^i]` for
/// `i = idx − 512`, `idx ∈ 0..2048` ⇒ guaranteed magnitudes
/// `[γ^−512, γ^1535]`, `γ = (1 + α)/(1 − α)`
fn closed_form_range(alpha: f64, scale: f64) -> (f64, f64) {
    let g = (1.0 + alpha) / (1.0 - alpha);
    (scale * g.powi(-512), scale * g.powi(1535))
}

#[test]
fn accurate_range_matches_the_bin_layout() {
    let law = exact_line();
    let sk = ResidualSketch::new(&law, 0.01).unwrap();
    assert_eq!(sk.scale(), 1.0); // evidence RMS 0 ⇒ unit scale
    let (lo, hi) = sk.accurate_range();
    let (wlo, whi) = closed_form_range(0.01, 1.0);
    assert!(
        (lo / wlo - 1.0).abs() < 1e-9 && (hi / whi - 1.0).abs() < 1e-9,
        "{lo} {hi} vs {wlo} {whi}"
    );
    let s = sk.summary();
    assert_eq!(s.accurate_range, (lo, hi));
}

#[test]
fn explicit_scale_is_rounded_down_to_a_power_of_two() {
    let law = exact_line();
    let sk = ResidualSketch::with_scale(&law, 0.01, 3.0).unwrap();
    assert_eq!(sk.scale(), 2.0);
    let ((lo, hi), (wlo, whi)) = (sk.accurate_range(), closed_form_range(0.01, 2.0));
    assert!(
        (lo / wlo - 1.0).abs() < 1e-9 && (hi / whi - 1.0).abs() < 1e-9,
        "{lo} {hi} vs {wlo} {whi}"
    );
    let sk = ResidualSketch::with_scale(&law, 0.01, 0.75).unwrap();
    assert_eq!(sk.scale(), 0.5);
    let sk = ResidualSketch::with_scale(&law, 0.01, 1e-9).unwrap();
    assert_eq!(sk.scale(), 2f64.powi(-30)); // 2^-30 ≈ 9.31e-10 ≤ 1e-9 < 2^-29
}

/// a law fitted to noisy evidence with RMS ≈ 1e-7: residuals of that size are
/// far below the unit-scale floor γ^−512 ≈ 3.6e-5, so at unit scale they lose
/// the guarantee; the default scale follows the law's own residual
fn small_noise_law() -> SignalLaw {
    // y = 1 + 2x ± 1e-7, alternating, at x = 0..8
    let ev: Vec<(f64, f64)> = (0..=8)
        .map(|i| {
            let x = f64::from(i);
            (x, truth(x) + if i % 2 == 0 { 1e-7 } else { -1e-7 })
        })
        .collect();
    line_law(1.0, 16.0, ev)
}

#[test]
fn default_scale_follows_the_law_residual() {
    let law = small_noise_law();
    let rms = law.residual().rms;
    assert!((rms / 1e-7 - 1.0).abs() < 1e-6, "rms {rms}");
    let sk = ResidualSketch::new(&law, 0.01).unwrap();
    // largest power of two ≤ 1e-7 is 2^-24 ≈ 5.96e-8
    assert_eq!(sk.scale(), 2f64.powi(-24));

    let rs: Vec<f64> = (1..=1000).map(|i| f64::from(i) * 1e-9).collect(); // 1e-9 … 1e-6
    let pts: Vec<(f64, f64)> = rs.iter().map(|&r| (4.0, truth(4.0) + r)).collect();
    // y = 9 + r is rounded to f64 near 9 (ulp 1.8e-15), so the residual the
    // sketch sees is y − 9, not r: the exact set is recomputed in the same way
    let seen: Vec<f64> = pts.iter().map(|&(x, y)| y - truth(x)).collect();

    let mut sk = ResidualSketch::new(&law, 0.01).unwrap();
    sk.extend(&pts);
    let s = sk.summary();
    assert_eq!(s.outside_accuracy, 0);
    // min / max are exact through the power-of-two scale; the mean is the
    // two-pass mean of the same residuals up to summation rounding
    let d = s.distribution.unwrap();
    let lo = seen.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = seen.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert_eq!((d.min, d.max), (lo, hi));
    let mean = seen.iter().sum::<f64>() / seen.len() as f64;
    assert!(
        (d.mean / mean - 1.0).abs() < 1e-12,
        "mean {} vs {mean}",
        d.mean
    );
    for k in 1..=99 {
        let q = f64::from(k) / 100.0;
        assert_rel(
            sk.quantile(q).unwrap(),
            exact_quantile(&seen, q),
            0.01,
            &format!("q {q}"),
        );
    }

    // the same residuals at unit scale fall below the floor and are counted
    let mut unit = ResidualSketch::with_scale(&law, 0.01, 1.0).unwrap();
    unit.extend(&pts);
    assert_eq!(unit.summary().outside_accuracy, 1000);
}

#[test]
fn exact_zero_residual_is_not_outside_accuracy() {
    let s = residual_summary(&exact_line(), &points_with(&[0.0, 0.0, 1.0]));
    assert_eq!(s.outside_accuracy, 0);
    assert_eq!(s.distribution.unwrap().p50, 0.0);
}

#[test]
fn residual_above_the_accurate_range_is_counted() {
    // scale 2^-24 ⇒ the top of the accurate range is 2^-24·γ^1535 ≈ 1.2e6
    let law = small_noise_law();
    let mut sk = ResidualSketch::new(&law, 0.01).unwrap();
    let (_, hi) = closed_form_range(0.01, 2f64.powi(-24));
    assert!(hi < 1e7);
    sk.push(4.0, truth(4.0) + 1e-7);
    sk.push(4.0, truth(4.0) + 1e7);
    sk.push(4.0, truth(4.0) - 1e7);
    let s = sk.summary();
    assert_eq!((s.count, s.outside_accuracy), (3, 2));
}
