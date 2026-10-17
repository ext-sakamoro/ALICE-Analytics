//! The four library max sites follow maximumNumber on the inputs that reach
//! them, checked by bit pattern through the public API (ALICE-Law G2).
//!
//! oracle: IEEE 754-2019 maximumNumber: a NaN operand yields the other
//! operand, and of an equal pair of zeros the result is +0. A comparison
//! (`if a > b { a } else { b }`) differs on both: it returns the NaN when the
//! NaN is the second operand, and it returns either zero by operand order.
//!
//! Sites and what reaches them:
//! - `CompositeDetector::anomaly_score`: a non-finite value in the history
//!   makes the EWMA and z-score components NaN while the MAD component stays
//!   finite; the score must be the MAD score, not NaN.
//! - `PrivacyBudget::remaining`: the budget is not validated, so -0.0 and NaN
//!   budgets reach `max64(budget - spent, 0.0)`; both must give +0.0.
//! - `ResidualSummary::max_abs` takes the larger of two absolute values of
//!   finite residuals (non-finite residuals are refused), so neither a NaN
//!   nor a -0.0 operand reaches it; no input distinguishes a comparison there.
//! - the internal `f64_usize` casts the result to `usize`, which maps NaN,
//!   -0.0 and +0.0 to 0 alike; no input distinguishes a comparison there.
//!
//! With a constant +0.0 second operand, `if r > 0.0 { r } else { 0.0 }` is
//! maximumNumber itself (NaN, -0.0 and +0.0 all give +0.0, a positive r gives
//! r), so that rewrite of `remaining` or `f64_usize` is an equivalent mutant
//! and no test can or should fail on it; the `>=` and `< 0.0` rewrites differ
//! and fail here.

// PrivacyBudget is deprecated (removed in 0.5.0); its remaining() is one of
// the four routed max sites until then, so the tie tests still pin it
#![allow(deprecated)]

use alice_analytics::anomaly::{CompositeDetector, MadDetector};
use alice_analytics::privacy::PrivacyBudget;

fn history(poison: f64) -> impl Iterator<Item = f64> {
    (0..64).map(move |i| {
        if i == 20 {
            poison
        } else {
            f64::from(i % 7) * 1.5 - 3.0
        }
    })
}

/// A NaN component score is skipped, so the composite equals the finite MAD
/// score bit for bit (a comparison with the NaN second would return NaN).
#[test]
fn composite_score_skips_a_nan_component() {
    for poison in [f64::NAN, f64::INFINITY] {
        let mut composite = CompositeDetector::new();
        let mut mad = MadDetector::new(3.0);
        for v in history(poison) {
            composite.observe(v);
            mad.observe(v);
        }
        for q in [1.0, -0.0, 0.0, 100.0] {
            let expected = mad.anomaly_score(q);
            assert!(
                expected.is_finite(),
                "the MAD component must stay finite here"
            );
            let got = composite.anomaly_score(q);
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "poison {poison:?}, value {q:?}: composite {got:?} vs MAD {expected:?}"
            );
        }
    }
}

/// `remaining` of a -0.0 budget is +0.0 (a `>=` comparison would keep -0.0).
#[test]
fn remaining_of_a_negative_zero_budget_is_positive_zero() {
    let r = PrivacyBudget::new(-0.0).remaining();
    assert_eq!(r.to_bits(), 0.0_f64.to_bits(), "got {r:?}");
}

/// `remaining` of a NaN budget is +0.0 (a comparison that keeps the
/// difference unless it is below zero would return NaN).
#[test]
fn remaining_of_a_nan_budget_is_positive_zero() {
    let r = PrivacyBudget::new(f64::NAN).remaining();
    assert_eq!(r.to_bits(), 0.0_f64.to_bits(), "got {r:?}");
}

/// `remaining` of a +0.0 budget is +0.0.
#[test]
fn remaining_of_a_zero_budget_is_positive_zero() {
    let r = PrivacyBudget::new(0.0).remaining();
    assert_eq!(r.to_bits(), 0.0_f64.to_bits(), "got {r:?}");
}
