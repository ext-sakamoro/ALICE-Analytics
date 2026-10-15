//! `StreamingStats::observe`, `SlidingWindow::min`/`max` and `DDSketch`'s
//! running extrema (`insert` and `merge`) must report the
//! same `-0.0`/`+0.0` sign for the same multiset of observations regardless
//! of the order they were fed in.
//!
//! All of these sites compared with plain `<`/`>` (`if value < self.min { self.min =
//! value }`): IEEE 754 defines `-0.0 == +0.0`, so neither comparison fires
//! once a running extremum already holds *either* sign of zero — the
//! extremum sticks to whichever sign arrived first. Two logically identical
//! streams that differ only in row order can therefore report a different
//! min/max sign (same architecture, same build profile: this is not the
//! arch-dependent `f64::min`/`max` bug tracked elsewhere, it is the ordinary
//! ordering-comparison idiom applied to a value IEEE defines as a tie).
//!
//! Fixed by `alice_det_math::{min64, max64}`, which this crate already
//! depends on, so both the min and the max resolve to the IEEE
//! minimum/maximum (`-0`/`+0`) independent of arrival order.
//!
//! Every helper feeds its observations through `std::hint::black_box`. The
//! inputs are literals, so without it a release build constant-folds the
//! extremum update and never runs the comparison instruction at all; the
//! `x86_64` `minsd`/`maxsd` path (which returns the second operand on a zero
//! tie, unlike the arm64 `fminnm`/`fmaxnm` path) then goes unexercised and a
//! revert to the std `f64::min`/`max` passes in release.

use alice_analytics::sketch::{DDSketch, Mergeable};
use alice_analytics::stats::StreamingStats;
use alice_analytics::window::SlidingWindow;
use std::hint::black_box;

fn bits(v: f64) -> u64 {
    v.to_bits()
}

const NEG_ZERO: u64 = 0x8000_0000_0000_0000;
const POS_ZERO: u64 = 0x0000_0000_0000_0000;

fn stats_minmax(values: &[f64]) -> (f64, f64) {
    let mut s = StreamingStats::new();
    for &v in values {
        s.observe(black_box(v));
    }
    (s.min(), s.max())
}

fn window_minmax(values: &[f64]) -> (f64, f64) {
    let mut w: SlidingWindow<16> = SlidingWindow::new();
    for &v in values {
        w.push(black_box(v));
    }
    (w.min(), w.max())
}

fn sketch_minmax(values: &[f64]) -> (f64, f64) {
    let mut d = DDSketch::new(0.01);
    for &v in values {
        d.insert(black_box(v));
    }
    (d.min(), d.max())
}

/// Each value in its own sketch, merged in input order: exercises the
/// `merge` path's extrema update rather than `insert`'s.
fn merged_sketch_minmax(values: &[f64]) -> (f64, f64) {
    let mut acc = DDSketch::new(0.01);
    for &v in values {
        let mut one = DDSketch::new(0.01);
        one.insert(black_box(v));
        acc.merge(&one);
    }
    (acc.min(), acc.max())
}

#[test]
fn streaming_stats_min_max_of_mixed_zero_is_order_independent() {
    for values in [[-0.0_f64, 0.0], [0.0_f64, -0.0]] {
        let (min, max) = stats_minmax(&values);
        assert_eq!(
            bits(min),
            NEG_ZERO,
            "min(-0, +0) must be -0 regardless of order, got {:016x}",
            bits(min)
        );
        assert_eq!(
            bits(max),
            POS_ZERO,
            "max(-0, +0) must be +0 regardless of order, got {:016x}",
            bits(max)
        );
    }
}

#[test]
fn sliding_window_min_max_of_mixed_zero_is_order_independent() {
    for values in [[-0.0_f64, 0.0], [0.0_f64, -0.0]] {
        let (min, max) = window_minmax(&values);
        assert_eq!(bits(min), NEG_ZERO, "window min(-0, +0) must be -0");
        assert_eq!(bits(max), POS_ZERO, "window max(-0, +0) must be +0");
    }
}

#[test]
fn ddsketch_insert_min_max_of_mixed_zero_is_order_independent() {
    for values in [[-0.0_f64, 0.0], [0.0_f64, -0.0]] {
        let (min, max) = sketch_minmax(&values);
        assert_eq!(bits(min), NEG_ZERO, "sketch min(-0, +0) must be -0");
        assert_eq!(bits(max), POS_ZERO, "sketch max(-0, +0) must be +0");
    }
}

#[test]
fn ddsketch_merge_min_max_of_mixed_zero_is_order_independent() {
    for values in [[-0.0_f64, 0.0], [0.0_f64, -0.0]] {
        let (min, max) = merged_sketch_minmax(&values);
        assert_eq!(bits(min), NEG_ZERO, "merged sketch min(-0, +0) must be -0");
        assert_eq!(bits(max), POS_ZERO, "merged sketch max(-0, +0) must be +0");
    }
}

#[test]
fn ddsketch_skips_non_finite_before_the_extrema_update() {
    // A rejected NaN must not disturb the zero tie that follows it.
    for values in [[f64::NAN, -0.0_f64, 0.0], [f64::NAN, 0.0_f64, -0.0]] {
        let (min, max) = sketch_minmax(&values);
        assert_eq!(bits(min), NEG_ZERO);
        assert_eq!(bits(max), POS_ZERO);
    }
}

#[test]
fn mixed_zero_alongside_a_third_point_still_resolves_to_ieee_minimum_maximum() {
    for values in [[-0.0_f64, 0.0, 1.0], [0.0_f64, -0.0, 1.0]] {
        let (min, _) = stats_minmax(&values);
        assert_eq!(bits(min), NEG_ZERO);
        let (wmin, _) = window_minmax(&values);
        assert_eq!(bits(wmin), NEG_ZERO);
        let (dmin, _) = sketch_minmax(&values);
        assert_eq!(bits(dmin), NEG_ZERO);
        let (merged_min, _) = merged_sketch_minmax(&values);
        assert_eq!(bits(merged_min), NEG_ZERO);
    }
    for values in [[-1.0_f64, -0.0, 0.0], [-1.0_f64, 0.0, -0.0]] {
        let (_, max) = stats_minmax(&values);
        assert_eq!(bits(max), POS_ZERO);
        let (_, wmax) = window_minmax(&values);
        assert_eq!(bits(wmax), POS_ZERO);
        let (_, dmax) = sketch_minmax(&values);
        assert_eq!(bits(dmax), POS_ZERO);
        let (_, merged_max) = merged_sketch_minmax(&values);
        assert_eq!(bits(merged_max), POS_ZERO);
    }
}

#[test]
fn non_zero_observations_are_unaffected() {
    let (min, max) = stats_minmax(&[1.0, 2.0, 3.0]);
    assert_eq!(bits(min), 1.0_f64.to_bits());
    assert_eq!(bits(max), 3.0_f64.to_bits());

    let (min, max) = stats_minmax(&[-3.0, -2.0, -1.0]);
    assert_eq!(bits(min), (-3.0_f64).to_bits());
    assert_eq!(bits(max), (-1.0_f64).to_bits());

    for f in [sketch_minmax, merged_sketch_minmax] {
        let (min, max) = f(&[5.0, -7.0, 2.5]);
        assert_eq!(bits(min), (-7.0_f64).to_bits());
        assert_eq!(bits(max), 5.0_f64.to_bits());
    }
}
