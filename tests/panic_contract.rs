//! Degenerate-input contracts: what each public entry point does with an
//! empty, out-of-domain, non-finite or extreme input.
//!
//! The crate takes numbers off a live telemetry stream, so the inputs here
//! are not hypothetical: an upstream division by zero sends `inf`, a missing
//! sample sends `NaN`, a clock that has not been set sends a timestamp near
//! `u64::MAX`, and a configuration file can set a relative accuracy of 0.
//!
//! Each test states which of the three possible answers is the correct one
//! for that input and asserts it, rather than asserting only that nothing
//! panicked:
//!
//! * **a value** — the input is inside the contract and has a defined answer
//!   (an empty estimator returns its documented neutral value; a magnitude
//!   beyond the accurate range is filed in the edge bin, keeping its rank but
//!   losing the α bound);
//! * **a saturated value** — the input is outside the representable range and
//!   the answer is the nearest representable one;
//! * **a panic** — the input violates a precondition the caller cannot
//!   recover from (a zero-width window), so the contract is a panic and
//!   `#[should_panic]` pins the message.
//!
//! "Does not panic" on its own is never the assertion: a counter that wraps
//! silently also does not panic, and that is the defect these tests exist to
//! keep out (see `bucket_index` in `src/sketch.rs`, where an `i32` overflow
//! panicked in a debug build and wrapped a magnitude into bin 0 in a release
//! one).
//!
//! The teeth are measured, not assumed: removing the `saturating_add` in
//! `bucket_index` or in `TumblingWindow::insert`, or making the zero-width
//! `push` return early instead of panicking, turns the corresponding test
//! red.

#![allow(clippy::float_cmp)]

use alice_analytics::anomaly::{EwmaDetector, MadDetector, StreamingMedian, ZScoreDetector};
use alice_analytics::pipeline::RingBuffer;
use alice_analytics::privacy::{
    LaplaceNoise, PrivacyBudget, PrivateAggregator, RandomizedResponse,
};
use alice_analytics::sketch::{CountMinSketch, DDSketch, HyperLogLog};
use alice_analytics::stats::{iqr, percentile_rank, quantile_sorted, StreamingStats};
use alice_analytics::streaming_ops::{
    ChangeRate, ExponentialMovingAverage, LinearRegression, SimpleMovingAverage,
};
use alice_analytics::window::{HierarchicalRollup, SlidingWindow, TumblingWindow};

// ---------------------------------------------------------------------------
// Preconditions the caller cannot recover from: a zero-width buffer
// ---------------------------------------------------------------------------
//
// `N` is a type parameter, so `N == 0` is a programming error rather than bad
// data: there is no value the call could return and no error the caller could
// handle. The contract is a panic, and these tests pin it so that a future
// change to a silent early return is a red test rather than a window that
// accepts values and forgets them.

#[test]
#[should_panic(expected = "SlidingWindow needs N > 0")]
fn sliding_window_of_width_zero_panics_on_push() {
    let mut w = SlidingWindow::<0>::new();
    w.push(1.0);
}

#[test]
#[should_panic(expected = "SimpleMovingAverage needs N > 0")]
fn simple_moving_average_of_width_zero_panics_on_observe() {
    let mut sma = SimpleMovingAverage::<0>::new();
    sma.observe(1.0);
}

#[test]
#[should_panic(expected = "RingBuffer needs N > 0")]
fn ring_buffer_of_capacity_zero_panics_on_push() {
    let mut rb = RingBuffer::<u8, 0>::new();
    let _ = rb.push(1);
}

/// Constructing one is still fine, and the read-only accessors answer for the
/// empty state: only the write paths above have a precondition.
#[test]
fn zero_width_buffers_construct_and_read_as_empty() {
    let w = SlidingWindow::<0>::new();
    assert_eq!(w.len(), 0);
    assert!(w.is_empty());
    assert_eq!(w.mean(), 0.0);
    assert_eq!(w.sum(), 0.0);

    let sma = SimpleMovingAverage::<0>::new();
    assert_eq!(sma.value(), 0.0);
    assert!(sma.is_empty());

    let mut rb = RingBuffer::<u8, 0>::new();
    assert_eq!(rb.capacity(), 0, "no usable slot");
    assert!(
        rb.is_full(),
        "a buffer with no usable slot is full from the start"
    );
    assert!(rb.is_empty());
    assert_eq!(rb.len(), 0);
    assert_eq!(rb.pop(), None);
    // N = 1 is the smallest instantiation that can be pushed to; it also has
    // no usable slot, but it refuses the push instead of panicking
    let mut rb1 = RingBuffer::<u8, 1>::new();
    assert_eq!(rb1.capacity(), 0);
    assert!(!rb1.push(7), "a full buffer refuses the item");
    assert_eq!(rb1.dropped(), 1);
}

// ---------------------------------------------------------------------------
// Documented argument contracts
// ---------------------------------------------------------------------------
//
// `ExponentialMovingAverage` already declares these panics in its `# Panics`
// section. They are pinned here because the assertion is the published
// contract: a smoothing factor outside (0, 1] is not a value the type can
// represent, and silently clamping it would produce an average that is not
// the one the caller asked for.

#[test]
#[should_panic(expected = "alpha must be in (0.0, 1.0]")]
fn ema_rejects_a_zero_smoothing_factor() {
    let _ = ExponentialMovingAverage::new(0.0);
}

#[test]
#[should_panic(expected = "alpha must be in (0.0, 1.0]")]
fn ema_rejects_a_smoothing_factor_above_one() {
    let _ = ExponentialMovingAverage::new(1.000_001);
}

/// `NaN` fails every comparison, so it is rejected by the same assertion
/// rather than slipping through as "not out of range".
#[test]
#[should_panic(expected = "alpha must be in (0.0, 1.0]")]
fn ema_rejects_a_nan_smoothing_factor() {
    let _ = ExponentialMovingAverage::new(f64::NAN);
}

#[test]
#[should_panic(expected = "span must be > 0")]
fn ema_rejects_a_zero_span() {
    let _ = ExponentialMovingAverage::from_span(0);
}

/// `alpha = 1` is the boundary and is inside the contract: the average is the
/// latest value.
#[test]
fn ema_accepts_the_boundary_factor_of_one() {
    let mut ema = ExponentialMovingAverage::new(1.0);
    ema.observe(3.0);
    ema.observe(11.0);
    assert_eq!(ema.value(), 11.0);
    // span = u64::MAX makes alpha the smallest positive value the formula can
    // produce; it is still inside (0, 1] and must not be rejected
    let tiny = ExponentialMovingAverage::from_span(u64::MAX);
    assert!(tiny.alpha() > 0.0 && tiny.alpha() <= 1.0);
}

// ---------------------------------------------------------------------------
// Non-finite and extreme magnitudes in a sketch
// ---------------------------------------------------------------------------

/// `inf` is the realistic shape of an upstream division by zero. The correct
/// answer is the documented one for a magnitude outside `accurate_range()`:
/// it is counted, filed in the edge bin so later ranks do not shift, and the
/// reported quantile is clamped to the observed range. It must not overflow
/// the bucket index.
#[test]
fn an_infinite_magnitude_is_counted_and_filed_in_the_edge_bin() {
    let mut dd = DDSketch::new(0.01);
    dd.insert(f64::INFINITY);
    assert_eq!(dd.count(), 1);
    // the only observation is +inf, so every order statistic is +inf
    assert_eq!(dd.quantile(0.0), f64::INFINITY);
    assert_eq!(dd.quantile(1.0), f64::INFINITY);

    // with finite company the ranks are preserved and the edge-bin
    // representative is the top (bottom) of the accurate range, not bin 0
    let mut dd = DDSketch::new(0.01);
    dd.insert(1.0);
    dd.insert(f64::INFINITY);
    dd.insert(f64::NEG_INFINITY);
    assert_eq!(dd.count(), 3);
    let (lo, hi) = dd.accurate_range();
    let (q0, q50, q100) = (dd.quantile(0.0), dd.quantile(0.5), dd.quantile(1.0));
    // the top bin covers (γ^(i−1), γ^i] with γ^i = hi, and its representative
    // is 2γ^i/(γ + 1) — the point whose relative distance to both edges is α
    let top = top_edge_representative(0.01, hi);
    assert!(
        (q100 / top - 1.0).abs() < 1e-9,
        "+inf must be filed in the top edge bin (representative {top:e}), got {q100:e}"
    );
    assert!(
        (q0 / -top - 1.0).abs() < 1e-9,
        "-inf must be filed in the bottom edge bin (representative {:e}), got {q0:e}; \
         a wrapped bucket index would instead report about {lo:e} and shift every later rank",
        -top
    );
    // the middle rank is the finite value, within the published bound
    assert!((q50 - 1.0).abs() <= 0.01, "q50 {q50}");
}

/// Representative of the top bin of a `DDSketch`: the bin covers
/// `(γ^(i−1), γ^i]` with `γ^i` the top of `accurate_range()`, and the
/// representative is `2γ^i/(γ + 1)`.
fn top_edge_representative(alpha: f64, hi: f64) -> f64 {
    let gamma = (1.0 + alpha) / (1.0 - alpha);
    2.0 * hi / (gamma + 1.0)
}

/// `f64::MAX` takes the same path: a finite magnitude far above the accurate
/// range must also saturate rather than wrap.
#[test]
fn the_largest_finite_magnitudes_are_filed_in_the_edge_bins() {
    let mut dd = DDSketch::new(0.01);
    dd.insert(f64::MAX);
    dd.insert(f64::MIN_POSITIVE);
    dd.insert(-f64::MAX);
    assert_eq!(dd.count(), 3);
    let (_, hi) = dd.accurate_range();
    let top = top_edge_representative(0.01, hi);
    assert!((dd.quantile(1.0) / top - 1.0).abs() < 1e-9);
    assert!((dd.quantile(0.0) / -top - 1.0).abs() < 1e-9);
}

/// A relative accuracy of 0 is a degenerate configuration (γ = 1, so the
/// logarithm base is 1 and `inv_ln_gamma` is infinite). It must not panic:
/// the accurate range collapses to the single magnitude 1, which is what a
/// zero-width bin layout means, and the reported quantile stays finite.
#[test]
fn a_zero_relative_accuracy_collapses_the_range_instead_of_overflowing() {
    let mut dd = DDSketch::new(0.0);
    dd.insert(2.0);
    dd.insert(0.5);
    assert_eq!(dd.count(), 2);
    assert_eq!(dd.alpha(), 0.0);
    assert_eq!(dd.accurate_range(), (1.0, 1.0));
    assert!(dd.quantile(0.5).is_finite());
}

/// Quantile arguments outside the unit interval and `NaN` are clamped, not
/// rejected: the reported value stays inside the observed range.
#[test]
fn quantile_arguments_outside_the_unit_interval_are_clamped() {
    let mut dd = DDSketch::new(0.01);
    for v in 1..=100 {
        dd.insert(f64::from(v));
    }
    for q in [-1.0f64, 2.0, f64::NAN] {
        let v = dd.quantile(q);
        assert!(
            (1.0..=100.0).contains(&v),
            "quantile({q}) returned {v}, outside the observed [1, 100]"
        );
    }
}

/// Empty estimators report their documented neutral value rather than
/// dividing by zero.
#[test]
fn empty_estimators_report_their_neutral_value() {
    let dd = DDSketch::new(0.01);
    assert_eq!(dd.count(), 0);
    assert_eq!(dd.quantile(0.5), 0.0);

    let hll = HyperLogLog::new();
    assert_eq!(hll.cardinality(), 0.0);

    let cm = CountMinSketch::new();
    assert_eq!(cm.estimate(&7u64), 0);

    let st = StreamingStats::new();
    assert_eq!(st.count(), 0);
    assert_eq!(st.mean(), 0.0);
    assert_eq!(st.variance(), 0.0);
    assert_eq!(st.std_dev(), 0.0);
    // fewer than 3 (4) observations cannot define g₁ (excess kurtosis)
    assert_eq!(st.skewness(), 0.0);
    assert_eq!(st.kurtosis(), 0.0);

    let mut med = StreamingMedian::new();
    assert_eq!(med.count(), 0);
    assert_eq!(med.median(), 0.0);

    let z = ZScoreDetector::new(3.0);
    assert_eq!(z.variance(), 0.0);
    assert_eq!(z.std_dev(), 0.0);
    // with no observed spread the standardised distance is infinite, which is
    // the honest answer; the detector still reports no anomaly, because it
    // has not seen enough observations to judge
    assert_eq!(z.z_score(1.0), f64::INFINITY);
    assert!(!z.is_anomaly(1e9));

    let mut mad = MadDetector::new(3.0);
    assert_eq!(mad.mad(), 0.0);
    assert!(!mad.is_anomaly(1e9));

    let agg = PrivateAggregator::new(1.0);
    assert_eq!(agg.estimate_mean(), 0.0);
    assert_eq!(agg.estimate_sum(), 0.0);
    // the standard error scales as 1/√n, so with no observations the
    // uncertainty is unbounded: infinity, not a misleading 0
    assert_eq!(agg.standard_error(), f64::INFINITY);
}

/// Empty and short slices: `quantile_sorted` and `percentile_rank` return 0
/// for an empty input, and `iqr` returns `None` below the four points the
/// quartiles need.
#[test]
fn sorted_slice_statistics_handle_empty_and_short_inputs() {
    assert_eq!(quantile_sorted(&[], 0.5), 0.0);
    assert_eq!(percentile_rank(&[], 1.0), 0.0);
    assert_eq!(quantile_sorted(&[42.0], 0.5), 42.0);
    assert!(iqr(&[]).is_none());
    assert!(iqr(&[1.0, 2.0, 3.0]).is_none());
    assert!(iqr(&[1.0, 2.0, 3.0, 4.0]).is_some());
    // a NaN or out-of-range q is clamped into [0, 1] and indexes a real element
    for q in [-1.0f64, 2.0, f64::NAN] {
        let v = quantile_sorted(&[1.0, 2.0, 3.0], q);
        assert!((1.0..=3.0).contains(&v), "q {q} gave {v}");
    }
}

// ---------------------------------------------------------------------------
// Timestamps near the end of the u64 range
// ---------------------------------------------------------------------------

/// A clock that has not been set reports a timestamp near `u64::MAX`. Adding
/// the window width to it overflows; the correct answer is the saturated
/// boundary. Wrapping instead would move the boundary backwards and emit
/// windows whose `end_ms` precedes their `start_ms`, and would silently lose
/// or duplicate events.
#[test]
fn window_boundaries_saturate_at_the_end_of_the_timestamp_range() {
    let mut tw = TumblingWindow::new(1_000, 0.01);
    let mut emitted = 0u64;
    for (v, ts) in [(1.0, u64::MAX), (2.0, u64::MAX), (3.0, u64::MAX)] {
        if let Some(r) = tw.insert(v, ts) {
            assert_eq!(r.end_ms, u64::MAX, "the boundary must saturate, not wrap");
            assert!(
                r.end_ms >= r.start_ms,
                "a wrapped boundary gives end_ms {} < start_ms {}",
                r.end_ms,
                r.start_ms
            );
            emitted += r.event_count;
        }
    }
    let r = tw.flush();
    assert_eq!(r.end_ms, u64::MAX);
    assert!(r.end_ms >= r.start_ms);
    // the saturated boundary is reached by every insert at the very end of
    // the range, so a window closes each time; what must hold is that the
    // three events are accounted for exactly once between the emitted
    // windows and the final flush
    assert_eq!(
        emitted + r.event_count,
        3,
        "three inserts must be accounted for exactly once"
    );

    // The boolean that decides whether a window has ended is where the two
    // behaviours differ observably. With a window starting at `u64::MAX - 500`
    // the boundary is past the end of the range: a later timestamp that is
    // still inside the window must not close it. A wrapping `+` would compute
    // a boundary of 499 and close the window on every subsequent sample —
    // silently, in a release build, where the addition does not panic.
    let mut tw = TumblingWindow::new(1_000, 0.01);
    assert!(
        tw.insert(1.0, u64::MAX - 500).is_none(),
        "the first sample opens the window"
    );
    assert!(
        tw.insert(2.0, u64::MAX - 400).is_none(),
        "u64::MAX - 400 is inside the window that opened at u64::MAX - 500; \
         a wrapped boundary (499) would close it here"
    );
    // the boundary saturates to u64::MAX exactly, and the comparison is `>=`,
    // so a sample *at* u64::MAX does close the window
    assert!(tw.insert(3.0, u64::MAX).is_some());
    let r = tw.flush();
    assert_eq!(
        r.event_count, 1,
        "only the third sample is in the new window"
    );
    assert_eq!(r.end_ms, u64::MAX, "the boundary saturates");

    let mut hr = HierarchicalRollup::new(1, 2, 3, 0.01);
    hr.insert(1.0, u64::MAX);
    hr.insert(2.0, u64::MAX);
    assert_eq!(hr.level_count(), 3);
}

/// A width of 0 is clamped to 1 by the constructor, so the modulo inside
/// `insert` can never divide by zero.
#[test]
fn a_zero_window_width_is_clamped_to_one() {
    let mut tw = TumblingWindow::new(0, 0.01);
    tw.insert(1.0, 5);
    tw.insert(2.0, 9);
    let r = tw.flush();
    assert!(r.end_ms > r.start_ms);
}

/// Two samples with the same timestamp give no new rate (dt = 0): the last
/// rate is repeated rather than dividing by zero.
#[test]
fn a_repeated_timestamp_repeats_the_last_change_rate() {
    let mut cr = ChangeRate::new();
    assert_eq!(cr.observe(1.0, 10), None, "the first sample has no rate");
    assert_eq!(cr.observe(3.0, 20), Some(0.2));
    assert_eq!(
        cr.observe(999.0, 20),
        Some(0.2),
        "dt = 0 keeps the last rate"
    );
    // a timestamp going backwards saturates dt to 0 and behaves the same way
    assert_eq!(cr.observe(999.0, 5), Some(0.2));
}

/// One point cannot define a slope; the documented answer is 0 rather than a
/// division by a zero second moment.
#[test]
fn a_single_point_regression_reports_a_zero_slope() {
    let mut lr = LinearRegression::new();
    lr.observe(1.0, 2.0);
    assert_eq!(lr.slope(), 0.0);
    assert_eq!(lr.r_squared(), 0.0);
    assert_eq!(lr.intercept(), 2.0, "the intercept is the single y");
    assert_eq!(lr.predict(100.0), 2.0);
}

// ---------------------------------------------------------------------------
// Privacy parameters
// ---------------------------------------------------------------------------

/// ε = 0 means an infinite noise scale. The mechanism stays usable (the
/// sample is infinite or `NaN`, which is the honest answer for "no privacy
/// budget spent"), and the integer form saturates instead of wrapping.
#[test]
fn a_zero_epsilon_gives_an_infinite_scale_without_wrapping() {
    let mut ln = LaplaceNoise::with_seed(1.0, 0.0, 7);
    assert_eq!(ln.scale(), f64::INFINITY);
    let s = ln.sample();
    assert!(s.is_infinite(), "scale inf gives an infinite sample: {s}");

    // the i64 conversion saturates (Rust `as` semantics, pinned here because
    // the noise can exceed the i64 range by construction)
    let mut big = LaplaceNoise::with_seed(1e300, 1e-300, 7);
    let v = big.privatize_int(0);
    assert!(
        v == i64::MAX || v == i64::MIN,
        "expected saturation, got {v}"
    );
}

/// The truthfulness probability must be inside [0.5, 1]: below 0.5 the
/// estimator inverts the sign of every reported proportion.
///
/// The domain is enforced the way `ExponentialMovingAverage::new` enforces
/// its own — an assertion rather than a silent correction. Measured
/// precedent (`cargo test`, every boundary): that constructor rejects a
/// negative factor, `-0.0`, `0.0`, anything above `1.0` including
/// `1.0 + f64::EPSILON`, `NaN`, `inf` and `-inf`, and accepts everything from
/// `f64::MIN_POSITIVE` up to and including `1.0`. A single `&&` of two
/// comparisons covers the non-finite cases for free, because every comparison
/// against `NaN` is false.
#[test]
fn a_randomized_response_accepts_its_whole_domain_and_only_that() {
    // the closed ends are inside the domain: 0.5 means "always answer at
    // random" (which `new(0.0)` produces exactly), 1.0 means "always truthful"
    assert_eq!(RandomizedResponse::with_probability(0.5, 7).p_true(), 0.5);
    assert_eq!(RandomizedResponse::with_probability(1.0, 7).p_true(), 1.0);
    // and the value is kept, not corrected
    assert_eq!(RandomizedResponse::with_probability(0.75, 7).p_true(), 0.75);

    // p_true = 0.5 carries no information about the input, so the estimator
    // returns the prior 0.5 rather than dividing by zero
    assert_eq!(RandomizedResponse::estimate_proportion(0.5, 10, 5), 0.5);
    // no responses at all: 0, not 0/0
    assert_eq!(RandomizedResponse::estimate_proportion(0.75, 0, 0), 0.0);
}

#[test]
#[should_panic(expected = "p_true must be in [0.5, 1.0]")]
fn a_randomized_response_rejects_a_probability_below_the_domain() {
    let _ = RandomizedResponse::with_probability(0.0, 7);
}

#[test]
#[should_panic(expected = "p_true must be in [0.5, 1.0]")]
fn a_randomized_response_rejects_a_probability_above_the_domain() {
    let _ = RandomizedResponse::with_probability(2.0, 7);
}

/// ⚠️ Until 2026-10-07 this was the one input the old `clamp(0.5, 1.0)` let
/// through: `f64::clamp` returns `NaN` when the value is `NaN`, so a
/// misconfigured probability was not corrected and `p_true()` reported `NaN`.
#[test]
#[should_panic(expected = "p_true must be in [0.5, 1.0]")]
fn a_randomized_response_rejects_a_nan_probability() {
    let _ = RandomizedResponse::with_probability(f64::NAN, 7);
}

#[test]
#[should_panic(expected = "p_true must be in [0.5, 1.0]")]
fn a_randomized_response_rejects_an_infinite_probability() {
    let _ = RandomizedResponse::with_probability(f64::INFINITY, 7);
}

/// A malformed spend request is refused and leaves the ledger byte for byte
/// unchanged.
///
/// This is the invariant the type exists for. Until 2026-10-07 the only guard
/// was the budget comparison `total + epsilon <= max`, which a negative value
/// passes trivially: `try_spend(-10.0)` returned `true` and *raised* the
/// remaining budget from 1 to 11, and `-inf` raised it to infinity — so a
/// caller could slip a negative epsilon between two honest queries and escape
/// the limit entirely. `NaN` and `+inf` were refused, but only as a side
/// effect of the comparison being false.
///
/// Refusal is asserted on every field, not just on the returned `bool`: a
/// guard that returns `false` after charging the ledger would pass a
/// `bool`-only test.
#[test]
fn a_malformed_spend_is_refused_and_leaves_the_ledger_untouched() {
    for bad in [
        -10.0f64,
        -f64::MIN_POSITIVE,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ] {
        let mut b = PrivacyBudget::new(1.0);
        // spend something first, so "untouched" is distinguishable from "reset"
        assert!(b.try_spend(0.25));
        let (spent, remaining, queries) = (b.spent(), b.remaining(), b.query_count());
        assert_eq!((spent, remaining, queries), (0.25, 0.75, 1));

        assert!(!b.try_spend(bad), "{bad:e} must be refused");
        assert_eq!(b.spent(), spent, "{bad:e} moved `spent`");
        assert_eq!(b.remaining(), remaining, "{bad:e} moved `remaining`");
        assert_eq!(b.query_count(), queries, "{bad:e} counted a query");
        assert!(!b.is_exhausted(), "{bad:e} changed exhaustion");
    }
}

/// Zero is well formed — it charges nothing and counts a query — and the
/// limit is enforced at the boundary.
#[test]
fn a_well_formed_spend_is_monotone_and_bounded() {
    let mut b = PrivacyBudget::new(1.0);
    assert_eq!(b.remaining(), 1.0);

    // 0.0 and -0.0 are both zero; `-0.0 < 0.0` is false in IEEE 754, so both
    // are accepted and neither moves the ledger
    for zero in [0.0f64, -0.0] {
        assert!(b.try_spend(zero), "{zero:e} is a well-formed request");
        assert_eq!(b.spent(), 0.0);
        assert_eq!(b.remaining(), 1.0);
    }
    assert_eq!(b.query_count(), 2, "a zero spend still counts as a query");

    assert!(b.try_spend(0.5));
    assert_eq!(b.spent(), 0.5);
    assert_eq!(b.remaining(), 0.5);

    assert!(!b.try_spend(0.75), "over budget is refused");
    assert_eq!(b.spent(), 0.5, "the refused amount is not charged");
    assert_eq!(b.remaining(), 0.5);

    // exactly the remainder fits, and then the budget is exhausted
    assert!(b.try_spend(0.5));
    assert_eq!(b.remaining(), 0.0);
    assert!(b.is_exhausted());
    assert!(!b.try_spend(f64::EPSILON), "nothing fits once exhausted");
    // ⚠️ a request below the ulp of the accumulated total is absorbed by the
    // addition and therefore accepted: `1.0 + f64::MIN_POSITIVE` rounds back
    // to `1.0`, so the comparison passes. It charges nothing, which is why
    // this is a property of f64 rather than a hole in the limit
    assert!(b.try_spend(f64::MIN_POSITIVE));
    assert_eq!(b.spent(), 1.0, "an absorbed request charges nothing");
    assert_eq!(b.remaining(), 0.0);

    // a budget that starts non-positive accepts nothing above zero
    let mut none = PrivacyBudget::new(-1.0);
    assert!(!none.try_spend(0.5));
    assert_eq!(none.spent(), 0.0);
    assert_eq!(none.query_count(), 0);
}

/// `EwmaDetector` takes its smoothing factor without an assertion (unlike
/// `ExponentialMovingAverage`), so both ends of the interval must stay
/// defined rather than producing `NaN`.
#[test]
fn ewma_detector_boundary_factors_stay_defined() {
    // alpha = 0: the warm-up seeds the estimate, so both the level and the
    // spread stay finite
    let mut slow = EwmaDetector::new(0.0, 3.0);
    for i in 0..10 {
        slow.observe(f64::from(i));
    }
    assert!(slow.ewma().is_finite(), "ewma {}", slow.ewma());
    assert!(slow.std_dev().is_finite() && slow.std_dev() >= 0.0);
    assert!(slow.anomaly_score(100.0).is_finite());

    // alpha = 1: no smoothing at all, so the variance estimate collapses to 0
    // and the standardised distance is infinite — defined, and never NaN
    let mut fast = EwmaDetector::new(1.0, 3.0);
    for i in 0..10 {
        fast.observe(f64::from(i));
    }
    assert_eq!(
        fast.ewma(),
        9.0,
        "alpha = 1 tracks the latest value exactly"
    );
    assert_eq!(fast.std_dev(), 0.0);
    assert_eq!(fast.anomaly_score(100.0), f64::INFINITY);
    assert!(fast.is_anomaly(100.0));
}

/// `NaN` observations are total (no panic) and keep their place in the count.
#[test]
fn nan_observations_do_not_panic_the_estimators() {
    let mut st = StreamingStats::new();
    for _ in 0..8 {
        st.observe(f64::NAN);
    }
    assert_eq!(st.count(), 8, "NaN still counts as an observation");

    let mut med = StreamingMedian::new();
    med.push(f64::NAN);
    med.push(1.0);
    assert_eq!(med.count(), 2);
}

/// How `DDSketch` classifies a `NaN`, pinned exactly as it behaves today.
///
/// ⚠️ This is the current behaviour, not a considered design, and it is
/// expected to change: `NaN` fails both `value > 0.0` and `value < 0.0`, so
/// it is filed with the exact zeros and a stream of nothing but `NaN`
/// reports a median of `0.0`. The same crate's `law::ResidualSummary`
/// already counts a non-finite residual separately as `non_finite`, and the
/// two will be reconciled together rather than one at a time. Until then
/// this test exists so that a change to the classification is a deliberate
/// edit of a stated contract rather than a silent one — a reader who comes
/// to fix the behaviour should expect to rewrite this test, not to work
/// around it.
#[test]
fn ddsketch_currently_files_nan_with_the_exact_zeros() {
    let mut dd = DDSketch::new(0.01);
    for _ in 0..5 {
        dd.insert(f64::NAN);
    }
    assert_eq!(dd.count(), 5, "NaN is counted");
    assert!(dd.mean().is_nan(), "NaN poisons the running sum");
    // filed as an exact zero: every reported order statistic is 0.0
    for q in [0.0f64, 0.5, 1.0] {
        assert_eq!(dd.quantile(q), 0.0, "quantile({q}) of an all-NaN stream");
    }
    // min / max never move, because every comparison against NaN is false
    assert_eq!(dd.min(), f64::INFINITY);
    assert_eq!(dd.max(), f64::NEG_INFINITY);
}
