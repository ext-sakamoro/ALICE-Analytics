//! Cross-platform golden hashes for every module whose arithmetic runs in
//! IEEE `f64`.
//!
//! `tests/analytic_oracle.rs` and `tests/law_residual.rs` check that the
//! numbers are *right*; this file checks that they are the **same bits** on
//! every target. The two properties are independent: a correct result can
//! still differ in the last ulp between macOS, glibc, MSVC and wasm, which is
//! what breaks a sketch merged across machines, a replayed telemetry stream
//! and any audit that re-derives a quantile from the raw events.
//!
//! Why bit-exactness holds here:
//!
//! 1. `+ - * /` and `sqrt` are required by IEEE 754 to be correctly rounded,
//!    so they produce identical bits on every target Rust supports (SSE2+,
//!    aarch64, wasm32).
//! 2. Every transcendental (`ln`, `exp`, `powf`) goes through
//!    [`alice_det_math`], whose kernels are built from those operations only,
//!    in a fixed evaluation order.
//! 3. `mul_add` is a fused multiply-add with a single rounding by IEEE 754,
//!    and a target without an FMA instruction gets the correctly rounded
//!    software `fma`, so it is bit-identical everywhere and is used freely.
//!    Integer powers use binary exponentiation with a fixed association
//!    order rather than `powi`, whose multiplication tree is unspecified.
//! 4. `clippy.toml` `disallowed-methods` rejects the inherent `f32` / `f64`
//!    forms of all of the above, and CI runs clippy with `--all-targets ...
//!    -D warnings`, so a regression is a compile error rather than a silent
//!    divergence.
//!
//! Each scenario drives a module through its public entry points with fixed
//! inputs, serialises every output bit-for-bit (`to_bits().to_le_bytes()`,
//! little-endian, so the hash does not depend on the host byte order) and
//! compares the SHA-256 with a constant recorded on macOS `aarch64`. CI runs
//! this file on macOS `aarch64`, Linux `x86_64`, Linux `aarch64` and Windows
//! `x86_64`; a mismatch on any of them means a platform-dependent operation
//! crept in.
//!
//! Updating a golden (only after an intentional algorithm change): run the
//! failing test, copy the `actual` hex into the constant, and record the
//! change in CHANGELOG under `[Unreleased] / Changed`. A mismatch that is
//! *not* explained by a deliberate change in this repository is a defect:
//! find the operation that left the list above instead of re-recording.
// `privacy` is deprecated (not differentially private) and still pinned here until 0.4.0 removes it
#![allow(deprecated)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp
)]

use core::fmt::Write as _;

use sha2::{Digest, Sha256};

// ---------------------------------------------------------------------------
// Byte sink
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Sink(Vec<u8>);

impl Sink {
    fn f64(&mut self, v: f64) {
        self.0.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn i64(&mut self, v: i64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn bool(&mut self, v: bool) {
        self.0.push(u8::from(v));
    }
    fn byte(&mut self, v: u8) {
        self.0.push(v);
    }
    fn len(&self) -> usize {
        self.0.len()
    }
    fn finish(self) -> String {
        let mut hex = String::with_capacity(64);
        for b in Sha256::digest(&self.0) {
            write!(hex, "{b:02x}").expect("writing to a String cannot fail");
        }
        hex
    }
}

/// Deterministic pseudo-random `f64` in `[0, 1)` from a counter: integer
/// mixing plus one exact division by a power of two, so the inputs
/// themselves are bit-identical everywhere.
fn prand(i: u64) -> f64 {
    let mut h = i
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0x1234_5678_9ABC_DEF0);
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    h ^= h >> 29;
    (h >> 11) as f64 / (1u64 << 53) as f64
}

/// A hash is only evidence if bytes went into it. Every scenario asserts a
/// non-zero, expected payload length before comparing, so a scenario that
/// silently stopped producing output (an API that started returning `None`,
/// a loop whose bound became 0) fails here rather than passing with the
/// hash of an empty buffer.
fn assert_golden(scenario: &str, sink: Sink, min_bytes: usize, expected: &str) {
    let bytes = sink.len();
    assert!(
        bytes >= min_bytes,
        "scenario `{scenario}` serialised {bytes} bytes, expected at least \
         {min_bytes}: the scenario stopped exercising the module, so its hash \
         proves nothing"
    );
    let actual = sink.finish();
    assert_eq!(
        actual, expected,
        "\n\nGolden hash mismatch for scenario `{scenario}` ({bytes} bytes).\n\
         actual:   {actual}\n\
         expected: {expected}\n\n\
         If an algorithm in this repository changed on purpose, update the\n\
         GOLDEN_* constant in tests/determinism_golden.rs and record it in\n\
         CHANGELOG. If it did not, a platform-dependent float operation was\n\
         introduced: check clippy.toml disallowed-methods and that every\n\
         transcendental goes through alice_det_math.\n"
    );
}

// ---------------------------------------------------------------------------
// 1. sketch — HyperLogLog (ln), DDSketch (ln + integer powers, non-finite
//    counter), Count-Min (exp)
// ---------------------------------------------------------------------------

const GOLDEN_SKETCH: &str = "49078f4c720eedc79aab35ba79d55f18f6f923fca3069fa3fdb08b264e3719b4";

#[test]
fn golden_sketch() {
    use alice_analytics::sketch::{
        CountMinSketch, DDSketch, DDSketch256, HeavyHitters10, HyperLogLog, Mergeable,
    };
    let mut s = Sink::default();

    // HyperLogLog: the small-range correction is m·ln(m/zeros)
    for n in [1u64, 7, 100, 5_000, 40_000] {
        let mut hll = HyperLogLog::new();
        for id in 0..n {
            hll.insert(&id);
        }
        s.f64(hll.cardinality());
        s.u64(n);
    }
    // merge path
    let mut a = HyperLogLog::new();
    let mut b = HyperLogLog::new();
    for id in 0..3_000u64 {
        a.insert(&id);
    }
    for id in 2_000..6_000u64 {
        b.insert(&id);
    }
    a.merge(&b);
    s.f64(a.cardinality());

    // DDSketch: bucket_index uses ln, bucket_representative an integer power
    for alpha in [0.01f64, 0.02, 0.05] {
        let mut dd = DDSketch::new(alpha);
        for i in 0..5_000u64 {
            // six decades, both signs, plus exact zero
            let v = prand(i) * 1e3;
            dd.insert(v);
            dd.insert(-v);
        }
        dd.insert(0.0);
        // non-finite samples are counted on their own and must not perturb
        // anything below (law::PointClass::NonFinite classification)
        dd.insert(f64::NAN);
        dd.insert(f64::INFINITY);
        dd.insert(f64::NEG_INFINITY);
        for q in [0.0f64, 0.01, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0] {
            s.f64(dd.quantile(q));
        }
        let (lo, hi) = dd.accurate_range();
        s.f64(lo);
        s.f64(hi);
        s.f64(dd.alpha());
        s.u64(dd.count());
        s.u64(dd.non_finite());
        s.f64(dd.sum());
        s.f64(dd.min());
        s.f64(dd.max());
    }
    // the 256-bin alias has a different offset, so its range is a separate law
    let dd = DDSketch256::new(0.05);
    let (lo, hi) = dd.accurate_range();
    s.f64(lo);
    s.f64(hi);

    // Count-Min: confidence is 1 − e^(−d)
    let mut cm = CountMinSketch::new();
    for i in 0..20_000u64 {
        cm.insert(&(i % 997));
    }
    s.f64(cm.error_bound());
    s.f64(cm.confidence());
    s.u64(cm.estimate(&13u64));

    // Heavy hitters keep integer counts, but the ranking is order-sensitive,
    // so pin the top-k as reported
    let mut hh = HeavyHitters10::new();
    for i in 0..5_000u64 {
        hh.insert_hash((i % 23) * 0x9E37_79B9);
    }
    for e in hh.top() {
        s.u64(e.hash);
        s.u64(e.count);
    }

    assert_golden("sketch", s, 400, GOLDEN_SKETCH);
}

// ---------------------------------------------------------------------------
// 2. stats — Welford moments (plain multiply-add chains), quantiles, IQR
// ---------------------------------------------------------------------------

const GOLDEN_STATS: &str = "bf98ab7a17423f29706e4a6dc5479b4d4188d4d7318dde69bc9b7f176ea90850";

#[test]
fn golden_stats() {
    use alice_analytics::stats::{
        iqr, percentile_rank, quantile_sorted, CovarianceMatrix, StreamingStats,
    };
    let mut s = Sink::default();

    // StreamingStats: mean / variance / skewness (m2·sqrt(m2)) / kurtosis
    let mut st = StreamingStats::new();
    for i in 0..2_000u64 {
        st.observe(prand(i) * 10.0 - 3.0);
    }
    s.f64(st.mean());
    s.f64(st.variance());
    s.f64(st.sample_variance());
    s.f64(st.std_dev());
    s.f64(st.skewness());
    s.f64(st.kurtosis());
    s.u64(st.count());

    // sorted-array statistics
    let mut xs: Vec<f64> = (0..501u64).map(|i| prand(i) * 100.0).collect();
    xs.sort_by(f64::total_cmp);
    for q in [0.0f64, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] {
        s.f64(quantile_sorted(&xs, q));
    }
    for v in [0.0f64, 12.5, 50.0, 99.9, 1000.0] {
        s.f64(percentile_rank(&xs, v));
    }
    let r = iqr(&xs).expect("501 samples is more than the 4 the API needs");
    s.f64(r.q1);
    s.f64(r.median);
    s.f64(r.q3);
    s.f64(r.iqr);
    s.f64(r.lower_fence);
    s.f64(r.upper_fence);

    // covariance / correlation (sqrt only)
    let mut cov = CovarianceMatrix::<3>::new();
    for i in 0..1_000u64 {
        let x = prand(i);
        cov.observe(&[x, 2.0 * x + 1.0, prand(i + 1_000_000)]);
    }
    for i in 0..3 {
        s.f64(cov.mean(i));
        s.f64(cov.variance(i));
        for j in 0..3 {
            s.f64(cov.covariance(i, j));
            s.f64(cov.correlation(i, j));
        }
    }

    assert_golden("stats", s, 350, GOLDEN_STATS);
}

// ---------------------------------------------------------------------------
// 3. window — tumbling / sliding / hierarchical aggregates
// ---------------------------------------------------------------------------

const GOLDEN_WINDOW: &str = "a0930b7ef27cd7b0ae96c9e31dd85b5e0963ced9dfb837da7d535d79e9d05a7b";

#[test]
fn golden_window() {
    use alice_analytics::window::{HierarchicalRollup, SlidingWindow, TumblingWindow};
    let mut s = Sink::default();

    let mut tw = TumblingWindow::new(1_000, 0.01);
    for i in 0..5_000u64 {
        if let Some(r) = tw.insert(prand(i) * 50.0, i * 7) {
            s.f64(r.counter);
            s.f64(r.gauge);
            s.f64(r.mean);
            s.f64(r.min);
            s.f64(r.max);
            s.f64(r.p50);
            s.f64(r.p99);
            s.u64(r.event_count);
            s.u64(r.start_ms);
            s.u64(r.end_ms);
        }
    }
    let f = tw.flush();
    s.f64(f.mean);
    s.f64(f.p50);
    s.f64(f.p99);
    s.u64(f.event_count);

    let mut sw = SlidingWindow::<16>::new();
    for i in 0..200u64 {
        sw.push(prand(i) * 9.0 - 4.0);
        s.f64(sw.mean());
        s.f64(sw.variance());
        s.f64(sw.std_dev());
        s.f64(sw.min());
        s.f64(sw.max());
        s.f64(sw.sum());
    }

    let mut hr = HierarchicalRollup::new(1_000, 10_000, 60_000, 0.02);
    for i in 0..30_000u64 {
        hr.insert(prand(i) * 3.0, i * 11);
    }
    for level in 0..hr.level_count() {
        s.u64(hr.window_size(level));
        if let Some(r) = hr.result(level) {
            s.f64(r.mean);
            s.f64(r.p50);
            s.f64(r.p99);
            s.u64(r.event_count);
        }
    }

    assert_golden("window", s, 4_000, GOLDEN_WINDOW);
}

// ---------------------------------------------------------------------------
// 4. anomaly — median / MAD / EWMA / EWMV / z-score
// ---------------------------------------------------------------------------

const GOLDEN_ANOMALY: &str = "0e13fcff4b706f052b6ce3208eb548d8be526df1e5f7712e035ed0e9b4e050d3";

#[test]
fn golden_anomaly() {
    use alice_analytics::anomaly::{
        CompositeDetector, EwmaDetector, MadDetector, StreamingMedian, ZScoreDetector,
    };
    let mut s = Sink::default();

    let mut med = StreamingMedian::new();
    for i in 0..300u64 {
        med.push(prand(i) * 20.0);
        s.f64(med.median());
    }

    let mut mad = MadDetector::new(3.0);
    for i in 0..300u64 {
        mad.observe(prand(i) * 20.0);
    }
    s.f64(mad.median());
    s.f64(mad.mad());
    for v in [0.0f64, 5.0, 10.0, 19.5, 1e4] {
        s.f64(mad.anomaly_score(v));
        s.bool(mad.is_anomaly(v));
    }

    let mut ew = EwmaDetector::new(0.2, 2.5);
    for i in 0..500u64 {
        ew.observe(prand(i) * 6.0 + 1.0);
        s.f64(ew.ewma());
        s.f64(ew.std_dev());
    }
    for v in [0.0f64, 4.0, 100.0] {
        s.f64(ew.anomaly_score(v));
        s.bool(ew.is_anomaly(v));
    }

    let mut zs = ZScoreDetector::new(3.0);
    for i in 0..500u64 {
        zs.observe(prand(i) * 6.0 + 1.0);
    }
    s.f64(zs.mean());
    s.f64(zs.variance());
    s.f64(zs.std_dev());
    for v in [0.0f64, 4.0, 100.0] {
        s.f64(zs.z_score(v));
        s.bool(zs.is_anomaly(v));
    }

    let mut cd = CompositeDetector::with_thresholds(3.0, 0.2, 2.5, 3.0);
    for i in 0..500u64 {
        let v = prand(i) * 6.0 + 1.0;
        cd.observe(v);
        s.bool(cd.is_anomaly(v));
        s.f64(cd.anomaly_score(v));
    }

    assert_golden("anomaly", s, 4_000, GOLDEN_ANOMALY);
}

// ---------------------------------------------------------------------------
// 5. privacy — Laplace inverse transform (ln), randomized response (exp)
// ---------------------------------------------------------------------------

const GOLDEN_PRIVACY: &str = "50f95bbc492118d85de552a3a950673e3c762b9005d1217c81c042cab9494ca5";

#[test]
fn golden_privacy() {
    use alice_analytics::privacy::{
        LaplaceNoise, PrivacyBudget, PrivateAggregator, RandomizedResponse, Rappor, XorShift64,
    };
    let mut s = Sink::default();

    // the PRNG itself is integer, but pin it: every mechanism below rides on it
    let mut rng = XorShift64::new(0xDEAD_BEEF_CAFE_F00D);
    for _ in 0..64 {
        s.u64(rng.next_u64());
    }
    for _ in 0..64 {
        s.f64(rng.next_f64());
    }
    for _ in 0..16 {
        s.f64(rng.next_f64_range(-3.5, 7.25));
        s.bool(rng.next_bool(0.3));
    }

    // Laplace: X = −b·sign(u)·ln(1 − 2|u|)
    for (sensitivity, epsilon) in [(1.0f64, 1.0f64), (2.5, 0.5), (0.1, 4.0)] {
        let mut ln = LaplaceNoise::with_seed(sensitivity, epsilon, 7);
        s.f64(ln.scale());
        for _ in 0..200 {
            s.f64(ln.sample());
        }
        for v in [0.0f64, -12.5, 1e6] {
            s.f64(ln.privatize(v));
        }
        for v in [0i64, -7, 1_000_000] {
            s.i64(ln.privatize_int(v));
        }
    }

    // Randomized response: p = 1/(1 + e^(−ε)) via the std constructor
    // (the hash changed on 2026-10-07 when the algebraically equivalent but
    // overflow-free form replaced e^ε/(1 + e^ε); it differs by 1 ulp at ε = 3)
    for epsilon in [0.5f64, 1.0, 3.0] {
        let rr = RandomizedResponse::new(epsilon);
        s.f64(rr.p_true());
    }
    let mut rr = RandomizedResponse::with_probability(0.75, 11);
    for i in 0..200u64 {
        s.bool(rr.privatize(i % 3 == 0));
    }
    for (n, k) in [(1_000u64, 700u64), (10u64, 1u64)] {
        s.f64(RandomizedResponse::estimate_proportion(0.75, n, k));
    }

    // RAPPOR
    let mut rap = Rappor::with_seed_params(0.5, 0.75, 0.25, 23);
    let (f, p, q) = rap.params();
    s.f64(f);
    s.f64(p);
    s.f64(q);
    for i in 0..32u64 {
        for b in rap.privatize(i * 0x0001_2345) {
            s.byte(b);
        }
    }

    // budget and aggregator
    let mut budget = PrivacyBudget::new(2.0);
    for e in [0.5f64, 0.75, 0.9, 0.1] {
        s.bool(budget.try_spend(e));
        s.f64(budget.remaining());
        s.f64(budget.spent());
    }
    s.bool(budget.is_exhausted());

    let mut agg = PrivateAggregator::new(1.5);
    for i in 0..500u64 {
        agg.add(prand(i) * 40.0 - 20.0);
    }
    s.f64(agg.estimate_mean());
    s.f64(agg.estimate_sum());
    s.f64(agg.standard_error());

    assert_golden("privacy", s, 3_000, GOLDEN_PRIVACY);
}

// ---------------------------------------------------------------------------
// 6. streaming_ops — change rate, EMA, SMA, regression
// ---------------------------------------------------------------------------

const GOLDEN_STREAMING_OPS: &str =
    "331765f15f622e6ee6a7a37813bef01e82b88cf838ad5094f996677f4d143632";

#[test]
fn golden_streaming_ops() {
    use alice_analytics::streaming_ops::{
        ChangeRate, ExponentialMovingAverage, LinearRegression, LinearRegressionFull,
        SimpleMovingAverage,
    };
    let mut s = Sink::default();

    let mut cr = ChangeRate::new();
    for i in 0..500u64 {
        if let Some(rate) = cr.observe(prand(i) * 1_000.0, i * 250) {
            s.f64(rate);
        }
        s.f64(cr.rate());
    }

    for span in [2u64, 5, 20, 100] {
        let mut ema = ExponentialMovingAverage::from_span(span);
        s.f64(ema.alpha());
        for i in 0..300u64 {
            ema.observe(prand(i) * 8.0);
            s.f64(ema.value());
        }
    }

    let mut sma = SimpleMovingAverage::<32>::new();
    for i in 0..300u64 {
        sma.observe(prand(i) * 8.0);
        s.f64(sma.value());
    }

    let mut lr = LinearRegression::new();
    let mut lrf = LinearRegressionFull::new();
    for i in 0..1_000u64 {
        let x = f64::from(i as u32) * 0.25;
        let y = 3.5 * x - 2.0 + (prand(i) - 0.5) * 0.1;
        lr.observe(x, y);
        lrf.observe(x, y);
    }
    s.f64(lr.slope());
    s.f64(lr.intercept());
    s.f64(lr.r_squared());
    s.f64(lrf.slope());
    s.f64(lrf.intercept());
    s.f64(lrf.r_squared());
    for x in [0.0f64, 1.0, 123.75, -50.0] {
        s.f64(lr.predict(x));
        s.f64(lrf.predict(x));
    }

    assert_golden("streaming_ops", s, 4_000, GOLDEN_STREAMING_OPS);
}

// ---------------------------------------------------------------------------
// 7. law (feature `law`) — residual summary of a fitted signal law
// ---------------------------------------------------------------------------

#[cfg(feature = "law")]
const GOLDEN_LAW: &str = "f017cc3eb4e268c47f52d045607ee0d82d256cc9813805fe9e051469dc2c907d";

#[cfg(feature = "law")]
#[test]
fn golden_law() {
    use alice_analytics::law::{residual_summary, ResidualSketch};
    use alice_zip::law::{Provenance, ResidualStats, SignalLaw, SignalLawParts, ValidRange};
    let mut s = Sink::default();

    // f(x) = c0 + c1·u with u = (x − 0) / (8 − 0): the division is by a power
    // of two, so y = 1 + 2x is exact in f64 for dyadic x
    let evidence: Vec<(f64, f64)> = (0..=8)
        .map(|i| {
            let x = f64::from(i);
            (x, 2.0 * x + 1.0)
        })
        .collect();
    let law = SignalLaw::from_parts(SignalLawParts {
        coefficients: vec![1.0, 16.0],
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
    .expect("the parts describe a valid linear law");

    let points: Vec<(f64, f64)> = (1..=500u64)
        .map(|i| {
            let x = prand(i) * 8.0;
            (x, 2.0 * x + 1.0 + (prand(i + 7_777) - 0.5))
        })
        .collect();

    let summary = residual_summary(&law, &points);
    s.u64(summary.count);
    s.u64(summary.out_of_range);
    s.u64(summary.non_finite);
    s.u64(summary.outside_accuracy);
    s.f64(summary.relative_accuracy);
    s.f64(summary.accurate_range.0);
    s.f64(summary.accurate_range.1);
    let d = summary
        .distribution
        .expect("500 in-range points produce a distribution");
    s.f64(d.mean);
    s.f64(d.min);
    s.f64(d.max);
    s.f64(d.max_abs);
    s.f64(d.p50);
    s.f64(d.p90);
    s.f64(d.p99);

    let mut sk = ResidualSketch::new(&law, 0.01).expect("0.01 is a valid relative accuracy");
    sk.extend(&points);
    for q in [0.0f64, 0.25, 0.5, 0.75, 1.0] {
        s.f64(sk.quantile(q).expect("q is inside the unit interval"));
    }
    let (lo, hi) = sk.accurate_range();
    s.f64(lo);
    s.f64(hi);
    s.f64(sk.scale());

    assert_golden("law", s, 150, GOLDEN_LAW);
}

// ---------------------------------------------------------------------------
// 8. semantics_id — the identifier of the arithmetic the scenarios above are
//    computed with, and the law identifier that mixes it in
// ---------------------------------------------------------------------------

/// `alice_analytics::SEMANTICS_ID` as hex
///
/// Unlike the scenarios above, this is not a hash of this crate's output: it
/// is the constant `alice-det-math` publishes to name its own numeric
/// behaviour, re-exported here. The value is transcribed from the dependency's
/// released source, so it pins *which* arithmetic this crate is built against.
///
/// A mismatch means the resolved `alice-det-math` is not the one this crate
/// was pinned to. That is not automatically wrong — the upstream re-records
/// the constant when it changes a function's output on purpose — but it must
/// not pass unnoticed, because every quantile, cardinality and noise sample in
/// this crate is only reproducible under one arithmetic. On a deliberate
/// upgrade: verify the golden hashes above are unchanged (they are the actual
/// outputs), then update this constant and record both in CHANGELOG.
const GOLDEN_SEMANTICS_ID: &str =
    "d2209b30f6f1f45baa1b638bcdfee34ac64773b2e63b9c083b2e77afc691398e";

#[test]
fn golden_semantics_id() {
    let mut s = Sink::default();
    for b in alice_analytics::SEMANTICS_ID {
        s.byte(b);
    }
    let bytes = s.len();
    assert_eq!(bytes, 32, "the identifier is 32 bytes, got {bytes}");

    let mut hex = String::with_capacity(64);
    for b in alice_analytics::SEMANTICS_ID {
        write!(hex, "{b:02x}").expect("writing to a String cannot fail");
    }
    assert_eq!(
        hex, GOLDEN_SEMANTICS_ID,
        "\n\nThe arithmetic this crate computes with changed.\n\
         actual:   {hex}\n\
         expected: {GOLDEN_SEMANTICS_ID}\n\n\
         Check which alice-det-math version resolved, confirm the golden\n\
         scenario hashes above are unchanged, then update GOLDEN_SEMANTICS_ID\n\
         and record the change in CHANGELOG.\n"
    );
}

/// The law identifier a residual summary reports mixes the arithmetic in, so
/// pinning it pins both halves at once
///
/// The closed-form properties of that identifier (what it covers, what it
/// ignores) are checked in `tests/law_identity.rs` against an independent
/// SHA-256 of the published encoding; this is the change detector.
///
/// The value below was computed outside this toolchain from the published
/// encoding (`SHA-256` over the length-prefixed tags, the arithmetic
/// identifier, the domain bits and the coefficient bits of the law built in
/// the test) and matches what the implementation returns, so it pins a
/// verified value rather than whatever happened to come out.
#[cfg(feature = "law")]
const GOLDEN_LAW_ID: &str = "bd2c0c2407fcf7ad8f424a25d71c33a68b2223e4bcb95c3601b2dbc7c7ddd549";

#[cfg(feature = "law")]
#[test]
fn golden_law_id() {
    use alice_analytics::law::residual_summary;
    use alice_zip::law::{Provenance, ResidualStats, SignalLaw, SignalLawParts, ValidRange};

    let law = SignalLaw::from_parts(SignalLawParts {
        coefficients: vec![1.0, 2.0, -0.5],
        domain: ValidRange { lo: 0.0, hi: 8.0 },
        evidence: vec![(0.0, 1.0), (4.0, 1.75), (8.0, 2.5)],
        residual: ResidualStats {
            n: 0,
            rms: 0.0,
            max_abs: 0.0,
        },
        provenance: Provenance::new("golden", "closed form"),
        oracles: Vec::new(),
    })
    .expect("valid parts");

    let summary = residual_summary(&law, &[(0.0, 1.5), (4.0, 1.75), (8.0, 3.0)]);
    assert!(
        summary.count > 0,
        "the summary must have summarised something, or the identifier below \
         is the only thing this test measures"
    );

    let mut hex = String::with_capacity(64);
    for b in summary.law_id {
        write!(hex, "{b:02x}").expect("writing to a String cannot fail");
    }
    assert_eq!(hex.len(), 64, "a law identifier is 32 bytes of hex");
    assert_eq!(
        hex, GOLDEN_LAW_ID,
        "\n\nThe identifier reported for a fixed law changed.\n\
         actual:   {hex}\n\
         expected: {GOLDEN_LAW_ID}\n\n\
         Either the arithmetic changed (see golden_semantics_id) or the\n\
         encoding in alice-zip did. Both change what a stored summary means:\n\
         confirm which, then update GOLDEN_LAW_ID and record it in CHANGELOG.\n"
    );
}
