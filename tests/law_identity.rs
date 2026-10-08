//! Oracles for the identity a residual summary reports: which law produced the
//! residuals, and under which arithmetic.
//!
//! A summary is a handful of numbers. On its own it cannot say what it is a
//! summary *of*, so a consumer that stores or forwards one cannot later tell
//! whether two summaries describe the same law, nor whether they were computed
//! with the same transcendental implementations. `law_id` closes both: it is
//! `alice_zip::law::SignalLaw::law_id` taken under this crate's own
//! [`alice_analytics::SEMANTICS_ID`], so it changes when the law changes *and*
//! when the arithmetic changes.
//!
//! Every expected value here is built from the encoding published in
//! `SignalLaw::law_id`'s documentation with an independent SHA-256, not by
//! calling the function under test.
#![cfg(feature = "law")]

use alice_analytics::law::{residual_summary, ResidualSketch, DEFAULT_RELATIVE_ACCURACY};
use alice_zip::law::{
    Provenance, ResidualStats, SignalLaw, SignalLawParts, ValidRange, LAW_ID_DOMAIN,
    SIGNAL_LAW_KIND,
};
use sha2::{Digest, Sha256};

/// The two domain-separation tags as published, written out here so that an
/// unannounced change upstream shows up as a failure rather than as a silently
/// renamed identifier.
const EXPECTED_LAW_ID_DOMAIN: &[u8] = b"alice-zip/law-id/v1";
const EXPECTED_SIGNAL_LAW_KIND: &[u8] = b"signal-law/polynomial/v1";

/// `len` as a big-endian `u64`, then the bytes
fn length_prefixed(h: &mut Sha256, bytes: &[u8]) {
    h.update((bytes.len() as u64).to_be_bytes());
    h.update(bytes);
}

/// The identifier as documented, computed here from first principles
///
/// ```text
/// len(LAW_ID_DOMAIN)     LAW_ID_DOMAIN
/// len(SIGNAL_LAW_KIND)   SIGNAL_LAW_KIND
///                        semantics_id                 (32 bytes)
///                        domain.lo as big-endian bits (8 bytes)
///                        domain.hi as big-endian bits (8 bytes)
/// len(coefficients)      each coefficient, big-endian bits
/// ```
fn expected_id(coefficients: &[f64], lo: f64, hi: f64, semantics_id: &[u8; 32]) -> [u8; 32] {
    let mut h = Sha256::new();
    length_prefixed(&mut h, EXPECTED_LAW_ID_DOMAIN);
    length_prefixed(&mut h, EXPECTED_SIGNAL_LAW_KIND);
    h.update(semantics_id);
    h.update(lo.to_bits().to_be_bytes());
    h.update(hi.to_bits().to_be_bytes());
    h.update((coefficients.len() as u64).to_be_bytes());
    for c in coefficients {
        h.update(c.to_bits().to_be_bytes());
    }
    h.finalize().into()
}

/// `f(x) = c0 + c1·u`, `u = (x − lo) / 8`: the division is by a power of two,
/// so `f` is exact for dyadic `x` and the residuals below are exact too
fn line_law(c0: f64, c1: f64, evidence: Vec<(f64, f64)>, source: &str) -> SignalLaw {
    SignalLaw::from_parts(SignalLawParts {
        coefficients: vec![c0, c1],
        domain: ValidRange { lo: 0.0, hi: 8.0 },
        evidence,
        residual: ResidualStats {
            n: 0,
            rms: 0.0,
            max_abs: 0.0,
        },
        provenance: Provenance::new(source, "closed form"),
        oracles: Vec::new(),
    })
    .expect("valid parts")
}

fn points() -> Vec<(f64, f64)> {
    // f(x) = 1 + 2u with u = x/8, so f(0) = 1, f(4) = 2, f(8) = 3
    vec![(0.0, 1.5), (4.0, 1.5), (8.0, 3.5)]
}

/// The tags the published encoding is built from have not changed
#[test]
fn the_published_tags_are_the_ones_this_oracle_encodes() {
    assert_eq!(
        LAW_ID_DOMAIN, EXPECTED_LAW_ID_DOMAIN,
        "the law-id domain tag changed upstream, so every identifier changed"
    );
    assert_eq!(
        SIGNAL_LAW_KIND, EXPECTED_SIGNAL_LAW_KIND,
        "the law-kind tag changed upstream, so every identifier changed"
    );
}

/// The crate re-exports the arithmetic identifier of the kernels it actually
/// calls, so a consumer can record it next to the numbers
#[test]
fn the_re_exported_arithmetic_identifier_is_the_one_the_kernels_come_from() {
    assert_eq!(
        alice_analytics::SEMANTICS_ID,
        alice_det_math::SEMANTICS_ID,
        "the re-export must name the arithmetic that ln / exp / powf go through"
    );
}

/// A summary says which law it summarises, under which arithmetic
#[test]
fn a_summary_carries_the_identity_of_its_law_and_arithmetic() {
    let law = line_law(1.0, 2.0, points(), "run 1");
    let summary = residual_summary(&law, &points());

    let expected = expected_id(&[1.0, 2.0], 0.0, 8.0, &alice_analytics::SEMANTICS_ID);
    assert_eq!(
        summary.law_id, expected,
        "the summary's identifier must be the law's identifier under this crate's arithmetic"
    );
}

/// The identifier describes what the law computes, not how it was obtained:
/// the same coefficients over the same domain share it
#[test]
fn the_identity_ignores_evidence_and_provenance() {
    let a = line_law(1.0, 2.0, points(), "run 1");
    let b = line_law(
        1.0,
        2.0,
        vec![(1.0, 1.25), (2.0, 1.5), (3.0, 1.75)],
        "a different measurement, a different method",
    );

    let ia = residual_summary(&a, &points()).law_id;
    let ib = residual_summary(&b, &points()).law_id;
    assert_eq!(
        ia, ib,
        "two laws that evaluate identically must share one identifier"
    );
}

/// A law that evaluates differently gets a different identifier
#[test]
fn a_changed_coefficient_changes_the_identity() {
    let base = residual_summary(&line_law(1.0, 2.0, points(), "run 1"), &points()).law_id;

    // The neighbour is taken from the bit pattern, not written as a decimal:
    // the ulp at 2.0 is 2^-51, so a hand-written `2.000_000_000_000_000_2`
    // rounds back to 2.0 and would compare two identical laws.
    let one_ulp_up = f64::from_bits(2.0_f64.to_bits() + 1);
    assert_ne!(
        one_ulp_up.to_bits(),
        2.0_f64.to_bits(),
        "the neighbour must be a different f64"
    );

    for (c0, c1) in [(1.0, one_ulp_up), (-1.0, 2.0), (1.0, -2.0), (0.0, 2.0)] {
        let other = residual_summary(&line_law(c0, c1, points(), "run 1"), &points()).law_id;
        assert_ne!(
            base, other,
            "f(x) = {c0} + {c1}u evaluates differently, so its identifier must differ"
        );
    }
}

/// `-0.0` is observable in `evaluate`, so it is not folded into `+0.0`
#[test]
fn a_negative_zero_coefficient_is_a_different_law() {
    let plus = residual_summary(&line_law(0.0, 2.0, points(), "run 1"), &points()).law_id;
    let minus = residual_summary(&line_law(-0.0, 2.0, points(), "run 1"), &points()).law_id;
    assert_ne!(
        plus, minus,
        "the sign of a zero coefficient is observable, so it must reach the identifier"
    );
}

/// The arithmetic identifier is part of the identity: the same law under a
/// different arithmetic is a different identity
#[test]
fn the_arithmetic_identifier_participates() {
    let law = line_law(1.0, 2.0, points(), "run 1");
    let actual = residual_summary(&law, &points()).law_id;

    let mut other_arithmetic = alice_analytics::SEMANTICS_ID;
    other_arithmetic[0] ^= 0x01;
    let under_other = expected_id(&[1.0, 2.0], 0.0, 8.0, &other_arithmetic);

    assert_ne!(
        actual, under_other,
        "changing the arithmetic must change the identifier the summary reports"
    );
}

/// The streaming and one-shot entry points report the same identity, and the
/// identity does not depend on how many points were pushed
#[test]
fn streaming_and_one_shot_agree_and_the_identity_is_not_data_dependent() {
    let law = line_law(1.0, 2.0, points(), "run 1");
    let one_shot = residual_summary(&law, &points()).law_id;

    let mut sketch = ResidualSketch::new(&law, DEFAULT_RELATIVE_ACCURACY).expect("valid alpha");
    let empty = sketch.summary().law_id;
    assert_eq!(
        empty, one_shot,
        "a summary of no points still names the law it would summarise"
    );

    sketch.extend(&points());
    assert_eq!(
        sketch.summary().law_id,
        one_shot,
        "pushing points must not change which law the summary names"
    );
    assert!(
        sketch.summary().count > 0,
        "the streaming path must have summarised something, or this test compares two empty runs"
    );
}
