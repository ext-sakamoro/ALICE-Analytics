//! Sketches as bytes: each sketch encodes to a fixed-length, versioned byte
//! form, decodes back to the same sketch, refuses bytes that are not one, and
//! the merge of decoded parts is the sketch of the union.
//!
//! oracle: the byte layout is written by hand from the format in the docs of
//! `sketch::bytes` (`ALAS`, version 1, kind, parameters, little-endian
//! fields); "the sketch of the union" is built by inserting every item into
//! one sketch, independently of the merge; the error bounds are the ones the
//! sketches state (`HyperLogLog` 1.04/sqrt(m), Count-Min e/w of the total,
//! `DDSketch` relative alpha), checked against exact counts and sorted values.

// test arithmetic on counts and exact float values, and sketches built on the stack as the
// crate's own constructors do
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::float_cmp,
    clippy::large_stack_arrays
)]

use alice_analytics::sketch::{
    CountMinSketch1024x5, DDSketch512, HyperLogLog12, HyperLogLog14, MergeError, Mergeable,
    SketchBytesError,
};
use alice_analytics::ANALYTICS_SEMANTICS_ID;

/// The header: `ALAS`, version 1, kind, two parameter bytes, then the crate's
/// meaning identifier (32 bytes).
const HEADER: usize = 8 + 32;
/// The exact sum of a `DDSketch`: 34 little-endian `u64` limbs, two's complement,
/// holding the sum times 2^1074.
const SUM_LIMBS: usize = 34;

/// An identifier this crate had before (54cbe6e9…): bytes written under it are refused.
const OLD_ID: [u8; 32] = [
    0x54, 0xcb, 0xe6, 0xe9, 0xc9, 0xa3, 0xaa, 0x11, 0x70, 0xf2, 0xb9, 0xcd, 0xb2, 0x2c, 0x2a, 0x95,
    0x03, 0xfa, 0x78, 0xbe, 0x5e, 0x07, 0x9a, 0xad, 0x5b, 0x42, 0x79, 0x5f, 0xee, 0x6c, 0xbf, 0x53,
];

// ---- HyperLogLog -----------------------------------------------------------

#[test]
fn an_empty_hyperloglog_is_its_header_and_zero_registers() {
    let h = HyperLogLog12::new();
    let mut out = vec![0u8; HyperLogLog12::ENCODED_LEN];
    h.encode(&mut out).unwrap();
    assert_eq!(HyperLogLog12::ENCODED_LEN, HEADER + 4096);
    assert_eq!(&out[..8], &[b'A', b'L', b'A', b'S', 1, 1, 12, 0]);
    assert_eq!(&out[8..HEADER], &ANALYTICS_SEMANTICS_ID);
    assert!(out[HEADER..].iter().all(|&b| b == 0));
}

fn hll14(items: impl Iterator<Item = u64>) -> HyperLogLog14 {
    let mut h = HyperLogLog14::new();
    for i in items {
        h.insert(&i);
    }
    h
}

fn bytes_hll14(h: &HyperLogLog14) -> Vec<u8> {
    let mut out = vec![0u8; HyperLogLog14::ENCODED_LEN];
    h.encode(&mut out).unwrap();
    out
}

#[test]
fn a_hyperloglog_round_trips_bit_equal() {
    let h = hll14(0..20_000);
    let b = bytes_hll14(&h);
    let back = HyperLogLog14::decode(&b).unwrap();
    assert_eq!(bytes_hll14(&back), b);
    assert_eq!(back.cardinality().to_bits(), h.cardinality().to_bits());
}

#[test]
fn merged_hyperloglog_parts_are_the_sketch_of_the_union() {
    let union = hll14(0..30_000);
    // three overlapping parts, each through bytes, merged in two orders
    let parts = [
        hll14(0..12_000),
        hll14(10_000..21_000),
        hll14(20_000..30_000),
    ];
    let bytes: Vec<Vec<u8>> = parts.iter().map(bytes_hll14).collect();
    for order in [[0usize, 1, 2], [2, 0, 1]] {
        let mut m = HyperLogLog14::decode(&bytes[order[0]]).unwrap();
        for &i in &order[1..] {
            m.merge(&HyperLogLog14::decode(&bytes[i]).unwrap()).unwrap();
        }
        assert_eq!(
            bytes_hll14(&m),
            bytes_hll14(&union),
            "register max is exact"
        );
    }
    let rel = (union.cardinality() - 30_000.0).abs() / 30_000.0;
    assert!(
        rel <= 3.0 * 1.04 / (HyperLogLog14::M as f64).sqrt(),
        "relative error {rel}"
    );
}

#[test]
fn hyperloglog_bytes_that_are_not_one_are_refused() {
    let good = bytes_hll14(&hll14(0..100));
    let mut bad = good.clone();
    bad[0] = b'X';
    assert_eq!(
        HyperLogLog14::decode(&bad).unwrap_err(),
        SketchBytesError::Magic
    );
    let mut bad = good.clone();
    bad[4] = 2;
    assert_eq!(
        HyperLogLog14::decode(&bad).unwrap_err(),
        SketchBytesError::Version(2)
    );
    let mut bad = good.clone();
    bad[5] = 2;
    assert_eq!(
        HyperLogLog14::decode(&bad).unwrap_err(),
        SketchBytesError::Kind {
            expected: 1,
            found: 2
        }
    );
    // bytes written under another meaning identifier are refused, an old one included
    let mut bad = good.clone();
    bad[8..HEADER].copy_from_slice(&OLD_ID);
    assert_eq!(
        HyperLogLog14::decode(&bad).unwrap_err(),
        SketchBytesError::Semantics { found: OLD_ID }
    );
    let mut bad = good.clone();
    bad[HEADER - 1] ^= 1;
    assert!(matches!(
        HyperLogLog14::decode(&bad).unwrap_err(),
        SketchBytesError::Semantics { .. }
    ));
    // a HyperLogLog12 is not a HyperLogLog14
    let mut small = vec![0u8; HyperLogLog12::ENCODED_LEN];
    HyperLogLog12::new().encode(&mut small).unwrap();
    assert_eq!(
        HyperLogLog14::decode(&small).unwrap_err(),
        SketchBytesError::Shape
    );
    assert_eq!(
        HyperLogLog14::decode(&good[..good.len() - 1]).unwrap_err(),
        SketchBytesError::Length {
            expected: HyperLogLog14::ENCODED_LEN,
            found: good.len() - 1
        }
    );
    // a register above 64 - P + 1 cannot come from any insert
    let mut bad = good.clone();
    bad[HEADER] = (64 - 14 + 2) as u8;
    assert!(matches!(
        HyperLogLog14::decode(&bad).unwrap_err(),
        SketchBytesError::Invalid(_)
    ));
    // the output buffer must be exactly the encoded length
    let mut short = vec![0u8; HyperLogLog14::ENCODED_LEN - 1];
    assert!(matches!(
        hll14(0..1).encode(&mut short).unwrap_err(),
        SketchBytesError::Length { .. }
    ));
}

// ---- Count-Min -------------------------------------------------------------

fn cms(items: &[(u64, u64)]) -> CountMinSketch1024x5 {
    let mut c = CountMinSketch1024x5::new();
    for &(k, n) in items {
        c.insert_hash(k.wrapping_mul(0x9e37_79b9_7f4a_7c15), n);
    }
    c
}

fn bytes_cms(c: &CountMinSketch1024x5) -> Vec<u8> {
    let mut out = vec![0u8; CountMinSketch1024x5::ENCODED_LEN];
    c.encode(&mut out).unwrap();
    out
}

#[test]
fn count_min_layout_round_trip_and_merge() {
    assert_eq!(
        CountMinSketch1024x5::ENCODED_LEN,
        HEADER + 4 + 4 + 8 + 8 * 1024 * 5
    );
    let empty = bytes_cms(&CountMinSketch1024x5::new());
    assert_eq!(&empty[..8], &[b'A', b'L', b'A', b'S', 1, 3, 0, 0]);
    assert_eq!(&empty[8..HEADER], &ANALYTICS_SEMANTICS_ID);
    assert_eq!(&empty[HEADER..HEADER + 4], &1024u32.to_le_bytes());
    assert_eq!(&empty[HEADER + 4..HEADER + 8], &5u32.to_le_bytes());

    let all: Vec<(u64, u64)> = (0..3_000u64).map(|k| (k, 1 + k % 7)).collect();
    let union = cms(&all);
    let b = bytes_cms(&union);
    assert_eq!(bytes_cms(&CountMinSketch1024x5::decode(&b).unwrap()), b);

    let (x, y) = all.split_at(1_234);
    let mut m = CountMinSketch1024x5::decode(&bytes_cms(&cms(y))).unwrap();
    m.merge(&CountMinSketch1024x5::decode(&bytes_cms(&cms(x))).unwrap())
        .unwrap();
    assert_eq!(bytes_cms(&m), b, "counter addition is exact");

    // the stated bound: true <= estimate <= true + (e / w) * total
    let total: u64 = all.iter().map(|&(_, n)| n).sum();
    assert_eq!(m.total(), total);
    let bound = (std::f64::consts::E / 1024.0) * total as f64;
    let mut checked = 0;
    for &(k, n) in &all {
        let est = m.estimate_hash(k.wrapping_mul(0x9e37_79b9_7f4a_7c15));
        assert!(est >= n);
        if (est - n) as f64 <= bound {
            checked += 1;
        }
    }
    // the bound holds with probability 1 - e^-depth per item
    assert!(
        checked as f64 >= 0.99 * all.len() as f64,
        "{checked} of {}",
        all.len()
    );
}

#[test]
fn count_min_with_counters_that_do_not_add_up_is_refused() {
    let mut b = bytes_cms(&cms(&[(1, 5), (2, 9)]));
    // total follows width and depth; a total smaller than one row's sum cannot come from inserts
    b[HEADER + 8..HEADER + 16].copy_from_slice(&1u64.to_le_bytes());
    assert!(matches!(
        CountMinSketch1024x5::decode(&b).unwrap_err(),
        SketchBytesError::Invalid(_)
    ));
}

// ---- DDSketch --------------------------------------------------------------

fn values(range: std::ops::Range<u64>) -> Vec<f64> {
    // positive values over four decades, a few negatives and zeros
    range
        .map(|i| {
            let x = (i.wrapping_mul(2_654_435_761) % 10_000) as f64 + 1.0;
            match i % 10 {
                0 => 0.0,
                1 => -x / 10.0,
                _ => x / 10.0,
            }
        })
        .collect()
}

fn dd(vs: &[f64]) -> DDSketch512 {
    let mut d = DDSketch512::new(0.02);
    for &v in vs {
        d.insert(v);
    }
    d
}

fn bytes_dd(d: &DDSketch512) -> Vec<u8> {
    let mut out = vec![0u8; DDSketch512::ENCODED_LEN];
    d.encode(&mut out).unwrap();
    out
}

#[test]
fn ddsketch_layout_and_round_trip() {
    // bins, alpha, count, zero_count, non_finite, min, max, the exact sum, the bins
    assert_eq!(
        DDSketch512::ENCODED_LEN,
        HEADER + 4 + 8 * 6 + 8 * SUM_LIMBS + 16 * 512
    );
    let e = bytes_dd(&DDSketch512::new(0.02));
    assert_eq!(&e[..8], &[b'A', b'L', b'A', b'S', 1, 2, 0, 0]);
    assert_eq!(&e[8..HEADER], &ANALYTICS_SEMANTICS_ID);
    assert_eq!(&e[HEADER..HEADER + 4], &512u32.to_le_bytes());
    assert_eq!(
        &e[HEADER + 4..HEADER + 12],
        &0.02f64.to_bits().to_le_bytes()
    );

    let d = dd(&values(0..5_000));
    let b = bytes_dd(&d);
    let back = DDSketch512::decode(&b).unwrap();
    assert_eq!(bytes_dd(&back), b);
    for q in [0.0, 0.1, 0.5, 0.9, 0.99, 1.0] {
        assert_eq!(back.quantile(q).to_bits(), d.quantile(q).to_bits());
    }
}

#[test]
fn merged_ddsketch_parts_answer_like_the_sketch_of_the_union() {
    let all = values(0..9_000);
    let union = dd(&all);
    let parts = [dd(&all[..2_000]), dd(&all[2_000..6_500]), dd(&all[6_500..])];
    let bytes: Vec<Vec<u8>> = parts.iter().map(bytes_dd).collect();
    let mut sorted = all.clone();
    sorted.sort_by(f64::total_cmp);
    for order in [[0usize, 1, 2], [2, 1, 0]] {
        let mut m = DDSketch512::decode(&bytes[order[0]]).unwrap();
        for &i in &order[1..] {
            m.merge(&DDSketch512::decode(&bytes[i]).unwrap()).unwrap();
        }
        assert_eq!(m.count(), union.count());
        for q in [0.05, 0.25, 0.5, 0.75, 0.95, 0.99] {
            // the bins add exactly, so a quantile is the union's to the bit
            assert_eq!(
                m.quantile(q).to_bits(),
                union.quantile(q).to_bits(),
                "q {q}"
            );
            // and within the stated relative error of the exact value
            let rank = (q * (sorted.len() - 1) as f64).floor() as usize;
            let exact = sorted[rank];
            if exact != 0.0 {
                assert!(
                    (m.quantile(q) - exact).abs() <= 0.02 * exact.abs() * 1.000_001,
                    "q {q}"
                );
            }
        }
        // the sum is exact, so the merged sketch is the union's, to the byte
        assert_eq!(bytes_dd(&m), bytes_dd(&union));
    }
}

#[test]
fn ddsketch_bytes_that_are_not_one_are_refused() {
    let good = bytes_dd(&dd(&values(0..100)));
    let mut bad = good.clone();
    bad[HEADER + 4..HEADER + 12].copy_from_slice(&1.5f64.to_bits().to_le_bytes());
    assert!(
        matches!(
            DDSketch512::decode(&bad).unwrap_err(),
            SketchBytesError::Invalid(_)
        ),
        "alpha outside (0, 1)"
    );
    let mut bad = good.clone();
    bad[HEADER..HEADER + 4].copy_from_slice(&256u32.to_le_bytes());
    assert_eq!(
        DDSketch512::decode(&bad).unwrap_err(),
        SketchBytesError::Shape
    );
    // count (after bins and alpha) must be the zeros plus every bin
    let mut bad = good.clone();
    bad[HEADER + 12..HEADER + 20].copy_from_slice(&7u64.to_le_bytes());
    assert!(matches!(
        DDSketch512::decode(&bad).unwrap_err(),
        SketchBytesError::Invalid(_)
    ));
    assert!(matches!(
        DDSketch512::decode(&good[..40]).unwrap_err(),
        SketchBytesError::Length { .. }
    ));
}

// ---- merge: order, refusal, overflow -----------------------------------------

#[test]
fn ddsketch_parts_merged_in_every_order_give_the_same_bytes() {
    let all = values(0..4_000);
    let parts = [
        dd(&all[..700]),
        dd(&all[700..2_100]),
        dd(&all[2_100..3_000]),
        dd(&all[3_000..]),
    ];
    let union = bytes_dd(&dd(&all));
    let mut orders = 0;
    // every permutation of four parts
    for a in 0..4 {
        for b in 0..4 {
            for c in 0..4 {
                for d in 0..4 {
                    let o = [a, b, c, d];
                    if (0..4).any(|i| !o.contains(&i)) {
                        continue;
                    }
                    let mut m = parts[o[0]].clone();
                    for &i in &o[1..] {
                        m.merge(&parts[i]).unwrap();
                    }
                    assert_eq!(bytes_dd(&m), union, "order {o:?}");
                    orders += 1;
                }
            }
        }
    }
    assert_eq!(orders, 24);
}

#[test]
fn the_ddsketch_sum_is_exact() {
    // oracle: the exact sums of these doubles, which a running float sum loses
    assert_eq!(dd(&[1e16, 1.0, -1e16]).sum(), 1.0);
    assert_eq!(dd(&[1e308, 1e308, -1e308]).sum(), 1e308);
    assert_eq!(dd(&[0.5, 0.25, 0.125]).sum(), 0.875);
    // the exact sum of the doubles 0.1 and 0.2 rounds to 0.30000000000000004
    assert_eq!(dd(&[0.1, 0.2]).sum().to_bits(), 0x3FD3_3333_3333_3334);
    assert_eq!(dd(&[]).sum(), 0.0);
}

#[test]
fn merging_ddsketches_of_different_alpha_is_refused_and_changes_nothing() {
    let mut a = dd(&values(0..100));
    let before = bytes_dd(&a);
    let mut other = DDSketch512::new(0.05);
    other.insert(3.0);
    assert_eq!(a.merge(&other), Err(MergeError::Alpha));
    assert_eq!(bytes_dd(&a), before);
}

/// A sketch of `n` zeros, written as bytes (`count` = `zero_count` = n), read back.
fn dd_of_zeros(n: u64) -> DDSketch512 {
    let mut b = bytes_dd(&DDSketch512::new(0.02));
    b[HEADER + 12..HEADER + 20].copy_from_slice(&n.to_le_bytes());
    b[HEADER + 20..HEADER + 28].copy_from_slice(&n.to_le_bytes());
    DDSketch512::decode(&b).unwrap()
}

#[test]
fn a_ddsketch_merge_that_would_overflow_a_count_is_refused_and_changes_nothing() {
    let mut a = dd_of_zeros(u64::MAX - 1);
    let before = bytes_dd(&a);
    assert_eq!(a.merge(&dd_of_zeros(5)), Err(MergeError::Overflow));
    assert_eq!(bytes_dd(&a), before);
    // one less fits
    assert_eq!(a.merge(&dd_of_zeros(1)), Ok(()));
    assert_eq!(a.count(), u64::MAX);
}

#[test]
fn a_count_min_merge_that_would_overflow_is_refused_and_changes_nothing() {
    let big = u64::MAX - 1;
    let mut b = bytes_cms(&CountMinSketch1024x5::new());
    b[HEADER + 8..HEADER + 16].copy_from_slice(&big.to_le_bytes());
    for row in 0..5 {
        let at = HEADER + 16 + 8 * 1024 * row;
        b[at..at + 8].copy_from_slice(&big.to_le_bytes());
    }
    let mut a = CountMinSketch1024x5::decode(&b).unwrap();
    assert_eq!(a.merge(&cms(&[(1, 5)])), Err(MergeError::Overflow));
    assert_eq!(bytes_cms(&a), b);
}

/// The handshake of every reader (ALICE-Law G8): bytes stamped with another
/// meaning identifier, an earlier one of this crate included, are refused by
/// each kind's `decode`, and bytes stamped with this build's are read.
#[test]
fn every_sketch_decode_checks_the_meaning_identifier() {
    fn stamped(mut b: Vec<u8>, id: &[u8; 32]) -> Vec<u8> {
        b[8..HEADER].copy_from_slice(id);
        b
    }
    let mut checked = 0;
    let h = bytes_hll14(&hll14(0..10));
    let c = bytes_cms(&cms(&[(1, 2)]));
    let d = bytes_dd(&dd(&[1.0, 2.0]));
    for id in [OLD_ID, [0u8; 32]] {
        let found = SketchBytesError::Semantics { found: id };
        assert_eq!(
            HyperLogLog14::decode(&stamped(h.clone(), &id)).unwrap_err(),
            found
        );
        assert_eq!(
            CountMinSketch1024x5::decode(&stamped(c.clone(), &id)).unwrap_err(),
            found
        );
        assert_eq!(
            DDSketch512::decode(&stamped(d.clone(), &id)).unwrap_err(),
            found
        );
        checked += 3;
    }
    assert!(HyperLogLog14::decode(&stamped(h, &ANALYTICS_SEMANTICS_ID)).is_ok());
    assert!(CountMinSketch1024x5::decode(&stamped(c, &ANALYTICS_SEMANTICS_ID)).is_ok());
    assert!(DDSketch512::decode(&stamped(d, &ANALYTICS_SEMANTICS_ID)).is_ok());
    assert_eq!(checked, 6);
}
