//! `ANALYTICS_SEMANTICS_ID` is the fold of its pins, every pin is the source
//! it claims to be, and every pin moves the identifier.
//!
//! oracle: the fold is recomputed here from the documented procedure (sort
//! by name, length-prefixed name, 32 digest bytes, SHA-256), the golden
//! entries are read from `tests/determinism_golden.rs` (the source, not a
//! copy), and the upstream entries are compared with the upstream crates'
//! own constants.

use alice_analytics::semantics::{
    ANALYTICS_SEMANTICS_ID, ANALYTICS_SEMANTICS_PINS, PREVIOUS_ANALYTICS_SEMANTICS_IDS,
};
use sha2::{Digest, Sha256};

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

fn fold(pins: &[(&str, [u8; 32])]) -> [u8; 32] {
    let mut sorted = pins.to_vec();
    sorted.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
    assert!(!sorted.is_empty(), "empty table");
    for w in sorted.windows(2) {
        assert_ne!(w[0].0, w[1].0, "repeated name");
    }
    let mut h = Sha256::new();
    for (name, digest) in &sorted {
        assert!(!name.is_empty() && name.is_ascii(), "bad name {name:?}");
        h.update(u32::try_from(name.len()).unwrap().to_be_bytes());
        h.update(name.as_bytes());
        h.update(digest);
    }
    h.finalize().into()
}

fn pin(name: &str) -> [u8; 32] {
    ANALYTICS_SEMANTICS_PINS
        .iter()
        .find(|(n, _)| *n == name)
        .unwrap_or_else(|| panic!("no pin {name}"))
        .1
}

/// The hex in the module documentation, the constant and the fold agree.
#[test]
fn the_identifier_is_the_fold_of_its_pins() {
    assert_eq!(
        hex(&ANALYTICS_SEMANTICS_ID),
        hex(&fold(&ANALYTICS_SEMANTICS_PINS)),
        "ANALYTICS_SEMANTICS_ID is not the fold of its pins"
    );
    let doc = include_str!("../src/semantics.rs");
    assert!(
        doc.contains(&format!("//! {}", hex(&ANALYTICS_SEMANTICS_ID))),
        "the module doc does not show the identifier's hex"
    );
}

/// The golden entries are the digests `tests/determinism_golden.rs` records.
#[test]
fn golden_pins_are_the_recorded_goldens() {
    let src = include_str!("determinism_golden.rs");
    let goldens = [
        ("sketch", "GOLDEN_SKETCH"),
        ("stats", "GOLDEN_STATS"),
        ("window", "GOLDEN_WINDOW"),
        ("anomaly", "GOLDEN_ANOMALY"),
        ("privacy", "GOLDEN_PRIVACY"),
        ("streaming_ops", "GOLDEN_STREAMING_OPS"),
        ("law", "GOLDEN_LAW"),
    ];
    for (name, constant) in goldens {
        let key = format!("const {constant}: &str =");
        let at = src
            .find(&key)
            .unwrap_or_else(|| panic!("{constant} not found"));
        let quote = at + src[at..].find('"').expect("opening quote") + 1;
        let recorded = &src[quote..quote + 64];
        assert_eq!(
            hex(&pin(name)),
            recorded,
            "pin {name} differs from {constant}"
        );
    }
    let names: Vec<_> = ANALYTICS_SEMANTICS_PINS.iter().map(|(n, _)| *n).collect();
    let mut expected: Vec<_> = goldens.iter().map(|(n, _)| *n).collect();
    expected.extend(["alice-det-math", "alice-zip"]);
    assert_eq!(names, expected);
}

#[test]
fn det_math_pin_is_the_resolved_arithmetic_identifier() {
    assert_eq!(
        hex(&pin("alice-det-math")),
        hex(&alice_analytics::SEMANTICS_ID)
    );
}

#[cfg(feature = "law")]
#[test]
fn zip_pin_is_the_resolved_zip_identifier() {
    assert_eq!(hex(&pin("alice-zip")), hex(&alice_zip::law::SEMANTICS_ID));
}

/// Changing any single pin (each golden and each upstream identifier) moves
/// the identifier.
#[test]
fn every_pin_moves_the_identifier() {
    for i in 0..ANALYTICS_SEMANTICS_PINS.len() {
        let mut changed = ANALYTICS_SEMANTICS_PINS;
        changed[i].1[31] ^= 1;
        assert_ne!(
            fold(&changed),
            fold(&ANALYTICS_SEMANTICS_PINS),
            "pin {} does not move the identifier",
            changed[i].0
        );
    }
}

#[test]
fn the_previous_list_does_not_contain_the_current_identifier() {
    assert!(!PREVIOUS_ANALYTICS_SEMANTICS_IDS.contains(&ANALYTICS_SEMANTICS_ID));
}

/// The previous identifier is this crate's fold with the upstream identifiers of the generation
/// before (alice-det-math 0.5.1 `85e08846…`, alice-zip `51ffeb07…`): the same scenario goldens, so
/// data stamped with it is reproduced here; removing it from the previous list fails this test
#[test]
fn the_previous_identifier_is_the_fold_under_the_previous_upstream_identifiers() {
    let unhex = |s: &str| -> [u8; 32] {
        core::array::from_fn(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
    };
    let before: Vec<(&str, [u8; 32])> = ANALYTICS_SEMANTICS_PINS
        .iter()
        .map(|&(n, d)| match n {
            "alice-det-math" => (
                n,
                unhex("85e0884623b154a3b2b264374907b6cbac09307ca3f9a08339e48c567d3cb315"),
            ),
            "alice-zip" => (
                n,
                unhex("51ffeb079769ea34bd3cb3df756b6cb940b8d2d6a6a0f1e1115ab8139415935e"),
            ),
            _ => (n, d),
        })
        .collect();
    let previous = fold(&before);
    assert_eq!(
        hex(&previous),
        "f8ded5961a8ef8be2d79ea4bfafd68f494c2443d725fb926cf9775824e043d4d"
    );
    assert!(
        alice_analytics::semantics::PREVIOUS_ANALYTICS_SEMANTICS_IDS.contains(&previous),
        "the identifier of the previous generation is not in PREVIOUS_ANALYTICS_SEMANTICS_IDS"
    );
}

/// The previous identifier `2def9a11…` is this crate's fold with the upstream identifiers of the
/// generation before (alice-det-math at 39507d1b `a2e46794…`, alice-zip `50a40cf6…`): the same
/// scenario goldens, so data stamped with it is reproduced here; removing it from the previous
/// list fails this test
#[test]
fn the_previous_identifier_before_det_math_2d01c63f_is_the_fold_under_its_upstream_identifiers() {
    let unhex = |s: &str| -> [u8; 32] {
        core::array::from_fn(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
    };
    let before: Vec<(&str, [u8; 32])> = ANALYTICS_SEMANTICS_PINS
        .iter()
        .map(|&(n, d)| match n {
            "alice-det-math" => (
                n,
                unhex("a2e46794a49b62730d8eb75d10e6dbfdd59d6562a985ae94c716fc637a57e3cf"),
            ),
            "alice-zip" => (
                n,
                unhex("50a40cf64625018acc99c940e544353b901457b4a7a80bf903544ed4ef9af406"),
            ),
            _ => (n, d),
        })
        .collect();
    let previous = fold(&before);
    assert_eq!(
        hex(&previous),
        "2def9a114cb31abd2918f95a57f5ad084917006a9c95810a41db197fffebdfa0"
    );
    assert!(
        alice_analytics::semantics::PREVIOUS_ANALYTICS_SEMANTICS_IDS.contains(&previous),
        "the identifier before det-math 2d01c63f is not in PREVIOUS_ANALYTICS_SEMANTICS_IDS"
    );
}
