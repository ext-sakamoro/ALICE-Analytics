//! `ANALYTICS_SEMANTICS_ID` is the fold of its pins, every pin is the source
//! it claims to be, and every pin moves the identifier.
//!
//! oracle: the fold is recomputed here from the documented procedure (sort
//! by name, length-prefixed name, 32 digest bytes, SHA-256), the golden
//! entries are read from `tests/determinism_golden.rs` (the source, not a
//! copy), and the upstream entries are compared with the upstream crates'
//! own constants.

use alice_analytics::semantics::{
    ANALYTICS_SEMANTICS_ID, ANALYTICS_SEMANTICS_PINS, DET_MATH_FUNCTIONS, DET_MATH_PINS,
    PREVIOUS_ANALYTICS_SEMANTICS_IDS,
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
        hex(&fold_entries(&entries())),
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
    expected.extend(["alice-zip"]);
    assert_eq!(names, expected);
}

#[test]
fn every_det_math_entry_is_the_resolved_function() {
    assert!(!DET_MATH_PINS.is_empty());
    assert_eq!(DET_MATH_FUNCTIONS.len(), DET_MATH_PINS.len());
    for (want, (name, digest)) in DET_MATH_FUNCTIONS.iter().zip(DET_MATH_PINS) {
        assert_eq!(want, name);
        assert_eq!(
            hex(digest),
            hex(&alice_det_math::function_pin(name)),
            "{name}"
        );
    }
}

/// The fold's entries: a det-math entry goes in under `det-math <function>`
fn entries() -> Vec<(String, [u8; 32])> {
    ANALYTICS_SEMANTICS_PINS
        .iter()
        .map(|&(n, d)| (n.to_string(), d))
        .chain(
            DET_MATH_PINS
                .iter()
                .map(|&(n, d)| (format!("det-math {n}"), d)),
        )
        .collect()
}

fn fold_entries(e: &[(String, [u8; 32])]) -> [u8; 32] {
    let named: Vec<(&str, [u8; 32])> = e.iter().map(|(n, d)| (n.as_str(), *d)).collect();
    fold(&named)
}

/// The fold of an earlier generation: this table's goldens, with the whole identifier of alice-det-math and
/// the identifier of alice-zip of then
fn fold_of_generation(det_math: &str, zip: &str) -> [u8; 32] {
    let unhex = |s: &str| -> [u8; 32] {
        core::array::from_fn(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
    };
    let mut before: Vec<(&str, [u8; 32])> = ANALYTICS_SEMANTICS_PINS
        .iter()
        .map(|&(n, d)| {
            if n == "alice-zip" {
                (n, unhex(zip))
            } else {
                (n, d)
            }
        })
        .collect();
    before.push(("alice-det-math", unhex(det_math)));
    fold(&before)
}

/// `38e4dba4…` folded alice-det-math's whole identifier: the same scenario goldens, so data stamped with it is
/// reproduced here; removing it from the previous list fails this test
#[test]
fn the_previous_identifier_is_the_fold_before_the_function_identifiers() {
    let previous = fold_of_generation(
        "a2a11b9b826227fa7a9ed8ab2c1ce63dbb07ad4c39aa2c46a363ddea6b47dbe9",
        "17d0df1d7c0a6e63d1a7380129f12386b677ac96e7ac193dece546805f4fe84c",
    );
    assert_eq!(
        hex(&previous),
        "38e4dba4aeda3ea1a291d620c0183470aaffb9c9d3cd9a101b1435b0d93909df"
    );
    assert!(PREVIOUS_ANALYTICS_SEMANTICS_IDS.contains(&previous));
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
    let previous = fold_of_generation(
        "85e0884623b154a3b2b264374907b6cbac09307ca3f9a08339e48c567d3cb315",
        "51ffeb079769ea34bd3cb3df756b6cb940b8d2d6a6a0f1e1115ab8139415935e",
    );
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
    let previous = fold_of_generation(
        "a2e46794a49b62730d8eb75d10e6dbfdd59d6562a985ae94c716fc637a57e3cf",
        "50a40cf64625018acc99c940e544353b901457b4a7a80bf903544ed4ef9af406",
    );
    assert_eq!(
        hex(&previous),
        "2def9a114cb31abd2918f95a57f5ad084917006a9c95810a41db197fffebdfa0"
    );
    assert!(
        alice_analytics::semantics::PREVIOUS_ANALYTICS_SEMANTICS_IDS.contains(&previous),
        "the identifier before det-math 2d01c63f is not in PREVIOUS_ANALYTICS_SEMANTICS_IDS"
    );
}
