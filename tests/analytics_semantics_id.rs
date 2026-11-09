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

/// No earlier identifier is reproduced (the `DDSketch` sum became exact, which
/// moved the `window` and `law` goldens): the previous list is empty, and an
/// identifier of an earlier generation is not in it.
#[test]
fn no_earlier_identifier_is_reproduced_after_the_exact_sum() {
    assert!(PREVIOUS_ANALYTICS_SEMANTICS_IDS.is_empty());
    assert_eq!(
        hex(&ANALYTICS_SEMANTICS_ID),
        "847535174fa305517aa11fe8b98f604d1b1ec48abaa23d2b1dce9ccc8aa8884f"
    );
}
