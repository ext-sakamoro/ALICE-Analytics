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

/// The fold entries of `84753517…` (alice-zip `65f48962…`), frozen as they were: the
/// 8 pins and the 4 det-math entries
const ENTRIES_84753517: &[(&str, &str)] = &[
    (
        "sketch",
        "49078f4c720eedc79aab35ba79d55f18f6f923fca3069fa3fdb08b264e3719b4",
    ),
    (
        "stats",
        "bf98ab7a17423f29706e4a6dc5479b4d4188d4d7318dde69bc9b7f176ea90850",
    ),
    (
        "window",
        "5f8fe58f25070514326759ff28497c7f5cd1221ad680ac10c4d092c1d461c147",
    ),
    (
        "anomaly",
        "0e13fcff4b706f052b6ce3208eb548d8be526df1e5f7712e035ed0e9b4e050d3",
    ),
    (
        "privacy",
        "50f95bbc492118d85de552a3a950673e3c762b9005d1217c81c042cab9494ca5",
    ),
    (
        "streaming_ops",
        "331765f15f622e6ee6a7a37813bef01e82b88cf838ad5094f996677f4d143632",
    ),
    (
        "law",
        "8dd341a44001b19a881e1879faf09f0bb58df759761d8333897f7f21da6b57b6",
    ),
    (
        "alice-zip",
        "65f48962c757ebb619d252f8784f830aca5f3698fbd22f34150f9887863f38dd",
    ),
    (
        "det-math exp64",
        "f87502b37c757ae5ea59a3cc991d02c3fe19ff3c75c9edd294216055372bf2a8",
    ),
    (
        "det-math ln64",
        "51c11488d360bb972761d2cecf192c173249f846f8127ef950e580b99a590cdc",
    ),
    (
        "det-math max64",
        "aac586eacc9acd69cf5988ffe736283c71fad8146027e86034d3e9b881db28b6",
    ),
    (
        "det-math min64",
        "1bef418073152f5e1e76f41718aa29b2517de8966a8c8ea8c1c2b5ba92fab3cb",
    ),
];

/// `84753517…` is the previous identifier, and the reason is exactly alice-zip's identifier
/// moving (the kind tag of its causal 2-D law became an entry): every other entry is unchanged
/// and none was added or removed, so its pinned bits are reproduced and it is listed in
/// `PREVIOUS_ANALYTICS_SEMANTICS_IDS` (a pin-only move keeps the old identifier).
#[test]
fn the_84753517_identifier_is_previous_because_only_the_alice_zip_pin_moved() {
    let unhex = |s: &str| -> [u8; 32] {
        core::array::from_fn(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).expect("hex"))
    };
    let frozen: Vec<(&str, [u8; 32])> = ENTRIES_84753517
        .iter()
        .map(|&(n, d)| (n, unhex(d)))
        .collect();
    assert_eq!(
        hex(&fold(&frozen)),
        "847535174fa305517aa11fe8b98f604d1b1ec48abaa23d2b1dce9ccc8aa8884f",
        "the frozen table does not fold to its identifier"
    );
    let now = entries();
    let mut moved = Vec::new();
    let mut compared = 0usize;
    for (name, digest) in ENTRIES_84753517 {
        compared += 1;
        let current = now.iter().find(|(n, _)| n == name);
        if current.is_none_or(|(_, d)| hex(d) != *digest) {
            moved.push(*name);
        }
    }
    assert_eq!(compared, 12, "entries compared");
    assert_eq!(moved, ["alice-zip"]);
    assert_eq!(
        now.len(),
        ENTRIES_84753517.len(),
        "no entry added or removed"
    );
    assert_eq!(
        PREVIOUS_ANALYTICS_SEMANTICS_IDS,
        [unhex(
            "847535174fa305517aa11fe8b98f604d1b1ec48abaa23d2b1dce9ccc8aa8884f"
        )]
    );
}

/// No identifier before `84753517…` is reproduced (the `DDSketch` sum became exact, which
/// moved the `window` and `law` goldens): `54cbe6e9…` and the generations before it are not in
/// the previous list.
#[test]
fn no_earlier_identifier_is_reproduced_after_the_exact_sum() {
    let unhex = |s: &str| -> [u8; 32] {
        core::array::from_fn(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).expect("hex"))
    };
    assert!(!PREVIOUS_ANALYTICS_SEMANTICS_IDS.contains(&unhex(
        "54cbe6e9c9a3aa1170f2b9cdb22c2a9503fa78be5e079aad5b42795fee6cbf53"
    )));
    assert_eq!(PREVIOUS_ANALYTICS_SEMANTICS_IDS.len(), 1);
    assert_eq!(
        hex(&ANALYTICS_SEMANTICS_ID),
        "67650b7f63ebe71670b7fd9b5781fe276f515dbcee4497169cdd74f1c19a08d4"
    );
}
