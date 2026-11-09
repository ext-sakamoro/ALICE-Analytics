//! Content hash of this crate's Law-bearing semantics: [`ANALYTICS_SEMANTICS_ID`].
//!
//! This module folds the crate's own Law pins, the cross-platform goldens of
//! `tests/determinism_golden.rs` (one per estimator family), with
//! [`DET_MATH_PINS`], the identifier of each `alice-det-math` function this
//! crate calls ([`DET_MATH_FUNCTIONS`]), and the identifier of `alice-zip`,
//! into one 32-byte value. Two builds with the same identifier produce the same bits on every
//! pinned scenario; a build that changes any of them, or that resolves an
//! upstream crate with a different identifier, has a different value.
//!
//! # Hex form
//!
//! ```text
//! 847535174fa305517aa11fe8b98f604d1b1ec48abaa23d2b1dce9ccc8aa8884f
//! ```
//!
//! # Entries
//!
//! [`ANALYTICS_SEMANTICS_PINS`]: one entry per scenario golden (`"sketch"`,
//! `"stats"`, `"window"`, `"anomaly"`, `"privacy"`, `"streaming_ops"`, and
//! `"law"`, the last run with the `law` feature), plus `"alice-zip"` (its
//! `law::SEMANTICS_ID`, the `law` feature). Each [`DET_MATH_PINS`] entry goes in
//! under the name `det-math <function>`. The identifier goldens (`GOLDEN_SEMANTICS_ID`,
//! `GOLDEN_LAW_ID`) are not pins: they are functions of the entries.
//!
//! # Fold
//!
//! The same procedure as the `SEMANTICS_ID` fold of `alice-det-math`: sort
//! the entries by name (ascending byte order), reject an empty table, a
//! repeated name, an empty name or a name that is not ASCII, then SHA-256
//! over, for each entry in that order, the name length as 4 bytes big-endian
//! (`u32`), the name bytes and the 32 digest bytes.
//!
//! # Independent of cargo features
//!
//! The table and the identifier are constants that no `cfg` gates; upstream
//! entries are recorded as hex. `tests/analytics_semantics_id.rs` checks
//! them against the resolved constants (`alice-zip` with the `law` feature),
//! so resolving another upstream identifier is a failing test.
//!
//! # Version and toolchain
//!
//! The identifier is a constant of this crate version: a dependency bump or
//! a toolchain change that moved any pinned bits fails a golden, and moving
//! the golden moves the identifier. Earlier identifiers whose pinned bits
//! are still reproduced are listed in [`PREVIOUS_ANALYTICS_SEMANTICS_IDS`].

/// Decodes 64 hex digits into 32 bytes at compile time; anything else is a
/// compile error.
pub(crate) const fn hex32(s: &str) -> [u8; 32] {
    const fn nibble(c: u8) -> u8 {
        match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            _ => panic!("not a lower-case hex digit"),
        }
    }
    let b = s.as_bytes();
    assert!(b.len() == 64, "a digest is 64 hex digits");
    let mut out = [0u8; 32];
    let mut i = 0;
    while i < 32 {
        out[i] = nibble(b[2 * i]) * 16 + nibble(b[2 * i + 1]);
        i += 1;
    }
    out
}

/// The pins folded into [`ANALYTICS_SEMANTICS_ID`], as `(name, digest)`.
pub const ANALYTICS_SEMANTICS_PINS: [(&str, [u8; 32]); 8] = [
    (
        "sketch",
        hex32("49078f4c720eedc79aab35ba79d55f18f6f923fca3069fa3fdb08b264e3719b4"),
    ),
    (
        "stats",
        hex32("bf98ab7a17423f29706e4a6dc5479b4d4188d4d7318dde69bc9b7f176ea90850"),
    ),
    (
        "window",
        hex32("5f8fe58f25070514326759ff28497c7f5cd1221ad680ac10c4d092c1d461c147"),
    ),
    (
        "anomaly",
        hex32("0e13fcff4b706f052b6ce3208eb548d8be526df1e5f7712e035ed0e9b4e050d3"),
    ),
    (
        "privacy",
        hex32("50f95bbc492118d85de552a3a950673e3c762b9005d1217c81c042cab9494ca5"),
    ),
    (
        "streaming_ops",
        hex32("331765f15f622e6ee6a7a37813bef01e82b88cf838ad5094f996677f4d143632"),
    ),
    (
        "law",
        hex32("8dd341a44001b19a881e1879faf09f0bb58df759761d8333897f7f21da6b57b6"),
    ),
    (
        "alice-zip",
        hex32("65f48962c757ebb619d252f8784f830aca5f3698fbd22f34150f9887863f38dd"),
    ),
];

/// The functions of `alice-det-math` this crate calls (ALICE-Law's
/// `scripts/fold_set_check.py` compares this with the calls in the source and
/// writes it).
pub const DET_MATH_FUNCTIONS: &[&str] = &["exp64", "ln64", "max64", "min64"];

/// The identifier of each function of [`DET_MATH_FUNCTIONS`], in that order, as
/// recorded when [`ANALYTICS_SEMANTICS_ID`] was computed. A function whose
/// identifier moves in the resolved `alice-det-math` stops the compile.
pub const DET_MATH_PINS: &[(&str, [u8; 32])] = &[
    (
        "exp64",
        hex32("f87502b37c757ae5ea59a3cc991d02c3fe19ff3c75c9edd294216055372bf2a8"),
    ),
    (
        "ln64",
        hex32("51c11488d360bb972761d2cecf192c173249f846f8127ef950e580b99a590cdc"),
    ),
    (
        "max64",
        hex32("aac586eacc9acd69cf5988ffe736283c71fad8146027e86034d3e9b881db28b6"),
    ),
    (
        "min64",
        hex32("1bef418073152f5e1e76f41718aa29b2517de8966a8c8ea8c1c2b5ba92fab3cb"),
    ),
];

const _: () = alice_det_math::assert_function_pins(DET_MATH_PINS);

/// The semantics identifier of this crate: the fold of
/// [`ANALYTICS_SEMANTICS_PINS`] (see the module documentation).
pub const ANALYTICS_SEMANTICS_ID: [u8; 32] =
    hex32("847535174fa305517aa11fe8b98f604d1b1ec48abaa23d2b1dce9ccc8aa8884f");

/// Earlier values of [`ANALYTICS_SEMANTICS_ID`] whose pinned bits this
/// release still reproduces, newest first
///
/// Empty: `54cbe6e9…` and the generations before it (`38e4dba4…`, `2def9a11…`,
/// `f8ded596…`) folded the `window` and `law` goldens of a `DDSketch` whose sum
/// was a running float sum; the sum is now exact (see
/// [`ExactSum`](crate::exact_sum::ExactSum)), so `mean` and `sum` of those
/// scenarios have other bits and no earlier identifier's pinned bits are reproduced.
pub const PREVIOUS_ANALYTICS_SEMANTICS_IDS: &[[u8; 32]] = &[];
