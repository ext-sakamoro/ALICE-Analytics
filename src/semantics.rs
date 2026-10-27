//! Content hash of this crate's Law-bearing semantics: [`ANALYTICS_SEMANTICS_ID`].
//!
//! [`crate::SEMANTICS_ID`] is the arithmetic identifier of `alice-det-math`,
//! re-exported; it names the transcendental functions, not this crate's own
//! rules. This module folds the crate's own Law pins, the cross-platform
//! goldens of `tests/determinism_golden.rs` (one per estimator family), with
//! the identifiers of the Law-bearing crates it depends on, into one 32-byte
//! value. Two builds with the same identifier produce the same bits on every
//! pinned scenario; a build that changes any of them, or that resolves an
//! upstream crate with a different identifier, has a different value.
//!
//! # Hex form
//!
//! ```text
//! 38e4dba4aeda3ea1a291d620c0183470aaffb9c9d3cd9a101b1435b0d93909df
//! ```
//!
//! # Entries
//!
//! [`ANALYTICS_SEMANTICS_PINS`]: one entry per scenario golden (`"sketch"`,
//! `"stats"`, `"window"`, `"anomaly"`, `"privacy"`, `"streaming_ops"`, and
//! `"law"`, the last run with the `law` feature), plus `"alice-det-math"`
//! (its `SEMANTICS_ID`) and `"alice-zip"` (its `law::SEMANTICS_ID`, the
//! `law` feature). The identifier goldens (`GOLDEN_SEMANTICS_ID`,
//! `GOLDEN_LAW_ID`) are not pins: they are functions of the entries.
//!
//! `alice-zip` 0.11 defines its own `law::SEMANTICS_ID` (a fold of its own
//! Law pins with the arithmetic identifier), so its entry differs from the
//! `"alice-det-math"` entry. Under 0.10 the two entries were equal (0.10
//! re-exported `alice-det-math`'s identifier); the `"law"` golden and every
//! scenario golden are unchanged between the two, only this entry moved.
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
pub const ANALYTICS_SEMANTICS_PINS: [(&str, [u8; 32]); 9] = [
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
        hex32("a0930b7ef27cd7b0ae96c9e31dd85b5e0963ced9dfb837da7d535d79e9d05a7b"),
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
        hex32("f017cc3eb4e268c47f52d045607ee0d82d256cc9813805fe9e051469dc2c907d"),
    ),
    (
        "alice-det-math",
        hex32("a2a11b9b826227fa7a9ed8ab2c1ce63dbb07ad4c39aa2c46a363ddea6b47dbe9"),
    ),
    (
        "alice-zip",
        hex32("17d0df1d7c0a6e63d1a7380129f12386b677ac96e7ac193dece546805f4fe84c"),
    ),
];

/// The semantics identifier of this crate: the fold of
/// [`ANALYTICS_SEMANTICS_PINS`] (see the module documentation).
pub const ANALYTICS_SEMANTICS_ID: [u8; 32] =
    hex32("38e4dba4aeda3ea1a291d620c0183470aaffb9c9d3cd9a101b1435b0d93909df");

/// Earlier values of [`ANALYTICS_SEMANTICS_ID`] whose pinned bits this
/// release still reproduces, newest first
///
/// `2def9a11…` is the identifier with alice-det-math 0.5.2 at 39507d1b (`a2e46794…`) and
/// alice-zip at `50a40cf6…`: the move to det-math 2d01c63f (sin / cos at |x| ≥ 2^13, powf of a
/// negative base, powf64, hypot of two subnormals, erf added) changed no function this crate
/// calls (`exp64` / `ln64` / `max64` / `min64`), so every scenario golden is unchanged.
///
/// `f8ded596…` is the identifier with alice-det-math 0.5.1 and alice-zip 0.11 at
/// `51ffeb07…`: the move to det-math 0.5.2 at 39507d1b (whose clamp follows the G2 order)
/// changed no function this crate calls, so every scenario golden is unchanged.
pub const PREVIOUS_ANALYTICS_SEMANTICS_IDS: &[[u8; 32]] = &[
    hex32("2def9a114cb31abd2918f95a57f5ad084917006a9c95810a41db197fffebdfa0"),
    hex32("f8ded5961a8ef8be2d79ea4bfafd68f494c2443d725fb926cf9775824e043d4d"),
];
