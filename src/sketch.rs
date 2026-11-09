//! Probabilistic Sketch Data Structures
//!
//! - HyperLogLog++: Cardinality estimation with ~1.04/√m standard error
//! - `DDSketch`: Relative-error quantile estimation
//! - Count-Min Sketch: Frequency estimation for heavy hitters

// a no_std unit-test build links std (libtest), whose inherent methods win;
// FloatExt only covers correctly rounded operations, so the bits are the same
#[cfg(not(any(feature = "std", test)))]
use crate::math::FloatExt;
use crate::math::{f64_i32, f64_u64, hash_index, i64_f64, ipow64, u64_f64, usize_f64};
use alice_det_math::{exp64, ln64};
use core::hash::{Hash, Hasher};

// ============================================================================
// Lookup Tables for Fast Computation
// ============================================================================

/// Precomputed 2^{-k} values for k = 0..64 (`HyperLogLog` optimization)
/// Eliminates expensive `powi()` calls in cardinality estimation
const POW2_NEG_LUT: [f64; 65] = [
    1.0,                         // 2^-0
    0.5,                         // 2^-1
    0.25,                        // 2^-2
    0.125,                       // 2^-3
    0.0625,                      // 2^-4
    0.031_25,                    // 2^-5
    0.015_625,                   // 2^-6
    0.007_812_5,                 // 2^-7
    0.003_906_25,                // 2^-8
    0.001_953_125,               // 2^-9
    0.000_976_562_5,             // 2^-10
    0.000_488_281_25,            // 2^-11
    0.000_244_140_625,           // 2^-12
    0.000_122_070_312_5,         // 2^-13
    6.103_515_625e-5,            // 2^-14
    3.051_757_812_5e-5,          // 2^-15
    1.525_878_906_25e-5,         // 2^-16
    7.629_394_531_25e-6,         // 2^-17
    3.814_697_265_625e-6,        // 2^-18
    1.907_348_632_812_5e-6,      // 2^-19
    9.536_743_164_062_5e-7,      // 2^-20
    4.768_371_582_031_25e-7,     // 2^-21
    2.384_185_791_015_625e-7,    // 2^-22
    1.192_092_895_507_812_5e-7,  // 2^-23
    5.960_464_477_539_063e-8,    // 2^-24
    2.980_232_238_769_531_2e-8,  // 2^-25
    1.490_116_119_384_765_6e-8,  // 2^-26
    7.450_580_596_923_828e-9,    // 2^-27
    3.725_290_298_461_914e-9,    // 2^-28
    1.862_645_149_230_957e-9,    // 2^-29
    9.313_225_746_154_785e-10,   // 2^-30
    4.656_612_873_077_393e-10,   // 2^-31
    2.328_306_436_538_696_3e-10, // 2^-32
    1.164_153_218_269_348_1e-10, // 2^-33
    5.820_766_091_346_741e-11,   // 2^-34
    2.910_383_045_673_370_4e-11, // 2^-35
    1.455_191_522_836_685_2e-11, // 2^-36
    7.275_957_614_183_426e-12,   // 2^-37
    3.637_978_807_091_713e-12,   // 2^-38
    1.818_989_403_545_856_5e-12, // 2^-39
    9.094_947_017_729_282e-13,   // 2^-40
    4.547_473_508_864_641e-13,   // 2^-41
    2.273_736_754_432_320_6e-13, // 2^-42
    1.136_868_377_216_160_3e-13, // 2^-43
    5.684_341_886_080_802e-14,   // 2^-44
    2.842_170_943_040_401e-14,   // 2^-45
    1.421_085_471_520_200_4e-14, // 2^-46
    7.105_427_357_601_002e-15,   // 2^-47
    3.552_713_678_800_501e-15,   // 2^-48
    1.776_356_839_400_250_5e-15, // 2^-49
    8.881_784_197_001_252e-16,   // 2^-50
    4.440_892_098_500_626e-16,   // 2^-51
    2.220_446_049_250_313e-16,   // 2^-52
    1.110_223_024_625_156_5e-16, // 2^-53
    5.551_115_123_125_783e-17,   // 2^-54
    2.775_557_561_562_891_4e-17, // 2^-55
    1.387_778_780_781_445_7e-17, // 2^-56
    6.938_893_903_907_228e-18,   // 2^-57
    3.469_446_951_953_614e-18,   // 2^-58
    1.734_723_475_976_807e-18,   // 2^-59
    8.673_617_379_884_035e-19,   // 2^-60
    4.336_808_689_942_018e-19,   // 2^-61
    2.168_404_344_971_009e-19,   // 2^-62
    1.084_202_172_485_504_4e-19, // 2^-63
    5.421_010_862_427_522e-20,   // 2^-64
];

/// Reciprocal of 2^52 for fast log2 approximation (avoids division in hot path)
const RCP_POW2_52: f64 = 1.0 / 4_503_599_627_370_496.0; // 1/2^52, precomputed

/// Fast approximate log2 using IEEE 754 bit extraction
/// Returns floor(log2(x)) for positive x, useful for bucket indexing
#[inline]
fn fast_log2_approx(x: f64) -> f64 {
    // IEEE 754 double: sign(1) | exponent(11) | mantissa(52)
    // For positive x: log2(x) ≈ exponent - 1023 + mantissa_fraction
    let bits = x.to_bits();
    // 11 bit の指数なので i64 に収まる (u64 → i64 の wrap は起きない)
    #[allow(clippy::cast_possible_wrap)]
    let exponent = ((bits >> 52) & 0x7FF) as i64;
    let mantissa = bits & 0x000F_FFFF_FFFF_FFFF;

    // exponent - 1023 gives the integer part of log2
    // mantissa * (1/2^52) gives a value in [0, 1) for linear interpolation
    let int_part = exponent - 1023;
    let frac_part = u64_f64(mantissa) * RCP_POW2_52;

    i64_f64(int_part) + frac_part
}

// ============================================================================
// Mergeable Trait - All sketches can be merged across distributed nodes
// ============================================================================

/// Trait for mergeable probabilistic data structures
pub trait Mergeable {
    /// Merge another sketch into this one.
    ///
    /// # Errors
    ///
    /// A [`MergeError`] when the two cannot be merged (a count would pass
    /// `u64::MAX`, or the sketches were made with different parameters); then
    /// `self` is left exactly as it was.
    fn merge(&mut self, other: &Self) -> Result<(), MergeError>;
}

/// Why two sketches were not merged; the merge changed nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeError {
    /// A count would pass `u64::MAX`.
    Overflow,
    /// The two `DDSketch`es have different `alpha` (their bins are different buckets).
    Alpha,
}

// ============================================================================
// Bytes - a fixed-length, versioned form of each sketch
// ============================================================================

/// Why bytes are not a sketch of the type asked for.
///
/// The byte form of every sketch here is: the magic `ALAS`, the format
/// version (`1`), the kind (`1` `HyperLogLog`, `2` `DDSketch`, `3` Count-Min),
/// two parameter bytes, this build's
/// [`ANALYTICS_SEMANTICS_ID`](crate::ANALYTICS_SEMANTICS_ID) (32 bytes), then
/// the sketch's fields in little-endian order (see each type's `encode`). The length is fixed per type (`ENCODED_LEN`), so a
/// store can keep it in a fixed column and merge parts after decoding them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SketchBytesError {
    /// The bytes or the output buffer do not have the type's length.
    Length {
        /// The type's `ENCODED_LEN`.
        expected: usize,
        /// The length given.
        found: usize,
    },
    /// The first four bytes are not `ALAS`.
    Magic,
    /// A format version this crate does not read.
    Version(u8),
    /// Written under another meaning identifier than this build's
    /// [`ANALYTICS_SEMANTICS_ID`](crate::ANALYTICS_SEMANTICS_ID) (an earlier
    /// one included): the hash or the buckets may differ, so the values do not
    /// mean the same thing and are not read.
    Semantics {
        /// The identifier in the bytes.
        found: [u8; 32],
    },
    /// The bytes are another kind of sketch.
    Kind {
        /// The kind of the type asked for.
        expected: u8,
        /// The kind in the bytes.
        found: u8,
    },
    /// The same kind of sketch, but another size (registers, bins, width or depth).
    Shape,
    /// A field no sequence of inserts and merges can produce.
    Invalid(&'static str),
}

const BYTES_MAGIC: [u8; 4] = *b"ALAS";
const BYTES_VERSION: u8 = 1;
const KIND_HLL: u8 = 1;
const KIND_DDSKETCH: u8 = 2;
const KIND_COUNTMIN: u8 = 3;
/// Header length: magic, version, kind, two parameter bytes, the identifier.
const HEADER_LEN: usize = 8 + 32;

fn bytes_header(out: &mut [u8], kind: u8, params: [u8; 2]) {
    out[..4].copy_from_slice(&BYTES_MAGIC);
    out[4] = BYTES_VERSION;
    out[5] = kind;
    out[6] = params[0];
    out[7] = params[1];
    out[8..HEADER_LEN].copy_from_slice(&crate::ANALYTICS_SEMANTICS_ID);
}

/// The checks every decode starts with: length, magic, version, kind. Returns
/// the two parameter bytes.
fn bytes_check(bytes: &[u8], expected_len: usize, kind: u8) -> Result<[u8; 2], SketchBytesError> {
    if bytes.len() < 4 {
        return Err(SketchBytesError::Length {
            expected: expected_len,
            found: bytes.len(),
        });
    }
    if bytes[..4] != BYTES_MAGIC {
        return Err(SketchBytesError::Magic);
    }
    if bytes.len() < HEADER_LEN {
        return Err(SketchBytesError::Length {
            expected: expected_len,
            found: bytes.len(),
        });
    }
    if bytes[4] != BYTES_VERSION {
        return Err(SketchBytesError::Version(bytes[4]));
    }
    if bytes[5] != kind {
        return Err(SketchBytesError::Kind {
            expected: kind,
            found: bytes[5],
        });
    }
    if bytes[8..HEADER_LEN] != crate::ANALYTICS_SEMANTICS_ID {
        let mut found = [0u8; 32];
        found.copy_from_slice(&bytes[8..HEADER_LEN]);
        return Err(SketchBytesError::Semantics { found });
    }
    Ok([bytes[6], bytes[7]])
}

const fn bytes_len_check(found: usize, expected: usize) -> Result<(), SketchBytesError> {
    if found == expected {
        Ok(())
    } else {
        Err(SketchBytesError::Length { expected, found })
    }
}

fn put_u32(out: &mut [u8], at: &mut usize, v: u32) {
    out[*at..*at + 4].copy_from_slice(&v.to_le_bytes());
    *at += 4;
}

fn put_u64(out: &mut [u8], at: &mut usize, v: u64) {
    out[*at..*at + 8].copy_from_slice(&v.to_le_bytes());
    *at += 8;
}

fn get_u32(bytes: &[u8], at: &mut usize) -> u32 {
    let mut b = [0u8; 4];
    b.copy_from_slice(&bytes[*at..*at + 4]);
    *at += 4;
    u32::from_le_bytes(b)
}

fn get_u64(bytes: &[u8], at: &mut usize) -> u64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&bytes[*at..*at + 8]);
    *at += 8;
    u64::from_le_bytes(b)
}

// ============================================================================
// Simple Hash Function (FNV-1a variant for determinism)
// ============================================================================

/// FNV-1a hash for deterministic, fast hashing
#[derive(Clone, Copy, Debug)]
pub struct FnvHasher {
    state: u64,
}

impl FnvHasher {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0100_0000_01b3;

    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            state: Self::FNV_OFFSET,
        }
    }

    /// Avalanche bit mixer (from `MurmurHash3` finalizer)
    /// Ensures all bits are well-distributed for `HyperLogLog`
    #[inline]
    const fn mix(mut h: u64) -> u64 {
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        h ^= h >> 33;
        h
    }

    #[inline]
    #[must_use]
    pub fn hash_bytes(data: &[u8]) -> u64 {
        let mut hasher = Self::new();
        hasher.write(data);
        Self::mix(hasher.state)
    }

    #[inline]
    #[must_use]
    pub fn hash_u64(value: u64) -> u64 {
        Self::hash_bytes(&value.to_le_bytes())
    }

    #[inline]
    #[must_use]
    pub fn hash_u128(value: u128) -> u64 {
        Self::hash_bytes(&value.to_le_bytes())
    }
}

impl Default for FnvHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl Hasher for FnvHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.state ^= u64::from(byte);
            self.state = self.state.wrapping_mul(Self::FNV_PRIME);
        }
    }

    #[inline]
    fn finish(&self) -> u64 {
        // Apply avalanche mixer for better bit distribution
        Self::mix(self.state)
    }
}

// ============================================================================
// HyperLogLog++ - Cardinality Estimation
// ============================================================================

/// Macro to generate `HyperLogLog` implementations for specific sizes
macro_rules! impl_hyperloglog {
    ($name:ident, $p:expr, $m:expr) => {
        /// `HyperLogLog`++ for cardinality (unique count) estimation
        ///
        /// Memory: $m bytes
        /// Error: ~1.04 / sqrt($m)
        #[derive(Clone, Debug)]
        pub struct $name {
            /// Registers storing maximum leading zeros + 1
            registers: [u8; $m],
        }

        impl $name {
            /// Number of registers (m = 2^P)
            pub const M: usize = $m;
            /// Bits used for register index
            pub const P: usize = $p;

            /// Alpha constant for bias correction
            const ALPHA: f64 = 0.7213 / (1.0 + 1.079 / ($m as f64));

            /// Create a new empty `HyperLogLog`
            #[inline]
            pub const fn new() -> Self {
                #[allow(clippy::large_stack_arrays)]
                Self {
                    registers: [0u8; $m],
                }
            }

            /// Length of [`encode`](Self::encode)'s output: the 40 header bytes, then one byte per register.
            pub const ENCODED_LEN: usize = HEADER_LEN + $m;

            /// Write the sketch to `out` (exactly [`ENCODED_LEN`](Self::ENCODED_LEN) bytes):
            /// the header (kind `1`, parameters `P`, `0`), then the `M` registers.
            ///
            /// # Errors
            ///
            /// [`SketchBytesError::Length`] when `out` is not `ENCODED_LEN` long.
            pub fn encode(&self, out: &mut [u8]) -> Result<(), SketchBytesError> {
                bytes_len_check(out.len(), Self::ENCODED_LEN)?;
                #[allow(clippy::cast_possible_truncation)]
                let p = $p as u8;
                bytes_header(out, KIND_HLL, [p, 0]);
                out[HEADER_LEN..].copy_from_slice(&self.registers);
                Ok(())
            }

            /// The sketch written by [`encode`](Self::encode).
            ///
            /// # Errors
            ///
            /// A [`SketchBytesError`] when the bytes are not this type's: wrong
            /// length, magic, version, kind or identifier, another `P`, or a register above
            /// `64 - P + 1` (no insert writes one).
            pub fn decode(bytes: &[u8]) -> Result<Self, SketchBytesError> {
                let params = bytes_check(bytes, Self::ENCODED_LEN, KIND_HLL)?;
                if usize::from(params[0]) != $p || params[1] != 0 {
                    return Err(SketchBytesError::Shape);
                }
                bytes_len_check(bytes.len(), Self::ENCODED_LEN)?;
                let mut out = Self::new();
                out.registers.copy_from_slice(&bytes[HEADER_LEN..]);
                if out.registers.iter().any(|&r| usize::from(r) > 64 - $p + 1) {
                    return Err(SketchBytesError::Invalid("a register above 64 - P + 1"));
                }
                Ok(out)
            }

            /// Insert an already-hashed value
            #[inline]
            pub const fn insert_hash(&mut self, hash: u64) {
                let idx = hash_index(hash) & (Self::M - 1);
                let w = hash >> $p;
                // rho = position of first 1 bit in the (64-P) remaining bits
                // leading_zeros(w) includes the P bits we shifted away, so subtract them
                let rho = if w == 0 {
                    // P ≤ 16 なので 64 - P + 1 ≤ 65 = u8 に収まる (rho は 1..=65)
                    #[allow(clippy::cast_possible_truncation)]
                    let rho = (64 - $p + 1) as u8;
                    rho
                } else {
                    #[allow(clippy::cast_possible_truncation)]
                    let rho = (w.leading_zeros() as usize - $p + 1) as u8;
                    rho
                };
                if rho > self.registers[idx] {
                    self.registers[idx] = rho;
                }
            }

            /// Insert a hashable value
            #[inline]
            pub fn insert<T: Hash>(&mut self, value: &T) {
                let mut hasher = FnvHasher::new();
                value.hash(&mut hasher);
                self.insert_hash(hasher.finish());
            }

            /// Insert raw bytes
            #[inline]
            pub fn insert_bytes(&mut self, bytes: &[u8]) {
                self.insert_hash(FnvHasher::hash_bytes(bytes));
            }

            /// Estimate cardinality using `HyperLogLog`++ algorithm
            /// Optimized with LUT for 2^{-k} values
            pub fn cardinality(&self) -> f64 {
                let mut sum = 0.0f64;
                let mut zeros = 0usize;

                // Use LUT instead of expensive powi() calls
                for &reg in &self.registers {
                    // LUT has 65 entries (0..=64), clamp to be safe
                    let idx = (reg as usize).min(64);
                    sum += POW2_NEG_LUT[idx];
                    if reg == 0 {
                        zeros += 1;
                    }
                }

                let m = usize_f64(Self::M);
                let inv_sum = 1.0 / sum;
                let raw_estimate = Self::ALPHA * m * m * inv_sum;

                if raw_estimate <= 2.5 * m && zeros > 0 {
                    let inv_zeros = 1.0 / usize_f64(zeros);
                    m * ln64(m * inv_zeros)
                } else {
                    raw_estimate
                }
            }

            /// Get raw registers
            #[inline]
            pub const fn registers(&self) -> &[u8] {
                &self.registers
            }

            /// Reset all registers to zero
            #[inline]
            pub const fn clear(&mut self) {
                #[allow(clippy::large_stack_arrays)]
                {
                    self.registers = [0u8; $m];
                }
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl Mergeable for $name {
            fn merge(&mut self, other: &Self) -> Result<(), MergeError> {
                for (dst, &src) in self.registers.iter_mut().zip(other.registers.iter()) {
                    if src > *dst {
                        *dst = src;
                    }
                }
                Ok(())
            }
        }
    };
}

// Generate common HyperLogLog sizes
impl_hyperloglog!(HyperLogLog10, 10, 1024); // 1KB, ~3.2% error
impl_hyperloglog!(HyperLogLog12, 12, 4096); // 4KB, ~1.6% error
impl_hyperloglog!(HyperLogLog14, 14, 16384); // 16KB, ~0.8% error
impl_hyperloglog!(HyperLogLog16, 16, 65536); // 64KB, ~0.4% error

/// Type alias for the most common `HyperLogLog` size (16KB, ~0.8% error)
pub type HyperLogLog = HyperLogLog14;

// ============================================================================
// DDSketch - Relative Error Quantile Estimation
// ============================================================================

/// `DDSketch` for quantile estimation with relative error guarantee
///
/// Guarantees that for any quantile q, the returned value v satisfies:
/// |v - `true_value`| <= α * `true_value`
///
/// where α is the relative accuracy (e.g., 0.01 for 1% error)
///
/// # Example
/// ```
/// use alice_analytics::sketch::DDSketch256;
///
/// let mut sketch = DDSketch256::new(0.01); // 1% relative error
///
/// for latency in [10.0, 20.0, 30.0, 100.0, 500.0] {
///     sketch.insert(latency);
/// }
///
/// let p99 = sketch.quantile(0.99);
/// // p99 ≈ 500.0 (within 1% relative error)
/// ```
/// Macro to generate `DDSketch` implementations for specific bin counts
macro_rules! impl_ddsketch {
    ($name:ident, $bins:expr) => {
        #[derive(Clone, Debug)]
        pub struct $name {
            positive_bins: [u64; $bins],
            negative_bins: [u64; $bins],
            zero_count: u64,
            count: u64,
            non_finite: u64,
            min: f64,
            max: f64,
            sum: crate::exact_sum::ExactSum,
            gamma: f64,
            ln_gamma: f64,
            inv_ln_gamma: f64,
            alpha: f64,
            offset: i32,
        }

        impl $name {
            pub const BINS: usize = $bins;

            pub fn new(alpha: f64) -> Self {
                let gamma = (1.0 + alpha) / (1.0 - alpha);
                let ln_gamma = ln64(gamma);
                // Offset to center around 1.0 (ln(1.0) = 0)
                // For typical latencies (1ms - 10s), we want indices to fit in BINS
                // With offset at BINS/4, we can handle values from gamma^(-BINS/4) to gamma^(3*BINS/4)
                let offset = ($bins / 4) as i32;

                Self {
                    positive_bins: [0u64; $bins],
                    negative_bins: [0u64; $bins],
                    zero_count: 0,
                    count: 0,
                    non_finite: 0,
                    min: f64::INFINITY,
                    max: f64::NEG_INFINITY,
                    sum: crate::exact_sum::ExactSum::ZERO,
                    gamma,
                    ln_gamma,
                    inv_ln_gamma: 1.0 / ln_gamma,
                    alpha,
                    offset,
                }
            }

            /// Length of [`encode`](Self::encode)'s output.
            pub const ENCODED_LEN: usize =
                HEADER_LEN + 4 + 8 * 6 + 8 * crate::exact_sum::LIMBS + 16 * $bins;

            /// Write the sketch to `out` (exactly [`ENCODED_LEN`](Self::ENCODED_LEN) bytes):
            /// the header (kind `2`, parameters `0`, `0`), then (little-endian) the
            /// bin count (`u32`), `alpha`, `count`, `zero_count`, `non_finite`,
            /// `min`, `max` (`f64` as bits, `u64` counts), the exact sum (its
            /// [`LIMBS`](crate::exact_sum::LIMBS) limbs, see
            /// [`ExactSum`](crate::exact_sum::ExactSum)), the positive bins and the
            /// negative bins (`u64` each). `gamma` and the offset are not
            /// stored: [`decode`](Self::decode) derives them from `alpha` as
            /// [`new`](Self::new) does.
            ///
            /// # Errors
            ///
            /// [`SketchBytesError::Length`] when `out` is not `ENCODED_LEN` long.
            pub fn encode(&self, out: &mut [u8]) -> Result<(), SketchBytesError> {
                bytes_len_check(out.len(), Self::ENCODED_LEN)?;
                bytes_header(out, KIND_DDSKETCH, [0, 0]);
                let mut at = HEADER_LEN;
                #[allow(clippy::cast_possible_truncation)]
                put_u32(out, &mut at, $bins as u32);
                put_u64(out, &mut at, self.alpha.to_bits());
                put_u64(out, &mut at, self.count);
                put_u64(out, &mut at, self.zero_count);
                put_u64(out, &mut at, self.non_finite);
                put_u64(out, &mut at, self.min.to_bits());
                put_u64(out, &mut at, self.max.to_bits());
                for &limb in self.sum.limbs() {
                    put_u64(out, &mut at, limb);
                }
                for &b in self.positive_bins.iter().chain(self.negative_bins.iter()) {
                    put_u64(out, &mut at, b);
                }
                Ok(())
            }

            /// The sketch written by [`encode`](Self::encode).
            ///
            /// # Errors
            ///
            /// A [`SketchBytesError`] when the bytes are not this type's: wrong
            /// length, magic, version, kind or identifier, another bin count, an `alpha`
            /// outside `(0, 1)`, or a `count` that is not the zeros plus every bin.
            pub fn decode(bytes: &[u8]) -> Result<Self, SketchBytesError> {
                bytes_check(bytes, Self::ENCODED_LEN, KIND_DDSKETCH)?;
                if bytes.len() < HEADER_LEN + 4 {
                    return Err(SketchBytesError::Length {
                        expected: Self::ENCODED_LEN,
                        found: bytes.len(),
                    });
                }
                let mut at = HEADER_LEN;
                if get_u32(bytes, &mut at) as usize != $bins {
                    return Err(SketchBytesError::Shape);
                }
                bytes_len_check(bytes.len(), Self::ENCODED_LEN)?;
                let alpha = f64::from_bits(get_u64(bytes, &mut at));
                if !(alpha > 0.0 && alpha < 1.0) {
                    return Err(SketchBytesError::Invalid("alpha outside (0, 1)"));
                }
                let mut out = Self::new(alpha);
                out.count = get_u64(bytes, &mut at);
                out.zero_count = get_u64(bytes, &mut at);
                out.non_finite = get_u64(bytes, &mut at);
                out.min = f64::from_bits(get_u64(bytes, &mut at));
                out.max = f64::from_bits(get_u64(bytes, &mut at));
                let mut limbs = [0u64; crate::exact_sum::LIMBS];
                for limb in &mut limbs {
                    *limb = get_u64(bytes, &mut at);
                }
                out.sum = crate::exact_sum::ExactSum::from_limbs(limbs);
                let mut filed = u128::from(out.zero_count);
                for b in out
                    .positive_bins
                    .iter_mut()
                    .chain(out.negative_bins.iter_mut())
                {
                    *b = get_u64(bytes, &mut at);
                    filed += u128::from(*b);
                }
                if filed != u128::from(out.count) {
                    return Err(SketchBytesError::Invalid(
                        "count is not the zeros plus every bin",
                    ));
                }
                Ok(out)
            }

            /// Add one observation.
            ///
            /// Contract for inputs outside the usual range:
            ///
            /// * **a non-finite sample (`NaN`, `+inf`, `-inf`) is not a
            ///   magnitude**: it is counted in
            ///   [`non_finite`](Self::non_finite) and nothing else is touched
            ///   — not [`count`](Self::count), not [`sum`](Self::sum), not
            ///   [`min`](Self::min) / [`max`](Self::max), and no bin. The
            ///   classification follows `law::PointClass::NonFinite`
            ///   (feature `law`), which groups "NaN or infinite" and keeps
            ///   such points out of the summary;
            /// * a **finite** magnitude outside
            ///   [`accurate_range`](Self::accurate_range) is counted and filed
            ///   in the nearest edge bin, so later ranks do not shift; the `α`
            ///   bound does not hold for it;
            /// * `0.0` is counted exactly and needs no bin.
            ///
            /// Until 2026-10-07 a `NaN` was counted in `count`, poisoned `sum`
            /// (and therefore `mean`), and was filed **with the exact zeros**,
            /// because it fails both `value > 0.0` and `value < 0.0`: a stream
            /// of nothing but `NaN` reported `quantile(q) == 0.0`. Counting a
            /// `NaN` sample as a zero sample is a misclassification, not a
            /// rounding question.
            #[inline]
            pub fn insert(&mut self, value: f64) {
                if !value.is_finite() {
                    self.non_finite += 1;
                    return;
                }

                self.count += 1;
                self.sum.add(value);

                self.min = alice_det_math::min64(self.min, value);
                self.max = alice_det_math::max64(self.max, value);

                // Out-of-range magnitudes collapse into the edge bins (the
                // α guarantee holds inside `accurate_range()`);
                // until 2026-09-17 they were counted but stored nowhere, so
                // every quantile above them shifted by one rank per lost value.
                if value > 0.0 {
                    let idx = self.bucket_index(value).min($bins - 1);
                    self.positive_bins[idx] += 1;
                } else if value < 0.0 {
                    let idx = self.bucket_index(-value).min($bins - 1);
                    self.negative_bins[idx] += 1;
                } else {
                    self.zero_count += 1;
                }
            }

            /// Bucket index calculation
            /// Uses standard `ln()` for quantile accuracy (`DDSketch` requires precise buckets)
            ///
            /// `saturating_add` rather than `+`: an extreme **finite**
            /// magnitude reaches this cast with a product large enough to
            /// saturate `as i32` at `i32::MAX`, and adding the offset to that
            /// overflows. A small `alpha` is enough on its own, because
            /// `inv_ln_gamma` grows without bound as `alpha` approaches 0
            /// (`alpha = 0` makes it infinite); non-finite samples no longer
            /// reach here at all, they are counted in `non_finite`.
            /// Before 2026-10-07 the overflow panicked in a
            /// debug build and wrapped to a negative index in a release one,
            /// where `max(0)` then filed the value in bin 0 — the same
            /// rank shift that the 2026-09-17 edge-bin fix removed, in the
            /// opposite direction. Saturating keeps it in the top edge bin,
            /// which is what `accurate_range()` already documents for
            /// magnitudes outside the guaranteed range.
            #[inline]
            fn bucket_index(&self, value: f64) -> usize {
                let idx =
                    f64_i32((ln64(value) * self.inv_ln_gamma).ceil()).saturating_add(self.offset);
                usize::try_from(idx.max(0)).unwrap_or(0)
            }

            /// Fast bucket index using IEEE 754 bit extraction (for non-critical paths)
            /// ~10x faster than `ln()` but has ~1-2% error
            #[inline(always)]
            #[allow(dead_code)]
            fn bucket_index_fast(&self, value: f64) -> usize {
                let log2_gamma = self.ln_gamma / core::f64::consts::LN_2;
                let inv_log2_gamma = 1.0 / log2_gamma;
                let log2_value = fast_log2_approx(value);
                let idx = f64_i32((log2_value * inv_log2_gamma).ceil()).saturating_add(self.offset);
                usize::try_from(idx.max(0)).unwrap_or(0)
            }

            /// Representative value of bin `idx` (which covers
            /// `(γ^(i−1), γ^i]`, `i = idx − offset`): `2γ^i / (γ + 1)`, the point
            /// whose relative distance to both bin edges is exactly α — the
            /// `DDSketch` guarantee (Masson, Rim & Lee 2019).
            ///
            /// History (2026-09-17, oracle `tests/analytic_oracle.rs`): the
            /// **lower bound** γ^(i−1) was returned, so a value near the top of
            /// its bin came back up to γ − 1 = 2α/(1−α) too small (3.1 % at
            /// α = 0.02 on a 6-decade stream) — outside the published bound.
            #[inline]
            fn bucket_representative(&self, idx: usize) -> f64 {
                let exp = i32::try_from(idx).unwrap_or(i32::MAX) - self.offset;
                2.0 * ipow64(self.gamma, exp) / (self.gamma + 1.0)
            }

            pub fn quantile(&self, q: f64) -> f64 {
                if self.count == 0 {
                    return 0.0;
                }
                let rank = f64_u64((q * u64_f64(self.count)).ceil()).max(1);
                // a bin representative may lie outside the observed range;
                // the true order statistic never does
                let clamp = |v: f64| v.clamp(self.min, self.max);
                let mut cumulative = 0u64;
                // negative values ascend from the largest magnitude (most
                // negative) — until 2026-09-17 the bins were walked from the
                // smallest magnitude, so the order of every negative quantile
                // was reversed
                for (idx, &count) in self.negative_bins.iter().enumerate().rev() {
                    cumulative += count;
                    if cumulative >= rank {
                        return clamp(-self.bucket_representative(idx));
                    }
                }
                cumulative += self.zero_count;
                if cumulative >= rank {
                    return 0.0;
                }
                for (idx, &count) in self.positive_bins.iter().enumerate() {
                    cumulative += count;
                    if cumulative >= rank {
                        return clamp(self.bucket_representative(idx));
                    }
                }
                self.max
            }
            /// Samples summarised: finite values only. A non-finite sample
            /// is in [`non_finite`](Self::non_finite) instead, so the ranks
            /// `quantile` computes from this count are the ranks of the
            /// values that actually reached a bin.
            #[inline]
            pub const fn count(&self) -> u64 {
                self.count
            }

            /// Samples rejected as non-finite (`NaN`, `+inf`, `-inf`)
            ///
            /// Named after `law::ResidualSummary::non_finite` (feature `law`),
            /// which counts the same class the same way. A non-zero value
            /// means the stream carried something that is not a magnitude —
            /// an upstream division by zero, a missing sample encoded as
            /// `NaN` — and that those samples are in no other figure here.
            #[inline]
            pub const fn non_finite(&self) -> u64 {
                self.non_finite
            }

            /// Exact sum of the summarised (finite) samples
            ///
            /// Finite in, finite out: a `NaN` sample can no longer poison it.
            #[inline]
            pub fn sum(&self) -> f64 {
                self.sum.value()
            }

            /// Mean of the summarised (finite) samples, `0.0` when there are
            /// none
            #[inline(always)]
            pub fn mean(&self) -> f64 {
                if self.count == 0 {
                    0.0
                } else {
                    self.sum.value() * (1.0 / u64_f64(self.count))
                }
            }

            /// Smallest summarised (finite) sample, `f64::INFINITY` when there
            /// are none
            #[inline]
            pub const fn min(&self) -> f64 {
                self.min
            }

            /// Largest summarised (finite) sample, `f64::NEG_INFINITY` when
            /// there are none
            #[inline]
            pub const fn max(&self) -> f64 {
                self.max
            }

            #[inline]
            pub const fn alpha(&self) -> f64 {
                self.alpha
            }

            /// Magnitudes `[γ^(−offset), γ^(BINS − 1 − offset)]` for which the
            /// relative-error guarantee holds (`offset = BINS / 4`)
            ///
            /// Bin `idx` covers `(γ^(i−1), γ^i]` with `i = idx − offset`; a
            /// magnitude below the first bin or above the last one is counted
            /// in that edge bin, so its quantile keeps the rank but not the
            /// `α` bound. Zero is counted exactly and needs no bin.
            #[must_use]
            pub fn accurate_range(&self) -> (f64, f64) {
                let lowest = -self.offset;
                let highest = i32::try_from($bins - 1).unwrap_or(i32::MAX) - self.offset;
                (ipow64(self.gamma, lowest), ipow64(self.gamma, highest))
            }

            pub const fn clear(&mut self) {
                self.positive_bins = [0u64; $bins];
                self.negative_bins = [0u64; $bins];
                self.zero_count = 0;
                self.count = 0;
                self.non_finite = 0;
                self.min = f64::INFINITY;
                self.max = f64::NEG_INFINITY;
                self.sum = crate::exact_sum::ExactSum::ZERO;
            }
        }

        impl Mergeable for $name {
            /// Bins, counts and the exact sum add; `min` and `max` combine.
            /// Merging in any order gives the same sketch, bit for bit.
            fn merge(&mut self, other: &Self) -> Result<(), MergeError> {
                if self.alpha.to_bits() != other.alpha.to_bits() {
                    return Err(MergeError::Alpha);
                }
                // every count is checked before anything changes
                let fits = |a: u64, b: u64| a.checked_add(b).is_some();
                let ok = fits(self.count, other.count)
                    && fits(self.zero_count, other.zero_count)
                    && fits(self.non_finite, other.non_finite)
                    && self
                        .positive_bins
                        .iter()
                        .zip(other.positive_bins.iter())
                        .all(|(&a, &b)| fits(a, b))
                    && self
                        .negative_bins
                        .iter()
                        .zip(other.negative_bins.iter())
                        .all(|(&a, &b)| fits(a, b));
                if !ok {
                    return Err(MergeError::Overflow);
                }
                for (dst, &src) in self
                    .positive_bins
                    .iter_mut()
                    .zip(other.positive_bins.iter())
                {
                    *dst += src;
                }
                for (dst, &src) in self
                    .negative_bins
                    .iter_mut()
                    .zip(other.negative_bins.iter())
                {
                    *dst += src;
                }
                self.zero_count += other.zero_count;
                self.count += other.count;
                self.non_finite += other.non_finite;
                self.sum.merge(&other.sum);
                self.min = alice_det_math::min64(self.min, other.min);
                self.max = alice_det_math::max64(self.max, other.max);
                Ok(())
            }
        }
    };
}

// Generate common DDSketch sizes
impl_ddsketch!(DDSketch128, 128); // Small, use alpha >= 0.1
impl_ddsketch!(DDSketch256, 256); // Medium, use alpha >= 0.05
impl_ddsketch!(DDSketch512, 512); // Good balance
impl_ddsketch!(DDSketch1024, 1024); // High accuracy, alpha >= 0.02
impl_ddsketch!(DDSketch2048, 2048); // Very high accuracy, alpha >= 0.01

/// Type alias for the most common `DDSketch` size (good for alpha=0.01)
pub type DDSketch = DDSketch2048;

// ============================================================================
// Count-Min Sketch - Frequency Estimation
// ============================================================================

/// Macro to generate `CountMinSketch` implementations
macro_rules! impl_countmin {
    ($name:ident, $w:expr, $d:expr) => {
        /// Count-Min Sketch for frequency estimation
        #[derive(Clone, Debug)]
        pub struct $name {
            counters: [[u64; $w]; $d],
            total: u64,
        }

        impl $name {
            pub const WIDTH: usize = $w;
            pub const DEPTH: usize = $d;

            #[inline]
            pub const fn new() -> Self {
                #[allow(clippy::large_stack_arrays)]
                Self {
                    counters: [[0u64; $w]; $d],
                    total: 0,
                }
            }

            /// Length of [`encode`](Self::encode)'s output.
            pub const ENCODED_LEN: usize = HEADER_LEN + 4 + 4 + 8 + 8 * $w * $d;

            /// Write the sketch to `out` (exactly [`ENCODED_LEN`](Self::ENCODED_LEN) bytes):
            /// the header (kind `3`, parameters `0`, `0`), then (little-endian) the
            /// width and depth (`u32`), `total` (`u64`), and the counters row by
            /// row (`u64` each).
            ///
            /// # Errors
            ///
            /// [`SketchBytesError::Length`] when `out` is not `ENCODED_LEN` long.
            pub fn encode(&self, out: &mut [u8]) -> Result<(), SketchBytesError> {
                bytes_len_check(out.len(), Self::ENCODED_LEN)?;
                bytes_header(out, KIND_COUNTMIN, [0, 0]);
                let mut at = HEADER_LEN;
                #[allow(clippy::cast_possible_truncation)]
                {
                    put_u32(out, &mut at, $w as u32);
                    put_u32(out, &mut at, $d as u32);
                }
                put_u64(out, &mut at, self.total);
                for row in &self.counters {
                    for &c in row {
                        put_u64(out, &mut at, c);
                    }
                }
                Ok(())
            }

            /// The sketch written by [`encode`](Self::encode).
            ///
            /// # Errors
            ///
            /// A [`SketchBytesError`] when the bytes are not this type's: wrong
            /// length, magic, version, kind or identifier, another width or depth,
            /// or a row whose counters do not add up to `total` (every insert and
            /// every merge adds the same count once to each row).
            pub fn decode(bytes: &[u8]) -> Result<Self, SketchBytesError> {
                bytes_check(bytes, Self::ENCODED_LEN, KIND_COUNTMIN)?;
                if bytes.len() < HEADER_LEN + 8 {
                    return Err(SketchBytesError::Length {
                        expected: Self::ENCODED_LEN,
                        found: bytes.len(),
                    });
                }
                let mut at = HEADER_LEN;
                if get_u32(bytes, &mut at) as usize != $w || get_u32(bytes, &mut at) as usize != $d
                {
                    return Err(SketchBytesError::Shape);
                }
                bytes_len_check(bytes.len(), Self::ENCODED_LEN)?;
                let mut out = Self::new();
                out.total = get_u64(bytes, &mut at);
                for row in &mut out.counters {
                    let mut sum = 0u128;

                    for c in row.iter_mut() {
                        *c = get_u64(bytes, &mut at);
                        sum += u128::from(*c);
                    }
                    let total = u128::from(out.total);
                    if sum != total {
                        return Err(SketchBytesError::Invalid("a row does not add up to total"));
                    }
                }
                Ok(out)
            }

            #[inline]
            const fn hash_for_row(hash: u64, row: usize) -> usize {
                let h = hash.wrapping_add((row as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
                let mixed = h ^ (h >> 33);
                let mixed = mixed.wrapping_mul(0xff51_afd7_ed55_8ccd);
                let mixed = mixed ^ (mixed >> 33);
                hash_index(mixed) & ($w - 1)
            }

            #[inline]
            pub fn insert_hash(&mut self, hash: u64, count: u64) {
                self.total += count;
                for row in 0..$d {
                    let col = Self::hash_for_row(hash, row);
                    self.counters[row][col] = self.counters[row][col].saturating_add(count);
                }
            }

            #[inline]
            pub fn insert<T: Hash>(&mut self, item: &T) {
                let mut hasher = FnvHasher::new();
                item.hash(&mut hasher);
                self.insert_hash(hasher.finish(), 1);
            }

            #[inline]
            pub fn insert_bytes(&mut self, bytes: &[u8]) {
                self.insert_hash(FnvHasher::hash_bytes(bytes), 1);
            }

            #[inline]
            pub fn estimate_hash(&self, hash: u64) -> u64 {
                let mut min_count = u64::MAX;
                for row in 0..$d {
                    let col = Self::hash_for_row(hash, row);
                    min_count = min_count.min(self.counters[row][col]);
                }
                min_count
            }

            #[inline]
            pub fn estimate<T: Hash>(&self, item: &T) -> u64 {
                let mut hasher = FnvHasher::new();
                item.hash(&mut hasher);
                self.estimate_hash(hasher.finish())
            }

            #[inline]
            pub fn estimate_bytes(&self, bytes: &[u8]) -> u64 {
                self.estimate_hash(FnvHasher::hash_bytes(bytes))
            }

            #[inline]
            pub const fn total(&self) -> u64 {
                self.total
            }

            #[inline]
            pub const fn clear(&mut self) {
                #[allow(clippy::large_stack_arrays)]
                {
                    self.counters = [[0u64; $w]; $d];
                }
                self.total = 0;
            }

            #[inline(always)]
            pub fn error_bound(&self) -> f64 {
                core::f64::consts::E * (1.0 / f64::from($w))
            }

            #[inline]
            pub fn confidence(&self) -> f64 {
                1.0 - exp64(-f64::from($d))
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl Mergeable for $name {
            /// Totals and counters add; a count that would pass `u64::MAX` is
            /// refused before anything changes.
            fn merge(&mut self, other: &Self) -> Result<(), MergeError> {
                let Some(total) = self.total.checked_add(other.total) else {
                    return Err(MergeError::Overflow);
                };
                // every counter is at most its row's sum, which is total (decode checks
                // it, inserts and merges keep it), so a total that fits keeps every
                // counter in range
                self.total = total;
                for (row, other_row) in self.counters.iter_mut().zip(other.counters.iter()) {
                    for (c, &o) in row.iter_mut().zip(other_row.iter()) {
                        *c += o;
                    }
                }
                Ok(())
            }
        }
    };
}

// Generate common CountMinSketch sizes
impl_countmin!(CountMinSketch1024x5, 1024, 5);
impl_countmin!(CountMinSketch2048x7, 2048, 7);
impl_countmin!(CountMinSketch4096x5, 4096, 5);

/// Type alias for the default Count-Min Sketch
pub type CountMinSketch = CountMinSketch1024x5;

// ============================================================================
// Heavy Hitters (Top-K using Count-Min Sketch + Heap)
// ============================================================================

/// Entry for heavy hitters tracking
#[derive(Clone, Copy, Debug)]
pub struct HeavyHitterEntry {
    /// Hash of the item
    pub hash: u64,
    /// Estimated frequency
    pub count: u64,
}

/// Macro to generate `HeavyHitters` implementations
macro_rules! impl_heavy_hitters {
    ($name:ident, $cms_name:ident, $k:expr) => {
        /// Heavy Hitters tracker using Count-Min Sketch
        #[derive(Clone, Debug)]
        pub struct $name {
            cms: $cms_name,
            top_k: [HeavyHitterEntry; $k],
            count: usize,
        }

        impl $name {
            pub const K: usize = $k;

            #[inline]
            pub const fn new() -> Self {
                Self {
                    cms: $cms_name::new(),
                    top_k: [HeavyHitterEntry { hash: 0, count: 0 }; $k],
                    count: 0,
                }
            }

            pub fn insert_hash(&mut self, hash: u64) {
                self.cms.insert_hash(hash, 1);
                let estimated = self.cms.estimate_hash(hash);

                let mut found_idx = None;
                for i in 0..self.count {
                    if self.top_k[i].hash == hash {
                        found_idx = Some(i);
                        break;
                    }
                }

                if let Some(idx) = found_idx {
                    self.top_k[idx].count = estimated;
                    self.sort_top_k();
                } else if self.count < $k {
                    self.top_k[self.count] = HeavyHitterEntry {
                        hash,
                        count: estimated,
                    };
                    self.count += 1;
                    self.sort_top_k();
                } else if estimated > self.top_k[0].count {
                    self.top_k[0] = HeavyHitterEntry {
                        hash,
                        count: estimated,
                    };
                    self.sort_top_k();
                }
            }

            fn sort_top_k(&mut self) {
                for i in 1..self.count {
                    let entry = self.top_k[i];
                    let mut j = i;
                    while j > 0 && self.top_k[j - 1].count > entry.count {
                        self.top_k[j] = self.top_k[j - 1];
                        j -= 1;
                    }
                    self.top_k[j] = entry;
                }
            }

            pub fn top(&self) -> impl Iterator<Item = &HeavyHitterEntry> {
                self.top_k[..self.count].iter().rev()
            }

            #[inline]
            pub const fn cms(&self) -> &$cms_name {
                &self.cms
            }

            pub const fn clear(&mut self) {
                self.cms.clear();
                self.top_k = [HeavyHitterEntry { hash: 0, count: 0 }; $k];
                self.count = 0;
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}

// Generate HeavyHitters variants
impl_heavy_hitters!(HeavyHitters10, CountMinSketch1024x5, 10);
impl_heavy_hitters!(HeavyHitters20, CountMinSketch2048x7, 20);
impl_heavy_hitters!(HeavyHitters5, CountMinSketch1024x5, 5);

/// Type alias for the default Heavy Hitters tracker
pub type HeavyHitters = HeavyHitters10;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_fnv_hash() {
        let h1 = FnvHasher::hash_u64(12345);
        let h2 = FnvHasher::hash_u64(12345);
        let h3 = FnvHasher::hash_u64(12346);

        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_hyperloglog_basic() {
        // Use HyperLogLog16 (64K registers) for best accuracy
        let mut hll = HyperLogLog16::new();

        // Insert 1000 unique values
        for i in 0..1000u64 {
            hll.insert_hash(FnvHasher::hash_u64(i));
        }

        let estimate = hll.cardinality();
        // Relaxed tolerance due to statistical nature
        assert!(
            estimate > 800.0 && estimate < 1200.0,
            "estimate = {estimate}"
        );
    }

    #[test]
    fn test_hyperloglog_merge() {
        let mut hll1 = HyperLogLog16::new();
        let mut hll2 = HyperLogLog16::new();

        for i in 0..500u64 {
            hll1.insert_hash(FnvHasher::hash_u64(i));
        }
        for i in 500..1000u64 {
            hll2.insert_hash(FnvHasher::hash_u64(i));
        }

        hll1.merge(&hll2).unwrap();
        let estimate = hll1.cardinality();
        assert!(
            estimate > 800.0 && estimate < 1200.0,
            "estimate = {estimate}"
        );
    }

    #[test]
    fn test_hyperloglog_16_large() {
        let mut hll = HyperLogLog16::new();

        // Use explicit hash for consistency with other tests
        for i in 0..100_000_u64 {
            hll.insert_hash(FnvHasher::hash_u64(i));
        }

        let estimate = hll.cardinality();
        // With 100K values in 64K registers, expect reasonable accuracy (±25%)
        assert!(
            estimate > 75000.0 && estimate < 125_000.0,
            "estimate = {estimate}"
        );
    }

    #[test]
    fn test_ddsketch_basic() {
        // Use DDSketch2048 for alpha=0.01 to ensure enough bins
        let mut sketch = DDSketch2048::new(0.01);

        let values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0];
        for v in values {
            sketch.insert(v);
        }

        assert_eq!(sketch.count(), 10);
        assert!((sketch.mean() - 55.0).abs() < 0.001);

        let p50 = sketch.quantile(0.5);
        assert!(p50 > 40.0 && p50 < 70.0, "p50 = {p50}");

        let p99 = sketch.quantile(0.99);
        assert!(p99 > 80.0 && p99 <= 100.0, "p99 = {p99}");
    }

    #[test]
    fn test_countmin_basic() {
        let mut cms = CountMinSketch1024x5::new();

        for _ in 0..100 {
            cms.insert_hash(1, 1);
        }
        for _ in 0..50 {
            cms.insert_hash(2, 1);
        }
        for _ in 0..10 {
            cms.insert_hash(3, 1);
        }

        assert!(cms.estimate_hash(1) >= 100);
        assert!(cms.estimate_hash(2) >= 50);
        assert!(cms.estimate_hash(3) >= 10);
    }

    #[test]
    fn test_countmin_merge() {
        let mut cms1 = CountMinSketch1024x5::new();
        let mut cms2 = CountMinSketch1024x5::new();

        for _ in 0..50 {
            cms1.insert_hash(1, 1);
        }
        for _ in 0..50 {
            cms2.insert_hash(1, 1);
        }

        cms1.merge(&cms2).unwrap();
        assert!(cms1.estimate_hash(1) >= 100);
    }

    // `format!` / `Vec` / the entropy-seeded constructors are `std`-only,
    // so this case cannot run on the `no_std` build
    #[cfg(feature = "std")]
    #[test]
    fn test_heavy_hitters() {
        let mut hh = HeavyHitters5::new();

        for _ in 0..100 {
            hh.insert_hash(1);
        }
        for _ in 0..50 {
            hh.insert_hash(2);
        }
        for _ in 0..30 {
            hh.insert_hash(3);
        }
        for _ in 0..10 {
            hh.insert_hash(4);
        }
        for _ in 0..5 {
            hh.insert_hash(5);
        }

        let top: Vec<_> = hh.top().collect();
        assert_eq!(top.len(), 5);

        assert_eq!(top[0].hash, 1);
        assert_eq!(top[1].hash, 2);
        assert_eq!(top[2].hash, 3);
    }

    #[test]
    fn test_fnv_hash_bytes() {
        let h1 = FnvHasher::hash_bytes(b"hello");
        let h2 = FnvHasher::hash_bytes(b"hello");
        let h3 = FnvHasher::hash_bytes(b"world");
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_fnv_hash_u128() {
        let h1 = FnvHasher::hash_u128(12345u128);
        let h2 = FnvHasher::hash_u128(12345u128);
        let h3 = FnvHasher::hash_u128(12346u128);
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_fnv_hasher_default() {
        let h = FnvHasher::default();
        assert_eq!(h.state, FnvHasher::FNV_OFFSET);
    }

    #[test]
    fn test_fnv_hasher_finish() {
        let mut h = FnvHasher::new();
        h.write(b"test");
        let result = h.finish();
        assert_ne!(result, 0);
    }

    #[test]
    fn test_hyperloglog_empty() {
        let hll = HyperLogLog16::new();
        let estimate = hll.cardinality();
        assert_eq!(estimate, 0.0);
    }

    #[test]
    fn test_hyperloglog_single() {
        let mut hll = HyperLogLog16::new();
        hll.insert_hash(FnvHasher::hash_u64(42));
        let estimate = hll.cardinality();
        assert!(estimate > 0.0 && estimate < 5.0, "estimate = {estimate}");
    }

    #[test]
    fn test_hyperloglog_clear() {
        let mut hll = HyperLogLog16::new();
        for i in 0..100u64 {
            hll.insert_hash(FnvHasher::hash_u64(i));
        }
        assert!(hll.cardinality() > 0.0);
        hll.clear();
        assert_eq!(hll.cardinality(), 0.0);
    }

    #[test]
    fn test_hyperloglog_registers() {
        let hll = HyperLogLog16::new();
        assert_eq!(hll.registers().len(), HyperLogLog16::M);
    }

    #[test]
    fn test_hyperloglog_default() {
        let hll = HyperLogLog16::default();
        assert_eq!(hll.cardinality(), 0.0);
    }

    #[test]
    fn test_hyperloglog_insert_bytes() {
        let mut hll = HyperLogLog16::new();
        hll.insert_bytes(b"test_item");
        assert!(hll.cardinality() > 0.0);
    }

    #[test]
    fn test_hyperloglog_insert_generic() {
        let mut hll = HyperLogLog16::new();
        hll.insert(&42u64);
        hll.insert(&"hello");
        assert!(hll.cardinality() > 0.0);
    }

    #[test]
    fn test_hyperloglog10_accuracy() {
        let mut hll = HyperLogLog10::new();
        for i in 0..1000u64 {
            hll.insert_hash(FnvHasher::hash_u64(i));
        }
        let estimate = hll.cardinality();
        // HLL10 has ~3.2% error, allow ±50%
        assert!(
            estimate > 500.0 && estimate < 1500.0,
            "estimate = {estimate}"
        );
    }

    #[test]
    fn test_hyperloglog12_accuracy() {
        let mut hll = HyperLogLog12::new();
        for i in 0..1000u64 {
            hll.insert_hash(FnvHasher::hash_u64(i));
        }
        let estimate = hll.cardinality();
        assert!(
            estimate > 700.0 && estimate < 1300.0,
            "estimate = {estimate}"
        );
    }

    #[test]
    fn test_ddsketch_empty() {
        let sketch = DDSketch2048::new(0.01);
        assert_eq!(sketch.count(), 0);
        assert_eq!(sketch.quantile(0.5), 0.0);
        assert_eq!(sketch.mean(), 0.0);
    }

    #[test]
    fn test_ddsketch_single() {
        let mut sketch = DDSketch2048::new(0.01);
        sketch.insert(42.0);
        assert_eq!(sketch.count(), 1);
        assert_eq!(sketch.sum(), 42.0);
        assert_eq!(sketch.min(), 42.0);
        assert_eq!(sketch.max(), 42.0);
    }

    #[test]
    fn test_ddsketch_clear() {
        let mut sketch = DDSketch2048::new(0.01);
        for v in [10.0, 20.0, 30.0] {
            sketch.insert(v);
        }
        assert_eq!(sketch.count(), 3);
        sketch.clear();
        assert_eq!(sketch.count(), 0);
        assert_eq!(sketch.sum(), 0.0);
    }

    #[test]
    fn test_ddsketch_alpha() {
        let sketch = DDSketch2048::new(0.05);
        assert!((sketch.alpha() - 0.05).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ddsketch_merge() {
        let mut s1 = DDSketch2048::new(0.01);
        let mut s2 = DDSketch2048::new(0.01);
        for v in [10.0, 20.0, 30.0] {
            s1.insert(v);
        }
        for v in [40.0, 50.0] {
            s2.insert(v);
        }
        s1.merge(&s2).unwrap();
        assert_eq!(s1.count(), 5);
        assert!((s1.sum() - 150.0).abs() < 0.001);
        assert_eq!(s1.min(), 10.0);
        assert_eq!(s1.max(), 50.0);
    }

    #[test]
    fn test_ddsketch128() {
        let mut sketch = DDSketch128::new(0.1);
        sketch.insert(100.0);
        assert_eq!(sketch.count(), 1);
    }

    #[test]
    fn test_ddsketch_negative() {
        let mut sketch = DDSketch2048::new(0.01);
        sketch.insert(-5.0);
        sketch.insert(-10.0);
        sketch.insert(0.0);
        assert_eq!(sketch.count(), 3);
    }

    #[test]
    fn test_countmin_empty() {
        let cms = CountMinSketch1024x5::new();
        assert_eq!(cms.estimate_hash(42), 0);
        assert_eq!(cms.total(), 0);
    }

    #[test]
    fn test_countmin_total() {
        let mut cms = CountMinSketch1024x5::new();
        cms.insert_hash(1, 10);
        cms.insert_hash(2, 20);
        assert_eq!(cms.total(), 30);
    }

    #[test]
    fn test_countmin_clear() {
        let mut cms = CountMinSketch1024x5::new();
        cms.insert_hash(1, 100);
        assert!(cms.estimate_hash(1) >= 100);
        cms.clear();
        assert_eq!(cms.estimate_hash(1), 0);
        assert_eq!(cms.total(), 0);
    }

    #[test]
    fn test_countmin_default() {
        let cms = CountMinSketch1024x5::default();
        assert_eq!(cms.total(), 0);
    }

    #[test]
    fn test_countmin_insert_generic() {
        let mut cms = CountMinSketch1024x5::new();
        cms.insert(&"hello");
        assert!(cms.estimate(&"hello") >= 1);
    }

    #[test]
    fn test_countmin_insert_bytes() {
        let mut cms = CountMinSketch1024x5::new();
        cms.insert_bytes(b"test_key");
        assert!(cms.estimate_bytes(b"test_key") >= 1);
    }

    #[test]
    fn test_countmin_error_bound() {
        let cms = CountMinSketch1024x5::new();
        let error = cms.error_bound();
        assert!(error > 0.0);
    }

    #[test]
    fn test_countmin_confidence() {
        let cms = CountMinSketch1024x5::new();
        let conf = cms.confidence();
        assert!(conf > 0.0 && conf < 1.0);
    }

    #[test]
    fn test_countmin_2048x7() {
        let mut cms = CountMinSketch2048x7::new();
        cms.insert_hash(42, 5);
        assert!(cms.estimate_hash(42) >= 5);
    }

    #[test]
    fn test_countmin_4096x5() {
        let mut cms = CountMinSketch4096x5::new();
        cms.insert_hash(42, 5);
        assert!(cms.estimate_hash(42) >= 5);
    }

    #[test]
    fn test_heavy_hitters_default() {
        let hh = HeavyHitters5::default();
        assert_eq!(hh.top().count(), 0);
    }

    #[test]
    fn test_heavy_hitters_clear() {
        let mut hh = HeavyHitters5::new();
        for _ in 0..10 {
            hh.insert_hash(1);
        }
        assert!(hh.top().count() > 0);
        hh.clear();
        assert_eq!(hh.top().count(), 0);
    }

    #[test]
    fn test_heavy_hitters_cms_access() {
        let mut hh = HeavyHitters5::new();
        hh.insert_hash(42);
        assert!(hh.cms().estimate_hash(42) >= 1);
    }

    #[test]
    fn test_heavy_hitters_10() {
        let mut hh = HeavyHitters10::new();
        for i in 0..10u64 {
            for _ in 0..(10 - i) {
                hh.insert_hash(i);
            }
        }
        assert_eq!(hh.top().count(), 10);
    }

    #[test]
    fn test_heavy_hitters_20() {
        let mut hh = HeavyHitters20::new();
        hh.insert_hash(1);
        assert_eq!(hh.top().count(), 1);
    }

    #[test]
    fn test_fast_log2_approx() {
        let approx = fast_log2_approx(1024.0);
        assert!((approx - 10.0).abs() < 0.1, "approx = {approx}");
    }

    #[test]
    fn test_fast_log2_approx_one() {
        let approx = fast_log2_approx(1.0);
        assert!(approx.abs() < 0.01, "approx = {approx}");
    }

    #[test]
    fn test_hyperloglog14_type_alias() {
        let mut hll = HyperLogLog::new();
        hll.insert_hash(FnvHasher::hash_u64(1));
        assert!(hll.cardinality() > 0.0);
    }

    #[test]
    fn test_ddsketch_type_alias() {
        let mut sketch = DDSketch::new(0.01);
        sketch.insert(42.0);
        assert_eq!(sketch.count(), 1);
    }

    #[test]
    fn test_countmin_type_alias() {
        let mut cms = CountMinSketch::new();
        cms.insert_hash(1, 1);
        assert!(cms.estimate_hash(1) >= 1);
    }

    #[test]
    fn test_heavy_hitters_type_alias() {
        let mut hh = HeavyHitters::new();
        hh.insert_hash(1);
        assert_eq!(hh.top().count(), 1);
    }
}
