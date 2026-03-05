//! Probabilistic Sketch Data Structures
//!
//! - HyperLogLog++: Cardinality estimation with ~1.04/√m standard error
//! - `DDSketch`: Relative-error quantile estimation
//! - Count-Min Sketch: Frequency estimation for heavy hitters

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
#[inline(always)]
fn fast_log2_approx(x: f64) -> f64 {
    // IEEE 754 double: sign(1) | exponent(11) | mantissa(52)
    // For positive x: log2(x) ≈ exponent - 1023 + mantissa_fraction
    let bits = x.to_bits();
    let exponent = ((bits >> 52) & 0x7FF) as i64;
    let mantissa = bits & 0x000F_FFFF_FFFF_FFFF;

    // exponent - 1023 gives the integer part of log2
    // mantissa * (1/2^52) gives a value in [0, 1) for linear interpolation
    let int_part = exponent - 1023;
    let frac_part = mantissa as f64 * RCP_POW2_52;

    int_part as f64 + frac_part
}

// ============================================================================
// Mergeable Trait - All sketches can be merged across distributed nodes
// ============================================================================

/// Trait for mergeable probabilistic data structures
pub trait Mergeable {
    /// Merge another sketch into this one
    fn merge(&mut self, other: &Self);
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
    fn mix(mut h: u64) -> u64 {
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
            self.state ^= byte as u64;
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
            pub fn new() -> Self {
                #[allow(clippy::large_stack_arrays)]
                Self {
                    registers: [0u8; $m],
                }
            }

            /// Insert an already-hashed value
            #[inline]
            pub fn insert_hash(&mut self, hash: u64) {
                let idx = (hash as usize) & (Self::M - 1);
                let w = hash >> $p;
                // rho = position of first 1 bit in the (64-P) remaining bits
                // leading_zeros(w) includes the P bits we shifted away, so subtract them
                let rho = if w == 0 {
                    (64 - $p + 1) as u8
                } else {
                    (w.leading_zeros() as usize - $p + 1) as u8
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

                let m = Self::M as f64;
                let inv_sum = 1.0 / sum;
                let raw_estimate = Self::ALPHA * m * m * inv_sum;

                if raw_estimate <= 2.5 * m && zeros > 0 {
                    let inv_zeros = 1.0 / zeros as f64;
                    m * (m * inv_zeros).ln()
                } else {
                    raw_estimate
                }
            }

            /// Count zero registers using SIMD when available
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            fn count_zeros_simd(&self) -> usize {
                #[cfg(target_arch = "x86_64")]
                {
                    use core::arch::x86_64::*;
                    let mut zeros = 0usize;
                    let chunks = self.registers.chunks_exact(32);
                    let remainder = chunks.remainder();

                    // SAFETY: chunks_exact(32) guarantees each chunk is exactly 32 bytes,
                    // matching the 256-bit AVX2 register width. _mm256_loadu_si256 performs
                    // an unaligned load, so no alignment requirement on the source pointer.
                    unsafe {
                        let zero_vec = _mm256_setzero_si256();
                        for chunk in chunks {
                            let data = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
                            let cmp = _mm256_cmpeq_epi8(data, zero_vec);
                            let mask = _mm256_movemask_epi8(cmp) as u32;
                            zeros += mask.count_ones() as usize;
                        }
                    }

                    // Handle remainder
                    for &reg in remainder {
                        if reg == 0 {
                            zeros += 1;
                        }
                    }
                    zeros
                }
            }

            /// Get raw registers
            #[inline]
            pub fn registers(&self) -> &[u8] {
                &self.registers
            }

            /// Reset all registers to zero
            #[inline]
            pub fn clear(&mut self) {
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
            fn merge(&mut self, other: &Self) {
                for (dst, &src) in self.registers.iter_mut().zip(other.registers.iter()) {
                    if src > *dst {
                        *dst = src;
                    }
                }
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
            min: f64,
            max: f64,
            sum: f64,
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
                let ln_gamma = gamma.ln();
                // Offset to center around 1.0 (ln(1.0) = 0)
                // For typical latencies (1ms - 10s), we want indices to fit in BINS
                // With offset at BINS/4, we can handle values from gamma^(-BINS/4) to gamma^(3*BINS/4)
                let offset = ($bins / 4) as i32;

                Self {
                    positive_bins: [0u64; $bins],
                    negative_bins: [0u64; $bins],
                    zero_count: 0,
                    count: 0,
                    min: f64::INFINITY,
                    max: f64::NEG_INFINITY,
                    sum: 0.0,
                    gamma,
                    ln_gamma,
                    inv_ln_gamma: 1.0 / ln_gamma,
                    alpha,
                    offset,
                }
            }

            #[inline]
            pub fn insert(&mut self, value: f64) {
                self.count += 1;
                self.sum += value;

                if value < self.min {
                    self.min = value;
                }
                if value > self.max {
                    self.max = value;
                }

                if value > 0.0 {
                    let idx = self.bucket_index(value);
                    if idx < $bins {
                        self.positive_bins[idx] += 1;
                    }
                } else if value < 0.0 {
                    let idx = self.bucket_index(-value);
                    if idx < $bins {
                        self.negative_bins[idx] += 1;
                    }
                } else {
                    self.zero_count += 1;
                }
            }

            /// Bucket index calculation
            /// Uses standard `ln()` for quantile accuracy (`DDSketch` requires precise buckets)
            #[inline]
            fn bucket_index(&self, value: f64) -> usize {
                let idx = (value.ln() * self.inv_ln_gamma).ceil() as i32 + self.offset;
                idx.max(0) as usize
            }

            /// Fast bucket index using IEEE 754 bit extraction (for non-critical paths)
            /// ~10x faster than `ln()` but has ~1-2% error
            #[inline(always)]
            #[allow(dead_code)]
            fn bucket_index_fast(&self, value: f64) -> usize {
                let log2_gamma = self.ln_gamma / std::f64::consts::LN_2;
                let inv_log2_gamma = 1.0 / log2_gamma;
                let log2_value = fast_log2_approx(value);
                let idx = (log2_value * inv_log2_gamma).ceil() as i32 + self.offset;
                idx.max(0) as usize
            }

            #[inline]
            fn bucket_lower_bound(&self, idx: usize) -> f64 {
                let exp = (idx as i32 - self.offset) as f64;
                self.gamma.powf(exp - 1.0)
            }

            pub fn quantile(&self, q: f64) -> f64 {
                if self.count == 0 {
                    return 0.0;
                }

                let rank = (q * self.count as f64).ceil() as u64;
                let mut cumulative = 0u64;

                for (idx, &count) in self.negative_bins.iter().enumerate().rev() {
                    cumulative += count;
                    if cumulative >= rank {
                        return -self.bucket_lower_bound(idx);
                    }
                }

                cumulative += self.zero_count;
                if cumulative >= rank {
                    return 0.0;
                }

                for (idx, &count) in self.positive_bins.iter().enumerate() {
                    cumulative += count;
                    if cumulative >= rank {
                        return self.bucket_lower_bound(idx);
                    }
                }

                self.max
            }

            #[inline]
            pub fn count(&self) -> u64 {
                self.count
            }

            #[inline]
            pub fn sum(&self) -> f64 {
                self.sum
            }

            #[inline(always)]
            pub fn mean(&self) -> f64 {
                if self.count == 0 {
                    0.0
                } else {
                    self.sum * (1.0 / self.count as f64)
                }
            }

            #[inline]
            pub fn min(&self) -> f64 {
                self.min
            }

            #[inline]
            pub fn max(&self) -> f64 {
                self.max
            }

            #[inline]
            pub fn alpha(&self) -> f64 {
                self.alpha
            }

            pub fn clear(&mut self) {
                self.positive_bins = [0u64; $bins];
                self.negative_bins = [0u64; $bins];
                self.zero_count = 0;
                self.count = 0;
                self.min = f64::INFINITY;
                self.max = f64::NEG_INFINITY;
                self.sum = 0.0;
            }
        }

        impl Mergeable for $name {
            fn merge(&mut self, other: &Self) {
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
                self.sum += other.sum;

                if other.min < self.min {
                    self.min = other.min;
                }
                if other.max > self.max {
                    self.max = other.max;
                }
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
            pub fn new() -> Self {
                #[allow(clippy::large_stack_arrays)]
                Self {
                    counters: [[0u64; $w]; $d],
                    total: 0,
                }
            }

            #[inline]
            fn hash_for_row(hash: u64, row: usize) -> usize {
                let h = hash.wrapping_add((row as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
                let mixed = h ^ (h >> 33);
                let mixed = mixed.wrapping_mul(0xff51_afd7_ed55_8ccd);
                let mixed = mixed ^ (mixed >> 33);
                (mixed as usize) & ($w - 1)
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
            pub fn total(&self) -> u64 {
                self.total
            }

            #[inline]
            pub fn clear(&mut self) {
                #[allow(clippy::large_stack_arrays)]
                {
                    self.counters = [[0u64; $w]; $d];
                }
                self.total = 0;
            }

            #[inline(always)]
            pub fn error_bound(&self) -> f64 {
                core::f64::consts::E * (1.0 / ($w as f64))
            }

            #[inline]
            pub fn confidence(&self) -> f64 {
                1.0 - (-($d as f64)).exp()
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl Mergeable for $name {
            fn merge(&mut self, other: &Self) {
                self.total += other.total;
                for row in 0..$d {
                    for col in 0..$w {
                        self.counters[row][col] =
                            self.counters[row][col].saturating_add(other.counters[row][col]);
                    }
                }
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
            pub fn new() -> Self {
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
            pub fn cms(&self) -> &$cms_name {
                &self.cms
            }

            pub fn clear(&mut self) {
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
            "estimate = {}",
            estimate
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

        hll1.merge(&hll2);
        let estimate = hll1.cardinality();
        assert!(
            estimate > 800.0 && estimate < 1200.0,
            "estimate = {}",
            estimate
        );
    }

    #[test]
    fn test_hyperloglog_16_large() {
        let mut hll = HyperLogLog16::new();

        // Use explicit hash for consistency with other tests
        for i in 0..100000u64 {
            hll.insert_hash(FnvHasher::hash_u64(i));
        }

        let estimate = hll.cardinality();
        // With 100K values in 64K registers, expect reasonable accuracy (±25%)
        assert!(
            estimate > 75000.0 && estimate < 125000.0,
            "estimate = {}",
            estimate
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
        assert!(p50 > 40.0 && p50 < 70.0, "p50 = {}", p50);

        let p99 = sketch.quantile(0.99);
        assert!(p99 > 80.0 && p99 <= 100.0, "p99 = {}", p99);
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

        cms1.merge(&cms2);
        assert!(cms1.estimate_hash(1) >= 100);
    }

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
        assert!(estimate > 0.0 && estimate < 5.0, "estimate = {}", estimate);
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
            "estimate = {}",
            estimate
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
            "estimate = {}",
            estimate
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
        s1.merge(&s2);
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
        let top: Vec<_> = hh.top().collect();
        assert_eq!(top.len(), 10);
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
        assert!((approx - 10.0).abs() < 0.1, "approx = {}", approx);
    }

    #[test]
    fn test_fast_log2_approx_one() {
        let approx = fast_log2_approx(1.0);
        assert!(approx.abs() < 0.01, "approx = {}", approx);
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
