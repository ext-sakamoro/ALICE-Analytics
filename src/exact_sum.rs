//! The exact sum of finite doubles, independent of the order they are added in.
//!
//! Every finite double is an integer multiple of `2^-1074` (the smallest
//! subnormal), so the sum of any number of them is an integer `N` times
//! `2^-1074`. [`ExactSum`] keeps `N` as a two's-complement integer of
//! [`LIMBS`] 64-bit limbs. Adding a double, or another sum, is integer
//! addition: exact, associative and commutative, so a sum built in any order,
//! or from parts merged in any order, has the same limbs. [`ExactSum::value`]
//! rounds `N * 2^-1074` to the nearest double (ties to even), the correctly
//! rounded sum; beyond the largest double it is `±inf`.
//!
//! Size: one double is below `2^1024`, that is `N < 2^2098`; `2^64` of them
//! stay below `2^2162`; with the sign bit that is 2163 bits, so 34 limbs
//! (2176 bits) hold any sum of up to `2^64` finite doubles without wrapping.

/// Limbs of [`ExactSum`]: 34 × 64 = 2176 bits.
pub const LIMBS: usize = 34;

/// An exact sum of finite doubles (see the module documentation).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExactSum {
    /// `N`, little-endian limbs, two's complement.
    limbs: [u64; LIMBS],
}

impl Default for ExactSum {
    fn default() -> Self {
        Self::ZERO
    }
}

impl ExactSum {
    /// The empty sum.
    pub const ZERO: Self = Self { limbs: [0; LIMBS] };

    /// The sum with these limbs (as written by [`limbs`](Self::limbs)).
    #[must_use]
    pub const fn from_limbs(limbs: [u64; LIMBS]) -> Self {
        Self { limbs }
    }

    /// `N`, little-endian, two's complement.
    #[must_use]
    pub const fn limbs(&self) -> &[u64; LIMBS] {
        &self.limbs
    }

    /// Add a finite double exactly. A non-finite value is not a number to add;
    /// it leaves the sum as it is (the caller counts it on its own).
    pub fn add(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        let bits = value.to_bits();
        let negative = bits >> 63 == 1;
        let field = (bits >> 52) & 0x7ff;
        let fraction = bits & ((1u64 << 52) - 1);
        // value = mantissa * 2^(shift - 1074)
        let (mantissa, shift) = if field == 0 {
            (fraction, 0)
        } else {
            (fraction | (1u64 << 52), field - 1)
        };
        if mantissa == 0 {
            return;
        }
        // shift < 2046, so the limb index is below 32 and the high part below 33
        #[allow(clippy::cast_possible_truncation)]
        let (limb, bit) = ((shift / 64) as usize, (shift % 64) as u32);
        let lo = mantissa << bit;
        let hi = if bit == 0 { 0 } else { mantissa >> (64 - bit) };
        let mut term = [0u64; LIMBS];
        term[limb] = lo;
        term[limb + 1] = hi;
        if negative {
            negate(&mut term);
        }
        add_into(&mut self.limbs, &term);
    }

    /// Add another sum.
    pub fn merge(&mut self, other: &Self) {
        add_into(&mut self.limbs, &other.limbs);
    }

    /// `N * 2^-1074` rounded to the nearest double, ties to even; `±inf` past
    /// the largest double; `+0.0` for the empty sum and for a sum that cancels.
    #[must_use]
    pub fn value(&self) -> f64 {
        let negative = self.limbs[LIMBS - 1] >> 63 == 1;
        let mut n = self.limbs;
        if negative {
            negate(&mut n);
        }
        let Some(top) = highest_bit(&n) else {
            return 0.0;
        };
        let magnitude = if top <= 52 {
            // N < 2^53: N * 2^-1074 is a double whose bits are N itself
            // (a subnormal below 2^52, the smallest binade at 2^52)
            n[0]
        } else {
            let drop = top - 52;
            let mut q = bits_at(&n, drop);
            let half = bit(&n, drop - 1);
            let sticky = (0..drop - 1).any(|i| bit(&n, i));
            let mut exponent = drop + 1;
            if half && (sticky || q & 1 == 1) {
                q += 1;
                if q == 1u64 << 53 {
                    q >>= 1;
                    exponent += 1;
                }
            }
            if exponent >= 0x7ff {
                return if negative {
                    f64::NEG_INFINITY
                } else {
                    f64::INFINITY
                };
            }
            ((exponent as u64) << 52) | (q - (1u64 << 52))
        };
        let v = f64::from_bits(magnitude);
        if negative {
            -v
        } else {
            v
        }
    }
}

fn add_into(acc: &mut [u64; LIMBS], term: &[u64; LIMBS]) {
    let mut carry = false;
    for (a, &t) in acc.iter_mut().zip(term.iter()) {
        let (s1, c1) = a.overflowing_add(t);
        let (s2, c2) = s1.overflowing_add(u64::from(carry));
        *a = s2;
        carry = c1 || c2;
    }
}

fn negate(x: &mut [u64; LIMBS]) {
    let mut carry = true;
    for limb in x.iter_mut() {
        let (s, c) = (!*limb).overflowing_add(u64::from(carry));
        *limb = s;
        carry = c;
    }
}

fn highest_bit(x: &[u64; LIMBS]) -> Option<usize> {
    x.iter()
        .enumerate()
        .rev()
        .find(|(_, &l)| l != 0)
        .map(|(i, &l)| i * 64 + 63 - l.leading_zeros() as usize)
}

fn bit(x: &[u64; LIMBS], i: usize) -> bool {
    x[i / 64] >> (i % 64) & 1 == 1
}

/// The 53 bits of `x` starting at bit `from` (the top one is set by the caller's choice).
fn bits_at(x: &[u64; LIMBS], from: usize) -> u64 {
    let (limb, off) = (from / 64, from % 64);
    let lo = x[limb] >> off;
    let hi = if off == 0 || limb + 1 >= LIMBS {
        0
    } else {
        x[limb + 1] << (64 - off)
    };
    (lo | hi) & ((1u64 << 53) - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sum(vs: &[f64]) -> f64 {
        let mut s = ExactSum::ZERO;
        for &v in vs {
            s.add(v);
        }
        s.value()
    }

    // oracle: sums of doubles whose exact value is known by hand
    #[test]
    fn exact_sums() {
        assert_eq!(sum(&[]).to_bits(), f64::to_bits(0.0));
        assert_eq!(sum(&[1e16, 1.0, -1e16]).to_bits(), f64::to_bits(1.0));
        assert_eq!(sum(&[1e308, 1e308, -1e308]).to_bits(), f64::to_bits(1e308));
        assert_eq!(sum(&[1e308, 1e308]).to_bits(), f64::to_bits(f64::INFINITY));
        assert_eq!(
            sum(&[-1e308, -1e308]).to_bits(),
            (f64::NEG_INFINITY).to_bits()
        );
        assert_eq!(
            sum(&[f64::MIN_POSITIVE / 4.0, f64::MIN_POSITIVE / 4.0]).to_bits(),
            f64::to_bits(f64::MIN_POSITIVE / 2.0)
        );
        assert_eq!(sum(&[5e-324, 5e-324, 5e-324]).to_bits(), 3);
        assert_eq!(sum(&[-2.5, 1.25]).to_bits(), f64::to_bits(-1.25));
        assert_eq!(sum(&[f64::MAX, -f64::MAX]).to_bits(), f64::to_bits(0.0));
        assert_eq!(sum(&[f64::MAX]).to_bits(), f64::to_bits(f64::MAX));
        // 2^53 + 1 is a tie between 2^53 and 2^53 + 2: to even (2^53)
        assert_eq!(
            sum(&[9_007_199_254_740_992.0, 1.0]).to_bits(),
            f64::to_bits(9_007_199_254_740_992.0)
        );
        // 2^53 + 3 ties between 2^53 + 2 and 2^53 + 4: to even (2^53 + 4)
        assert_eq!(
            sum(&[9_007_199_254_740_992.0, 3.0]).to_bits(),
            f64::to_bits(9_007_199_254_740_996.0)
        );
        // just above the tie goes up
        assert_eq!(
            sum(&[9_007_199_254_740_992.0, 1.0, 1e-300]).to_bits(),
            f64::to_bits(9_007_199_254_740_994.0)
        );
        assert_eq!(sum(&[0.1, 0.2]).to_bits(), 0x3FD3_3333_3333_3334);
    }

    #[test]
    fn order_and_merge_do_not_change_the_limbs() {
        let vs = [
            3.5,
            -1e200,
            7e-310,
            1e200,
            0.1,
            -0.3,
            f64::from_bits(0x7830_0000_0000_0000), // 2^900
            -f64::from_bits(0x7830_0000_0000_0000),
        ];
        let mut fwd = ExactSum::ZERO;
        for &v in &vs {
            fwd.add(v);
        }
        let mut rev = ExactSum::ZERO;
        for &v in vs.iter().rev() {
            rev.add(v);
        }
        assert_eq!(fwd, rev);
        let (mut a, mut b) = (ExactSum::ZERO, ExactSum::ZERO);
        for &v in &vs[..3] {
            a.add(v);
        }
        for &v in &vs[3..] {
            b.add(v);
        }
        b.merge(&a);
        assert_eq!(b, fwd);
    }

    #[test]
    fn non_finite_values_are_not_added() {
        assert_eq!(
            sum(&[1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY]).to_bits(),
            f64::to_bits(1.0)
        );
    }
}
