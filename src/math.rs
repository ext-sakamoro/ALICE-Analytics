//! `no_std` 用の float 数学関数 shim
//!
//! `std` あり: `f64` の inherent method をそのまま使う (本 module は空)
//! `std` なし (`--no-default-features`): core には float の超越関数が無いため、
//! 同名 method を [`FloatExt`] trait で提供し [`libm`] (pure Rust、`no_std`) に委譲する
//! 各 module は `#[cfg(not(feature = "std"))] use crate::math::FloatExt;` で取り込む
//!
//! 精度注意: `libm` と platform libm は最終 ulp で異なりうる (HyperLogLog / DDSketch の
//! 推定値が std build と `no_std` build で完全一致することは保証しない)

#[cfg(not(feature = "std"))]
pub trait FloatExt: Sized {
    fn mul_add(self, a: Self, b: Self) -> Self;
    fn sqrt(self) -> Self;
    fn ln(self) -> Self;
    fn exp(self) -> Self;
    fn powf(self, n: Self) -> Self;
    fn ceil(self) -> Self;
    fn floor(self) -> Self;
    fn round(self) -> Self;
}

#[cfg(not(feature = "std"))]
impl FloatExt for f64 {
    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        libm::fma(self, a, b)
    }
    #[inline]
    fn sqrt(self) -> Self {
        libm::sqrt(self)
    }
    #[inline]
    fn ln(self) -> Self {
        libm::log(self)
    }
    #[inline]
    fn exp(self) -> Self {
        libm::exp(self)
    }
    #[inline]
    fn powf(self, n: Self) -> Self {
        libm::pow(self, n)
    }
    #[inline]
    fn ceil(self) -> Self {
        libm::ceil(self)
    }
    #[inline]
    fn floor(self) -> Self {
        libm::floor(self)
    }
    #[inline]
    fn round(self) -> Self {
        libm::round(self)
    }
}
