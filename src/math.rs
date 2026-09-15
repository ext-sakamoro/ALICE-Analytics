//! `no_std` 用の float 数学関数 shim
//!
//! `std` あり: `f64` の inherent method をそのまま使う (本 module は空)
//! `std` なし (`--no-default-features`): core には float の超越関数が無いため、
//! 同名 method を [`FloatExt`] trait で提供し [`libm`] (pure Rust、`no_std`) に委譲する
//! 各 module は `#[cfg(not(feature = "std"))] use crate::math::FloatExt;` で取り込む
//!
//! 精度注意: `libm` と platform libm は最終 ulp で異なりうる (`HyperLogLog` / `DDSketch` の
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

// ---------------------------------------------------------------------------
// 整数 ↔ float の単一 audit 点 (std / no_std 共通、clippy cast_* lint の集約)
// ---------------------------------------------------------------------------

/// カウンタ / timestamp (`u64`) → `f64` 統計値は 2^53 未満が前提 (ms timestamp は 2^43 程度)
#[inline]
#[allow(clippy::cast_precision_loss)]
pub(crate) const fn u64_f64(n: u64) -> f64 {
    n as f64
}

/// 要素数 / index (`usize`) → `f64` (2^53 未満が前提)
#[inline]
#[allow(clippy::cast_precision_loss)]
pub(crate) const fn usize_f64(n: usize) -> f64 {
    n as f64
}

/// 非負 `f64` (floor / ceil 済の index 位置) → `usize` 負 / NaN は 0 に寄せる
#[inline]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub(crate) fn f64_usize(x: f64) -> usize {
    x.max(0.0) as usize
}

/// hash (`u64`) → bucket index 用 `usize` 呼出側で `& (M - 1)` / `% N` を掛ける前提
/// (32-bit target では上位 bit を捨てる = 下位 bit で mask する用途のみ)
#[inline]
#[allow(clippy::cast_possible_truncation)]
pub(crate) const fn hash_index(h: u64) -> usize {
    h as usize
}

/// `f64` (`round` / `ceil` 済) → `u64` 負 / NaN は 0、2^64 超は saturate (Rust `as` の挙動を明示)
#[inline]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub(crate) fn f64_u64(x: f64) -> u64 {
    x as u64
}

/// `f64` (`round` 済) → `i64` (Rust `as` は saturating、NaN → 0)
#[inline]
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn f64_i64(x: f64) -> i64 {
    x as i64
}

/// `i64` → `f64` (|x| < 2^53 が前提: log2 の指数 / privatize の整数値)
#[inline]
#[allow(clippy::cast_precision_loss)]
pub(crate) const fn i64_f64(x: i64) -> f64 {
    x as f64
}

/// `f64` (`ceil` 済の bucket 位置) → `i32` (`DDSketch` の bucket index、±2^31 内が前提、`as` は saturating)
#[inline]
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn f64_i32(x: f64) -> i32 {
    x as i32
}
