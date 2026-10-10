//! float 数学関数: 決定論的な超越関数と、`no_std` 用の丸め / `sqrt` shim
//!
//! # 決定論
//!
//! `ln` / `exp` / `powf` は platform libm の実装差で最終 ulp が OS / CPU / compiler
//! ごとに変わる ⇒ 本 crate は [`alice_det_math`] (IEEE 754 の基本演算のみで構成、
//! 固定の演算順) に委譲する std / `no_std` のどちらでも同じ bit を返す
//! `clippy.toml` の `disallowed-methods` が inherent method 側を禁止して再発を止め、
//! `tests/determinism_golden.rs` が出力の SHA-256 を pin する
//!
//! `+ - * / sqrt` / `mul_add` と `ceil` / `floor` / `round` は IEEE 754 が正確丸めを
//! 要求するので target 非依存 (`mul_add` は融合積和の単一丸めが規定で、FMA 命令が
//! 無い target では正確丸めの software fma になる) `no_std` には inherent method が
//! 無いため [`FloatExt`] で [`libm`] (pure Rust、正確丸め) に委譲する

/// 固定順の整数冪 (2 進法による繰り返し二乗、`n` の bit を下位から)
///
/// `f64::powi` は LLVM が乗算木の結合順を自由に選べるので target 間で bit が
/// 一致しない ⇒ 演算順をここで固定する `|n|` に対して O(log n) 回の乗算なので
/// `alice_det_math::powf64` (≤ 16 ulp) より誤差が小さい (`γ^1535` で実測 9e-14 → 0)
#[inline]
pub(crate) fn ipow64(x: f64, n: i32) -> f64 {
    let mut base = x;
    let mut e = n.unsigned_abs();
    let mut acc = 1.0f64;
    while e > 0 {
        if e & 1 == 1 {
            acc *= base;
        }
        e >>= 1;
        if e > 0 {
            base *= base;
        }
    }
    if n < 0 {
        1.0 / acc
    } else {
        acc
    }
}

// unused in a no_std unit-test build, where libtest links std (see the imports)
#[cfg(not(any(feature = "std", test)))]
pub trait FloatExt: Sized {
    fn mul_add(self, a: Self, b: Self) -> Self;
    fn sqrt(self) -> Self;
    fn ceil(self) -> Self;
    fn floor(self) -> Self;
    fn round(self) -> Self;
}

#[cfg(not(any(feature = "std", test)))]
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
