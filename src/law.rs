//! Residual distribution of a fitted law (`law` feature)
//!
//! An [`alice_zip::law::SignalLaw`] carries `y = f(x)`, the closed `x`
//! interval its evidence covers and the RMS / largest residual over that
//! evidence. This module summarises how a *set of points* deviates from the
//! law: each residual `y − f(x)` is streamed into a [`DDSketch2048`], and the
//! summary reports its quantiles together with the accuracy the sketch
//! guarantees for them.
//!
//! | reported | how it is obtained |
//! |----------|--------------------|
//! | `count`, `mean`, `min`, `max`, `max_abs` | exact (running sum / extremes) |
//! | `p50`, `p90`, `p99`, [`ResidualSketch::quantile`] | `DDSketch`: the order statistic of rank `⌈q·n⌉`, within `α·\|value\|` |
//! | `accurate_range` | magnitudes for which that bound holds ([`DDSketch2048::accurate_range`] times the scale) |
//! | `outside_accuracy` | non-zero residuals whose magnitude lies outside `accurate_range` |
//! | `out_of_range` | points whose `x` lies outside the law's valid range (or is not finite) |
//! | `non_finite` | points inside the range whose `y − f(x)` is NaN or infinite |
//!
//! Points outside the valid range are counted and **not** summarised: the law
//! is never evaluated there, so nothing is extrapolated. Points with a
//! non-finite residual are counted and not summarised either, so one NaN does
//! not turn the mean into NaN.
//!
//! # Scale
//!
//! The sketch's bins are laid out around magnitude 1 (for `α = 0.01`,
//! `accurate_range` is about `[3.6e-5, 2.0e13]`). Residuals are therefore
//! divided by a power-of-two scale before they are inserted and multiplied
//! back on the way out; a power of two keeps both steps exact. By default the
//! scale is the largest power of two not above the law's own residual RMS
//! ([`alice_zip::law::ResidualStats::rms`]), or 1 when that RMS is 0. Use
//! [`ResidualSketch::with_scale`] to choose it.
//!
//! ```
//! use alice_analytics::law::residual_summary;
//! use alice_zip::law::{Provenance, SignalLaw};
//!
//! // y = 1 + 2x measured at x = 0..=10, with residuals of ±0.5
//! let evidence: Vec<(f64, f64)> = (0..=10)
//!     .map(|i| {
//!         let x = f64::from(i);
//!         (x, 1.0 + 2.0 * x + if i % 2 == 0 { 0.5 } else { -0.5 })
//!     })
//!     .collect();
//! let law = SignalLaw::fit_polynomial(&evidence, 1, Provenance::new("run 1", "least squares"))?;
//!
//! // new points: two inside the valid range [0, 10], one outside
//! let s = residual_summary(&law, &[(2.0, 5.25), (4.0, 9.0), (12.0, 25.0)]);
//! assert_eq!(s.count, 2);
//! assert_eq!(s.out_of_range, 1); // x = 12 is not evaluated
//! let d = s.distribution.expect("two points were summarised");
//! assert!(d.max_abs < 0.5);
//! # Ok::<(), alice_zip::law::LawError>(())
//! ```

extern crate alloc;

use alloc::boxed::Box;

use alice_zip::law::SignalLaw;

use crate::math::u64_f64;
use crate::sketch::DDSketch2048;

/// Relative accuracy used by [`residual_summary`]
pub const DEFAULT_RELATIVE_ACCURACY: f64 = 0.01;

/// Why a [`ResidualSketch`] could not be created
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ResidualError {
    /// The relative accuracy is not a finite number in `(0, 1)`
    InvalidAccuracy,
    /// The scale is not a finite, positive, normal number
    InvalidScale,
}

impl core::fmt::Display for ResidualError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::InvalidAccuracy => "relative accuracy must be finite and in (0, 1)",
            Self::InvalidScale => "scale must be finite, positive and normal",
        })
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ResidualError {}

/// What [`ResidualSketch::push`] did with a point
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PointClass {
    /// `x` is in the valid range and `y − f(x)` is finite; the residual was summarised
    Summarised {
        /// `y − f(x)`
        residual: f64,
    },
    /// `x` is outside the valid range or not finite; the law was not evaluated
    OutOfRange,
    /// `x` is in range but `y − f(x)` is NaN or infinite
    NonFinite,
}

/// Distribution of the summarised residuals (present when at least one was summarised)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidualDistribution {
    /// Mean residual (exact running sum / count)
    pub mean: f64,
    /// Smallest residual (exact)
    pub min: f64,
    /// Largest residual (exact)
    pub max: f64,
    /// Largest `|residual|` (exact)
    pub max_abs: f64,
    /// Median, within the relative accuracy when inside the accurate range
    pub p50: f64,
    /// 90th percentile, same bound
    pub p90: f64,
    /// 99th percentile, same bound
    pub p99: f64,
}

/// Summary of the residuals of a set of points about a law
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidualSummary {
    /// Points summarised (in range, finite residual)
    pub count: u64,
    /// Points not summarised because `x` is outside the valid range or not finite
    pub out_of_range: u64,
    /// Points not summarised because `y − f(x)` is NaN or infinite
    pub non_finite: u64,
    /// `α`: each reported quantile `q̂` of a residual `q` satisfies `|q̂ − q| ≤ α·|q|`
    /// when `|q|` lies in `accurate_range` (and `q̂ = q` when `q = 0`)
    pub relative_accuracy: f64,
    /// Residual magnitudes `[lo, hi]` for which the bound holds
    pub accurate_range: (f64, f64),
    /// Summarised non-zero residuals whose magnitude is outside `accurate_range`
    /// (0 means the bound holds for every reported quantile)
    pub outside_accuracy: u64,
    /// `None` when no point was summarised
    pub distribution: Option<ResidualDistribution>,
}

/// Streams the residuals `y − f(x)` of points about a law into a quantile sketch
///
/// See the [module docs](self) for what is exact and what is approximate.
#[derive(Debug, Clone)]
pub struct ResidualSketch<'a> {
    law: &'a SignalLaw,
    sketch: Box<DDSketch2048>,
    scale: f64,
    /// accurate range of the sketch in scaled units
    range: (f64, f64),
    out_of_range: u64,
    non_finite: u64,
    outside_accuracy: u64,
}

/// Largest power of two `≤ v` for a positive normal `v`, `None` otherwise
fn power_of_two_floor(v: f64) -> Option<f64> {
    if !v.is_normal() || v <= 0.0 {
        return None;
    }
    // keep the exponent, clear the mantissa
    Some(f64::from_bits(v.to_bits() & 0x7ff0_0000_0000_0000))
}

/// Largest power of two not above the law's residual RMS, or 1 when that RMS is 0
fn default_scale(law: &SignalLaw) -> f64 {
    power_of_two_floor(law.residual().rms).unwrap_or(1.0)
}

impl<'a> ResidualSketch<'a> {
    /// A sketch with relative accuracy `alpha`, scaled to the law's own residual RMS
    ///
    /// # Errors
    ///
    /// [`ResidualError::InvalidAccuracy`] unless `0 < alpha < 1`
    pub fn new(law: &'a SignalLaw, alpha: f64) -> Result<Self, ResidualError> {
        Self::build(law, alpha, default_scale(law))
    }

    /// A sketch with relative accuracy `alpha` and an explicit residual scale,
    /// rounded down to a power of two
    ///
    /// # Errors
    ///
    /// [`ResidualError::InvalidAccuracy`] unless `0 < alpha < 1`;
    /// [`ResidualError::InvalidScale`] unless `scale` is finite, positive and normal
    pub fn with_scale(law: &'a SignalLaw, alpha: f64, scale: f64) -> Result<Self, ResidualError> {
        let scale = power_of_two_floor(scale).ok_or(ResidualError::InvalidScale)?;
        Self::build(law, alpha, scale)
    }

    fn build(law: &'a SignalLaw, alpha: f64, scale: f64) -> Result<Self, ResidualError> {
        if !(alpha > 0.0 && alpha < 1.0) {
            return Err(ResidualError::InvalidAccuracy);
        }
        Ok(Self::unchecked(law, alpha, scale))
    }

    /// `alpha` in `(0, 1)` and `scale` a positive normal power of two
    fn unchecked(law: &'a SignalLaw, alpha: f64, scale: f64) -> Self {
        let sketch = Box::new(DDSketch2048::new(alpha));
        let range = sketch.accurate_range();
        Self {
            law,
            sketch,
            scale,
            range,
            out_of_range: 0,
            non_finite: 0,
            outside_accuracy: 0,
        }
    }

    /// Adds one point; returns whether it was summarised
    pub fn push(&mut self, x: f64, y: f64) -> PointClass {
        let Ok(fx) = self.law.evaluate(x) else {
            self.out_of_range += 1;
            return PointClass::OutOfRange;
        };
        let residual = y - fx;
        if !residual.is_finite() {
            self.non_finite += 1;
            return PointClass::NonFinite;
        }
        let scaled = residual / self.scale;
        let m = scaled.abs();
        if m != 0.0 && (m < self.range.0 || m > self.range.1) {
            self.outside_accuracy += 1;
        }
        self.sketch.insert(scaled);
        PointClass::Summarised { residual }
    }

    /// Adds every point
    pub fn extend(&mut self, points: &[(f64, f64)]) {
        for &(x, y) in points {
            self.push(x, y);
        }
    }

    /// Residual at quantile `q ∈ [0, 1]` (rank `⌈q·n⌉`); `None` when nothing was
    /// summarised or `q` is outside `[0, 1]`
    #[must_use]
    pub fn quantile(&self, q: f64) -> Option<f64> {
        if self.sketch.count() == 0 || !(0.0..=1.0).contains(&q) {
            return None;
        }
        Some(self.sketch.quantile(q) * self.scale)
    }

    /// The power-of-two scale residuals are divided by before insertion
    #[must_use]
    pub const fn scale(&self) -> f64 {
        self.scale
    }

    /// Residual magnitudes `[lo, hi]` for which the relative-accuracy bound holds
    #[must_use]
    pub fn accurate_range(&self) -> (f64, f64) {
        (self.range.0 * self.scale, self.range.1 * self.scale)
    }

    /// The summary of everything pushed so far
    #[must_use]
    pub fn summary(&self) -> ResidualSummary {
        let count = self.sketch.count();
        let distribution = (count > 0).then(|| {
            let min = self.sketch.min() * self.scale;
            let max = self.sketch.max() * self.scale;
            let q = |p: f64| self.sketch.quantile(p) * self.scale;
            ResidualDistribution {
                mean: self.sketch.sum() * self.scale / u64_f64(count),
                min,
                max,
                max_abs: min.abs().max(max.abs()),
                p50: q(0.50),
                p90: q(0.90),
                p99: q(0.99),
            }
        });
        ResidualSummary {
            count,
            out_of_range: self.out_of_range,
            non_finite: self.non_finite,
            relative_accuracy: self.sketch.alpha(),
            accurate_range: self.accurate_range(),
            outside_accuracy: self.outside_accuracy,
            distribution,
        }
    }
}

/// Summarises the residuals of `points` about `law` with
/// [`DEFAULT_RELATIVE_ACCURACY`] and the default scale
#[must_use]
pub fn residual_summary(law: &SignalLaw, points: &[(f64, f64)]) -> ResidualSummary {
    // DEFAULT_RELATIVE_ACCURACY lies in (0, 1), so the checked constructor is not needed
    let mut sketch = ResidualSketch::unchecked(law, DEFAULT_RELATIVE_ACCURACY, default_scale(law));
    sketch.extend(points);
    sketch.summary()
}
