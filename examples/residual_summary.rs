//! Residual distribution of a fitted law
//!
//! Fits `y = 1 + 2x` to noisy evidence on `x ∈ [0, 10]`, then summarises how a
//! second batch of points deviates from it. Points outside `[0, 10]` and points
//! with a non-finite `y` are counted, not summarised.
//!
//! `cargo run --example residual_summary --features law`

use alice_analytics::law::{residual_summary, ResidualSketch};
use alice_zip::law::{LawError, Provenance, SignalLaw};

/// deterministic noise in [-0.5, 0.5)
fn noise(i: u32) -> f64 {
    let h = i.wrapping_mul(2_654_435_761) >> 8;
    f64::from(h) / f64::from(1u32 << 24) - 0.5
}

fn main() -> Result<(), LawError> {
    let evidence: Vec<(f64, f64)> = (0..=100)
        .map(|i| {
            let x = f64::from(i) * 0.1;
            (x, 2.0f64.mul_add(x, 1.0) + 0.1 * noise(i))
        })
        .collect();
    let law = SignalLaw::fit_polynomial(
        &evidence,
        1,
        Provenance::new("example run", "least squares"),
    )?;
    let fit = law.residual();
    println!(
        "law: degree {}, valid x in [{}, {}]",
        law.degree(),
        law.domain().lo,
        law.domain().hi
    );
    println!(
        "fit residual over {} points: rms {:.4e}, max |r| {:.4e}",
        fit.n, fit.rms, fit.max_abs
    );

    // a second batch: twice the noise, plus points the law cannot judge
    let mut batch: Vec<(f64, f64)> = (0..1000)
        .map(|i| {
            let x = f64::from(i) * 0.01;
            (x, 2.0f64.mul_add(x, 1.0) + 0.2 * noise(i + 1000))
        })
        .collect();
    batch.push((12.0, 25.0)); // outside the valid range: not extrapolated
    batch.push((5.0, f64::NAN)); // non-finite y

    let s = residual_summary(&law, &batch);
    println!(
        "summarised {} points, {} outside the valid range, {} non-finite",
        s.count, s.out_of_range, s.non_finite
    );
    println!(
        "quantiles within {} x |value| for |residual| in [{:.3e}, {:.3e}] ({} outside)",
        s.relative_accuracy, s.accurate_range.0, s.accurate_range.1, s.outside_accuracy
    );
    if let Some(d) = s.distribution {
        println!(
            "mean {:+.4e}  min {:+.4e}  max {:+.4e}  max |r| {:.4e}",
            d.mean, d.min, d.max, d.max_abs
        );
        println!("p50 {:+.4e}  p90 {:+.4e}  p99 {:+.4e}", d.p50, d.p90, d.p99);
    }

    // the streaming form: push points one by one, read any quantile
    let mut sketch = ResidualSketch::new(&law, 0.005).expect("0.005 lies in (0, 1)");
    for &(x, y) in &batch {
        sketch.push(x, y);
    }
    if let Some(p999) = sketch.quantile(0.999) {
        println!(
            "p99.9 at alpha 0.005: {p999:+.4e} (scale {:e})",
            sketch.scale()
        );
    }
    Ok(())
}
