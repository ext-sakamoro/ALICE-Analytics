//! PyO3 Python Bindings for ALICE-Analytics
//!
//! Streaming analytics with probabilistic data structures.
//! Jupyter notebook integration with NumPy batch APIs.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};

use crate::anomaly::{EwmaDetector, MadDetector, ZScoreDetector};
use crate::privacy::{LaplaceNoise, RandomizedResponse};
use crate::sketch::{
    CountMinSketch2048x7, DDSketch256, FnvHasher, HyperLogLog14, Mergeable,
};

// ============================================================================
// HyperLogLog (Cardinality Estimation)
// ============================================================================

/// HyperLogLog cardinality estimator (~0.8% error, 16KB).
#[pyclass(name = "HyperLogLog")]
pub struct PyHyperLogLog {
    inner: HyperLogLog14,
}

#[pymethods]
impl PyHyperLogLog {
    #[new]
    fn new() -> Self {
        Self {
            inner: HyperLogLog14::new(),
        }
    }

    /// Insert a pre-hashed u64 value.
    fn insert_hash(&mut self, hash: u64) {
        self.inner.insert_hash(hash);
    }

    /// Insert a string value (hashed internally).
    fn insert_str(&mut self, value: &str) {
        self.inner.insert_hash(FnvHasher::hash_bytes(value.as_bytes()));
    }

    /// Batch insert u64 hashes from NumPy array (GIL released).
    fn insert_batch<'py>(&mut self, _py: Python<'py>, hashes: PyReadonlyArray1<'py, u64>) -> PyResult<()> {
        let slice = hashes.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
        // Cannot release GIL: &mut self
        for &h in slice {
            self.inner.insert_hash(h);
        }
        Ok(())
    }

    /// Estimated number of unique elements.
    fn cardinality(&self) -> f64 {
        self.inner.cardinality()
    }

    /// Merge another HyperLogLog into this one (for distributed counting).
    fn merge(&mut self, other: &PyHyperLogLog) {
        self.inner.merge(&other.inner);
    }

    fn clear(&mut self) {
        self.inner.clear();
    }

    fn __repr__(&self) -> String {
        format!("HyperLogLog(cardinality={:.0})", self.inner.cardinality())
    }
}

// ============================================================================
// DDSketch (Quantile Estimation)
// ============================================================================

/// DDSketch quantile estimator (relative error guarantee).
#[pyclass(name = "DDSketch")]
pub struct PyDDSketch {
    inner: DDSketch256,
}

#[pymethods]
impl PyDDSketch {
    #[new]
    #[pyo3(signature = (alpha=0.02))]
    fn new(alpha: f64) -> Self {
        Self {
            inner: DDSketch256::new(alpha),
        }
    }

    fn insert(&mut self, value: f64) {
        self.inner.insert(value);
    }

    /// Batch insert values from NumPy array.
    fn insert_batch(&mut self, values: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let slice = values.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
        for &v in slice {
            self.inner.insert(v);
        }
        Ok(())
    }

    /// Get quantile value (e.g., 0.50 for median, 0.99 for P99).
    fn quantile(&self, q: f64) -> f64 {
        self.inner.quantile(q)
    }

    fn count(&self) -> u64 {
        self.inner.count()
    }
    fn sum(&self) -> f64 {
        self.inner.sum()
    }
    fn mean(&self) -> f64 {
        self.inner.mean()
    }
    fn min(&self) -> f64 {
        self.inner.min()
    }
    fn max(&self) -> f64 {
        self.inner.max()
    }

    fn merge(&mut self, other: &PyDDSketch) {
        self.inner.merge(&other.inner);
    }

    fn clear(&mut self) {
        self.inner.clear();
    }

    fn __repr__(&self) -> String {
        format!(
            "DDSketch(count={}, p50={:.2}, p99={:.2})",
            self.inner.count(),
            self.inner.quantile(0.50),
            self.inner.quantile(0.99)
        )
    }
}

// ============================================================================
// Count-Min Sketch (Frequency Estimation)
// ============================================================================

/// Count-Min Sketch frequency estimator (never underestimates).
#[pyclass(name = "CountMinSketch")]
pub struct PyCountMinSketch {
    inner: CountMinSketch2048x7,
}

#[pymethods]
impl PyCountMinSketch {
    #[new]
    fn new() -> Self {
        Self {
            inner: CountMinSketch2048x7::new(),
        }
    }

    fn insert_hash(&mut self, hash: u64) {
        self.inner.insert_hash(hash, 1);
    }

    fn insert_str(&mut self, value: &str) {
        self.inner.insert_hash(FnvHasher::hash_bytes(value.as_bytes()), 1);
    }

    /// Batch insert u64 hashes.
    fn insert_batch(&mut self, hashes: PyReadonlyArray1<'_, u64>) -> PyResult<()> {
        let slice = hashes.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
        for &h in slice {
            self.inner.insert_hash(h, 1);
        }
        Ok(())
    }

    fn estimate_hash(&self, hash: u64) -> u64 {
        self.inner.estimate_hash(hash)
    }

    fn estimate_str(&self, value: &str) -> u64 {
        self.inner.estimate_hash(FnvHasher::hash_bytes(value.as_bytes()))
    }

    fn total(&self) -> u64 {
        self.inner.total()
    }

    fn merge(&mut self, other: &PyCountMinSketch) {
        self.inner.merge(&other.inner);
    }

    fn clear(&mut self) {
        self.inner.clear();
    }

    fn __repr__(&self) -> String {
        format!("CountMinSketch(total={})", self.inner.total())
    }
}

// ============================================================================
// MAD Detector (Median Absolute Deviation)
// ============================================================================

/// Streaming anomaly detector using Median Absolute Deviation.
#[pyclass(name = "MadDetector")]
pub struct PyMadDetector {
    inner: MadDetector,
}

#[pymethods]
impl PyMadDetector {
    #[new]
    #[pyo3(signature = (threshold_k=3.0))]
    fn new(threshold_k: f64) -> Self {
        Self {
            inner: MadDetector::new(threshold_k),
        }
    }

    fn observe(&mut self, value: f64) {
        self.inner.observe(value);
    }

    /// Batch observe values (sequential, stateful).
    fn observe_batch(&mut self, values: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let slice = values.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
        for &v in slice {
            self.inner.observe(v);
        }
        Ok(())
    }

    fn is_anomaly(&mut self, value: f64) -> bool {
        self.inner.is_anomaly(value)
    }

    fn anomaly_score(&mut self, value: f64) -> f64 {
        self.inner.anomaly_score(value)
    }

    /// Batch anomaly detection: returns bool array.
    fn is_anomaly_batch<'py>(
        &mut self,
        py: Python<'py>,
        values: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let slice = values.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
        let result: Vec<bool> = slice.iter().map(|&v| self.inner.is_anomaly(v)).collect();
        Ok(result.into_pyarray(py))
    }

    /// Batch anomaly scores.
    fn anomaly_score_batch<'py>(
        &mut self,
        py: Python<'py>,
        values: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let slice = values.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
        let result: Vec<f64> = slice.iter().map(|&v| self.inner.anomaly_score(v)).collect();
        Ok(result.into_pyarray(py))
    }

    fn median(&mut self) -> f64 {
        self.inner.median()
    }
    fn mad(&mut self) -> f64 {
        self.inner.mad()
    }
    fn count(&self) -> usize {
        self.inner.count()
    }

    fn clear(&mut self) {
        self.inner.clear();
    }
}

// ============================================================================
// EWMA Detector
// ============================================================================

/// EWMA-based anomaly detector (statistically optimal for gradual drift).
#[pyclass(name = "EwmaDetector")]
pub struct PyEwmaDetector {
    inner: EwmaDetector,
}

#[pymethods]
impl PyEwmaDetector {
    #[new]
    #[pyo3(signature = (alpha=0.1, threshold_k=3.0))]
    fn new(alpha: f64, threshold_k: f64) -> Self {
        Self {
            inner: EwmaDetector::new(alpha, threshold_k),
        }
    }

    fn observe(&mut self, value: f64) {
        self.inner.observe(value);
    }

    fn observe_batch(&mut self, values: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let slice = values.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
        for &v in slice {
            self.inner.observe(v);
        }
        Ok(())
    }

    fn is_anomaly(&self, value: f64) -> bool {
        self.inner.is_anomaly(value)
    }

    fn anomaly_score(&self, value: f64) -> f64 {
        self.inner.anomaly_score(value)
    }

    /// Batch anomaly check (GIL released — &self methods).
    fn is_anomaly_batch<'py>(
        &self,
        py: Python<'py>,
        values: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let slice = values.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
        let detector = &self.inner;
        let result = py.detach(|| {
            slice.iter().map(|&v| detector.is_anomaly(v)).collect::<Vec<bool>>()
        });
        Ok(result.into_pyarray(py))
    }

    /// Batch anomaly scores (GIL released).
    fn anomaly_score_batch<'py>(
        &self,
        py: Python<'py>,
        values: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let slice = values.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
        let detector = &self.inner;
        let result = py.detach(|| {
            slice.iter().map(|&v| detector.anomaly_score(v)).collect::<Vec<f64>>()
        });
        Ok(result.into_pyarray(py))
    }

    fn ewma(&self) -> f64 {
        self.inner.ewma()
    }
    fn std_dev(&self) -> f64 {
        self.inner.std_dev()
    }
    fn count(&self) -> u64 {
        self.inner.count()
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

// ============================================================================
// Z-Score Detector
// ============================================================================

/// Z-Score based anomaly detector (classical statistical method).
#[pyclass(name = "ZScoreDetector")]
pub struct PyZScoreDetector {
    inner: ZScoreDetector,
}

#[pymethods]
impl PyZScoreDetector {
    #[new]
    #[pyo3(signature = (threshold_k=3.0))]
    fn new(threshold_k: f64) -> Self {
        Self {
            inner: ZScoreDetector::new(threshold_k),
        }
    }

    fn observe(&mut self, value: f64) {
        self.inner.observe(value);
    }

    fn observe_batch(&mut self, values: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let slice = values.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
        for &v in slice {
            self.inner.observe(v);
        }
        Ok(())
    }

    fn is_anomaly(&self, value: f64) -> bool {
        self.inner.is_anomaly(value)
    }

    fn anomaly_score(&self, value: f64) -> f64 {
        self.inner.z_score(value)
    }

    /// Batch anomaly check (GIL released).
    fn is_anomaly_batch<'py>(
        &self,
        py: Python<'py>,
        values: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let slice = values.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
        let detector = &self.inner;
        let result = py.detach(|| {
            slice.iter().map(|&v| detector.is_anomaly(v)).collect::<Vec<bool>>()
        });
        Ok(result.into_pyarray(py))
    }

    fn anomaly_score_batch<'py>(
        &self,
        py: Python<'py>,
        values: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let slice = values.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
        let detector = &self.inner;
        let result = py.detach(|| {
            slice.iter().map(|&v| detector.z_score(v)).collect::<Vec<f64>>()
        });
        Ok(result.into_pyarray(py))
    }

    fn count(&self) -> u64 {
        self.inner.count()
    }
}

// ============================================================================
// Laplace Noise (Differential Privacy)
// ============================================================================

/// Laplace noise generator for differential privacy.
#[pyclass(name = "LaplaceNoise")]
pub struct PyLaplaceNoise {
    inner: LaplaceNoise,
}

#[pymethods]
impl PyLaplaceNoise {
    #[new]
    fn new(sensitivity: f64, epsilon: f64) -> Self {
        Self {
            inner: LaplaceNoise::new(sensitivity, epsilon),
        }
    }

    /// Create with deterministic seed (for reproducibility).
    #[staticmethod]
    fn with_seed(sensitivity: f64, epsilon: f64, seed: u64) -> Self {
        Self {
            inner: LaplaceNoise::with_seed(sensitivity, epsilon, seed),
        }
    }

    fn privatize(&mut self, value: f64) -> f64 {
        self.inner.privatize(value)
    }

    /// Batch privatize values.
    fn privatize_batch<'py>(
        &mut self,
        py: Python<'py>,
        values: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let slice = values.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
        let result: Vec<f64> = slice.iter().map(|&v| self.inner.privatize(v)).collect();
        Ok(result.into_pyarray(py))
    }

    fn scale(&self) -> f64 {
        self.inner.scale()
    }
}

// ============================================================================
// Randomized Response
// ============================================================================

/// Randomized response for binary differential privacy.
#[pyclass(name = "RandomizedResponse")]
pub struct PyRandomizedResponse {
    inner: RandomizedResponse,
}

#[pymethods]
impl PyRandomizedResponse {
    #[new]
    fn new(epsilon: f64) -> Self {
        Self {
            inner: RandomizedResponse::new(epsilon),
        }
    }

    fn privatize(&mut self, value: bool) -> bool {
        self.inner.privatize(value)
    }

    fn p_true(&self) -> f64 {
        self.inner.p_true()
    }

    /// Estimate true proportion from noisy responses.
    #[staticmethod]
    fn estimate_proportion(p_true: f64, n: u64, k: u64) -> f64 {
        RandomizedResponse::estimate_proportion(p_true, n, k)
    }
}

// ============================================================================
// Hash Utility
// ============================================================================

/// FNV-1a hash function.
#[pyfunction]
fn fnv_hash_bytes(data: &[u8]) -> u64 {
    FnvHasher::hash_bytes(data)
}

/// FNV-1a hash for u64.
#[pyfunction]
fn fnv_hash_u64(value: u64) -> u64 {
    FnvHasher::hash_u64(value)
}

/// Batch hash u64 values (GIL released).
#[pyfunction]
fn fnv_hash_batch<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, u64>,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    let slice = values.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = py.detach(|| {
        slice.iter().map(|&v| FnvHasher::hash_u64(v)).collect::<Vec<u64>>()
    });
    Ok(result.into_pyarray(py))
}

// ============================================================================
// Module
// ============================================================================

#[pymodule]
pub fn alice_analytics(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Sketches
    m.add_class::<PyHyperLogLog>()?;
    m.add_class::<PyDDSketch>()?;
    m.add_class::<PyCountMinSketch>()?;

    // Anomaly Detectors
    m.add_class::<PyMadDetector>()?;
    m.add_class::<PyEwmaDetector>()?;
    m.add_class::<PyZScoreDetector>()?;

    // Privacy
    m.add_class::<PyLaplaceNoise>()?;
    m.add_class::<PyRandomizedResponse>()?;

    // Hash
    m.add_function(wrap_pyfunction!(fnv_hash_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(fnv_hash_u64, m)?)?;
    m.add_function(wrap_pyfunction!(fnv_hash_batch, m)?)?;

    Ok(())
}
