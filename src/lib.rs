use pyo3::prelude::*;
pub mod motor;

/// A Python module implemented in Rust.
#[pymodule]
fn _sfhmm_ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(motor::forward, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(motor::backward, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(motor::compute_log_xi_sum, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(motor::viterbi, m)?)?;
    Ok(())
}
