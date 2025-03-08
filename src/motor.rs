use pyo3::{prelude::*, Python};
use numpy::{
    ndarray::{Array1, Array2, ArrayView1},
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2
};
use std::f64::INFINITY;

type Dtype = f64;

fn argmax(x: ArrayView1<Dtype>) -> usize {
    let mut x_max = -INFINITY;
    let mut pos = 0;
    for (i, &val) in x.iter().enumerate() {
        if val > x_max {
            x_max = val;
            pos = i;
        }
    }
    pos
}

fn max(x: ArrayView1<Dtype>) -> Dtype {
    x[argmax(x)]
}

fn logsumexp(x: ArrayView1<Dtype>) -> Dtype {
    let x_max = max(x);

    if x_max.is_infinite() {
        return -INFINITY;
    }

    let acc: Dtype = x.iter().map(|&val| (val - x_max).exp()).sum();
    acc.ln() + x_max
}

fn logaddexp(a: Dtype, b: Dtype) -> Dtype {
    if a.is_infinite() && a < 0.0 {
        b
    } else if b.is_infinite() && b < 0.0 {
        a
    } else {
        a.max(b) + (-(a - b).abs()).exp().ln_1p()
    }
}

#[pyfunction]
pub fn forward<'py>(
    py: Python<'py>,
    n_samples: usize,
    n_components: usize,
    log_startprob: PyReadonlyArray1<Dtype>,
    log_transmat_kernel: PyReadonlyArray1<Dtype>,
    framelogprob: PyReadonlyArray2<Dtype>,
    fwdlattice: PyReadonlyArray2<Dtype>,
    max_stride: isize,
) -> PyResult<Py<PyArray2<Dtype>>> {
    let fwdlattice = fwdlattice.as_array();
    let log_transmat_kernel = log_transmat_kernel.as_array();
    let log_startprob = log_startprob.as_array();
    let framelogprob = framelogprob.as_array();
    let mut out = Array2::<Dtype>::zeros(fwdlattice.raw_dim());
    let mut work_buffer = Array1::zeros(log_transmat_kernel.len());

    for i in 0..n_components {
        out[[0, i]] = log_startprob[i] + framelogprob[[0, i]];
    }

    for t in 1..n_samples {
        for j in 0..n_components {
            let j_isize = j as isize;
            for i in (j_isize - max_stride)..=(j_isize + max_stride) {
                let p = (j_isize - i + max_stride) as usize;
                if i >= 0 && i < n_components as isize {
                    let i: usize = i.try_into().unwrap();
                    work_buffer[p] = fwdlattice[[t - 1, i]] + log_transmat_kernel[p];
                } else {
                    work_buffer[p] = -INFINITY;
                }
            }
            out[[t, j]] = logsumexp(work_buffer.view()) + framelogprob[[t, j]];
        }
    }
    Ok(out.into_pyarray_bound(py).unbind())
}

#[pyfunction]
pub fn backward<'py>(
    py: Python<'py>,
    n_samples: usize,
    n_components: usize,
    log_transmat_kernel: PyReadonlyArray1<Dtype>,
    framelogprob: PyReadonlyArray2<Dtype>,
    bwdlattice: PyReadonlyArray2<Dtype>,
    max_stride: isize,
) -> PyResult<Py<PyArray2<Dtype>>> {
    let framelogprob = framelogprob.as_array();
    let log_transmat_kernel = log_transmat_kernel.as_array();
    let bwdlattice = bwdlattice.as_array();
    let mut out = Array2::zeros(bwdlattice.raw_dim());
    let mut work_buffer = Array1::zeros(log_transmat_kernel.len());

    for i in 0..n_components {
        out[[n_samples - 1, i]] = 0.0;
    }

    for t in (0..n_samples - 1).rev() {
        for i in 0..n_components as isize {
            for j in (i - max_stride)..=(i + max_stride) {
                let p = (j - i + max_stride) as usize;
                if j >= 0 && j < n_components as isize {
                    work_buffer[p] =
                        log_transmat_kernel[p]
                        + framelogprob[[t + 1, j as usize]]
                        + bwdlattice[[t + 1, j as usize]];
                } else {
                    work_buffer[p] = -INFINITY;
                }
            }
            out[[t, i as usize]] = logsumexp(work_buffer.view());
        }
    }
    Ok(out.into_pyarray_bound(py).unbind())
}

#[pyfunction]
pub fn compute_log_xi_sum<'py>(
    py: Python<'py>,
    n_samples: usize,
    n_components: usize,
    fwdlattice: PyReadonlyArray2<Dtype>,
    log_transmat_kernel: PyReadonlyArray1<Dtype>,
    bwdlattice: PyReadonlyArray2<Dtype>,
    framelogprob: PyReadonlyArray2<Dtype>,
    log_xi_sum: PyReadonlyArray2<Dtype>,
    max_stride: usize,
) -> PyResult<Py<PyArray2<Dtype>>> {
    let fwdlattice = fwdlattice.as_array();
    let log_transmat_kernel = log_transmat_kernel.as_array();
    let bwdlattice = bwdlattice.as_array();
    let framelogprob = framelogprob.as_array();
    let log_xi_sum = log_xi_sum.as_array();
    let mut out = Array2::zeros(log_xi_sum.raw_dim());
    let mut work_buffer = Array2::from_elem((n_components, n_components), -INFINITY);
    let logprob = logsumexp(fwdlattice.row(n_samples - 1));

    for t in 0..n_samples - 1 {
        for i in 0..n_components {
            for j in (i as isize - max_stride as isize)..=(i as isize + max_stride as isize) {
                let p = (j as isize - i as isize + max_stride as isize) as usize;
                if j >= 0 && j < n_components as isize {
                    work_buffer[[i, j as usize]] =
                        fwdlattice[[t, i]]
                        + log_transmat_kernel[p]
                        + framelogprob[[t + 1, j as usize]]
                        + bwdlattice[[t + 1, j as usize]]
                        - logprob;
                }
            }
        }

        for i in 0..n_components {
            for j in (i as isize - max_stride as isize)..=(i as isize + max_stride as isize) {
                if j >= 0 && j < n_components as isize {
                    out[[i, j as usize]] = logaddexp(
                        log_xi_sum[[i, j as usize]], work_buffer[[i, j as usize]]
                    );
                }
            }
        }
    }
    Ok(out.into_pyarray_bound(py).unbind())
}

#[pyfunction]
pub fn viterbi<'py>(
    py: Python<'py>,
    n_samples: usize,
    n_components: usize,
    log_startprob: PyReadonlyArray1<Dtype>,
    log_transmat_kernel: PyReadonlyArray1<Dtype>,
    framelogprob: PyReadonlyArray2<Dtype>,
    max_stride: usize,
) -> PyResult<(Py<PyArray1<i32>>, Dtype)> {
    let framelogprob = framelogprob.as_array();
    let log_transmat_kernel = log_transmat_kernel.as_array();
    let log_startprob = log_startprob.as_array();

    let mut state_sequence = Array1::zeros(n_samples);
    let mut viterbi_lattice = Array2::zeros((n_samples, n_components));
    let mut work_buffer = Array1::zeros(log_transmat_kernel.len());

    for i in 0..n_components {
        viterbi_lattice[[0, i]] = log_startprob[i] + framelogprob[[0, i]];
    }

    for t in 1..n_samples {
        for i in 0..n_components {
            for j in (i as isize - max_stride as isize)..=(i as isize + max_stride as isize) {
                let p = (j as isize - i as isize + max_stride as isize) as usize;
                if j >= 0 && j < n_components as isize {
                    work_buffer[p] =
                        log_transmat_kernel[p] +
                        viterbi_lattice[[t - 1, j as usize]];
                } else {
                    work_buffer[p] = -INFINITY;
                }
            }
            viterbi_lattice[[t, i]] = max(work_buffer.view()) + framelogprob[[t, i]];
        }
    }

    let mut where_from = argmax(viterbi_lattice.row(n_samples - 1)) as isize;
    state_sequence[n_samples - 1] = where_from as i32;
    let logprob = viterbi_lattice[[n_samples - 1, where_from as usize]];

    for t in (0..n_samples - 1).rev() {
        for i in (where_from - max_stride as isize)..=(where_from + max_stride as isize) {
            let p = (where_from - i + max_stride as isize) as usize;
            if i >= 0 && i < n_components as isize {
                work_buffer[p] =
                    viterbi_lattice[[t, i as usize]]
                    + log_transmat_kernel[p];
            } else {
                work_buffer[p] = -INFINITY;
            }
        }
        where_from = -(argmax(work_buffer.view()) as isize) + max_stride as isize + where_from;
        state_sequence[t] = where_from as i32;
    }

    Ok((state_sequence.into_pyarray_bound(py).unbind(), logprob))
}
