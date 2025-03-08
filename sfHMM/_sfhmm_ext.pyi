from numpy import ndarray

def forward(
    n_samples: int,
    n_components: int,
    log_startprob: ndarray,
    log_transmat_kernel: ndarray,
    framelogprob: ndarray,
    max_stride: int,
) -> ndarray: ...
def backward(
    n_samples: int,
    n_components: int,
    log_transmat_kernel: ndarray,
    framelogprob: ndarray,
    max_stride: int,
) -> ndarray: ...
def compute_log_xi_sum(
    n_samples: int,
    n_components: int,
    fwdlattice: ndarray,
    log_transmat_kernel: ndarray,
    bwdlattice: ndarray,
    framelogprob: ndarray,
    max_stride: int,
) -> ndarray: ...
def viterbi(
    n_samples: int,
    n_components: int,
    log_startprob: ndarray,
    log_transmat_kernel: ndarray,
    framelogprob: ndarray,
    max_stride: int,
) -> tuple[ndarray, float]: ...
