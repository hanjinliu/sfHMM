from __future__ import annotations

from hmmlearn import hmm
import numpy as np

__all__ = ["hmm_sampling", "motor_sampling"]

def hmm_sampling(
    dim: int = 3,
    n_data: int = 500,
    trs: float = 0.05,
    sigma: float = 0.5,
    rand: int | None = None, 
    ans: bool = False,
    scale: float = 1,
    poi: bool = False,
):
    """
    Sampline function.py

    Parameters
    ----------
    dim : int, default is 3
        The number of states.
    n_data : int, default is 500
        The length of data.
    trs : float, default is 0.05
        Probability of transition.
    sigma : float, default is 0.5
        Standard deviation of noise.
    rand : int or None, optional
        Random seed.
    ans : bool, default is False
        If the answer of state sequence is returned.
    scale : int, default is 1
        Interval between mean values.
    poi : bool, default is False
        If Poisson distributed.

    """    
    startprob= np.full(dim, 1.0/dim)
    transmat = (
        np.full((dim, dim), trs / (dim - 1))
        + np.identity(dim) * (1.0 - trs / (dim - 1) - trs)
    )
    means = np.arange(1, dim+1).reshape(-1,1)*scale
    covars = np.full(dim, sigma*sigma)*scale*scale
    
    model = hmm.GaussianHMM(n_components=dim, covariance_type="spherical")
    model.startprob_ = startprob
    model.transmat_ = transmat
    model.means_ = means
    model.covars_ = covars
    
    data, states = model.sample(n_data,random_state=rand)
    
    answer = model.means_[states, 0]

    if poi:
        np.random.seed(rand)
        data_1 = np.random.poisson(lam=answer)
        np.random.seed()
    else:
        data_1 = data.flatten()

    if ans:
        return data_1, answer
    else:
        return data_1

def motor_sampling(
    pdf=[0.005, -1, 0.015],
    sigma: float = 0.5,
    n_data: int = 500,
    rand: int | None = None,
    ans: bool = False,
):
    """
    pdf: probability distribution
        [... , 2_steps_backward, 1_step_backward, stay, 1_step_forward, 2_steps_forward,
        ...]. one of pdf can be -1
    """
    pdf = np.array(pdf)
    if np.any(pdf == -1):
        rem = 1-np.sum(pdf[pdf>=0])
        pdf[pdf==-1] = rem
    
    if np.sum(pdf) != 1:
        raise ValueError("sum of pdf must be 1")
    
    np.random.seed(seed=rand)
    
    rands = np.random.rand(n_data)
    cum_pdf = np.cumsum(pdf).reshape(-1,1)
    assign_mat, _ = np.meshgrid(rands, np.arange(len(pdf))) > cum_pdf
    
    delta_y = np.sum(assign_mat.astype("int"), axis=0) - (len(pdf) - 1) / 2
    y_m = np.cumsum(delta_y)
    y = y_m + np.random.normal(scale=sigma, size=n_data)
    np.random.seed()
    if ans:
        return y, y_m    
    else:
        return y
