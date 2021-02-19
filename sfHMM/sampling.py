from hmmlearn import hmm
import numpy as np

def hmm_sampling(dim=3, num_of_data=500, trs=0.05, sigma=0.5, rand=None, ans=False, scale=1, poi=False):
    """
    Sampline function.py

    Parameters
    ----------
    dim : int, optional
        The number of states, by default 3
    num_of_data : int, optional
        The length of data, by default 500
    trs : float, optional
        Probability of transition, by default 0.05
    sigma : float, optional
        Standard deviation of noise, by default 0.5
    rand : int or None, optional
        Random seed, by default None
    ans : bool, optional
        If the answer of state sequence is returned, by default False
    scale : int, optional
        Interval between mean values, by default 1
    poi : bool, optional
        If Poisson distributed, by default False

    """    
    startprob= np.full(dim, 1.0/dim)
    transmat = np.full((dim,dim), trs/(dim-1)) + np.identity(dim)*(1.0 - trs/(dim-1) - trs)
    means = np.arange(1, dim+1).reshape(-1,1)*scale
    covars = np.full(dim, sigma*sigma)*scale*scale
    
    model = hmm.GaussianHMM(n_components=dim, covariance_type="spherical")
    model.startprob_ = startprob
    model.transmat_ = transmat
    model.means_ = means
    model.covars_ = covars
    
    data, states = model.sample(num_of_data,random_state=rand)
    
    answer = model.means_[states, 0]

    if (poi):
        np.random.seed(rand)
        data_1 = np.random.poisson(lam=answer)
        np.random.seed()
    else:
        data_1 = data.flatten()

    if (ans):
        return data_1, answer
    else:
        return data_1