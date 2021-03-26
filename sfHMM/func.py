import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.utils import normalize

def gauss_mix(x, gmm):
    return np.exp(gmm.score_samples(x.reshape(-1,1)))

def gauss(x, wt, mu, sg):
    y = wt * np.exp(-(x - mu)** 2 / (2 * sg * sg)) / np.sqrt(2 * np.pi) / sg
    return y

def concat(list_of_list):
    out = []
    for list_ in list_of_list:
        out += list(list_)
    return out

def plot2(data1, data2=None, ylim=None, legend=True, color1=None, **kwargs):
    """
    dict_ = {"color": ___, "label": ___}
    """
    if (ylim is None):
        ylim = [np.min(data1), np.max(data1)]
    
    plt.xlim(0, len(data1))
    plt.ylim(ylim)
    
    plt.plot(data1, color=color1, label="raw data")
    
    if data2 is not None:
        plt.plot(data2, **kwargs)

    if legend:
        plt.legend(bbox_to_anchor = (1.05, 1), loc = "upper left", borderaxespad = 0,)
    
    return None

def calc_covars(data_raw, states, n_components):
    if n_components <= 1:
        return np.array([np.var(data_raw)])
    
    covars = np.empty(n_components)
    
    for s in range(n_components):
        index = (states == s)
        if np.sum(index) > 0:
            covars[s] = np.var(data_raw[index])
            if covars[s] == 0:
                covars[s] = np.var(data_raw)
        else:
            covars[s] = np.var(data_raw)
    
    return covars

def calc_startprob(d0_list, gmm):
    logprob = gmm._estimate_weighted_log_prob(np.asarray(d0_list).reshape(-1,1))
    prob = np.exp(np.sum(logprob, axis=0))
    return prob / np.sum(prob)

def calc_transmat(states_list, n_components):
    transmat = np.zeros((n_components, n_components), dtype="float64")
    
    for states in states_list:
        for i in range(len(states) - 1):
            transmat[states[i], states[i+1]] += 1.0
    transmat += 1e-12
    normalize(transmat, axis=1)
    return transmat
