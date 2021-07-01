import numpy as np
from scipy.special import softmax
from scipy.stats import entropy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from hmmlearn.utils import normalize
from functools import wraps
from warnings import warn

class sfHMMAnalysisError(Exception):
    pass

def append_log(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if len(self._log)>0 and self._log[-1][0] == func.__name__:
            self._log[-1][1] += 1
            self._log[-1][2] = "Failed"
        else:
            self._log.append([func.__name__, 1, "Failed"])
        out = func(self, *args, **kwargs)
        self._log[-1][2] = "Passed"
        return out
    return wrapper

def under_development(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        msg = f"Method `{func.__name__}` is under development and its behavior may change in the future."
        warn(msg, FutureWarning)
        return func(self, *args, **kwargs)
    return wrapper
    
def gauss_mix(x, gmm):
    return np.exp(gmm.score_samples(x.reshape(-1, 1)))

def concat(list_of_list):
    out = []
    for list_ in list_of_list:
        out += list(list_)    
    return out

def plot2(data1, data2=None, ylim=None, legend=True, color1=None, **kwargs):
    """
    dict_ = {"color": ___, "label": ___}
    """
    if ylim is None:
        ylim = [np.min(data1), np.max(data1)]
    
    plt.xlim(0, len(data1))
    plt.ylim(ylim)
    
    plt.plot(data1, color=color1, label="raw data")
    
    if data2 is not None:
        plt.plot(data2, **kwargs)

    if legend:
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
    
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
    # Here, using `scipy.special.softmax` is very important because logprob always
    # composed of very large and very small values. This situation causes underflow
    # or overflow.
    logprob = gmm._estimate_weighted_log_prob(np.asarray(d0_list).reshape(-1,1))
    sumlogprob = np.sum(logprob, axis=0)
    return softmax(sumlogprob)

def calc_transmat(states_list, n_components):
    transmat = np.zeros((n_components, n_components), dtype=np.float64)
    
    for states in states_list:
        for i in range(len(states) - 1):
            transmat[states[i], states[i+1]] += 1.0
    transmat += 1e-12
    normalize(transmat, axis=1)
    return transmat

def normalized_mutual_information(a1, a2, bins, range):
    # (H(X) + H(Y)) / H(X,Y)
    a1, _ = np.histogram(a1, bins=bins, range=range, density=True)
    a2, _ = np.histogram(a2, bins=bins, range=range, density=True)
    if np.isnan(a2).any():
        return 0
    hist, _ = np.histogramdd([a1, a2], bins=bins)
    hist /= np.sum(hist)
    h1 = entropy(np.sum(hist, axis=0))
    h2 = entropy(np.sum(hist, axis=1))
    h12 = entropy(np.ravel(hist))
    
    return (h1 + h2) / h12

def optimize_ax(d1, d2, bins=None, range=None, bounds=None):
    def calc_nmi(a, d1, d2):
        return -normalized_mutual_information(d1, a*d2, bins, range)
    
    result = minimize(calc_nmi, 1, args=(d1, d2), method="Powell", bounds=bounds)
    return result.x, 0

def optimize_b(d1, d2, bins=None, range=None, bounds=None):
    def calc_nmi(b, d1, d2):
        return -normalized_mutual_information(d1, d2+b, bins, range)
    
    result = minimize(calc_nmi, 0, args=(d1, d2), method="Powell", bounds=bounds)
    return 1, result.x

