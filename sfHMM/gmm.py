import numpy as np
from sklearn import mixture

__doc__ = \
r"""
GMM clustering based on AIC/BIC minimization, or Dirichlet process.
"""

class GMM1:
    def __init__(self, n_components):
        self.n_components = n_components
        self._init()
    
    def _init(self):
        self.wt = None
        self.mu = None
        self.sg = None
        self.aic = np.inf
        self.bic = np.inf
        return self
    
    def fit(self, data, n_init=1, random_state=0):
        gmm = mixture.GaussianMixture(n_components=self.n_components, covariance_type="spherical",
                                      n_init=n_init, random_state=random_state)
        gmm.fit(data)
        wt_ = gmm.weights_
        mu_ = gmm.means_.flatten()
        sg_ = np.sqrt(gmm.covariances_)
        order = np.argsort(mu_)
        self.wt = wt_[order]
        self.mu = mu_[order]
        self.sg = sg_[order]
        self.aic = gmm.aic(data)
        self.bic = gmm.bic(data)
        return self


def interval_check(mu, thr=None):
    """
    check if peaks are too near to each other.
    """
    if (thr is None or len(mu)==1):
        return False
    elif ((np.diff(np.sort(mu)) < thr).any()):
        return True
    return False

def sg_check(sg, thr=None):
    """
    check if any standard deviations are too small.
    """
    if (thr is None or len(sg)==1):
        return False
    elif ((sg < thr).any()):
        return True
    return False


class GMMn:
    def __init__(self, data, krange):
        self.data = data
        self.klist = list(range(krange[0], krange[1] + 1))
        self.results = None
    
    def __getitem__(self, key):
        return self.results[key]
    
    def fit(self, min_interval=None, min_sg=None, n_init:int=1, random_state:int=0):
        d = np.asarray(self.data).reshape(-1, 1)
        self.results = {k: GMM1(k)._init() for k in self.klist}
        
        for gmm1 in self.results.values():
            gmm1.fit(d, n_init=n_init, random_state=random_state)
            
            if (interval_check(gmm1.mu, thr=min_interval) or
                sg_check(gmm1.sg, thr=min_sg)):
                gmm1.aic = np.inf
                gmm1.bic = np.inf
                
        return None
    
    def get_optimal(self, criterion="bic"):
        if (criterion == "bic"):
            argmin = np.argmin(self.get_bic())
        elif (criterion == "aic"):
            argmin = np.argmin(self.get_aic())
        else:
            raise ValueError("'criterion' must be either 'aic' or 'bic'.")
        k_best = self.klist[argmin]
        return self[k_best]
    
    def get_aic(self):
        return [self[k].aic for k in self.klist]

    def get_bic(self):
        return [self[k].bic for k in self.klist]


class DPGMM:
    def __init__(self, data):
        self.data = data
        
    def fit(self, n_init:int = 1, n_peak:int = 10, random_state=0):
        d = np.asarray(self.data).reshape(-1, 1)
        dpgmm = mixture.BayesianGaussianMixture(n_components=n_peak, covariance_type="spherical",
                    n_init=n_init, random_state=random_state).fit(d)
        clusters = list(set(dpgmm.predict(d)))
        
        wt_ = dpgmm.weights_[clusters]
        mu_ = dpgmm.means_.flatten()[clusters]
        sg_ = np.sqrt(dpgmm.covariances_)[clusters]
        self.n_components = len(wt_)
        order = np.argsort(mu_)
        self.wt = wt_[order]
        self.mu = mu_[order]
        self.sg = sg_[order]
    
        return None