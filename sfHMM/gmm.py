import numpy as np
from sklearn import mixture

class GMM1(mixture.GaussianMixture):
    """
    GaussianMixture with sorted parameters.
    """    
    def __init__(self, n_components, n_init, random_state):
        super().__init__(n_components=n_components,
                         covariance_type="spherical",
                         n_init=n_init,
                         random_state=random_state)
        self.valid = False
    
    def fit(self, data):
        super().fit(data)
        self.valid = True
        
        # sort all
        order = np.argsort(self.means_.flat)
        self.weights_ = self.weights_[order]
        self.means_ = self.means_[order]
        self.covariances_ = self.covariances_[order].reshape(-1, 1, 1)
        self.precisions_cholesky_ = self.precisions_cholesky_[order]
        self.precisions_ = self.precisions_[order]
        self.sigma_ = np.sqrt(self.covariances_.flat)
        
        return self


def interval_check(mu, thr=None):
    """
    check if peaks are too near to each other.
    """
    if (thr is None or len(mu)==1):
        return False
    elif ((np.diff(mu.flat) < thr).any()):
        return True
    else:
        return False

def sg_check(sg, thr=None):
    """
    check if any standard deviations are too small.
    """
    if (thr is None or len(sg)==1):
        return False
    elif ((sg < thr).any()):
        return True
    else:
        return False


class GMMs:
    """
    Gaussian mixture models with different states.
    The best model can be chosen by comparing AIC or BIC.
    """    
    def __init__(self, data, krange):
        self.data = data
        self.klist = list(range(krange[0], krange[1] + 1))
        self.results = None
    
    def __getitem__(self, key):
        return self.results[key]
    
    def __repr__(self):
        out = "Gaussian Mixture Models"
        line1 = "  n  ||"
        line2 = " AIC ||"
        line3 = " BIC ||"
        for n, aic, bic in zip(self.klist, self.get_aic(), self.get_bic()):
            line1 += f" {n:>7} |"
            line2 += f" {int(aic+0.5):>7} |"
            line3 += f" {int(bic+0.5):>7} |"
        return "\n".join([out, line1, line2, line3])
    
    def fit(self, min_interval=None, min_sg=None, n_init:int=1, random_state:int=0):
        d = np.asarray(self.data).reshape(-1, 1)
        self.results = {k: GMM1(k, n_init, random_state) for k in self.klist}
        
        for gmm1 in self.results.values():
            gmm1.fit(d)
            
            if (interval_check(gmm1.means_, thr=min_interval) or
                sg_check(gmm1.sigma_, thr=min_sg)):
                gmm1.valid = False
                
        return None
    
    def get_optimal(self, criterion="bic", only_valid=True):
        if (criterion == "bic"):
            cri_list = self.get_bic()
        elif (criterion == "aic"):
            cri_list = self.get_aic()
        else:
            raise ValueError("'criterion' must be either 'aic' or 'bic'.")
        
        if (only_valid):
            cri_list[~self.isvalid()] = np.inf
                    
        k_best = self.klist[np.argmin(cri_list)]
        return self[k_best]
    
    def get_aic(self):
        d = np.asarray(self.data).reshape(-1, 1)
        return np.array([self[k].aic(d) for k in self.klist])

    def get_bic(self):
        d = np.asarray(self.data).reshape(-1, 1)
        return np.array([self[k].bic(d) for k in self.klist])

    def isvalid(self):
        return np.array([gmm1.valid for gmm1 in self.results.values()])

class DPGMM:
    def __init__(self, data):
        raise NotImplementedError