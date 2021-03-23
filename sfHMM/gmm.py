import numpy as np
from sklearn import mixture

class GMM1(mixture.GaussianMixture):
    """
    GaussianMixture with sorted parameters. Also, parameter initialization is
    deterministic because initial centroids are set at regular intervals.
    """    
    def __init__(self, n_components, **kwargs):
        super().__init__(n_components=n_components,
                         **kwargs)
        self.valid = False
    
    def fit(self, data):
        super().fit(data)
        self.valid = True
        
        # sort all
        order = np.argsort(self.means_.flat)
        self.weights_ = self.weights_[order]
        self.means_ = self.means_[order]
        if self.covariance_type == "spherical":
            self.covariances_ = self.covariances_[order]
            self.precisions_cholesky_ = self.precisions_cholesky_[order]
            self.precisions_ = self.precisions_[order]
        self.sigma_ = np.sqrt(self.covariances_.flat)
        
        return self

class GMMs:
    """
    Gaussian mixture models with different states.
    The best model can be chosen by comparing AIC or BIC.
    """    
    def __init__(self, data, krange, covariance_type="spherical",
                 min_interval=None, min_sg=None):
        self.data = data
        self.klist = list(range(krange[0], krange[1] + 1))
        self.results = None
        self.covariance_type = covariance_type
        self.min_interval = min_interval if min_interval else -1
        self.min_sg = min_sg if min_sg else -1
    
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
    
    def fit(self, n_init=1, random_state=0):
        d = np.asarray(self.data).reshape(-1, 1)
            
        self.results = {k: GMM1(k, n_init=n_init, random_state=random_state, 
                                covariance_type=self.covariance_type) 
                        for k in self.klist}
        
        for gmm1 in self.results.values():
            gmm1.fit(d)
            
            if self._interval_check(gmm1.means_) or self._sg_check(gmm1.sigma_):
                gmm1.valid = False
                
        return None
    
    def get_optimal(self, criterion="bic", only_valid=True):
        if criterion == "bic":
            cri_list = self.get_bic()
        elif criterion == "aic":
            cri_list = self.get_aic()
        else:
            raise ValueError("'criterion' must be either 'aic' or 'bic'.")
        
        if only_valid:
            ng = self.isvalid()
            if ng.all():
                pass
            else:
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
    
    def _interval_check(self, mu):
        """
        check if peaks are too near to each other.
        """
        return (np.diff(mu.flat) < self.min_interval).any()

    def _sg_check(self, sg):
        """
        check if any standard deviations are too small.
        """
        return (sg < self.min_sg).any()



class DPGMM(mixture.BayesianGaussianMixture):
    def __init__(self, n_components, n_init, random_state, **kwargs):
        # change default
        kw = dict(weight_concentration_prior=1,
                  weight_concentration_prior_type="dirichlet_distribution",
                  covariance_type="spherical",
                  max_iter=1000
                  )
        kw.update(kwargs)
        
        # different covariance types require different input        
        if "covariance_prior" in kw:
            c = kw["covariance_prior"]
            if kw["covariance_type"] == "tied" and isinstance(c, (int, float)):
                kw["covariance_prior"] = [[c]]
            elif kw["covariance_type"] == "spherical" and isinstance(c, np.ndarray):
                kw["covariance_prior"] = float(c)
        
        super().__init__(n_components=n_components,
                         n_init=n_init,
                         random_state=random_state,
                         **kw)
    
    def fit(self, data):
        super().fit(data)
        
        labels = self.predict(data)
        unique_labels = np.unique(labels)
        self.n_components = len(unique_labels)
        
        # sort all
        self.means_ = self.means_[unique_labels]
        order = np.argsort(self.means_.flat)
        self.weights_ = self.weights_[unique_labels][order]
        self.means_ = self.means_[order]
        if self.covariance_type == "spherical":
            self.covariances_ = self.covariances_[unique_labels][order]
            self.precisions_cholesky_ = self.precisions_cholesky_[unique_labels][order]
            self.precisions_ = self.precisions_[unique_labels][order]
            self.degrees_of_freedom_ = self.degrees_of_freedom_[unique_labels][order]
            
        if self.weight_concentration_prior_type == "dirichlet_process":
            self.weight_concentration_ = (self.weight_concentration_[0][unique_labels][order],
                                        self.weight_concentration_[1][unique_labels][order])
        else:
            self.weight_concentration_ = self.weight_concentration_[unique_labels][order]
        self.mean_precision_ = self.mean_precision_[unique_labels][order]
        self.sigma_ = np.sqrt(self.covariances_.flat)
        
        return self
        