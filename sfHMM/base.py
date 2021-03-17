import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from .func import *
from .gmm import GMMs, DPGMM

class sfHMMBase(GaussianHMM):
    count = 0
    colors = {"raw data": "violet", 
              "step finding": "darkgreen",
              "denoised": "darkblue", 
              "Viterbi pass": "black",
              }
    styles = {"font.size": 16, 
              "lines.linewidth": 1,
              "axes.titlesize": 24,
              "font.family": "serif",
              "font.serif": "Arial",
              "axes.grid": True,
              "axes.labelsize": 16,
              "grid.linewidth": 0.5,
              "legend.frameon": False,
              "boxplot.meanprops.linewidth": 1,          
              }
    
    def __init__(self, sg0:float=-1, psf:float=-1, krange=[1, 6],
                 model:str="g", name:str="", **hmmlearn_params):
        sg0, psf, krange, model = check(sg0, psf, krange, model)
        self.sg0 = sg0
        self.psf = psf
        self.krange = krange
        self.model = model
        params = dict(covariance_type="spherical", init_params="")
        params.update(hmmlearn_params)
        super().__init__(self, **params)
        self.n_features = 1
        self.name = name if name else self._name()
    
    def get_params(self, deep=True):
        out = dict()
        for key in self._get_param_names():
            try:
                value = getattr(self, key)
            except AttributeError:
                value = None
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        out.pop("data_raw", None)
        return out

    def step_finding(self):
        pass
    
    def denoising(self):
        pass
    
    def gmmfit(self, method):
        pass
    
    def hmmfit(self):
        pass
    
    def plot(self):
        pass
    
    def run_all(self, plot:bool=True):
        """
        Conduct all the processes with default settings.

        Parameters
        ----------
        plot : bool, optional
            Specify if show the plot of each process, by default True.
        """
        self.step_finding()
        self.denoising()
        self.gmmfit()
        self.hmmfit()

        plot and self.plot()
        
        return self
    
    def _name(self):
        self.__class__.count += 1
        return f"{self.__class__.__name__}-{self.__class__.count - 1}"

    def _hist(self):
        """
        Draw a histogram that is composed of raw data, denoised data and fitting curve of GMM.
        """
        plt.ylim(self.ylim)
        n_bin = min(int(np.sqrt(self.data_raw.size*1.4)), 100)
        fit_x = np.linspace(self.ylim[0], self.ylim[1], 256)
        fit_y = gauss_mix(fit_x, self.gmm_opt.weights_, self.gmm_opt.means_.flatten(), self.gmm_opt.sigma_)
        peak_x = self.gmm_opt.means_.flatten()
        peak_y = gauss_mix(peak_x, self.gmm_opt.weights_, self.gmm_opt.means_.flatten(), self.gmm_opt.sigma_)
        peak_y += np.max(peak_y) * 0.1
        plt.plot(fit_y, fit_x, color="red", linestyle="-.")
        plt.plot(peak_y, peak_x, "<", color = "red", markerfacecolor='pink', markersize=10)
        plt.hist(self.data_raw, bins=n_bin, color=self.colors["raw data"],
                 orientation="horizontal", alpha=0.7, density=True)
        plt.hist(self.data_fil, bins=n_bin, color=self.colors["denoised"],
                 orientation="horizontal", histtype="step", density=True, lw=2)
        
        return None
    
    def _gmmfit(self, method, edges):
        """
        Fit the denoised data to Gaussian mixture model.
        
        Paramters
        ---------
        method: str, 'aic', 'bic' or 'Dirichlet'
            How to determine the number of states.
        
        Raises
        ------
        ValueError
            If 'method' got an inappropriate string.
        """
        # in case S.D. of noise was very small
        if len(self._sg_list) > 0:
            sg0_ = min(self.sg0, np.percentile(self._sg_list, 25))
        else:
            sg0_ = self.sg0
     
        if method in ("aic", "bic"):
            gmm_ = GMMs(self.data_fil, self.krange, min_interval=sg0_*1.5, min_sg=sg0_*0.8)
            gmm_.fit(edges=edges)
            self.gmm = gmm_
            self.gmm_opt = self.gmm.get_optimal(method)

        elif method == "Dirichlet":
            gmm_ = DPGMM(n_components=self.krange[1], n_init=1, 
                         random_state=0,
                         covariance_prior=sg0_**2)
            gmm_.fit(np.asarray(self.data_fil.reshape(-1,1)))
            self.gmm_opt = gmm_

        else:
            raise ValueError(f"method: {method}")
        
        self.n_components = self.gmm_opt.n_components
        
        return None
    
    def _set_hmm_params(self):
        if not hasattr(self, "n_components"):
            raise AttributeError("'n_components' has yet been specified.")
        
        hasattr(self, "covars_") or self._set_covars()
        hasattr(self, "means_") or self._set_means()
        hasattr(self, "startprob_") or self._set_startprob()
        hasattr(self, "transmat_") or self._set_transmat()
        
        return None
    