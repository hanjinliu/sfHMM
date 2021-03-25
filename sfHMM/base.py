import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from .func import *
from .gmm import GMMs, DPGMM
from .step import GaussStep, PoissonStep

class sfHMMBase(GaussianHMM):
    count = 0
    colors = {"raw data": "violet", 
              "step finding": "darkgreen",
              "denoised": "darkblue", 
              "Viterbi path": "black",
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
    
    def __init__(self, sg0:float=-1, psf:float=-1, krange=(1, 6),
                 model:str="g", name:str="", **hmmlearn_params):
        self.sg0 = sg0
        self.psf = psf
        self.krange = krange
        self.model = model
        params = dict(covariance_type="spherical", init_params="")
        params.update(hmmlearn_params)
        super().__init__(self, **params)
        self.n_features = 1
        self.name = name if name else self._name()
        
    @property
    def krange(self):
        return self._krange
    
    @krange.setter
    def krange(self, value):
        if isinstance(value, int):
            value = (value, value)
        else:
            kmin, kmax = value
            value = (kmin, kmax)
        self._krange = value
    
    @property
    def model(self):
        return self._model
    
    @model.setter
    def model(self, s):
        if s.lower() in ("g", "gauss"):
            self._model = "Gauss"
            self.StepClass = GaussStep
        elif s.lower() in ("p", "poisson"):
            self._model = "Poisson"
            self.StepClass = PoissonStep
        else:
            raise ValueError(f"Invalid model identifier: {s}")
    
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

    def step_finding(self): ...
    def denoising(self): ...
    def gmmfit(self, method): ...
    def hmmfit(self): ...
    def plot(self): ...
    
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
    
    def accumulate_transitions(self):
        """
        This function returns all the transitions occurred in the trajectory.
        """
    
    def _name(self):
        self.__class__.count += 1
        return f"{self.__class__.__name__}-{self.__class__.count - 1}"

    def _hist(self, sl=None, ylim=None):
        """
        Draw a histogram that is composed of raw data, denoised data and fitting curve of GMM.
        """
        if sl is None:
            sl = slice(None)
            ylim = self.ylim
            amp = 1
        elif isinstance(sl, slice):
            amp = self.data_raw.size / (sl.stop - sl.start)
        else:
            raise TypeError("sl must be a slice object.")
        plt.ylim(ylim)
        n_bin = min(int(np.sqrt(self.data_raw[sl].size*2)), 256)
        fit_x = np.linspace(self.ylim[0], self.ylim[1], 256)
        fit_y = gauss_mix(fit_x, self.gmm_opt) * amp
        peak_x = self.gmm_opt.means_.ravel()
        peak_y = gauss_mix(peak_x, self.gmm_opt) * amp
        peak_y += np.max(peak_y) * 0.1
        plt.plot(fit_y, fit_x, color="red", linestyle="-.")
        plt.plot(peak_y, peak_x, "<", color = "red", markerfacecolor='pink', markersize=10)
        plt.hist(self.data_raw[sl], bins=n_bin, color=self.colors["raw data"],
                 orientation="horizontal", alpha=0.7, density=True)
        plt.hist(self.data_fil[sl], bins=n_bin, color=self.colors["denoised"],
                 orientation="horizontal", histtype="step", density=True, lw=2)
        
        return None
    
    def _init_sg0(self, p:float=25):
        """
        Initialize 'sg0' if it is negative.
        """        
        if self.sg0 < 0:
            l = np.abs(self._accumulate_step_sizes())
            if len(l) > 0:
                self.sg0 = np.percentile(l, p) * 0.2
            else:
                self.sg0 = np.std(self.data_raw)
        
        return None
    
    def _gmmfit(self, method:str, n_init:int, random_state:int):
        # in case S.D. of noise was very small
        if len(self._sg_list) > 0:
            sg0_ = min(self.sg0, np.percentile(self._sg_list, 25))
        else:
            sg0_ = self.sg0
            
        if method.lower() in ("aic", "bic"):
            gmm_ = GMMs(self.data_fil, self.krange, min_interval=sg0_*1.5,  min_sg=sg0_*0.8, 
                        covariance_type=self.covariance_type)
            gmm_.fit(n_init=n_init, random_state=random_state)
            self.gmm = gmm_
            self.gmm_opt = self.gmm.get_optimal(method)
        elif method.lower() == "dirichlet":
            gmm_ = DPGMM(n_components=self.krange[1], n_init=1, 
                         random_state=random_state,
                         mean_precision_prior=1/np.var(self.data_raw),
                         covariance_prior=sg0_**2,
                         covariance_type=self.covariance_type)
            gmm_.fit(np.asarray(self.data_fil).reshape(-1,1))
            self.gmm_opt = gmm_
        else:
            raise ValueError(f"method: {method}")
        
        self.n_components = self.gmm_opt.n_components
        
        return None
    
    def tdp(self, **kwargs):
        """
        Pseudo transition density plot.
        """
        means = self.means_.ravel()
        cov = self.covars_.ravel()
        
        axlim = (np.min(means) - np.sqrt(cov.max()),
                 np.max(means) + np.sqrt(cov.max()))
                    
        tr = self.accumulate_transitions()
        
        axes = np.linspace(*axlim, 200)
        x, y = np.meshgrid(axes, axes)
        z = np.zeros((200, 200))

        for sx, sy in tr:
            mx = self.means_[sx]
            my = self.means_[sy]
            z += np.exp(-((x - mx)**2/(2*cov[sx]) + (y - my)**2/(2*cov[sy])))
        
        z /= np.max(z)
        
        kw = {"vmin":0, "cmap":"jet", "origin":"lower"}
        kw.update(kwargs)
        
        with plt.style.context(self.__class__.styles):
            plt.figure()
            plt.title("Transition Density Plot")
            plt.imshow(z.T, **kw)
            plt.colorbar()
            pos = ((means - axlim[0]) / (axlim[1] - axlim[0]) * 200).astype("int16")
            digit_0 = int(np.median(np.floor(np.log10(np.abs(means)))))
            plt.xticks(pos, np.round(means, -digit_0 + 1))
            plt.yticks(pos, np.round(means, -digit_0 + 1))
            plt.xlabel("Before")
            plt.ylabel("After")
            plt.show()
        return None
    
    def _set_hmm_params(self):
        if not hasattr(self, "n_components"):
            raise AttributeError("'n_components' has yet been specified.")
        
        hasattr(self, "covars_") or self._set_covars()
        hasattr(self, "means_") or self._set_means()
        hasattr(self, "startprob_") or self._set_startprob()
        hasattr(self, "transmat_") or self._set_transmat()
        
        return None

