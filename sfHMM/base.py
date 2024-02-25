from abc import abstractmethod
import os
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sfHMM.utils import gauss_mix
from sfHMM.gmm import GMMs, DPGMM
from sfHMM.step import GaussStep, PoissonStep, BaseStep

class sfHMMBase(GaussianHMM):
    count = 0
    colors = {
        "raw data": "#FF89F4", 
        "step finding": "#335426",
        "denoised": "#180CB4", 
        "GMM": "#ED1111",
        "GMM marker": "#FF81D8",
        "Viterbi path": "#3B252B",
    }
    styles = {
        "font.size": 16, 
        "lines.linewidth": 1,
        "axes.titlesize": 24,
        "font.family": "serif",
        "font.serif": "Arial",
        "axes.grid": True,
        "axes.labelsize": 16,
        "grid.linewidth": 0.5,
        "legend.framealpha": 0.8,
        "legend.frameon": False,
        "boxplot.meanprops.linewidth": 1,          
    }
    
    means_: NDArray[np.number]
    startprob_: NDArray[np.number]
    covars_: NDArray[np.number]
    transmat_: NDArray[np.number]
    
    def __init__(
        self,
        sg0: float = -1,
        psf: float = -1,
        krange: "int | tuple[int, int] | None" = None,
        model:str="g",
        name:str="", 
        **hmmlearn_params,
    ):
        self.sg0 = sg0
        self.psf = psf
        self.krange = krange
        self.model = model
        params = dict(covariance_type="spherical", init_params="")
        params.update(hmmlearn_params)
        super().__init__(self, **params)
        self.n_features = 1
        self.name = str(name) if name else self._name()
        self._log: list[tuple[str, str, str]] = []
    
    @property
    def log(self):
        out = ""
        for func, n, description in self._log:
            if n == 1:
                out += f"{func}: {description}\n"
            else:
                out += f"{func} ({n}): {description}\n"
        return out
    
    @property
    def krange(self):
        return self._krange
    
    @krange.setter
    def krange(self, value):
        if value is None:
            pass
        elif isinstance(value, int):
            value = max(1, value)
            value = (value, value)
        else:
            kmin, kmax = value
            kmin = max(1, kmin)
            kmax = max(kmin, kmax)
            value = (kmin, kmax)
        self._krange = value
    
    @property
    def model(self):
        return self._model
    
    @model.setter
    def model(self, s):
        if isinstance(s, str):
            if s.lower() in ("g", "gauss", "gaussstep"):
                self._model = "GaussStep"
                self.StepClass = GaussStep
            elif s.lower() in ("p", "poisson", "poissonstep"):
                self._model = "PoissonStep"
                self.StepClass = PoissonStep
            else:
                raise ValueError(f"Invalid model identifier: {s}")    
            
        elif issubclass(s, BaseStep):
            self._model = s.__name__
            self.StepClass = s
            
        else:
            raise TypeError("`model` must be string or BaseStep instance, "
                            f"but got {type(s)}")
            
    def get_params(self, deep:bool=True):
        out = super().get_params(deep=deep)
        out.pop("data_raw", None) # This is not parameter
        out.pop("name", None)     # This does not affect analysis
        return out
    
    def save(self, path: str = None, overwrite: bool = False):
        """
        Save the content.

        Parameters
        ----------
        path : str, optional
            Saving path. If not given, data will saved at the same directory where it was read.
        overwrite : bool, default is False
            Allow overwriting existing file.
            
        """        
        from sfHMM.io import save

        if path is None:
            try:
                source = self.source
            except AttributeError:
                raise AttributeError(
                    "Data was not read by 'read' function so that the location of "
                    "original data is unknown. 'path' argument is needed."
                )
            file, ext = os.path.splitext(source)
            path = "".join([file, "-sfHMMresult", ext])
        if os.path.exists(path) and not overwrite:
            raise FileExistsError(f"File {path} already exists. Change the name or set 'overwrite=True'.")
        save(self, path)
        return None

    @abstractmethod
    def step_finding(self): ...
    
    @abstractmethod
    def denoising(self): ...
    
    @abstractmethod
    def gmmfit(self, method, n_init, random_state): ...
    
    @abstractmethod
    def hmmfit(self): ...
    
    @abstractmethod
    def plot(self): ...
    
    def run_all(self, plot:bool=True, continue_:bool=False):
        """
        Conduct all the processes with default settings.

        Parameters
        ----------
        plot : bool, default is True
            Specify if show the plot of each process, by default True.
        continue_ : bool, default is False
            If True, and sfHMM analysis is half-way, then only run the rest of analysis.
        """
        if continue_:
            logs = [_l[0] for _l in self._log if _l[2] == "Passed"]
        else:
            logs = []
        
        for func in ["step_finding", "denoising", "gmmfit", "hmmfit"]:
            if func not in logs:
                getattr(self, func)()
        plot and self.plot()
        
        return self
    
    
    def accumulate_transitions(self):
        """
        This function returns all the transitions occurred in the trajectory.
        """
    
    def fit(self, X, lengths=None):
        # overloaded to make it scalable.
        scale = self.sg0*5
        scale = 1.0 if scale <= 0 else scale
        
        self.means_ /= scale
        self._covars_ /= scale**2
        super().fit(X/scale, lengths=lengths)
        self.means_ *= scale
        self._covars_ *= scale**2
        return self
    
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
        plt.plot(fit_y, fit_x, color=self.colors["GMM"], linestyle="-.")
        plt.plot(peak_y, peak_x, "<", color = self.colors["GMM"], 
                 markerfacecolor=self.colors["GMM marker"], markersize=10)
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
            _l = np.abs(self._accumulate_step_sizes())
            if len(_l) > 0:
                self.sg0 = np.percentile(_l, p) * 0.2
            else:
                self.sg0 = np.std(self.data_raw)
        
        return None
    
    def _gmmfit(self, method:str, n_init:int, random_state:int):
        # in case S.D. of noise was very small
        if len(self._sg_list) > 0:
            sg0_ = min(self.sg0, np.percentile(self._sg_list, 25))
        else:
            sg0_ = self.sg0
        
        if self.krange is None:
            self.krange = (1, 6) # Enough for most situations.
            
        if method.lower() in ("aic", "bic"):
            gmm_ = GMMs(self.data_fil, self.krange, min_interval=sg0_*1.5,  min_sg=sg0_*0.8, 
                        covariance_type=self.covariance_type)
            gmm_.fit(n_init=n_init, random_state=random_state, scale=self.sg0*5)
            self.gmm = gmm_
            self.gmm_opt = self.gmm.get_optimal(method)
        elif method.lower() == "dirichlet":
            gmm_ = DPGMM(n_components=self.krange[1], n_init=1, 
                         random_state=random_state,
                         covariance_prior=0.02,
                         covariance_type=self.covariance_type)
            gmm_.fit(np.asarray(self.data_fil).reshape(-1, 1), scale=self.sg0*5)
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
