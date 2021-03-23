import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from .func import *
from .gmm import GMMs, DPGMM
from . import _hmmc_motor
from hmmlearn.utils import log_mask_zero
from scipy import special

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
    
    def _init_sg0(self, p=25):
        """
        Initialize 'sg0' if sg0 is negative.
        """        
        if self.sg0 < 0:
            l = self._accumulate_step_sizes()
            if len(l) > 0:
                self.sg0 = np.percentile(l, p) * 0.2
            else:
                self.sg0 = np.std(self.data_raw)
        
        return None
    
    def _gmmfit(self, method, n_init, random_state):
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
                    
        z = self._accumulate_transitions(axlim, cov)
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


def normalize(A, axis=None, mask=None):
    """
    See: https://github.com/matthiasplappert/hmmlearn/tree/topology-fix
    """
    A += np.finfo(float).eps
    if mask is None:
        inverted_mask = None
    else:
        inverted_mask = np.invert(mask)
        A[inverted_mask] = 0.0
    Asum = A.sum(axis)

    if axis and A.ndim > 1:
        # Make sure we don't divide by zero.
        Asum[Asum == 0] = 1
        shape = list(A.shape)
        shape[axis] = 1
        Asum.shape = shape

    A /= Asum
    
    if inverted_mask is not None:
        A[inverted_mask] = np.finfo(float).eps
    return None

class sfHMMmotorBase(sfHMMBase):
    """
    This base class enables sparse transition probability matrix aiming at motor
    stepping trajectories. The attribute `transmat_` is generated from `transmat_kernel`
    every time it is called. Also, during M-step transmat_kernel is updated.
    """
    def __init__(self, sg0:float=-1, psf:float=-1, krange=(1, 6),
                 model:str="g", name:str="", max_stride:int=2):
        super().__init__(sg0, psf, krange, model, name, covariance_type="tied")
        self.max_stride = max_stride
        
    @property
    def transmat_(self):
        transmat = np.zeros((self.n_components, self.n_components))
        for i, p in enumerate(self.transmat_kernel):
            transmat += np.eye(self.n_components, k=i-self.max_stride)*p
        
        mask = transmat > 0
        normalize(transmat, axis=1, mask=mask)
        return transmat
    
    def tdp(self, **kwargs):
        dy_step, dy_vit = self._accumulate_transitions()
        
        with plt.style.context(self.__class__.styles):
            if dy_vit.size>0:
                xmin = min(dy_step.min(), dy_vit.min())
                xmax = max(dy_step.max(), dy_vit.max())
            else:
                xmin = dy_step.min()
                xmax = dy_step.max()
            plt.figure(figsize=(6, 4.4))
            # plot step sizes using step finding result
            plt.subplot(2, 1, 1)
            kw = dict(bins=int((self.max_stride*2+1)*5),
                      color=self.__class__.colors["step finding"])
            kw.update(kwargs)
            plt.hist(dy_step, **kw)
            plt.xlim(xmin, xmax)
            # plot step sizes using Viterbi path
            plt.subplot(2, 1, 2)
            kw = dict(bins=int((self.max_stride*2+1)*5),
                      color=self.__class__.colors["Viterbi path"])
            kw.update(kwargs)
            plt.hist(dy_vit, **kw)
            plt.xlim(xmin, xmax)
            
            plt.xlabel("step size")
            plt.show()
        return None
    
    def _init_sg0(self, p=50):
        """
        Initialize 'sg0' if sg0 is negative.
        """        
        return super()._init_sg0(p=p)
    
    def _check(self):
        self.startprob_ = np.asarray(self.startprob_)
        if len(self.startprob_) != self.n_components:
            raise ValueError("startprob_ must have length n_components")
        if not np.allclose(self.startprob_.sum(), 1.0):
            raise ValueError("startprob_ must sum to 1.0 (got {:.4f})"
                             .format(self.startprob_.sum()))

        # self.transmat_ = np.asarray(self.transmat_) <- skip
        if self.transmat_.shape != (self.n_components, self.n_components):
            raise ValueError(
                "transmat_ must have shape (n_components, n_components)")
        if not np.allclose(self.transmat_.sum(axis=1), 1.0):
            raise ValueError("rows of transmat_ must sum to 1.0 (got {})"
                             .format(self.transmat_.sum(axis=1)))
    
    def _do_mstep(self, stats):
        if 's' in self.params:
            startprob_ = np.maximum(self.startprob_prior - 1 + stats['start'], 0)
            self.startprob_ = np.where(self.startprob_ == 0, 0, startprob_)
            normalize(self.startprob_)
        if 't' in self.params:
            transmat_ = np.maximum(self.transmat_prior - 1 + stats['trans'], 0)
            transmat_ = np.where(self.transmat_ == 0, 0, transmat_)
            for i in range(len(self.transmat_kernel)):
                self.transmat_kernel[i] = np.sum(np.diag(transmat_, k=i-self.max_stride))
            self.transmat_kernel = self.transmat_kernel/np.sum(self.transmat_kernel)
    
    def _do_viterbi_pass(self, framelogprob):
        n_samples, n_components = framelogprob.shape
        state_sequence, logprob = _hmmc_motor._viterbi(
            n_samples, n_components, log_mask_zero(self.startprob_),
            log_mask_zero(self.transmat_kernel), framelogprob, self.max_stride)
        return logprob, state_sequence

    def _do_forward_pass(self, framelogprob):
        n_samples, n_components = framelogprob.shape
        fwdlattice = np.zeros((n_samples, n_components))
        _hmmc_motor._forward(n_samples, n_components,
                             log_mask_zero(self.startprob_),
                             log_mask_zero(self.transmat_kernel),
                             framelogprob, fwdlattice, self.max_stride)
        with np.errstate(under="ignore"):
            return special.logsumexp(fwdlattice[-1]), fwdlattice

    def _do_backward_pass(self, framelogprob):
        n_samples, n_components = framelogprob.shape
        bwdlattice = np.zeros((n_samples, n_components))
        _hmmc_motor._backward(n_samples, n_components,
                              log_mask_zero(self.startprob_),
                              log_mask_zero(self.transmat_kernel),
                              framelogprob, bwdlattice, self.max_stride)
        return bwdlattice
    
    def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        stats['nobs'] += 1
        if 's' in self.params:
            stats['start'] += posteriors[0]
        if 't' in self.params:
            n_samples, n_components = framelogprob.shape
            # when the sample is of length 1, it contains no transitions
            # so there is no reason to update our trans. matrix estimate
            if n_samples <= 1:
                return

            log_xi_sum = np.full((n_components, n_components), -np.inf)
            _hmmc_motor._compute_log_xi_sum(n_samples, n_components, fwdlattice,
                                      log_mask_zero(self.transmat_kernel),
                                      bwdlattice, framelogprob,
                                      log_xi_sum, self.max_stride)
            with np.errstate(under="ignore"):
                stats['trans'] += np.exp(log_xi_sum)
    
    def _get_n_fit_scalars_per_param(self):
        nc = self.n_components
        nf = self.n_features
        return {
            "s": nc - 1,
            "t": self.max_stride*2 + 1,
            "m": nc * nf,
            "c": {
                "spherical": nc,
                "diag": nc * nf,
                "full": nc * nf * (nf + 1) // 2,
                "tied": nf * (nf + 1) // 2,
            }[self.covariance_type],
        }