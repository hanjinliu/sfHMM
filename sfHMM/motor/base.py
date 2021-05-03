import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.utils import log_mask_zero, normalize
from ..base import sfHMMBase
from . import _hmmc_motor
from scipy import special

class sfHMMmotorBase(sfHMMBase):
    """
    This base class enables sparse transition probability matrix aiming at analyzing motor
    stepping trajectories. The attribute `transmat_` is generated from `transmat_kernel`
    every time it is called. Also, during M-step transmat_kernel is updated.
    """
    def __init__(self, sg0:float=-1, psf:float=-1, krange=None,
                 model:str="g", name:str="", max_stride:int=2):
        super().__init__(sg0=sg0, psf=psf, krange=krange, model=model, name=name,
                         covariance_type="tied")
        self.max_stride = max_stride
        
    @property
    def transmat_(self):
        transmat = np.zeros((self.n_components, self.n_components))
        for i, p in enumerate(self.transmat_kernel):
            transmat += np.eye(self.n_components, k=i-self.max_stride)*p
        
        normalize(transmat, axis=1)
        return transmat

    
    def gmmfit(self, method:str="Dirichlet", n_init:int=1, random_state:int=0, estimation:str="fast"):
        """
        estimation : str, optional
            How to estimate krange from step finding result.
            - "fast" ... narrower range but faster.
            - "safe" ... wider range but safer.
            or estimated krange can be directly given.            
        """        
        if self.krange is None:
            self._estimate_krange(estimation)
        return super().gmmfit(method, n_init, random_state)
    
    def _estimate_krange(self, estimation):
        dy = self._accumulate_step_sizes()
        nsmall, nlarge = sorted(map(int, [np.sum(dy>0), np.sum(dy<0)]))
        if estimation == "fast":
            self.krange = (nlarge - nsmall, nlarge - nsmall//2)
        elif estimation == "safe":
            self.krange = (nlarge - nsmall, nlarge)
        else:
            raise ValueError(f"Cannot interpret estimation method: {estimation}")
        return None
        
    def tdp(self, **kwargs):
        dy_step = self._accumulate_step_sizes()
        dy_vit = np.array([self.means_[sy]-self.means_[sx]
                           for sx, sy in self.accumulate_transitions()])
        
        with plt.style.context(self.__class__.styles):
            if dy_step.size == 0:
                print("no step found")
                return None
            elif dy_vit.size > 0:
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
    
    def _init_sg0(self, p:float=50):
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
            
        if not np.allclose(self.transmat_kernel.sum(), 1.0):
            raise ValueError("transmat_kernel must sum to 1.0"
                             f" (got {self.transmat_kernel.sum()})")
    
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
            normalize(self.transmat_kernel)
        
        # GaussianHMM._do_mstep must be copied.
        means_prior = self.means_prior
        means_weight = self.means_weight

        denom = stats['post'][:, None]
        if 'm' in self.params:
            self.means_ = ((means_weight * means_prior + stats['obs'])
                           / (means_weight + denom))

        if 'c' in self.params:
            covars_prior = self.covars_prior
            covars_weight = self.covars_weight
            meandiff = self.means_ - means_prior

        if self.covariance_type in ('spherical', 'diag'):
            c_n = (means_weight * meandiff**2
                    + stats['obs**2']
                    - 2 * self.means_ * stats['obs']
                    + self.means_**2 * denom)
            c_d = max(covars_weight - 1, 0) + denom
            self._covars_ = (covars_prior + c_n) / np.maximum(c_d, 1e-5)
            if self.covariance_type == 'spherical':
                self._covars_ = np.tile(self._covars_.mean(1)[:, None],
                                        (1, self._covars_.shape[1]))
        elif self.covariance_type in ('tied', 'full'):
            c_n = np.empty((self.n_components, self.n_features,
                            self.n_features))
            for c in range(self.n_components):
                obsmean = np.outer(stats['obs'][c], self.means_[c])
                c_n[c] = (means_weight * np.outer(meandiff[c],
                                                    meandiff[c])
                            + stats['obs*obs.T'][c]
                            - obsmean - obsmean.T
                            + np.outer(self.means_[c], self.means_[c])
                            * stats['post'][c])
            cvweight = max(covars_weight - self.n_features, 0)
            if self.covariance_type == 'tied':
                self._covars_ = ((covars_prior + c_n.sum(axis=0)) /
                                    (cvweight + stats['post'].sum()))
            elif self.covariance_type == 'full':
                self._covars_ = ((covars_prior + c_n) /
                                    (cvweight + stats['post'][:, None, None]))
    
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
    
    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob,
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
                
        if 'm' in self.params or 'c' in self.params:
            stats['post'] += posteriors.sum(axis=0)
            stats['obs'] += np.dot(posteriors.T, obs)

        if 'c' in self.params:
            if self.covariance_type in ('spherical', 'diag'):
                stats['obs**2'] += np.dot(posteriors.T, obs ** 2)
            elif self.covariance_type in ('tied', 'full'):
                # posteriors: (nt, nc); obs: (nt, nf); obs: (nt, nf)
                # -> (nc, nf, nf)
                stats['obs*obs.T'] += np.einsum(
                    'ij,ik,il->jkl', posteriors, obs, obs)
    
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