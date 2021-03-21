import matplotlib.pyplot as plt
import numpy as np
from .single_sfhmm import sfHMM1
from .multi_sfhmm import sfHMMn
from .base import sfHMMmotorBase
from .func import *

def normalize(A, axis=None, mask=None):
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

class sfHMM1motor(sfHMM1, sfHMMmotorBase):
    def __init__(self, data_raw, sg0:float=-1, psf:float=-1, krange=[1, 6],
                 model:str="g", name:str="", max_step_size:int=2):
        super().__init__(data_raw, sg0, psf, krange, model, name)
        self.max_step_size = max_step_size
        self.covariance_type = "tied"
        
    def gmmfit(self, method="bic", n_init=1, random_state=0, estimate_krange=True):
        if estimate_krange:
            n_forward_step = np.sum(self.step.step_size_list > 0)
            k = 2 * n_forward_step - len(self.step.step_size_list) + 1 # predicted number of new steps.
            self.krange = [int(k*0.9), int(k*1.1)]
        return super().gmmfit(method, n_init, random_state)
    
    def _set_covars(self):
        if self.states is None:
            raise RuntimeError("Cannot initialize 'covars_' because the state sequence 'states' has" 
                               "yet been determined.")
        self.covars_ = [[np.var(self.data_raw - self.step.fit)]]
        self.min_covar = np.min(self.covars_) * 0.015
        return None
    
    def _set_transmat(self):
        if self.states is None:
            raise RuntimeError("Cannot initialize 'transmat_' because the state sequence 'states' has" 
                               "yet been determined.")
        transmat_kernel = np.zeros(self.max_step_size*2 + 1)

        dy = np.diff(self.states)
        dy = np.clip(dy, -self.max_step_size, self.max_step_size)
        dy_unique, counts = np.unique(dy,return_counts=True)
        transmat_kernel[dy_unique + self.max_step_size] = counts
        self.transmat_kernel = transmat_kernel/np.sum(transmat_kernel)
        
        return None
    
    def tdp(self, **kwargs):
        dy = np.diff(self.viterbi)
        dy = dy[dy!=0]
        with plt.style.context(self.__class__.styles):
            plt.hist(dy, **kwargs)
            plt.xlabel("step size")
            plt.show()
        return None
    
class sfHMMnmotor(sfHMMn, sfHMMmotorBase):
    def __init__(self, sg0:float=-1, psf:float=-1, krange=[1, 6], 
                 model:str="g", name:str="", max_step_size:int=2):
        super().__init__(sg0, psf, krange, model, name)
        self.max_step_size = max_step_size
        self.covariance_type = "tied"
    
    def append(self, data):
        sf = sfHMM1motor(data, sg0=self.sg0, psf=self.psf, krange=self.krange,
                    model=self.model, name=self.name+f"[{self.n_data}]")
        self.n_data += 1
        self._sf_list.append(sf)
        self.ylim[0] = min(sf.ylim[0], self.ylim[0])
        self.ylim[1] = max(sf.ylim[1], self.ylim[1])
        return self
    
    def gmmfit(self, method="bic", n_init=1, random_state=0, estimate_krange=True):
        if estimate_krange:
            step_size_list = np.array(concat([sf.step.step_size_list for sf in self]))
            n_forward_step = np.sum(step_size_list > 0)
            k = 2 * n_forward_step - len(step_size_list) + 1 # predicted number of new steps.
            self.krange = [int(k*0.9), int(k*1.1)]
        return super().gmmfit(method, n_init, random_state)
    
    def hmmfit(self):
        """
        HMM paramter optimization by Forward-Backward algorithm, and state inference by Viterbi 
        algorithm.
        """
        if self.n_data <= 0:
            raise RuntimeError("Cannot start analysis before appending data.")
        
        self.data_raw_all = self.data_raw
        self.states_list = [sf.states for sf in self]
        
        self._set_hmm_params()
        
        _data_reshaped = np.asarray(self.data_raw_all).reshape(-1, 1)
        _lengths = [sf.data_raw.size for sf in self]
        self.fit(_data_reshaped, lengths=_lengths)
        
        for sf in self:
            sf.covars_ = [[self.covars_[0, 0, 0]]]
            sf.min_covar = self.min_covar
            sf.means_ = self.means_
            sf.startprob_ = self.startprob_
            sf.transmat_kernel = self.transmat_kernel
            sf.states = sf.predict(np.asarray(sf.data_raw).reshape(-1, 1))
            sf.viterbi = sf.means_[sf.states, 0]
        del self.data_raw_all, self.states_list
        return self
    
    def _set_covars(self):
        step_fit = np.array(concat([sf.step.fit for sf in self]))
        self.covars_ = [[np.var(self.data_raw - step_fit)]]
        self.min_covar = np.min(self.covars_) * 0.015
        return None
    
    def _set_transmat(self):
        transmat_kernel = np.zeros(self.max_step_size*2 + 1)
        dy = concat([np.diff(sf.states) for sf in self])
        dy = np.clip(dy, -self.max_step_size, self.max_step_size)
        dy_unique, counts = np.unique(dy,return_counts=True)
        transmat_kernel[dy_unique + self.max_step_size] = counts
        self.transmat_kernel = transmat_kernel/np.sum(transmat_kernel)
        
        return None
    
    def tdp(self, **kwargs):
        dy = concat([np.diff(sf.viterbi) for sf in self])
        dy = dy[dy!=0]
        with plt.style.context(self.__class__.styles):
            plt.hist(dy, **kwargs)
            plt.xlabel("step size")
            plt.show()
        return None