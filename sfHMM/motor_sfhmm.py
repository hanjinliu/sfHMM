import numpy as np
from .single_sfhmm import sfHMM1
from .multi_sfhmm import sfHMMn
from .base import sfHMMmotorBase
from .func import *


class sfHMM1Motor(sfHMM1, sfHMMmotorBase):
    def __init__(self, data_raw, sg0:float=-1, psf:float=-1, krange=(1, 6),
                 model:str="g", name:str="", max_stride:int=2):
        super().__init__(data_raw, sg0, psf, krange, model, name)
        self.max_stride = max_stride
        self.covariance_type = "tied"
        
    def gmmfit(self, method="bic", n_init=3, random_state=0, estimate_krange=True):
        if estimate_krange:
            k = self._get_movement_range()
            n = self.step.n_step
            self.krange = (max(1, int(k-n*0.1)), int(k+n*0.1)+1)
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
        transmat_kernel = np.zeros(self.max_stride*2 + 1)
        dy = np.diff(self.states)
        dy = dy[np.abs(dy)<=self.max_stride]
        dy_unique, counts = np.unique(dy,return_counts=True)
        transmat_kernel[dy_unique + self.max_stride] = counts
        self.transmat_kernel = transmat_kernel/np.sum(transmat_kernel)
        
        return None
    
    def _accumulate_transitions(self, axlim=None, cov=None):
        dy_step = np.diff(self.step.fit)
        dy_step = dy_step[dy_step!=0]
        dy_vit = np.diff(self.viterbi)
        dy_vit = dy_vit[dy_vit!=0]
        return dy_step, dy_vit
    
    def _get_movement_range(self):
        all_move = np.hstack(([0], np.cumsum(np.where(self.step.step_size_list > 0, 1, -1))))
        return all_move.max() - all_move.min() + 1
    
    
class sfHMMnMotor(sfHMMn, sfHMMmotorBase):
    def __init__(self, sg0:float=-1, psf:float=-1, krange=(1, 6), 
                 model:str="g", name:str="", max_stride:int=2):
        super().__init__(sg0, psf, krange, model, name)
        self.max_stride = max_stride
        self.covariance_type = "tied"
    
    def append(self, data):
        sf = sfHMM1Motor(data, sg0=self.sg0, psf=self.psf, krange=self.krange,
                         model=self.model, name=self.name+f"[{self.n_data}]",
                         max_stride=self.max_stride)
        self.n_data += 1
        self._sf_list.append(sf)
        self.ylim[0] = min(sf.ylim[0], self.ylim[0])
        self.ylim[1] = max(sf.ylim[1], self.ylim[1])
        return self
    
    def gmmfit(self, method="bic", n_init=3, random_state=0, estimate_krange=True):
        if estimate_krange:
            k_list = [sf._get_movement_range() for sf in self]
            n_list = [sf.step.n_step for sf in self]
            argkmin = np.argmin(k_list)
            argkmax = np.argmax(k_list)
            kmin = int(k_list[argkmin] - n_list[argkmin]*0.1)
            kmax = int(k_list[argkmax] + n_list[argkmax]*0.1)
            self.krange = (max(1, kmin), kmax)
        return super().gmmfit(method, n_init, random_state)
        
    def _set_covars(self):
        step_fit = np.array(concat([sf.step.fit for sf in self]))
        self.covars_ = [[np.var(self.data_raw - step_fit)]]
        self.min_covar = np.min(self.covars_) * 0.015
        return None
    
    def _set_transmat(self):
        transmat_kernel = np.zeros(self.max_stride*2 + 1)
        dy = np.array(concat([np.diff(sf.states) for sf in self]))
        dy = dy[np.abs(dy) <= self.max_stride]
        dy_unique, counts = np.unique(dy, return_counts=True)
        transmat_kernel[dy_unique + self.max_stride] = counts
        self.transmat_kernel = transmat_kernel/np.sum(transmat_kernel)
        
        return None
    
    def _accumulate_transitions(self, axlim=None, cov=None):
        dy_step = np.array(concat([np.diff(sf.step.fit) for sf in self]))
        dy_step = dy_step[dy_step!=0]
        dy_vit = np.array(concat([np.diff(sf.viterbi) for sf in self]))
        dy_vit = dy_vit[dy_vit!=0]
        return dy_step, dy_vit
    
    def _copy_params(self, sf):
        sf.covars_ = [[self.covars_[0,0,0]]]
        sf.min_covar = self.min_covar
        sf.means_ = self.means_
        sf.startprob_ = self.startprob_
        sf.transmat_kernel = self.transmat_kernel
        return None