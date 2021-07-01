from __future__ import annotations
import numpy as np
from ..single_sfhmm import sfHMM1
from ..multi_sfhmm import sfHMMn
from .base import sfHMMmotorBase
from ..utils import *

# TODO: estimate_krange is not a proper way to do so in sfHMMnMotor

class sfHMM1Motor(sfHMMmotorBase, sfHMM1):
    def __init__(self, data_raw=None, *, sg0:float=-1, psf:float=-1, krange=None,
                 model:str="g", name:str="", max_stride:int=2):
        """
        Parameters
        ----------
        data_raw : array like, optional
            Data for analysis.
        sg0 : float, optional
            Parameter used in filtering method. Expected to be 20% of signal change.
            If <= 0, sg0 will be determined automatically.
        psf : float, optional
            Transition probability used in step finding algorithm.
            if 0 < p < 0.5 is not satisfied, the original Kalafut-Visscher's algorithm is executed.
        krange : int or (int, int)
            Minimum and maximum number of states to search in GMM clustering. If it is integer, then
            it will be interpretted as (krange, krange).
        model: str, by default "g" (= Gaussian)
            Distribution of noise. Gauss and Poisson distribution are available for now.
        name : str, optional
            Name of the object.
        max_strides : int, default is 2.
            The largest step of motor. If max_stride = 2, then from 2-step backward to 2-step
            forward steps are considered. Larger value results in longer calculation time.
        """   
        super().__init__(sg0=sg0, psf=psf, krange=krange, model=model, name=name)
        self.data_raw = data_raw
        self.max_stride = max_stride
        self.covariance_type = "tied"
    
    def _set_covars(self):
        if self.states is None:
            raise sfHMMAnalysisError("Cannot initialize 'covars_' because the state sequence " 
                                     "'states' has yet been determined.")
        self.covars_ = [[np.var(self.data_raw - self.step.fit)]]
        self.min_covar = np.min(self.covars_) * 0.015
        return None
    
    def _set_transmat(self):
        if self.states is None:
            raise sfHMMAnalysisError("Cannot initialize 'transmat_' because the state sequence " 
                                     "'states' hasyet been determined.")
        transmat_kernel = np.zeros(self.max_stride*2 + 1)
        dy = np.diff(self.states)
        dy = dy[np.abs(dy)<=self.max_stride]
        dy_unique, counts = np.unique(dy,return_counts=True)
        transmat_kernel[dy_unique + self.max_stride] = counts
        transmat_kernel += 1e-12
        self.transmat_kernel = transmat_kernel/np.sum(transmat_kernel)
        
        return None
    
    
class sfHMMnMotor(sfHMMmotorBase, sfHMMn):
    def __init__(self, data_raw=None, *, sg0:float=-1, psf:float=-1, krange=None, 
                 model:str="g", name:str="", max_stride:int=2):
        """
        Parameters
        ----------
        data_raw : iterable, optional
            Datasets to be added.
        sg0 : float, optional
            Parameter used in filtering method. Expected to be 20% of signal change.
            If <= 0, sg0 will be determined automatically.
        psf : float, optional
            Transition probability used in step finding algorithm.
            if 0 < p < 0.5 is not satisfied, the original Kalafut-Visscher's algorithm is executed.
        krange : int or (int, int)
            Minimum and maximum number of states to search in GMM clustering. If it is integer, then
            it will be interpretted as (krange, krange).
        model: str, by default "g" (= Gaussian)
            Distribution of noise. Gauss and Poisson distribution are available for now.
        name : str, optional
            Name of the object.
        max_strides : int, default is 2.
            The largest step of motor. If max_stride = 2, then from 2-step backward to 2-step
            forward steps are considered. Larger value results in longer calculation time.
        """   
        super().__init__(sg0=sg0, psf=psf, krange=krange, model=model, name=name)
        self.max_stride = max_stride
        self.covariance_type = "tied"
        if data_raw is not None:
            self.appendn(data_raw)
    
    @append_log
    def append(self, data, name:str=None) -> sfHMMnMotor:
        if name is None:
            name = self.name + f"[{self.n_data}]"
            
        sf = sfHMM1Motor(data, sg0=self.sg0, psf=self.psf, krange=self.krange,
                         model=self.model, name=name, max_stride=self.max_stride)
        self.n_data += 1
        self._sf_list.append(sf)
        self.ylim[0] = min(sf.ylim[0], self.ylim[0])
        self.ylim[1] = max(sf.ylim[1], self.ylim[1])
        return self
            
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
        transmat_kernel += 1e-12
        self.transmat_kernel = transmat_kernel/np.sum(transmat_kernel)
        
        return None
    
    def _copy_params(self, sf):
        if self.covariance_type == "spherical":
            sf.covars_ = self.covars_.ravel()
        else:
            sf.covars_ = [[self.covars_[0,0,0]]]
        sf.min_covar = self.min_covar
        sf.means_ = self.means_
        sf.startprob_ = self.startprob_
        sf.transmat_kernel = self.transmat_kernel
        return None
