import numpy as np
import matplotlib.pyplot as plt
from .func import *
from .single_sfhmm import sfHMM1
from .step import GaussStep, PoissonStep
from .base import sfHMMBase

class sfHMMn(sfHMMBase):
    """
    Multi-trajectory sfHMM.
    This class shares all the attributes in hmmlearn.hmm.GaussianHMM.

    Parameters
    ----------
    sg0 : float, optional
        Parameter used in filtering method. Expected to be 20% of signal change.
        If <= 0, sg0 will be determined automatically.
    p : float, optional
        Transition probability used in step finding algorithm.
        if 0 < p < 0.5 is not satisfied, the original Kalafut-Visscher's algorithm is executed.
    krange : int or list
        Minimum and maximum number of states to search in GMM clustering. If it is integer, 
        then it will be interpretted as [1, krange].
    model: str, optional
        Distribution of noise. Gauss and Poisson distribution are available for now.
    
    Analysis Results
    ----------------
    gmm_opt : GMMn or DPGMM object
        The result of GMM clustering, which has following attributes:
        - wt ... Weights.
        - mu ... Means.
        - sg ... Standard deviations.
        - n_components ... The number of states.
        If AIC/BIC minimization of standard GMM clustering was conducted, the clustering
        results will be stored in 'gmm'.
    n_components : int
        The optimal number of states. Same as 'gmm_opt.n_components'.
    self[i] (accessed by indexing) : sfHMM object
        The i-th sfHMM object. See .\single_sfhmm.py.
    """
    
    def __init__(self, sg0:float=-1, psf:float=-1, krange=[1, 6], 
                 model:str="g", name:str=""):
        
        self.n_data = 0
        super().__init__(sg0, psf, krange, model, name)
        self.ylim = [np.inf, -np.inf]
        self._sf_list = []
    
    
    def __getitem__(self, key):
        return self._sf_list[key]
    
    def __iter__(self):
        return iter(self._sf_list)
    
    def append(self, data):
        """
        Append a trajectory as sfHMM object.
        """
        sf = sfHMM1(data, sg0=self.sg0, psf=self.psf, krange=self.krange,
                    model=self.model, name=self.name+f"[{self.n_data}]")
        self.n_data += 1
        self._sf_list.append(sf)
        self.ylim[0] = min(sf.ylim[0], self.ylim[0])
        self.ylim[1] = max(sf.ylim[1], self.ylim[1])
        return self
    
    def step_finding(self):
        """
        Step finding by extended version of Kalafut-Visscher's algorithm.
        Run independently for each sfHMM object.
        """
        if self.n_data <= 0:
            raise RuntimeError("Cannot start analysis before appending data.")
        
        StepMethod = {"Poisson": PoissonStep,
                      "Gauss": GaussStep,
                      }[self.model]
        for sf in self:
            sf.psf = self.psf
            sf.step = StepMethod(sf.data_raw, sf.psf)
            sf.step.multi_step_finding()
        return self
    
    def denoising(self):
        """
        Denoising by cutting of the standard deviation of noise to sg0.
        Run independently for each sfHMM object.
        """
        if self.n_data <= 0:
            raise RuntimeError("Cannot start analysis before appending data.")
        
        self._init_sg0()
        
        for sf in self:
            sf.sg0 = self.sg0
            sf.denoising()

        return self
    
    
    def gmmfit(self, method="bic", n_init=1, random_state=0):
        """
        Fit the denoised data to Gaussian mixture model.
        
        Paramters
        ---------
        method : str, optional
            How to determine the optimal number of states. This parameter must be
            'aic', 'bic' or 'Dirichlet'. By default "bic".
        n_init : int, optional
            How many times initialization will be performed in K-means, by default 1.
        random_state : int , optional
            Random seed for K-means initialization., by default 0.
        
        Raises
        ------
        ValueError
            If 'method' got an inappropriate string.
        """
        if self.n_data <= 0:
            raise RuntimeError("Cannot start analysis before appending data.")
        
        self._gmmfit(method, n_init, random_state)
        
        for sf in self:
            sf.states = self.gmm_opt.predict(np.asarray(sf.step.fit).reshape(-1, 1))
            sf.n_components = self.n_components
        return self
    
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
            self._copy_params(sf)
            sf.states = sf.predict(np.asarray(sf.data_raw).reshape(-1, 1))
            sf.viterbi = sf.means_[sf.states, 0]
        del self.data_raw_all, self.states_list
        return self

    def _set_covars(self):
        self.covars_ = calc_covars(self.data_raw_all, concat(self.states_list), self.n_components)
        self.min_covar = np.min(self.covars_) * 0.015
        return None
    
    def _set_means(self):
        self.means_ = self.gmm_opt.means_.copy()
        return None
    
    def _set_startprob(self):
        d0_list = [sf.data_raw[0] for sf in self]
        self.startprob_ = calc_startprob(d0_list, self.gmm_opt.weights_, self.gmm_opt.means_, self.covars_)
        return None
    
    def _set_transmat(self):
        self.transmat_ = calc_transmat(self.states_list, self.n_components)
        return None


    def plot(self):
        self.plot_hist()
        self.plot_traces()
        return None
    
    def plot_hist(self):
        """
        Plot histogram of data_raw, data_fil and GMM fitting result.
        """
        with plt.style.context(self.__class__.styles):
            plt.figure(figsize=(3, 4))
            plt.suptitle(self.name, fontweight="bold")
            self._hist()
            plt.show()
        return None
        
    def plot_traces(self, data:str="Viterbi path", n_col:int=4, filter_func=None):
        """
        Plot all the trajectories.

        Parameters
        ----------
        data : str, optional
            Which data to plot over the raw data trajectories, by default "Viterbi path"
        n_col : int, optional
            Number of columns of figure, by default 4
        filter_func : callable or None, optional
            If not None, only sfHMM objects that satisfy filter_func(sf)==True are plotted.

        """
        c_other = self.colors.get(data, None)
        
        # index list that satisfies filter_func
        if filter_func is None:
            indices = np.arange(self.n_data)
        else:
            indices = [i for (i, sf) in enumerate(self) if filter_func(sf)]

        n_row = (len(indices) - 1) // n_col + 1
        
        with plt.style.context(self.__class__.styles):
            plt.figure(figsize=(n_col * 2.7, n_row * 4))
            plt.suptitle(self.name, fontweight="bold")
            
            for i, ind in enumerate(indices):
                sf = self[ind]
                plt.subplot(n_row, n_col, i + 1)
                if data == "Viterbi path":
                    d = sf.viterbi
                elif data == "denoised":
                    d = sf.data_fil
                elif data == "step finding":
                    d = sf.step.fit
                elif data == "none":
                    d = None
                else:
                    raise ValueError("'data' must be 'step finding', 'denoised', "
                                    "'Viterbi path' or 'none'")

                plot2(sf.data_raw, d, ylim=self.ylim, legend=False,
                    color1 = self.colors["raw data"], color=c_other)
                plt.text(sf.data_raw.size*0.98, self.ylim[1]*0.98, str(ind), 
                        ha="right", va="top", color="gray")
            
            plt.tight_layout()
            plt.show()
        return None
    
    
    def _accumulate_transitions(self, axlim, cov):
        axes = np.linspace(*axlim, 200)
                    
        x, y = np.meshgrid(axes, axes)
        z = np.zeros((200, 200))

        for sf in self:
            for i in range(sf.viterbi.size - 1):
                mx = sf.viterbi[i]
                my = sf.viterbi[i + 1]
                if (mx != my):
                    z += np.exp(-((x - mx)** 2 + (y - my)** 2) / (2 * cov[sf.states[i]]))
        
        return z

    def _accumulate_step_sizes(self):
        if self[0].step is None:
                raise RuntimeError("Steps are not detected yet.")
        step_size_list = concat([sf.step.step_size_list for sf in self])
        return np.abs(step_size_list)
    
    def _copy_params(self, sf):
        sf.covars_ = self.covars_.ravel()
        sf.min_covar = self.min_covar
        sf.means_ = self.means_
        sf.startprob_ = self.startprob_
        sf.transmat_ = self.transmat_
        return None
        
    @property
    def data_raw(self):
        return np.array(concat([sf.data_raw for sf in self]))
    
    @property    
    def data_fil(self):
        return np.array(concat([sf.data_fil for sf in self]))
        
    @property
    def _sg_list(self):
        return np.array(concat([sf._sg_list for sf in self]))
    
    @property
    def n_list(self):
        return [sf.data_raw.size for sf in self]