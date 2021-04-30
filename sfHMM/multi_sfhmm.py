from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from .utils import *
from .single_sfhmm import sfHMM1
from .base import sfHMMBase
from typing import Iterable
import re
import copy

__all__ = ["sfHMMn"]
# TODO: rewrite docstring
class sfHMMn(sfHMMBase):
    """
    Multi-trajectory sfHMM.
    This class shares all the attributes in hmmlearn.hmm.GaussianHMM.

    Analysis Results
    ----------------
    gmm_opt : `GMMs` or `DPGMM` object
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
    
    def __init__(self, sg0:float=-1, psf:float=-1, krange=(1, 6), 
                 model:str="g", name:str="", **kwargs):
        """
        Parameters
        ----------
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
        """        
        self.n_data = 0
        super().__init__(sg0, psf, krange, model, name, **kwargs)
        self.ylim = [np.inf, -np.inf]
        self._sf_list = []
    
    
    def __getitem__(self, key) -> sfHMM1:
        return self._sf_list[key]
    
    def __iter__(self):
        return iter(self._sf_list)
    
    def __add__(self, other:sfHMMn) -> sfHMMn:
        """
        `+` is supported for two sfHMMn objects. a+b makes a new sfHMMn object with concatenated
        list of sfHMM1 objects.
        """        
        if self is other:
            new = self.__class__(sg0=self.sg0, psf=self.psf, krage=self.krange,
                                 model=self.model, name=self.name+"+"+other.name)
            new.n_data = self.n_data + other.n_data
            new.ylim = [min(self.ylim[0], other.ylim[0]),
                        max(self.ylim[1], other.ylim[1])]
            new._sf_list = copy.deepcopy(self._sf_list) + copy.deepcopy(other._sf_list)
            return new
        else:
            raise TypeError("Unsupported operand type(s) for +: "
                           f"{self.__class__.__name__} and {type(other)}.")
    
    @property
    def names(self) -> list[str]:
        return [sf.name for sf in self]
    
    @append_log
    def append(self, data, name:str=None) -> sfHMMn:
        """
        Append a trajectory as sfHMM object.

        Parameters
        ----------
        data : array
            Data to analyze.
        name : str, optional
            Name of the data.
        """        
        if name is None:
            name = self.name+f"[{self.n_data}]"
            
        sf = sfHMM1(data, sg0=self.sg0, psf=self.psf, krange=self.krange,
                    model=self.StepClass, name=name)
        self.n_data += 1
        self._sf_list.append(sf)
        self.ylim[0] = min(sf.ylim[0], self.ylim[0])
        self.ylim[1] = max(sf.ylim[1], self.ylim[1])
        return self
    
    
    def appendn(self, datasets:Iterable) -> sfHMMn:
        """
        Append all the data in the list

        Parameters
        ----------
        datasets : dict, list or any iterable objects except for np.ndarray.
            Datasets to be appended.
        """        
        if isinstance(datasets, dict):
            for name, data in datasets.items():
                self.append(data, name=name)
        elif isinstance(datasets, np.ndarray):
            raise TypeError("Datasets of ndarray is ambiguious. Please use self.append(a) for 1-D "
                            "ndarray, or explicitly specify along which axis to iterate by such as "
                            "list(a) or np.split(a, axis=1).")
        else:
            for data in datasets:
                self.append(data)
                
        return self
    
    @append_log
    def delete(self, indices:int|list[int]|tuple[int]) -> None:
        """
        Delete sfHMM1 object(s)

        Parameters
        ----------
        indices : int or iterable of int
            Indices to delete.
        """        
        if isinstance(indices, int):
            indices = [indices]
        indices = sorted(indices)
        
        for i in reversed(indices):
            self._sf_list.pop(i)
        
        data_raw_all = self.data_raw
        self.ylim = [data_raw_all.min(), data_raw_all.max()]
        self.n_data -= len(indices)
        
        return None
    
    
    def from_pandas(self, df, like:str=None, regex:str=None, melt:bool|str="infer") -> sfHMMn:
        """
        Load datasets from pandas.DataFrame.

        Parameters
        ----------
        df : DataFrame
            Input data.
        like : str, optional
            If given, dataset that contains this string is appended.
        regex : regular expression, optional
            If given, dataset that matches this regular expression is appended.
        melt : bool or "infer", optional
            If input DataFrame is melted, which is automatically determined when melt is "infer".
        """        
        if melt == "infer":
            # Determine if df is melted or not
            col0 = df.iloc[:,0]
            if df.shape[1] == 2 and len(col0.unique()) < len(col0)//20:
                return self.from_pandas(df, like, regex, melt=True)
            else:
                return self.from_pandas(df, like, regex, melt=False)
        
        if melt:
            if df.shape[1] != 2:
                raise ValueError("For melted DataFrame, it must composed of two columns, with names "
                                 "in the first and values in the second.")
            name_col, value_col = df.columns
            for name in df[name_col].unique():
                if like and not like in name:
                    continue
                elif regex and not re.match(regex, name):
                    continue
                self.append(df[df[name_col] == name][value_col], name)
        
        else:
            if like is not None or regex is not None:
                df = df.filter(like=like, regex=regex)
            for name in df:
                data = df[name].dropna()
                self.append(data, name)
        
        if self.n_data == 0:
            raise ValueError("No data appended. Confirm that input DataFrame is in a correct format.")
        
        return self
 
    @append_log
    def step_finding(self) -> sfHMMn:
        """
        Step finding by extended version of Kalafut-Visscher's algorithm.
        Run independently for each sfHMM object.
        """
        if self.n_data <= 0:
            raise sfHMMAnalysisError("Cannot start analysis before appending data.")
        
        for sf in self:
            sf.psf = self.psf
            sf.step_finding()
        return self
    
    @append_log
    def denoising(self) -> sfHMMn:
        """
        Denoising by cutting of the standard deviation of noise to sg0.
        Run independently for each sfHMM object.
        """
        if self.n_data <= 0:
            raise sfHMMAnalysisError("Cannot start analysis before appending data.")
        
        self._init_sg0()
        
        for sf in self:
            sf.sg0 = self.sg0
            sf.denoising()

        return self
    
    @append_log
    def gmmfit(self, method:str="bic", n_init:int=1, random_state:int=0) -> sfHMMn:
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
            raise sfHMMAnalysisError("Cannot start analysis before appending data.")
        
        self._gmmfit(method, n_init, random_state)
        
        for sf in self:
            sf.states = self.gmm_opt.predict(np.asarray(sf.step.fit).reshape(-1, 1))
            sf.n_components = self.n_components
        return self
    
    @append_log
    def hmmfit(self) -> sfHMMn:
        """
        HMM paramter optimization by Forward-Backward algorithm, and state inference by Viterbi 
        algorithm.
        """
        if self.n_data <= 0:
            raise sfHMMAnalysisError("Cannot start analysis before appending data.")
        
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
    
    def run_all_separately(self, plot_every:int=0) -> sfHMMn:
        """
        Run the function `run_all` for every sfHMM1 object.
        Paramters
        ---------
        plot_every : int
            Every `plot_every` result will be plotted during iteration.
        """        
        for i, sf in enumerate(self):
            pl = plot_every > 0 and i % plot_every == 0
            try:
                sf.run_all(plot=pl)
            except Exception as e:
                print(f"{e.__class__.name} during {i}-th trace: {e}")
        
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
        self.startprob_ = calc_startprob(d0_list, self.gmm_opt)
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
    
    
    def accumulate_transitions(self) -> list[float]:
        return concat([sf.accumulate_transitions() for sf in self])


    def _accumulate_step_sizes(self) -> np.ndarray:
        if self[0].step is None:
            raise sfHMMAnalysisError("Steps are not detected yet.")
        return np.array(concat([sf.step.step_size_list for sf in self]))
    
    def _copy_params(self, sf):
        if self.covariance_type == "spherical":
            sf.covars_ = self.covars_.ravel()
        else:
            sf.covars_ = [[self.covars_[0,0,0]]]
        sf.min_covar = self.min_covar
        sf.means_ = self.means_
        sf.startprob_ = self.startprob_
        sf.transmat_ = self.transmat_
        return None
        
    @property
    def data_raw(self) -> np.ndarray:
        return np.array(concat([sf.data_raw for sf in self]))
    
    @property    
    def data_fil(self) -> np.ndarray:
        return np.array(concat([sf.data_fil for sf in self]))
        
    @property
    def _sg_list(self) -> np.ndarray:
        return np.array(concat([sf._sg_list for sf in self]))
    
    @property
    def n_list(self) -> list[int]:
        return [sf.data_raw.size for sf in self]