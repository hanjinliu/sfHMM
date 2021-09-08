from __future__ import annotations
import re
import copy
from typing import Iterable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import *
from .single_sfhmm import sfHMM1
from .base import sfHMMBase

__all__ = ["sfHMMn"]

class sfHMMn(sfHMMBase):
    """
    Multi-trajectory sfHMM.
    This class shares many attributes in sfHMM1.

    Analysis Results
    ----------------
    gmm_opt : `GMMs` or `DPGMM` object. For more detail, see `sfHMM.gmm`.
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
    
    def __init__(self, data_raw=None, *, sg0:float=-1, psf:float=-1, krange=None, 
                 model:str="g", name:str="", **kwargs):
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
        """        
        self.n_data = 0
        super().__init__(sg0, psf, krange, model, name, **kwargs)
        self.ylim = [np.inf, -np.inf]
        self._sf_list = []
        self.gmm_opt = None
        data_raw is None or self.appendn(data_raw)
    
    
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
        Append all the data in `datasets`.

        Parameters
        ----------
        datasets : dict, list or any iterable objects except for np.ndarray or pd.DataFrame.
            Datasets to be appended. If it is dict, then keys are interpreted as names.
        """        
        if isinstance(datasets, dict):
            self.from_dict(datasets)
        elif isinstance(datasets, np.ndarray):
            raise TypeError("Datasets of ndarray is ambiguious. Please use self.append(a) for 1-D "
                            "ndarray, or explicitly specify along which axis to iterate by such as "
                            "list(a) or np.split(a, axis=1).")
        elif isinstance(datasets, pd.DataFrame):
            self.from_pandas(datasets)
        else:
            for data in datasets:
                self.append(data)
                
        return self
    
    @append_log
    def delete(self, indices:int|Iterable[int]) -> None:
        """
        Delete sfHMM1 object(s) from the list.

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
    
    def deleteif(self, filter_func, *args) -> list[int]:
        """
        Delete sfHMM1 object(s) from the list if certain conditions are
        satisfied.

        Parameters
        ----------
        filter_func : callable
            `filter_func(sf, *args)==True` then delete `sf`.
        *args : 
            Additional arguments that will be passed to `filter_func`.

        Returns
        -------
        list of int
            Deleted indices.
        """        
        if not callable(filter_func):
            raise TypeError("`filter_func` must be callable")
        
        indices = []
        for i, sf in enumerate(self):
            filter_func(sf, *args) and indices.append(i)
        
        len(indices) > 0 and self.delete(indices)
        
        return indices
    
    @append_log
    def pop(self, ind:int) -> sfHMM1:
        """
        Delete one sfHMM1 object and return it.

        Parameters
        ----------
        ind : int
            Indice to pop.
        """        
        out = self._sf_list.pop(ind)
        data_raw_all = self.data_raw
        self.ylim = [data_raw_all.min(), data_raw_all.max()]
        self.n_data -= 1
        return out
    
    def from_dict(self, d:dict, like:str=None, regex:str=None):
        """
        Load datasets from dict.

        Parameters
        ----------
        d : dict
            Input data.
        like : str, optional
            If given, key that contains this string is appended.
        regex : regular expression, optional
            If given, key that matches this regular expression is appended.
        """    
        for name, data in d.items():
            # only keys matched like or regex requirement are appended.
            if like and not like in name:
                continue
            if regex and not re.match(regex, name):
                continue
            
            self.append(data, name=str(name))
        
        if self.n_data == 0:
            raise ValueError("No data appended. Confirm that input dict is in a correct format.")
        
        return self
        
    def from_pandas(self, df:pd.DataFrame, like:str=None, regex:str=None, melt:bool|str="infer") -> sfHMMn:
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
                data = df[name].dropna() # delete NaN
                self.append(data, name)  # append data
        
        if self.n_data == 0:
            raise ValueError("No data appended. Confirm that input DataFrame is in a correct format.")
        
        return self
    
    def read(self, path:str, sep:str=None, encoding:str=None, header:int=0, **kwargs):
        """
        Read a file using appropriate function, and import its data to sfHMMn object.

        Parameters
        ----------
        path : str
            Path to file.
        sep, encoding, header
            Important arguments in pd.read_csv. Default is header=0 rather than "infer" 
            because header="infer" usually works in a wrong way.
        **kwargs
            Other keyword arguments that will passed to pd.read_csv() or pd.read_excel().
        """   
        from .io import read
        read(path, out=self, sep=sep, encoding=encoding, header=header, **kwargs)
        self.source = path
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
        
        self.fit(self.data_raw_all.reshape(-1, 1), lengths=self.size)
        
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
        if self.gmm_opt is not None:
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
        
    def plot_traces(self, data:str="Viterbi path", filter_func=None, sharex=False):
        """
        Plot all the trajectories. The figure will look like:
         __ __ __ __
        | 0| 1| 2| 3|
        |__|__|__|__|
        | 4| 5| 6| 7|
        |__|__|__|__|
        | 8| 9|10|11|
        :     :     :

        Parameters
        ----------
        data : str, default is "Viterbi path"
            Which data to plot over the raw data trajectories.
        filter_func : callable or None, optional
            If not None, only sfHMM objects that satisfy filter_func(sf)==True are plotted.
        sharex : bool, default is False
            If True, all the subplots will share x-limits.
        """
        c_other = self.colors.get(data, None)
        
        # index list that satisfies filter_func
        if filter_func is None:
            indices = np.arange(self.n_data)
        else:
            indices = [i for (i, sf) in enumerate(self) if filter_func(sf)]
        
        n_col = 4
        n_row = (len(indices) - 1) // n_col + 1
        
        if sharex:
            xlim = [0, max(sf.size for sf in self)]
        
        with plt.style.context(self.__class__.styles):
            plt.figure(figsize=(n_col * 2.9, n_row * 2.4))

            for i, ind in enumerate(indices):
                sf = self[ind]
                ax = plt.subplot(n_row, n_col, i+1)
                    
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
                
                sharex and plt.xlim(xlim)
                
                plt.text(ax.get_xlim()[1]*0.98, ax.get_ylim()[1]*0.98, str(ind), 
                         ha="right", va="top", color="gray")
            
            plt.tight_layout()
            plt.show()
        return None
    
    def view_in_qt(self, title:str|None=None):
        """
        Open a Qt viewer and plot the results.
        """
        from .viewer import TrajectoryViewer
        if title is None:
            title = self.name
            
        datalist = []
        for sf in self:
            data = dict()
            sf.data_raw is None or data.update({"raw data": sf.data_raw})
            sf.step is None or data.update({"step finding": sf.step.fit})
            sf.data_fil is None or data.update({"denoised": sf.data_fil})
            sf.viterbi is None or data.update({"Viterbi path": sf.viterbi})
            datalist.append(data)
            
        viewer = TrajectoryViewer(datalist, self.__class__.styles, self.__class__.colors)
        viewer.setWindowTitle(title)
        viewer.show()
        return viewer
    
    def accumulate_transitions(self) -> list[tuple[int, int]]:
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
    
    @under_development
    @append_log
    def align(self, bounds:tuple[float, float], bins:int=32, formula="y=ax") -> list:
        """
        Align step finding results with `formula` transformation. The optimal parameter is 
        determined by minimizing normalized mutual information of two step finding results:
        `self[0].step.fit` as the reference and `a * self[i].step.fit + b` as the variable.
        
        (1) y = ax
          __                  __    __
         |      __           |     |  |
        _|    _|  |__  -->  _|    _|  |__
        
        (2) y = x + b
        _                  _       __
         |__    __          |__  _|  |__
              _|  |__  -->


        Parameters
        ----------
        bounds : tuple of floats
            Bounds of parameter $a$ or $b$, i.e. optimal parameter is searched in the range from
            `bounds[0]` to `bounds[1]`.
        bins : int, default is 32
            Bin number for calculating shannon entropy and mutual information.
        formula: str, default is "y=ax"
            Formulation of transformation.

        """        
        
        if self.n_data < 2:
            raise sfHMMAnalysisError("Cannot align datasets because n_data < 2.")
        if self[0].step is None:
            raise sfHMMAnalysisError("Cannot align datasets before step finding.")
        
        formula = re.sub(" ", "", formula)
        optimization_func = {"ax": optimize_ax,
                             "y=ax": optimize_ax,
                             "b": optimize_b,
                             "y=x+b": optimize_b}[formula]
        
        for sf in self[1:]:
            result = optimization_func(self[0].step.fit, sf.step.fit, bins=bins, 
                                       range=self.ylim, bounds=[bounds])
            
            a, b = result
            sf.data_raw[:] = a*sf.data_raw + b
            sf.ylim = [a*sf.ylim[0] + b, a*sf.ylim[1] + b]
            sf.step.fit[:] = a*sf.step.fit + b
            sf.step.mu_list[:] = a*sf.step.mu_list + b
            sf.step.step_size_list[:] = a*sf.step.step_size_list
            if sf.data_fil is not None:
                sf.data_fil[:] = a*sf.data_fil + b
            
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
    def size(self) -> list[int]:
        return [sf.size for sf in self]
    
