from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from .base import sfHMMBase
from warnings import warn
from .utils import *

__all__ = ["sfHMM1"]

class sfHMM1(sfHMMBase):    
    """
    Step-finding based HMM for single trajectory.
    
    Analysis Results
    ----------------
    data_raw : np.ndarray
        Raw data.
    step : `GaussStep` or `PoissonStep` object
        The result of step finding, which has following attributes:
        - fit ... Fitting result.
        - n_step ... The number of steps (region between two signal change points).
        - step_list ... List of signal change points.
        - mu_list ... list of mean values of each step.
        - len_list ... list of step lengths (step_list[i+1] - step_list[i]).
        - step_size_list ... list of signal change (mu_list[i+1] - mu_list[i]).
    data_fil : np.ndarray
        Data after denoised.
    gmm_opt : `GMMs` or `DPGMM` object. For more detail, see `sfHMM.gmm`.
        The result of GMM clustering, which inherits `sklearn.mixture.GaussianMixture`.
        If AIC/BIC minimization of standard GMM clustering was conducted, the clustering
        results will be stored in `gmm`. See `sfHMM.gmm.GMMs`.
    n_components : int
        The optimal number of states. Same as 'gmm_opt.n_components'.
    states : np.ndarray
        The optimal state sequence. Before HMM fitting, this is determined from the results
        of step finding and GMM clustering. After HMM fitting, this is Viterbi path with
        values {0, 1, 2, ...}.
    viterbi : np.ndarray
        Viterbi path of 'data_raw', while takes values in 'means_'.
    """
    
    def __init__(self, data_raw=None, *, sg0:float=-1, psf:float=-1, krange=None,
                 model:str="g", name:str="", **kwargs):
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
        """        
        self.step = None
        self.data_fil = None
        self.gmm_opt = None
        self.states = None
        self.viterbi = None
        self._sg_list = []
        super().__init__(sg0, psf, krange, model, name, **kwargs)
        self.data_raw = data_raw
    
    @property
    def size(self):
        return self.data_raw.size
    
    @property
    def data_raw(self):
        return self._data_raw
    
    @data_raw.setter
    def data_raw(self, value):
        if value is None:
            self._data_raw = None
        elif np.isscalar(value):
            raise TypeError(f"Wrong type of input data: {type(value)}")
        else:
            d = np.asarray(value)
            if not np.issubdtype(d.dtype, np.number):
                raise TypeError("Input contains non-numeric object(s).")
            elif d.ndim == 1:
                pass
            elif d.ndim == 2 and (d.shape[0] == 1 or d.shape[1] == 1):
                d = d.ravel()
            elif self.krange is not None and d.size < self.krange[1]:
                raise ValueError(f"Input data size is too small: {d.size}")
            else:
                raise ValueError("Input data must be one-dimensonal or any arrays "
                                 "that can be converted to one-dimensional ones.")
            self._data_raw = d
            self.ylim = [np.min(self.data_raw), np.max(self.data_raw)]

    def read(self, path:str, sep:str=None, encoding:str=None, header:int=0, **kwargs):
        """
        Read a file using appropriate function, and import its data to sfHMM1 object. If
        multiple trajectories are found, ValueError will be raised.

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
        out = read(path, sep=sep, encoding=encoding, header=header, **kwargs)
        if len(out._sf_list) > 1:
            raise ValueError("More than one trajectory found. Use sfHMMn instead.")
        self.data_raw = out[0].data_raw
        self.source = path
        return self
    
    @append_log
    def step_finding(self) -> sfHMM1:
        """
        Step finding by extended version of Kalafut-Visscher's algorithm.
        """
        if np.all(np.diff(self.data_raw) > 0):
            msg = f"Data of {self.name} is monotonically increasing. Isn't it a time axis?"
            warn(msg, UserWarning)
        self.step = self.StepClass(self.data_raw, self.psf)
        self.step.multi_step_finding()
        self.psf = getattr(self.step, "p", -1)
        return self
    
    @append_log
    def denoising(self) -> sfHMM1:
        """
        Denoising by cutting of the standard deviation of noise to sg0.
        """
        if self.step is None:
            raise sfHMMAnalysisError("Cannot run denoising before step finding.")
        
        self._init_sg0()
        self.data_fil = np.empty_like(self.data_raw, dtype=np.float64)
        
        for i in range(self.step.n_step):
            x0 = self.step.step_list[i]
            x1 = self.step.step_list[i+1]
            mu = self.step.mu_list[i]
            sg = np.sqrt(np.mean((self.data_raw[x0:x1] - mu)**2))
            self._sg_list.append(sg)
            if self.sg0 < sg:
                self.data_fil[x0:x1] = (self.data_raw[x0:x1] - mu) * self.sg0 / sg + mu
            else:
                self.data_fil[x0:x1] = self.data_raw[x0:x1]

        return self
    
    @append_log
    def gmmfit(self, method:str="bic", n_init:int=1, random_state:int=0) -> sfHMM1:
        """
        Fit the denoised data to Gaussian mixture model, and the optimal number of states
        will be determined. After that, state sequence 'states' will be initialized.

        Parameters
        ----------
        method : str, optional
            How to determine the optimal number of states. This parameter must be
            'aic', 'bic' or 'Dirichlet'. By default "bic".
        n_init : int, optional
            How many times initialization will be performed in K-means, by default 1.
        random_state : int , optional
            Random seed for K-means initialization., by default 0.

        """
        # If denoising was passed.
        if self.data_fil is None:
            self.data_fil = self.data_raw
        
        # Start GMM clustering and determine optimal number of states.
        self._gmmfit(method, n_init, random_state)
        
        # If denoising is conducted without step finding, state sequence will be inferred
        # using 'self.data_fil'.
        if self.step is not None:
            self.states = self.gmm_opt.predict(np.asarray(self.step.fit).reshape(-1, 1))
        else:
            self.states = self.gmm_opt.predict(np.asarray(self.data_fil).reshape(-1, 1))
            
        return self
    
    @append_log
    def hmmfit(self) -> sfHMM1:
        """
        HMM paramter optimization by EM algorithm, and state inference by Viterbi 
        algorithm.
        """
        self._set_hmm_params()
        
        _data_reshaped = np.asarray(self.data_raw).reshape(-1, 1)
        self.fit(_data_reshaped)
        self.states = self.predict(_data_reshaped)
        self.viterbi = self.means_[self.states, 0]
        
        return self


    def plot(self, trange=None):
        """        
        Plot figures of:
            [1] data_raw & step_fit      ||  layout
            [2] data_raw & data_fil      ||  [ 1 ]
            [3] histograms of [2]        ||  [ 2 ][3]
            [4] data_raw & viterbi       ||  [ 4 ]
        
        Parameters
        ----------
        trange : array like, optional
            Range of x-axis to show, by default None
        """
        
        if trange is None:
            sl = slice(0, self.size)
            ylim = self.ylim
        elif isinstance(trange, (list, tuple, np.ndarray)):
            sl = slice(*trange)
            ylim = np.min(self.data_raw[sl]), np.max(self.data_raw[sl])
        elif isinstance(trange, slice):
            sl = trange
            ylim = np.min(self.data_raw[sl]), np.max(self.data_raw[sl])
        else:
            raise TypeError(f"`trange` must be a slice or an array-like object, but got {type(trange)}")
        
        tasks = []
        showhist = self.gmm_opt is not None
        self.step is None or tasks.append("step finding")
        self.data_fil is None or tasks.append("denoised")
        self.viterbi is None or tasks.append("Viterbi path")
        c_raw = self.__class__.colors["raw data"]
        n_row = max(len(tasks), 1)
        n_col = showhist + 1
        
        with plt.style.context(self.__class__.styles):
            plt.figure(figsize=(6*n_col, 4.2*n_row))
                        
            for i, task in enumerate(tasks):
                i += 1
                plt.subplot(n_row, n_col, (i-1)*n_col + 1)
                i == 1 and plt.title(self.name, fontweight="bold")
                kw = dict(ylim=ylim, color1=c_raw, color=self.__class__.colors[task], label=task)
                if task == "step finding":
                    plot2(self.data_raw[sl], self.step.fit[sl], **kw)
                elif task == "denoised":
                    plot2(self.data_raw[sl], self.data_fil[sl], legend=False, **kw)
                    if showhist:
                        plt.subplot(n_row, n_col*2, 2*(i-1)*n_col + 3)
                        self._hist(sl, ylim)
                elif task == "Viterbi path":
                    plot2(self.data_raw[sl], self.viterbi[sl], **kw)
                else:
                    raise NotImplementedError
            
            if trange is not None:
                ax = plt.subplot(6,4,7)
                ax.plot(self.data_raw, color="gray", alpha=0.3)
                ax.plot(np.arange(sl.start, sl.stop), self.data_raw[sl], color=self.__class__.colors["raw data"])
                ax.set_xlim(0, self.size)
                ax.set_ylim(self.ylim)
                ax.plot([sl.start, sl.stop, sl.stop, sl.start, sl.start],
                         [ylim[0], ylim[0], ylim[1], ylim[1], ylim[0]], color="black")
                ax.set_xticks([])
                ax.set_yticks([])
            
            len(tasks) == 0 and plot2(self.data_raw[sl], ylim=ylim, color1=c_raw)
            
            plt.show()
        
        return None
    
    def view_in_qt(self):
        """
        Open a Qt viewer and plot the results.
        """        
        from .viewer import TrajectoryViewer
        data = dict()
        self.data_raw is None or data.update({"raw data": self.data_raw})
        self.step is None or data.update({"step finding": self.step.fit})
        self.data_fil is None or data.update({"denoised": self.data_fil})
        self.viterbi is None or data.update({"Viterbi path": self.viterbi})
        app = TrajectoryViewer(data, self.__class__.styles, self.__class__.colors)
        app.show()
        return app
    
    def accumulate_transitions(self) -> list[tuple[int, int]]:
        """
        Accumulate all the transitions occurred in `self.states`, and return them in
        [(y0, y1), (y1, y2), (y2, y3), ...] format.

        Returns
        -------
        list[tuple[int, int]]
            List of transitions.
        """        
        if not hasattr(self, "states"):
            return np.array([], dtype=np.float64)
        return [(self.states[i], self.states[i+1]) 
                for i in range(self.states.size - 1)
                if self.states[i] != self.states[i+1]]
        
    def _accumulate_step_sizes(self):
        if self.step is None:
            raise sfHMMAnalysisError("Steps are not detected yet.")
        return self.step.step_size_list
    
    def _set_covars(self):
        if self.states is None:
            raise sfHMMAnalysisError("Cannot initialize 'covars_' because the state sequence " 
                                     "'states' hasyet been determined.")
        self.covars_ = calc_covars(self.data_raw, self.states, self.n_components)
        self.min_covar = np.min(self.covars_) * 0.015
        return None
    
    def _set_means(self):
        if self.gmm_opt is None:
            raise sfHMMAnalysisError("Cannot initialize 'means_'. You must run gmmfit() before "
                                     "hmmfit() or set 'means_' manually.")
        self.means_ = self.gmm_opt.means_.copy()
        return None
    
    def _set_startprob(self):
        if self.gmm_opt is None:
            raise sfHMMAnalysisError("Cannot initialize 'startprob_'. You must run gmmfit() "
                                     "before hmmfit() or set 'startprob_' manually.")
        self.startprob_ = calc_startprob([self.data_raw[0]], self.gmm_opt)
        return None
    
    def _set_transmat(self):
        if self.states is None:
            raise sfHMMAnalysisError("Cannot initialize 'transmat_' because the state sequence " 
                                     "'states' has yet been determined.")
        self.transmat_ = calc_transmat([self.states], self.n_components)
        return None
    