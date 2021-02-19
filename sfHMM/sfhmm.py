import matplotlib.pyplot as plt
import numpy as np
try:
    from .step_ext import GaussStep
    from .step_ext import PoissonStep
except ImportError as e:
    from .step import GaussStep, PoissonStep
    print(f"Could not import pyd file due to following ImportError: {e}\n"\
           "'step.py' was imported instead. Step finding will take much longer time.")
from .base import Base_sfHMM
from .func import *

class sfHMM(Base_sfHMM):    
    """
    Step-finding based HMM.

    Parameters
    ----------
    data : array like.
        Data for analysis.
    sg0 : float, optional
        Parameter used in filtering method. Expected to be 20% of signal change.
        If <= 0, sg0 will be determined automatically.
    p : float, optional
        Transition probability used in step finding algorithm.
        if 0 < p < 0.5 is not satisfied, the original Kalafut-Visscher's algorithm is executed.
    krange : int or list
        Minimum and maximum number of states to search in GMM clustering. If it is integer, then
        it will be interpretted as [1, krange].
    model: str, optional
        Distribution of noise. Gauss and Poisson distribution are available for now.
    
    Analysis Results
    ----------------
    data_raw : np.ndarray
        Raw data.
    step : GaussStep or PoissonStep object
        The result of step finding, which has following attributes:
        - fit ... Fitting result.
        - n_step ... The number of steps (region between two signal change points).
        - step_list ... List of signal change points.
        - mu_list ... list of mean values of each step.
        - len_list ... list of step lengths (step_list[i+1] - step_list[i]).
        - step_size_list ... list of signal change (mu_list[i+1] - mu_list[i]).
    data_fil : np.ndarray
        Data after denoised.
    gmm_opt : GMMn or DPGMM object
        The result of GMM clustering, which has following attributes:
        - wt ... Weights.
        - mu ... Means.
        - sg ... Standard deviations.
        - n_components ... The number of states.
        If AIC/BIC minimization of standard GMM clustering was conducted, the clustering
        results will be stored in 'gmm'. See .\gmm.py
    n_components : int
        The optimal number of states. Same as 'gmm_opt.n_components'.
    states : np.ndarray
        The optimal state sequence. Before HMM fitting, this is determined from the results
        of step finding and GMM clustering. After HMM fitting, this is Viterbi pass with
        values {0, 1, 2, ...}.
    viterbi : np.ndarray
        Viterbi pass of 'data_raw', while takes value in 'means_'.
    """
    
    def __init__(self, data_raw, sg0:float=-1, p:float=-1, krange=[1, 6],
                 model:str="g", name:str=""):

        self.data_raw = np.asarray(data_raw).flatten()
        self.step = None
        self.data_fil = None
        self.gmm_opt = None
        self.states = None
        self.viterbi = None
        self._sg_list = []
        super().__init__(sg0, p, krange, model, name)
        self.ylim = [np.min(self.data_raw), np.max(self.data_raw)]
    

    def step_finding(self):
        """
        Step finding by extended version of Kalafut-Visscher's algorithm.
        """
        if (self.p <= 0 or 0.5 <= self.p):
            self.p = 1/(1 + np.sqrt(self.data_raw.size))
            
        if (self.model == "Poisson"):
            self.step = PoissonStep(self.data_raw, self.p)
        elif (self.model == "Gauss"):
            self.step = GaussStep(self.data_raw.astype("float64"), self.p)
        else:
            raise ValueError
        
        self.step.multi_step_finding()
        
        return self
    
    def denoising(self):
        """
        Denoising by cutting of the standard deviation of noise to sg0.
        """
        self._init_sg0()
        self.data_fil = np.empty(self.data_raw.size, dtype="float64")
        
        for i in range(self.step.n_step):
            x0 = self.step.step_list[i]
            x1 = self.step.step_list[i+1]
            mu = self.step.mu_list[i]
            sg = np.sqrt(np.mean((self.data_raw[x0:x1] - mu)**2))
            self._sg_list.append(sg)
            if (self.sg0 < sg):
                self.data_fil[x0:x1] = (self.data_raw[x0:x1] - mu) * self.sg0 / sg + mu
            else:
                self.data_fil[x0:x1] = self.data_raw[x0:x1]

        return self
    
    
    def gmmfit(self, n_init=1, method="bic", random_state=0):
        """
        Fit the denoised data to Gaussian mixture model, and the optimal number of states
        will be determined. After that, state sequence 'states' will be initialized.

        Parameters
        ----------
        n_init : int, optional
            How many times initialization will be performed in K-means, by default 1.
        method : str, optional
            How to determine the optimal number of states. This parameter must be
            'aic', 'bic' or 'Dirichlet'. by default "bic".
        random_state : int , optional
            Random seed for K-means initialization., by default 0.

        Raises
        ------
        ValueError
            If 'method' got an inappropriate string.
        """
        # If denoising was passed.
        if (self.data_fil is None):
            self.data_fil = self.data_raw

        # Start GMM clustering and determine optimal number of states.
        self._gmmfit(n_init, method, random_state)
        
        # If denoising is conducted without step finding, state sequence will be inferred
        # using 'self.data_fil'.
        if (self.step is not None):
            self.states = infer_states(self.step.fit, self.gmm_opt.mu)
        else:
            self.states = infer_states(self.data_fil, self.gmm_opt.mu)
            
        return self
    
    
    def hmmfit(self):
        """
        HMM paramter optimization by EM algorithm, and state inference by Viterbi 
        algorithm.

        Raises
        ------
        AttributeError
            If 'n_components' is not specified yet.
        """
        self._set_hmm_params()
        
        _data_reshaped = np.asarray(self.data_raw).reshape(-1, 1)
        self.fit(_data_reshaped)
        self.states = self.predict(_data_reshaped)
        self.viterbi = self.means_[self.states, 0]
        
        return self


    def plot(self):
        """
        Plot figures of:
            [1] raw data and step finding result (data_raw & step_fit)      ||  layout
            [2] raw data and denoised data (data_raw & data_fil)            ||  [ 1 ]
            [3] histograms of [2]                                           ||  [ 2 ][3]
            [4] raw data and HMM fitted data (data_raw & viterbi)           ||  [ 4 ]
        """
        
        tasks = []
        showhist = self.gmm_opt is not None
        self.step is None or tasks.append("step finding")
        self.data_fil is None or tasks.append("denoised")
        self.viterbi is None or tasks.append("Viterbi pass")
        c_raw = self.__class__.colors["raw data"]
        n_row = max(len(tasks), 1)
        n_col = showhist + 1
        
        plt.figure(figsize=(6*n_col, 4.2*n_row))
        i = 0
        while tasks:
            task = tasks.pop(0)
            i += 1
            plt.subplot(n_row, n_col, (i-1)*n_col + 1)
            kw = dict(ylim=self.ylim, color1=c_raw, color=self.__class__.colors[task], label=task)
            if (task == "step finding"):
                plot2(self.data_raw, self.step.fit, **kw)
            elif (task == "denoised"):
                plot2(self.data_raw, self.data_fil, legend=False, **kw)
                if (showhist):
                    plt.subplot(n_row, n_col*2, 2*(i-1)*n_col + 3)
                    self._hist()
            elif (task == "Viterbi pass"):
                plot2(self.data_raw, self.viterbi, **kw)
            else:
                raise NotImplementedError
        
        i == 0 and plot2(self.data_raw)
        
        plt.show()
        
        return None
    
    
    def tdp(self, **kwargs):
        """
        Pseudo transition density plot.
        """
        plt.figure()
        means = self.means_.flatten()
        axlim = (np.min(means) - 3 * self.sg0,
                 np.max(means) + 3 * self.sg0)
        axes = np.linspace(*axlim, 200)
                    
        x, y = np.meshgrid(axes, axes)
        z = np.zeros((200, 200))

        for i in range(self.viterbi.size - 1):
            mx = self.viterbi[i]
            my = self.viterbi[i + 1]
            if (mx != my):
                z += np.exp(-((x - mx)** 2 + (y - my)** 2) / (2 * self.sg0 ** 2))
        
        z /= 2 * np.pi * self.sg0 ** 2
        
        kw = {"vmin":0, "cmap":"jet", "origin":"lower"}
        kw.update(kwargs)
        
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


    def _init_sg0(self):
        """
        Initialize 'sg0' if sg0 is negative.
        """        
        if (self.sg0 < 0):
            if (len(self.step.step_size_list) > 0):
                self.sg0 = np.percentile(np.abs(self.step.step_size_list), 25) * 0.2
            else:
                self.sg0 = np.std(self.data_raw)
        
        return None
    
    def _set_covars(self):
        if (self.states is None):
            raise ValueError("Cannot initialize 'covars_' because the state sequence 'states' has" 
                             "yet been determined.")
        self.covars_ = calc_covars(self.data_raw, self.states, self.n_components)
        self.min_covar = np.min(self.covars_) * 0.015
        return None
    
    def _set_means(self):
        if (self.gmm_opt is None):
            raise ValueError("Cannot initialize 'means_'. You must run gmmfit() before hmmfit() or" \
                             "set 'means_' manually.")
        self.means_ = self.gmm_opt.mu.reshape(-1, 1)
        return None
    
    def _set_startprob(self):
        if (self.gmm_opt is None):
            raise ValueError("Cannot initialize 'startprob_'. You must run gmmfit() before hmmfit() or" \
                             "set 'startprob_' manually.")
        self.startprob_ = calc_startprob([self.data_raw[0]], self.gmm_opt.wt, 
                                         self.gmm_opt.mu, self.covars_)
        return None
    
    def _set_transmat(self):
        if (self.states is None):
            raise ValueError("Cannot initialize 'transmat_' because the state sequence 'states' has" 
                             "yet been determined.")
        self.transmat_ = calc_transmat([self.states], self.n_components)
        return None
    