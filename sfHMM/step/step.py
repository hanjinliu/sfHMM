import numpy as np
import matplotlib.pyplot as plt
from .moment import *
from scipy.stats import t as student_t
from scipy.stats import norm
import heapq

r"""
    An efficient, extended version of step finding algorithm original in [1].
    Although similar summation of data is repeated many times during a single run, the 
original algorithm did not make full use of the results obtained in previous iterations. 
In short, before calculating such as square deviation, the 'generator' of it is stored and
passed to next iteration. The n-th-order moment E[X_i^n] is very useful for in this purpose, 
that's why `Moment` class is defined in moment.py. A `Moment` object corresponds to one constant
region (between two signal change points) and has n-th order moment with any step position as
attributes. Here in step finding classes, a corresponding `Moment` object is split into two 
fragments and they are passed to newly generated two steps.
    In this optimized implementation, both GaussStep and PoissonStep can analyze 100,000-point
data within 1 sec!

Notes
-----
`x0` and `dx` always denote positions shown below:
       
       x0+dx                     
 0  x0   v  oooooooooo           
 v   v   ooo    <- new step (2)  
 oooo                            
     .......    <- old step      
     oooo       <- new step (1)  

Inheritance Map
---------------
                         BaseStep
                      /            \
           RecursiveStep            GaussStep
            /       \
SDFixedGaussStep, PoissonStep, 
TtestStep, BayesianPoissonStep

"""

class Heap:
    """
    Priorty queue. I wrapped heapq because it is not intuitive.
    """    
    def __init__(self):
        self.heap = []
    
    def push(self, item):
        heapq.heappush(self.heap, item)
    
    def pop(self):
        return heapq.heappop(self.heap)

def estimate_sigma(data):
    p = norm.cdf(1) # = sigma for standard normal distribution.
    return np.quantile(np.diff(data), p)/np.sqrt(2)
    
    
class BaseStep:
    def __init__(self, data, p):
        self.data = np.asarray(data)
        self.len = self.data.size
        self.n_step = 1
        self.step_list = [0, self.len]
        if 0.0 < p < 0.5:
            self.p = p
        else:
            self.p = 1/(1 + np.sqrt(self.len))

        self.penalty = np.log(self.p/(1-self.p))
        

    def multi_step_finding(self):
        """
        Run step-finding algorithm and store all the information.
        """
    
    def _finalize(self):
        """
        Called at the end of fitting.
        """        
        self.step_list.sort()
        self.n_step = len(self.step_list) - 1 
        self.mu_list = np.zeros(self.n_step)
        
        for i in range(self.n_step):
            self.mu_list[i] = np.mean(self.data[self.step_list[i]:self.step_list[i+1]])
            self.fit[self.step_list[i]:self.step_list[i+1]] = self.mu_list[i]
        
        self.len_list = np.diff(self.step_list)
        self.step_size_list = np.diff(self.mu_list)
        
        return None
    
    def plot(self):
        plt.plot(self.data, color="lightgray", label="raw data")
        plt.plot(self.fit, color="red", label="fit")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
        plt.show()
        return None


class RecursiveStep(BaseStep):
    def _append_steps(self, mom, x0:int=0):
        if len(mom) < 3:
            return None
        s, dx = mom.get_optimal_splitter()
        if self._continue(s):
            self.step_list.append(x0+dx)
            mom1, mom2 = mom.split(dx)
            self._append_steps(mom1, x0=x0)
            self._append_steps(mom2, x0=x0+dx)
        else:
            pass
        return None
    
    def _init_momemt(self) -> Moment: ...
    
    def _continue(self, s) -> bool: ... 
    
    def multi_step_finding(self):
        mom = self._init_moment()
        self.fit = np.full(self.len, mom.total[0]/self.len)
        self._append_steps(mom)
        self._finalize()
        return self
        
        
class GaussStep(BaseStep):
    """
    Gauss-distribution step finding.
    
    Reference
    ---------
    Kalafut, B., & Visscher, K. (2008). An objective, model-independent method for 
    detection of non-uniform steps in noisy signals. Computer Physics Communications, 
    179(10), 716–723. https://doi.org/10.1016/j.cpc.2008.06.008
    """    
    def __init__(self, data, p=-1):
        """
        Parameters
        ----------
        data : array
            Input array.
        p : float, optional
            Probability of transition (signal change). If not in a proper range 0<p<0.5,
            then This algorithm will be identical to the original Kalafut-Visscher's.
        """
        super().__init__(np.asarray(data, dtype=np.float64), p)
        
    def multi_step_finding(self):
        g = GaussMoment().init(self.data)
        self.fit = np.full(self.len, g.total[0]/self.len)
        chi2 = g.chi2   # initialize total chi^2
        heap = Heap()   # chi^2 change (<0), dx, x0, GaussMoment object of the step
        heap.push(g.get_optimal_splitter() + (0, g))
        
        while True:
            dchi2, dx, x0, g = heap.pop()
            dlogL = self.penalty - self.len/2 * np.log(1 + dchi2/chi2)
            
            if dlogL > 0:
                x = x0 + dx
                g1, g2 = g.split(dx)
                len(g1) > 2 and heap.push(g1.get_optimal_splitter() + (x0, g1))
                len(g2) > 2 and heap.push(g2.get_optimal_splitter() + (x, g2))
                self.step_list.append(x)
                chi2 += dchi2
            else:
                break
        
        self._finalize()
        return self
    

class SDFixedGaussStep(RecursiveStep):
    """
    Gauss-distribution step finding with fixed standard deviation of noise.
    If standard deviation of noise is unknown then it will be estimated by
    wavelet method. Compared to GaussStep, this algorithm detects more steps
    in some cases and less in others.

    """    
    def __init__(self, data, p=-1, sigma=-1):
        super().__init__(np.asarray(data, dtype=np.float64), p)
        if sigma < 0:
            sigma = estimate_sigma(data)
        self.sigma = sigma
    
    def _init_moment(self):
        return SDFixedGaussMoment().init(self.data)
    
    def _continue(self, sq) -> bool:
        return self.penalty + sq/(2*self.sigma**2) > 0


class TtestStep(RecursiveStep):
    """
    T-test based step finding.
    
    Reference
    ---------
    Shuang, B., Cooper, D., Taylor, J. N., Kisley, L., Chen, J., Wang, W., ... & Landes, 
    C. F. (2014). Fast step transition and state identification (STaSI) for discrete
    single-molecule data analysis. The journal of physical chemistry letters, 5(18), 3157-3161.
    https://doi.org/10.1021/jz501435p
    """    
    def __init__(self, data, alpha=0.05, sigma=-1):
        self.data = np.asarray(data, dtype=np.float64)
        self.len = self.data.size
        self.n_step = 1
        self.step_list = [0, self.len]
        if not 0 < alpha < 0.5:
            alpha = 0.05
        self.alpha = alpha
        
        if sigma < 0:
            sigma = estimate_sigma(data)
        
        self.sigma = sigma
    
    def _init_moment(self):
        return TtestMoment().init(self.data)
        
    def _append_steps(self, mom, x0:int=0):
        if len(mom) < 3:
            return None
        tk, dx = mom.get_optimal_splitter()
        t_cri = student_t.ppf(1-self.alpha/2, len(mom))
        if t_cri < tk/self.sigma:
            self.step_list.append(x0+dx)
            mom1, mom2 = mom.split(dx)
            self._append_steps(mom1, x0=x0)
            self._append_steps(mom2, x0=x0+dx)
        else:
            pass
        return None


class PoissonStep(RecursiveStep):
    """
    Poisson distribution step finding. Input must be integer.
    """    
    def __init__(self, data, p=-1):
        super().__init__(data, p)
        if not np.issubdtype(self.data.dtype, np.integer):
            raise TypeError("In PoissonStep, non-integer data type is forbidden.")
        elif np.any(self.data < 0):
            raise ValueError("Input data contains negative values.")

    def _init_moment(self):
        return PoissonMoment().init(self.data)

    def _continue(self, dlogL):
        return self.penalty + dlogL > 0
    

class BayesianPoissonStep(RecursiveStep):
    """
    Poisson distribution step finding in a Bayesian method.
    
    Reference
    ---------
    Ensign, D. L., & Pande, V. S. (2010). Bayesian detection of intensity changes in 
    single molecule and molecular dynamics trajectories. Journal of Physical Chemistry
    B, 114(1), 280–292. https://doi.org/10.1021/jp906786b
    """    
    def __init__(self, data, skept=4):
        self.data = np.asarray(data)
        self.len = self.data.size
        self.n_step = 1
        self.step_list = [0, self.len]
        if skept <= 0:
            raise ValueError(f"`skept` must be larger than 0, but got {skept}")
        self.skept = skept
        if not np.issubdtype(self.data.dtype, np.integer):
            raise TypeError("In PoissonStep, non-integer data type is forbidden.")
        elif np.any(self.data < 0):
            raise ValueError("Input data contains negative values.")
    
    def _init_moment(self):
        return BayesianPoissonMoment().init(self.data)
    
    def _continue(self, logbf):
        return np.log(self.skept) < logbf