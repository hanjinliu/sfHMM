import numpy as np
from .moment import GaussMoment, PoissonMoment
import heapq

"""
    An efficient, extended version of step finding algorithm original in [1].
    Although similar summation of data is repeated many times during a single run, the 
original algorithm did not make full use of the results obtained in previous iterations. 
In short, before calculating such as square deviation, the 'generator' of it is stored and
passed to next iteration. The n-th-order moment E[X_i^n] is very useful for in this purpose, 
that's whiy Moment class is defined in moment.py. A moment object corresponds to one constant
region (between two signal change points) and has n-th order moment with any step position
as attributes. Here in GaussStep/PoissonStep class, GaussMoment/PoissonMoment is split into
two fragments and they are passed to newly generated two steps.
    In this optimized implementation, both GaussStep and PoissonStep can analyze 100,000-point
data within 1 sec!

Note that `x0` and `dx` always denotes positions shown below:
       
       x0+dx                     
 0  x0   v  oooooooooo           
 v   v   ooo    <- new step (2)  
 oooo                            
     .......    <- old step      
     oooo       <- new step (1)  

Reference
---------
[1] Kalafut, B., & Visscher, K. (2008). An objective, model-independent method for detection
of non-uniform steps in noisy signals. Computer Physics Communications, 179(10), 716–723.
https://doi.org/10.1016/j.cpc.2008.06.008

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
        pass
    
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

        
class GaussStep(BaseStep):
    """
    Step finding algorithm for trajectories that follow Gauss distribution.
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
        super().__init__(data, p)
        
    def multi_step_finding(self):
        g = GaussMoment().init(self.data, order=2)
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


class PoissonStep(BaseStep):
    """
    Step finding algorithm for trajectories that follow Poisson distribution.
    """    
    def __init__(self, data, p=-1):
        super().__init__(data, p)
        if not np.issubdtype(self.data.dtype, np.integer):
            raise TypeError("In PoissonStep, non-integer data type is forbidden.")
        elif np.any(self.data < 0):
            raise ValueError("Input data contains negative values.")

    def multi_step_finding(self):
        p = PoissonMoment().init(self.data, order=1)
        self.fit = np.full(self.len, p.total[0]/self.len)
        self._append_steps(p)
        self._finalize()
        return None
    
    def _append_steps(self, p:PoissonMoment, x0:int=0):
        if len(p) < 3:
            return None
        slogm, dx = p.get_optimal_splitter()
        dlogL = self.penalty + slogm
        if dlogL > 0:
            self.step_list.append(x0+dx)
            p1, p2 = p.split(dx)
            self._append_steps(p1, x0=x0)
            self._append_steps(p2, x0=x0+dx)
        else:
            pass
        return None
