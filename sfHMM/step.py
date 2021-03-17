import numpy as np
from .moment import GaussMoment, PoissonMoment
import heapq

class Heap:
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
            self.penalty = np.log(p/(1-p))
        else:
            self.penalty = -0.5 * np.log(self.len)

    def multi_step_finding(self):
        pass
    
    def _finalize(self):
        self.n_step = len(self.step_list) - 1 
        self.mu_list = np.zeros(self.n_step)
        
        for i in range(self.n_step):
            self.mu_list[i] = np.mean(self.data[self.step_list[i]:self.step_list[i+1]])
            self.fit[self.step_list[i]:self.step_list[i+1]] = self.mu_list[i]
        
        self.len_list = np.diff(self.step_list)
        self.step_size_list = np.diff(self.mu_list)
        
        return None

        
class GaussStep(BaseStep):
    def __init__(self, data, p=-1):
        super().__init__(data, p)
        
    def multi_step_finding(self):
        g = GaussMoment().init(self.data, order=2)
        self.fit = np.full(self.len, g.total[0]/self.len)
        chi2 = g.chi2 # initialize total chi^2
        heap = Heap() # chi^2 change (<0), dx, x0, moment
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
        
        self.step_list.sort()
        self._finalize()
        return self


class PoissonStep(BaseStep):
    def __init__(self, data, p=-1):
        super().__init__(data, p)

    def multi_step_finding(self):
        p = PoissonMoment().init(self.data, order=1)
        self.fit = np.full(self.len, p.total[0]/self.len)
        self._append_steps(p)
        self.step_list.sort()
        self._finalize()
        return None
    
    def _append_steps(self, p:PoissonMoment, start:int=0):
        if len(p) < 3:
            return None
        slogm, dx = p.get_optimal_splitter()
        dlogL = self.penalty + slogm
        if dlogL > 0:
            self.step_list.append(start + dx)
            p1, p2 = p.split(dx)
            self._append_steps(p1, start = start)
            self._append_steps(p2, start = start + dx)
        else:
            pass
        return None
