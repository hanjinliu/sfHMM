import numpy as np
from dataclasses import dataclass


class Base(object):
    def __init__(self, data, p):
        self.data = np.asarray(data)
        self.len = self.data.size
        self.n_step = 1
        self.step_list = [0, self.len]

        if (0.0 < p < 0.5):
            self.penalty = np.log(p/(1-p))
        else:
            self.penalty = -0.5 * np.log(self.len)
        
        self.fit: np.ndarray
        self.len_list: np.ndarray
        self.mu_list: np.ndarray
        self.step_size_list: np.ndarray

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

@dataclass
class Moment:
    """
    fw          bw
     0 * ****** 0
     1 ** ***** 1
     : :      : :
     n ****** * n

    
    """    
    fw: np.ndarray = None
    bw: np.ndarray = None
    total: float = None
    
    def __len__(self):
        """
        Get the length of the constant region
        """
        return self.fw.shape[1] + 1

    def complement(self):
        if self.fw is None:
            self.fw = self.total.reshape(-1,1) - self.bw
        elif self.bw is None:
            self.bw = self.total.reshape(-1,1) - self.fw
        else:
            pass
        return None
    
    def split(self, i:int):
        """
        Split a Moment object into to Moment objects at position i.
        This means :i-1 will be the former, and i: will be the latter.
        """        
        return (self.__class__(self.fw[:,:i-1], None, self.fw[:,i-1]),
                self.__class__(None, self.bw[:,i:], self.bw[:,i-1]))
    
    def init(self, data, order=1):
        orders = np.arange(1, order+1)
        self.fw = np.vstack([np.cumsum(data[:-1]**o) for o in orders])
        self.total = np.array(self.fw[:,-1] + data[-1]**orders)
        self.complement()
        return self
    
    def get_optimal_splitter(self):
        pass
    
@dataclass
class GaussMoment(Moment):    
    @property
    def chi2(self):
        return self.total[1] - self.total[0]**2/len(self)
    
    def get_optimal_splitter(self):
        chi2_fw = self.fw[1] - self.fw[0]**2 / np.arange(1, len(self))
        chi2_bw = self.bw[1] - self.bw[0]**2 / np.arange(len(self)-1, 0, -1)
        chi2 = chi2_fw + chi2_bw
        x = np.argmin(chi2)
        return chi2[x] - self.chi2, x + 1
    
        
class GaussStep(Base):
    def __init__(self, data, p=-1):
        super().__init__(data, p)
    
    def multi_step_finding(self):
        i = 0
        repeat = True
        last_updated = 0
        g = GaussMoment().init(self.data, order=2)
        self.fit = np.full(self.len, g.total[0]/self.len)
        moments = [g]
        chi2 = g.chi2
        dchi2, x = g.get_optimal_splitter()
        best_step = [x]
        best_dchi2 = [dchi2]
        
        while repeat:
            x0 = self.step_list[i]
            x1 = self.step_list[i+1]
            if x1 - x0 > 2:
                if best_step[i] > 0:
                    x = best_step[i]
                    dx = x - x0
                    dchi2 = best_dchi2[i]
                else:
                    moments[i].complement()
                    dchi2, dx = moments[i].get_optimal_splitter()
                    x = x0 + dx
                    best_step[i] = x
                    best_dchi2[i] = dchi2
                    
                dlogL = self.penalty - self.len/2 * np.log(1 + dchi2/chi2)
                if dlogL > 0:
                    # insert moments
                    g1, g2 = moments[i].split(dx)
                    moments[i] = g1
                    moments.insert(i+1, g2)
                    # insert step position
                    self.step_list.insert(i+1, x)
                    # initialize information
                    best_step[i] = 0
                    best_step.insert(i+1, 0)
                    best_dchi2.insert(i+1, 0)

                    chi2 += dchi2
                    last_updated = i + 1

                elif i + 1 == last_updated or len(self.step_list) == 2:
                    repeat = False
            elif i + 1 == last_updated:
                repeat = False
            i = i + 1
            if i >= len(self.step_list)-1:
                i = 0
        
        self._finalize()
        return self


@dataclass
class PoissonMoment(Moment):    
    @property
    def slogm(self):
        return self.total[0] * np.log(self.total[0] / len(self))
    
    def get_optimal_splitter(self):
        slogm_fw = self.fw[0] * np.log((self.fw[0]+1e-12) / np.arange(1, len(self)))
        slogm_bw = self.bw[0] * np.log((self.bw[0]+1e-12) / np.arange(len(self)-1, 0, -1))
        slogm = slogm_fw + slogm_bw
        x = np.argmax(slogm)
        return slogm[x] - self.slogm, x + 1
    

class PoissonStep(Base):
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
        p.complement()
        if len(p) < 3:
            return None
        slogm, dx = p.get_optimal_splitter()
        dL = self.penalty + slogm
        if dL > 0:
            self.step_list.append(start + dx)
            p1, p2 = p.split(dx)
            self._append_steps(p1, start = start)
            self._append_steps(p2, start = start + dx)
        else:
            pass
        return None
