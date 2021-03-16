import numpy as np
from dataclasses import dataclass

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
    