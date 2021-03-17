import numpy as np
from dataclasses import dataclass

# Ideally we should prepare `n = np.arange(1, len(data))` first, and view
# it many times in get_optimal_splitter like n[:len(self.fw)], but this
# does not improved efficiency so much.

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
        """
        Complement moments values.
        Calculate fw from bw, or vice versa.
        """        
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
        m1 = self.__class__(self.fw[:,:i-1], None, self.fw[:,i-1])
        m2 = self.__class__(None, self.bw[:,i:], self.bw[:,i-1])
        m1.complement()
        m2.complement()
        return (m1, m2)
    
    def init(self, data, order=1):
        """
        Initialization using total data.

        Parameters
        ----------
        data : array
            The input data.
        order : int, optional
            Up to what order of moment should be calculated, by default 1
        """
        self.__class__.fwn = np.arange(1, len(data))
        self.__class__.bwn = np.arange(len(data)-1, 0, -1)
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
        n = np.arange(1, len(self))
        chi2_fw = self.fw[1] - self.fw[0]**2 / n
        chi2_bw = self.bw[1] - self.bw[0]**2 / n[::-1]
        chi2 = chi2_fw + chi2_bw
        x = np.argmin(chi2)
        return chi2[x] - self.chi2, x + 1


@dataclass
class PoissonMoment(Moment):    
    @property
    def slogm(self):
        return self.total[0] * np.log(self.total[0] / len(self))
    
    def get_optimal_splitter(self):
        n = np.arange(1, len(self))
        slogm_fw = self.fw[0] * np.log((self.fw[0]+1e-12) / n)
        slogm_bw = self.bw[0] * np.log((self.bw[0]+1e-12) / n[::-1])
        slogm = slogm_fw + slogm_bw
        x = np.argmax(slogm)
        return slogm[x] - self.slogm, x + 1
    