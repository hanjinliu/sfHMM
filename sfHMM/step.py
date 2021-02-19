import numpy as np

__doc__ =\
r"""
Python coded step finding algorithm, in case step_ext does not work.
These takes 10-20 times longer time than the C++ coded ones.
"""

def _make_bool_mat(n):
    """
    large bool-like matrix used in _1_step_finding().
    
    a:
    [[  1, nan, ... , nan, nan],
     [  1,   1, ... , nan, nan],
      ...
     [  1,   1, ... ,   1, nan]]
    
    b:
    [[nan,   1, ... ,   1,   1],
     [nan, nan, ... ,   1,   1],
      ...
     [nan, nan, ... , nan,   1]]
    
    """
    a = np.tril(np.full((n - 1, n), 1, dtype = "float"))
    b = 1 - a
    a[a == 0] = np.nan
    b[b == 0] = np.nan
    return a, b


class Base(object):
    def __init__(self, data, p):
        
        data0 = np.array(data).flatten()
        length = len(data0)
        
        self.data = data0
        self.len = length
        self.arr123 = np.arange(length)+1
        self.n_step = 1
        self.step_list = [0, length]
        self.fit = np.full(length, np.mean(data0))
        self.mat0, self.mat1 = _make_bool_mat(self.len)

        if (0.0 < p and p < 1.0):
            self.penalty = np.log(p/(1-p))
        else:
            self.penalty = -0.5 * np.log(self.len)
        
        self.len_list: np.ndarray
        self.mu_list: np.ndarray
        self.step_size_list: np.ndarray

    def multi_step_finding(self):
        pass
    
    def _1_step_finding(self, X):
        """
        find the best step position in array X
        """
        pass
    
    def _append_steps(self):
        pass
    
    def _update_attribute(self):
        self.n_step = len(self.step_list) - 1 
        self.mu_list = np.zeros(self.n_step)
        
        for i in range(self.n_step):
            self.mu_list[i] = np.mean(self.data[self.step_list[i]:self.step_list[i+1]])
            self.fit[self.step_list[i]:self.step_list[i+1]] = self.mu_list[i]
        
        self.len_list = np.diff(self.step_list)
        self.step_size_list = np.diff(self.mu_list)
        
        return None


class GaussStep(Base):
    def __init__(self, data, p=-1):
        super().__init__(data, p)
        self.sqrsum = np.var(self.data) * self.len
    
    def multi_step_finding(self):
        self._append_steps()
        self._update_attribute()
        return None

    def _1_step_finding(self, X):
        n = len(X)
        old_sqrsum = np.var(X) * n
        mat0 = self.mat0[:n-1,:n]
        mat1 = self.mat1[:n-1,:n]
        _arr123 = self.arr123[:n - 1]
        data_mat = np.full((n-1,1),1)*X.reshape(1,-1)
        sqrsum = np.nanvar(data_mat*mat0, axis=1)*_arr123 + np.nanvar(data_mat*mat1, axis=1)*(n-_arr123)
        step_pos = np.argmin(sqrsum) + 1
        new_sqrsum = sqrsum[step_pos-1]
        return step_pos, new_sqrsum - old_sqrsum
    
    def _append_steps(self):
        i = 0
        repeat = True
        last_updated = 0
        beststep = [0]
        bestdsqrsum = [0]
        
        while(repeat):
            arr = self.data[self.step_list[i]:self.step_list[i+1]]
            if (len(arr) > 2):
                if (beststep[i] > 0):
                    step_pos = beststep[i]
                    dsqrsum = bestdsqrsum[i]
                else:
                    step_pos, dsqrsum = self._1_step_finding(arr)
                    beststep[i] = step_pos
                    bestdsqrsum[i] = dsqrsum

                dL = self.penalty - self.len / 2 * (np.log(1 + dsqrsum / self.sqrsum))
                if (dL > 0):
                    self.step_list.insert(i + 1, self.step_list[i] + step_pos)
                    beststep[i] = 0
                    beststep.insert(i + 1, 0)
                    bestdsqrsum.insert(i + 1, 0)

                    self.sqrsum += dsqrsum
                    last_updated = i + 1

                elif(i + 1 == last_updated or len(self.step_list) == 2):
                    repeat = False
            elif(i + 1 == last_updated):
                repeat = False
            i = i + 1
            if(i >= len(self.step_list)-1):
                i = 0
        
        return None

class PoissonStep(Base):
    def __init__(self, data, p=-1):
        super().__init__(data, p)

    def multi_step_finding(self):
        self._append_steps()
        self.step_list.sort()
        self._update_attribute()
        return None

    def _1_step_finding(self, X):
        n = len(X)
        old_XlogX = np.sum(X)*np.log(np.mean(X))
        mat0 = self.mat0[:n-1,:n]
        mat1 = self.mat1[:n-1,:n]
        data_mat = np.full((n-1,1),1)*X.reshape(1,-1)
        S0 = np.nansum(data_mat*mat0, axis=1)
        L0 = self.arr123[:n-1]
        S1 = np.nansum(data_mat*mat1, axis=1)
        L1 = n - L0
        S0[S0==0] = 1
        S1[S1==0] = 1   # 0log0 = 0 = 1log1
        XlogX = S0*np.log(S0/L0) + S1*np.log(S1/L1)
        step_pos = np.argmax(XlogX)+1
        new_XlogX = XlogX[step_pos-1]
        return step_pos, old_XlogX, new_XlogX
    
    def _append_steps(self, X = None, start = 0):
        if (X is None):
            X = self.data
        if(len(X)<3):
            return None
        step_pos, old, new = self._1_step_finding(X)
        dL = self.penalty + new - old
        if (dL > 0):
            self.step_list.append(start + step_pos)
            self._append_steps(X[:step_pos], start = start)
            self._append_steps(X[step_pos:], start = start + step_pos)
        else:
            pass
        return None
    
