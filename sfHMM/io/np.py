import numpy as np
import os
from .utils import *
from ..base import sfHMMBase
from ..single_sfhmm import sfHMM1
from ..multi_sfhmm import sfHMMn

__all__ = ["load_txt"]

def loadtxt(path, out:sfHMMn=None, sep:str=None, encoding:str=None, skiprows=0, **kwargs):
    out = check_ref(out, sfHMMn)
    
    arr = np.loadtxt(path, delimiter=sep, encoding=encoding, skiprows=skiprows, **kwargs)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    
    out.appendn([i for i in arr.T])
    return out

def savetxt(path, obj:sfHMMBase):
    pass