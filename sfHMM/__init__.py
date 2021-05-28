__version__ = "0.4.1"

from .single_sfhmm import sfHMM1
from .multi_sfhmm import sfHMMn

try:
    from . import io
except ImportError:
    pass

try:
	from . import motor
except ImportError:
    pass

from .sampling import hmm_sampling, motor_sampling


r"""
Inheritance Map
---------------

      (hmmlearn.hmm.GaussianHMM)
                  |
                  |
             (sfHMMBase)
           /      |      \
         /        |        \
   sfHMM1  (sfHMMmotorBase)  sfHMMn
       \        /   \        /
        \     /       \     /
     sfHMM1Motor    sfHMMnMotor

"""
