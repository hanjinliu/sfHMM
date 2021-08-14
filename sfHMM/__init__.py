__version__ = "0.5.1"

from .single_sfhmm import sfHMM1
from .multi_sfhmm import sfHMMn

class ModuleInsufficient:
    def __init__(self, module_name:str, error:ImportError):
        self.name = module_name
        self.error = error
        
    def __getattr__(self, name: str):
        raise ImportError(f"Cannot use {self.name} module due to following "
                          f"ImportError: {self.error}")
        
from . import io

try:
	from . import motor
except ImportError as e:
    motor = ModuleInsufficient("motor", e)

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

# TODO: hmmfit is not scalable now.
# - sfHMMBase._normalize, that rescale self.means_, self.covars_ etc.
# - sfHMMBase._hmmfit, that reshape and rescale data.
# - Don't specify min_covar in _set_covar.