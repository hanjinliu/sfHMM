__version__ = "0.6.4"

from sfHMM.single_sfhmm import sfHMM1
from sfHMM.multi_sfhmm import sfHMMn
from sfHMM import io
from sfHMM.sampling import hmm_sampling, motor_sampling

class ModuleInsufficient:
    def __init__(self, module_name:str, error:ImportError):
        self.name = module_name
        self.error = error
        
    def __getattr__(self, name: str):
        raise ImportError(f"Cannot use {self.name} module due to following "
                          f"ImportError: {self.error}")
        

try:
	from sfHMM import motor
except ImportError as e:
    motor = ModuleInsufficient("motor", e)  # type: ignore
    

# Inheritance Map
# ---------------

#       (hmmlearn.hmm.GaussianHMM)
#                   |
#                   |
#              (sfHMMBase)
#            /      |      \
#          /        |        \
#    sfHMM1  (sfHMMmotorBase)  sfHMMn
#        \        /   \        /
#         \     /       \     /
#      sfHMM1Motor    sfHMMnMotor

__all__ = ["sfHMM1", "sfHMMn", "io", "hmm_sampling", "motor_sampling", "motor"]
