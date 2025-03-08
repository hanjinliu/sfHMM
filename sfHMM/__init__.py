__version__ = "0.7.0"

from sfHMM.single_sfhmm import sfHMM1
from sfHMM.multi_sfhmm import sfHMMn
from sfHMM import io
from sfHMM.sampling import hmm_sampling, motor_sampling
from sfHMM import motor


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
