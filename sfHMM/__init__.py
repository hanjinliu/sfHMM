__version__ = "0.3.8"

from .single_sfhmm import sfHMM1
from .multi_sfhmm import sfHMMn
from .sampling import hmm_sampling, motor_sampling

__doc__ = \
r"""
sfHMM algorithm.
Trajectories are fitted to HMM in an objective manner,
or with user-defined parameters psf and sg0.

Usage
-----
(1) If you want to analyze single trajectory:

>>> data = ``array-like object``
>>> sf = sfHMM1(data)
>>> sf.run_all()

and if you want to analyze multiple trajectories with common parameters:

>>> data_list = ``list of array-like object``
>>> msf = sfHMMn()
>>> for data in data_list:
>>>     msf.append(data)
>>> msf.run_all()

(2) You can also use other filtering methods by simply substituting 'data_fil'.

>>> sf = sfHMM1(data)
>>> sf.data_fil = YourFilteringMethod(sf.data_raw, paramters)
>>> sf.gmmfit()
>>> sf.hmmfit()
>>> sf.plot()

or by overloading member function denoising() in sfHMM1 (which is recommended for
sfHMMn because you don't need to run filtering function for every trajectory).

(3) If signals are clear enough that denoising is not needed, then skip it.

>>> sf = sfHMM1(data)
>>> sf.gmmfit()
>>> sf.hmmfit()
>>> sf.plot()

(4) sfHMM1 can be simply used as GaussianHMM with manual initialization.

>>> sf = sfHMM1(data)
>>> sf.n_components = ...
>>> sf.means_ = ...
>>> sf.covars_ = ...
>>> sf.transmat_ = ...
>>> sf.startprob_ = ...
>>> sf.hmmfit()
>>> sf.plot()

(5) To manually specify which GMM to adopt, substitute gmm_opt.

>>> sf = sfHMM1(data)
>>> sf.run_all(plot=False) # fit anyway
>>> sf.gmm_opt = sf.gmm[3] # three-state model is chosen
>>> sf.hmmfit()
>>> sf.plot()

"""
