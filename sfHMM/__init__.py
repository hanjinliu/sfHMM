from .sfhmm import sfHMM
from .multi_sfhmm import Multi_sfHMM

__all__ = ["sfHMM", "Multi_sfHMM"]

__doc__ = \
r"""
sfHMM algorithm.
Trajectories can be fitted to HMM in an objective manner,
or with user-defined parameters p and sg0.

Usage
-----
(1) If you want to analyze single trajectory:

>>> data = ``array-like object``
>>> sf = sfHMM(data)
>>> sf.do_all()

and if you want to analyze multiple trajectories with common parameters:

>>> data_list = ``list of array-like object``
>>> msf = Multi_sfHMM()
>>> for data in data_list:
>>>     msf.append(data)
>>> msf.do_all()

(2) You can also use other filtering methods by simply substituting 'data_fil'.

>>> sf = sfHMM(data)
>>> sf.data_fil = YourFilteringMethod(sf.data_raw, paramters)
>>> sf.gmmfit()
>>> sf.hmmfit()
>>> sf.plot()

or by overloading member function denoising() in sfHMM (which is recommended for
Multi_sfHMM because you don't need to run filtering function for every trajectory).

(3) If signals are clear enough that denoising is not needed, then skip it.

>>> sf = sfHMM(data)
>>> sf.gmmfit()
>>> sf.hmmfit()
>>> sf.plot()

(4) sfHMM can be simply used as GaussianHMM with manual initialization.

>>> sf = sfHMM(data)
>>> sf.n_components = ...
>>> sf.means_ = ...
>>> sf.covars_ = ...
>>> sf.transmat_ = ...
>>> sf.startprob_ = ...
>>> sf.hmmfit()
>>> sf.plot()

(5) To manually specify which GMM to adopt, substitute gmm_opt.

>>> sf = sfHMM(data)
>>> sf.do_all(plot=False)  # fit anyway
>>> sf.gmm_opt = sf.gmm[3] # three-state model is chosen
>>> sf.hmmfit()
>>> sf.plot()

"""

