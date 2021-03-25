# sfHMM
Step finding based HMM. Classes for HMM analysis in this module inherit `GaussianHMM` class in [hmmlearn](https://github.com/hmmlearn/hmmlearn).

![ex1](animation.gif)

# Installation

```
pip install git+https://github.com/hanjinliu/sfHMM
```

# Contents

- `sfHMM` module contains basic classes and functions for sfHMM data analysis.
  - `sfHMM1` ... sfHMM for single trajectory.
  - `sfHMMn` ... sfHMM for multiple trajectories.
  - `hmm_sampling` ... sample data generation of FRET-like data.
  - `motor_sampling` ... sample data generation of motor-like data.
- `sfHMM.step` module contains several step finding algorithms in similar API. Our efficient implementation enables analysis of **100,000** data points within **~1 sec**!
  - `GaussStep` ... step finding for Gauss distribution [1].
  - `PoissonStep` ... step finding for Poisson distribution.
  - `SDFixedGaussStep` ... step finding for Gauss distribution with fixed standard deviation.
  - `TtestStep` ... step finding using T-test [2].
  - `BayesianPoissonStep` ... step finding in Bayesian method [3].
- `sfHMM.motor` module contains sfHMM classes aiming at analyzing motor-stepping.
  - `sfHMM1Motor` ... sfHMM for single motor-stepping trajectory.
  - `sfHMMnMotor` ... sfHMM for multiple motor-stepping trajectories.

# Inheritance Map

```
      (hmmlearn.hmm.GaussianHMM)
                  |
                  |
             (sfHMMBase)
           /      |      \
         /        |        \
   sfHMM1  (sfHMMmotorBase)  sfHMMn
       \        /   \        /
        \     /       \    /
     sfHMM1Motor    sfHMMnMotor
```

# Common Parameters

All the parameters are optional.
- `sg0` ... The parameter used in denoising process.
- `psf` ... The parameter used in step finding.
- `krange` ... Range of the number of hidden states.
- `model` ... Distribution of signal. Gaussian and Poissonian are supported now.
- `name` ... Name of the object.

# Common Methods

- `run_all()` ... run all the needed algorithm in the most plausible condition.
- `step_finding()` ... Step finding by likelihood maximization.
- `denoising()` ... The standard deviation of noise is cut off to `sg0`.
- `gmmfit()` ... Gaussian mixture model clustering.
- `hmmfit()` ... HMM parameter initialization and optimization.
- `plot()` ... visualize the results of sfHMM analysis.
- `tdp()` ... show the results in pseudo transition density plot.

# Citation
If you found sfHMM useful, please consider citing our paper.
 ...

# References
[1] Kalafut, B., & Visscher, K. (2008). An objective, model-independent method for detection of non-uniform steps in noisy signals. Computer Physics Communications, 179(10), 716-723.

[2] Shuang, B., Cooper, D., Taylor, J. N., Kisley, L., Chen, J., Wang, W., ... & Landes, C. F. (2014). Fast step transition and state identification (STaSI) for discrete single-molecule data analysis. The journal of physical chemistry letters, 5(18), 3157-3161.

[3] Ensign, D. L., & Pande, V. S. (2010). Bayesian detection of intensity changes in  single molecule and molecular dynamics trajectories. Journal of Physical Chemistry B, 114(1), 280–292.