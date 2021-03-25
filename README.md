# sfHMM
Step finding based HMM.

![ex1](animation.gif)

# Installation

```
pip install git+https://github.com/hanjinliu/sfHMM
```

# Contents

- `sfHMM` module
  - `sfHMM1` ... sfHMM for single trajectory.
  - `sfHMMn` ... sfHMM for multiple trajectories.
  - `hmm_sampling` ... sample data generation of FRET-like data.
  - `motor_sampling` ... sample data generation of motor-like data.
- `sfHMM.step` module
  - `GaussStep` ... step finding for Gauss distribution[^1].
  - `PoissonStep` ... step finding for Poisson distribution.
  - `SDFixedGaussStep` ... step finding for Gauss distribution with fixed standard deviation.
  - `TtestStep` ... step finding using T-test[^2].
  - `BayesianPoissonStep` ... step finding in Bayesian method[^3].
- `sfHMM.motor` module
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

- `step_finding()` ... Step finding by likelihood maximization.
- `denoising()` ... The standard deviation of noise is cut off to `sg0`.
- `gmmfit()` ... Gaussian mixture model clustering.
- `hmmfit()` ... HMM parameter initialization and optimization.
- `plot()` ... visualize the results of sfHMM analysis.
- `run_all()` ... run all the four steps and plot the results.
- `tdp()` ... show the results in pseudo transition density plot.

# Citation
If you found sfHMM useful, please consider citing our paper.
 ...

# References
[^1]: Kalafut, B., & Visscher, K. (2008). An objective, model-independent method for detection of non-uniform steps in noisy signals. Computer Physics Communications, 179(10), 716-723.

[^2]: Shuang B, Cooper D, Taylor JN, Kisley L, Chen J, Wang W, Li CB, Komatsuzaki T, Landes CF. 2014. Fast step transition and state identification (STaSI) for discrete Single-Molecule data analysis. The Journal of Physical Chemistry Letters 5:3157–3161.

[^3]: Ensign, D. L., & Pande, V. S. (2010). Bayesian detection of intensity changes in  single molecule and molecular dynamics trajectories. Journal of Physical Chemistry B, 114(1), 280–292.