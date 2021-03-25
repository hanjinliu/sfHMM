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
  - `GaussStep` ... step finding for Gauss distribution.
  - `PoissonStep` ... step finding for Poisson distribution.
  - `SDFixedGaussStep` ... step finding for Gauss distribution with fixed standard deviation.
  - `TtestStep` ... step finding using T-test.
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
- Kalafut, B., & Visscher, K. (2008). An objective, model-independent method for detection of non-uniform steps in noisy signals. Computer Physics Communications, 179(10), 716-723.
