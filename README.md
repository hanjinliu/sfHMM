# sfHMM
Step finding based HMM.

![ex1](animation.gif)

# Installation

```
pip install git+https://github.com/hanjinliu/sfHMM
```

# Basic Usage

First import `sfHMM1` class and `sfHMMn` class.

```python
from sfHMM import sfHMM1, sfHMMn
```

You can also use simulated data with `hmm_sampling` function.

```python
from sfHMM import hmm_sampling
data = hmm_sampling()
```

## Single trajectory

```python
sf = sfHMM1(data).run_all()
```

## Multiple trajectories

Append datasets one by one using `append` method.
```python
msf = sfHMMn()
for data in list_of_data:
    msf.append(data)
msf.run_all()
```

# Details of Attributes and Methods

`sfHMM1` class and `sfHMMn` class have a similar structure (both inherit `sfHMMBase`) so that they have many attributes and methods in common. Also, `sfHMMBase` inherits `GaussianHMM` in [hmmlearn](https://github.com/hmmlearn/hmmlearn) so that prediction, scoring methods of `GaussianHMM` are all supported.

## Parameters

All the parameters are optional.
- `sg0` ... The parameter used in denoising process.
- `psf` ... The parameter used in step finding.
- `krange` ... Range of the number of hidden states.
- `model` ... Distribution of signal. Gaussian and Poissonian are supported now.
- `name` ... Name of the object.

## Attributes and Methods

sfHMM is composed of four steps.

1. `step_finding()` ... Step finding by likelihood maximization.
2. `denoising()` ... The standard deviation of noise is cut off to `sg0`.
3. `gmmfit()` ... Gaussian mixture model clustering.
4. `hmmfit()` ... HMM parameter initialization and optimization.

Attributes are sequencially added to the object.

- After `step_finding`

  - `step` ... `GaussStep` or `PoissonStep` object, defined in `step` module. This object has following attributes:
    - `fit` ... Fitting result.
    - `n_step` ... The number of steps (region between two signal change points).
    - `step_list` ... List of signal change points.
    - `mu_list` ... list of mean values of each step.
    - `len_list` ... list of step lengths (`step_list[i+1] - step_list[i]`).
    - `step_size_list` ... list of signal change (`mu_list[i+1] - mu_list[i]`). 
  - `psf` ... Automatically determined here if needed.

- After `denoising`  

  - `data_fil` ... Data after denoised.
  - `sg0` ... Automatically determined here if needed.

- After `gmmfit`

  - `gmm_opt` ... The optimal Gaussian mixture model in the form of `GMM1` object. `GMM1` inherits `sklearn.mixture.GaussianMixture`. The only Difference is that all the parameters are sorted after fitting.

  - `gmm` ... `GMMn` object defined in `gmm` module. This object contains `GMM1` objects with different number of states. You can get $n$-state model by indexing like `gmm[n]`.
  - `n_components` ... The number of states.
  - `states` ... State sequence, predicted only with the results in 1-3.

- After `hmmfit`
    
  - `means_`, `covars_`, `transmat_`, `startprob_` ... Parameters in Gaussian HMM. For detailed definition, see [hmmlearn](https://github.com/hmmlearn/hmmlearn).
  - `states` (updated after `gmmfit`) ... State sequence, optimized using Viterbi algorithm. This array takes values {0, 1, 2, ...}.
  - `viterbi` ... Signal sequence of Viterbi pass. This array takes values {`means_[0]`, `means_[1]`, `means_[2]`, ...}, so that basically you can plot `viterbi` and `data_raw` on the same figure.

## Other Methods

- `plot()` ... visualize the results of sfHMM analysis.
- `run_all()` ... run all the four steps and plot the results.
- `tdp()` ... show the results in pseudo transition density plot.

## Customize Plots

The super class `sfHMMBase` has class attributes that is passed to `matplotlib` every time you plot. You can change them by updating the dictionaries.

- `colors` ... Line colors of each data.
- `styles` ... Styles of plot. See `rcParams` of `matplotlib`.

For example, if you want to use different color for raw data and smaller font size in `sfHMM1`, then run following codes:

```python
sfHMM1.colors["raw data"] = "gold"
sfHMM1.styles["font.size"] = 10
```

## Additional Attributes and Methods in sfHMMn

- `self[i]` ... The `sfHMM1` object of $i$-th trace. The real list of objects is `_sf_list`. Iteration is defined on this list.
  
```python
msf.do_all()
msf[0].plot() # plot the first trace and its analysis results.
for sf in msf:
  sf.plot()   # plot all the traces in msf and their analysis results.
```

- `n_list`(property) ... List of data lengths.
- `plot_hist()` ... Plot histogram only.
- `plot_traces()` ... Plot all the traces. If you want to plot traces that satisfies a certain condition.
  
```python
# Only plot traces longer than 50.
def fil(sf):
  return sf.data_raw.size > 50:

msf.do_all(plot=False)
plt.hist(msf.n_list) # show the histogram of data lengths.
msf.plot_traces(filter_func=fil)
```

# Application to Motor Stepping

sfHMM can be modified for application to motor stepping trajectories. `sfHMM1Motor` (for single trajectory) and `sfHMMnMotor` (for multiple trajectories) have similar API as `sfHMM1` and `sfHMMn` but there are slight differences due to specialization to motor stepping trajectories such as sparse transition and large number of states.

## Difference in Parameters

- `krange` ... Because it is hard to define the number of states, this parameter is not needed to be predefined. This parameter can be estimated in `gmmfit()` based on the step finding results.
- `max_stride` ... The maximum size of state transition. Transition further than this will be ignored because transition probability for it will be near zero. For most motors this parameter should be set to 1 or 2.

## Difference in Attributes and Methods

- `covariance_type` ... This is an attribute defined in `hmmlearn`. Because all the state should have the same distribution, this is set to `'tied'` here.
- `transmat_kernel` ... Independent paramter set in the transition probability matrix. The length of this array is equal to `max_stride*2+1`. This is passed to `transmat_` getter method every time to construct transition probability matrix. For example, when `transmat_kernel = [0.01, 0.97, 0.02]` then the generated `transmat_` will be:
```python
[[0.98, 0.02,    0,    0, ... ,    0], # on the boundaries, the diagonal
 [0.01, 0.97, 0.02,    0, ... ,    0], # components are larger.
 [   0, 0.01, 0.97, 0.02, ... ,    0],
 [   0, ...         ... , 0.01, 0.99]]
```
- `gmmfit()` ... `n_init=3` is default setting because of the large number of states. Also, if you want to use the predifined `krange`, you need to explicitly add keyward argument `estimate_krange=False`.
- `tdp()` ... In the case of motor stepping, transition desity plot is not a straightforward way to visualize transition. Histogram of transition frequency is plotted here.

## Example
```python
from sfHMM import sfHMM1Motor
sf = sfHMM1Motor(data, max_stride=2)
sf.run_all()
```

# Citation
If you found sfHMM useful, please consider citing our paper.
 ...

# References
- Kalafut, B., & Visscher, K. (2008). An objective, model-independent method for detection of non-uniform steps in noisy signals. Computer Physics Communications, 179(10), 716-723.
