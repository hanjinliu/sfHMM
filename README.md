# sfHMM
Step finding based HMM.

![ex1](animation.gif)

# Installation

```
pip install git+https://github.com/hanjinliu/sfHMM
```

# Basic Usage

First import `sfHMM` class and `Multi_sfHMM` class.

```python
from sfHMM import sfHMM, Multi_sfHMM
```

You can also use simulated data with `hmm_sampling` function.

```python
from sfHMM import hmm_sampling
data = hmm_sampling()
```

## Single trajectory

```python
sf = sfHMM(data)
sf.do_all()
```

## Multiple trajectories

Append data to analyze one by one using `append` method.
```python
msf = Multi_sfHMM()
for data in list_of_data:
    msf.append(data)
msf.do_all()
```

# Details of Attributes and Methods

`sfHMM` class and `Multi_sfHMM` class have a similar structure (both inherit `Base_sfHMM`) so that they have many attributes and methods in common. Also `Base_sfHMM` inherits `GaussianHMM` so that prediction, scoring methods in `hmmlearn` are all supported.

## Parameters

All the parameters are optional.
- `sg0` ... The parameter used in denoising process.
- `psf` ... The parameter used in step finding.
- `krange` ... Range of the number of hidden states.
- `model` ... Distribution of signal. Gaussian and Poissonian are supported now.
- `name` ... Name of the object.

## Attributes and Methods

Analysis based on sfHMM is composed of four steps.

1. `step_finding()` ... Step finding by likelihood maximization.
2. `denoising()` ... The standard deviation of noise is cut off to `sg0`.
3. `gmmfit()` ... Gaussian mixture model clustering.
4. `hmmfit()` ... HMM parameter initialization and optimization.

Attributes are sequencially added to the object.

- step_finding

  - `step` ... `GaussStep` or `PoissonStep` object, defined in `step` module or `stepc` extendsion module. This object has following attributes:
    - `fit` ... Fitting result.
    - `n_step` ... The number of steps (region between two signal change points).
    - `step_list` ... List of signal change points.
    - `mu_list` ... list of mean values of each step.
    - `len_list` ... list of step lengths (`step_list[i+1] - step_list[i]`).
    - `step_size_list` ... list of signal change (`mu_list[i+1] - mu_list[i]`). 
  - `psf` ... Automatically determined here if needed.

- denoising  

  - `data_fil` ... Data after denoised.
  - `sg0` ... Automatically determined here if needed.

- gmmfit

  - `gmm_opt` ... The optimal Gaussian mixture model in the form of `GMM1` object, defined in `gmm` module. This object has following attributes:

    - `wt` ... Weights of each Gaussian.
    - `mu` ... Means of each Gaussian.
    - `sg` ... Standard deviations of each Gaussian.
    - `aic` ... Akaike information criterion.
    - `bic` ... Bayes information criterion.

  - `gmm` ... `GMMn` object defined in `gmm` module. This object contains `GMM1` objects with different number of states. You can get `n`-state model by indexing like `gmm[n]`.
  - `n_components` ... The number of states.
  - `states` ... State sequence, predicted only with the results in 1-3.

- hmmfit
    
  - `means_` ... Mean values. See `hmmlearn`.
  - `covars_` ... Covariances. See `hmmlearn`.
  - `transmat_` ... Transition probability matrix. See `hmmlearn`.
  - `startprob_` ... Starting probability. See `hmmlearn`.
  - `states` ... State sequence, optimized using Viterbi algorithm. This array takes values {0, 1, 2, ...}.
  - `viterbi` ... Signal sequence of Viterbi pass. This array takes values {`means_[0]`, `means_[1]`, `means_[2]`, ...}.

## Other Methods

- `plot()` = visualize the results of sfHMM analysis.
- `do_all()` = conduct all the four steps and plot the results.
- `tdp()` = show the results in pseudo transition density plot.

## Customize Plots

The super class `Base_sfHMM` has class attributes that is passed to `matplotlib` every time you plot. You can change them by updating the dictionaries.

- `colors` ... Line colors of each data.
- `styles` ... Styles of plot. See `rcParams` of `matplotlib`.

For example, if you want to use different color for raw data and smaller font size in `sfHMM`, then run following codes:

```python
sfHMM.colors["raw data"] = "gold"
sfHMM.styles["font.size"] = 10
```

## Additional attributes and Methods in Multi_sfHMM
- `self[i]` ... `sfHMM` objects for `i`-th trace. The real list of objects is `_sf_list`. Iteration is defined on this list.
  
```python
msf.do_all()
msf[0].plot() # plot the first trace and its analysis results.
for sf in msf:
  sf.plot()   # plot all the traces in msf and their analysis results.
```

- `n_list` ... List of data lengths (property).
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

# Citation
 ...

# References
- Kalafut & Visscher
- hmmlearn
