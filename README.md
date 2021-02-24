# sfHMM
Step finding based HMM.

![ex1](animation.gif)

# Basic Usage

First import `sfHMM` class and `Multi_sfHMM` class.

```python
from sfHMM import sfHMM, Multi_sfHMM
```

You can also use simulated data with `hmm_sampling` function.

```python
from sfHMM import sfHMM
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
sfHMM class and Multi_sfHMM class have a similar structure (both have super class `Base_sfHMM`) so that they have many attributes and methods in common.

## Parameters
All the parameters are optional.
- `sg0` ... The parameter used in denoising process.
- `psf` ... The parameter used in step finding.
- `krange` ... Range of the number of hidden states.
- `model` ... Distribution of signal. Gaussian and Poissonian are supported now.
- `name` ... Name of the object.

## Attributes and Methods

sfHMM is composed of four steps. Attributes are sequencially added to the object.

1. `step_finding()`
   
    Step finding by likelihood maximization.
    
    $$\ln L=k\ln \frac{p_{sf}}{1-p_{sf}}-\frac{N}{2}\ln \hat \sigma^2 + const.$$

    *Attributes*

  - `step` ... `GaussStep` or `PoissonStep` object, defined in `step` module or `stepc` extendsion module. This object has following attributes:
    - `fit` ... Fitting result.
    - `n_step` ... The number of steps (region between two signal change points).
    - `step_list` ... List of signal change points.
    - `mu_list` ... list of mean values of each step.
    - `len_list` ... list of step lengths (`step_list[i+1] - step_list[i]`).
    - `step_size_list` ... list of signal change (`mu_list[i+1] - mu_list[i]`). 

2. `denoising()`

    The standard deviation of noise is cut off to `sg0`.
  
    *Attributes*

  - `data_fil` ... Data after denoised.

3. `gmmfit()`
   
    *Attributes*
    
  - `gmm_opt` ... The optimal Gaussian mixture model in the form of `GMM1` object, defined in `gmm` module. This object has following attributes:

    - `wt` ... Weights.
    - `mu` ... Means.
    - `sg` ... Standard deviations.
    - `aic` ... Akaike information criterion.
    - `bic` ... Bayes information criterion.

  - `gmm` ... `GMMn` object defined in `gmm` module. This object contains `GMM1` objects with different number of states. You can get `n`-state model by indexing like `gmm[n]`.
  - `n_components` ... The number of states.
  - `states` ... State sequence, predicted only with the results in 1-3.

4. `hmmfit()`

    *Attributes*

  - `means_` ... Mean values. See `hmmlearn`.
  - `covars_` ... Covariances. See `hmmlearn`.
  - `transmat_` ... Transition probability matrix. See `hmmlearn`.
  - `startprob_` ... Starting probability. See `hmmlearn`.
  - `states` ... State sequence, predicted by HMM.
  - `viterbi` ... Viterbi pass of optimized HMM parameters.

## Other Methods

- `plot()` = visualize the results of sfHMM analysis.
- `do_all()` = conduct all the four steps and plot the results.
- `tdp()` = show the results in pseudo transition density plot.

## Customize Preferences

The super class `Base_sfHMM` has class attributes that is passed to `matplotlib` every time you plot. You can change them by updating the dictionaries.

- `colors` ... Line colors of each data.
- `styles` ... See `rcParams` of `matplotlib`.

For example, if you want to use different color for raw data and smaller font size in `sfHMM`, then run following codes:

```python
sfHMM.colors["raw data"] = "gold"
sfHMM.styles["font.size"] *= 0.7
```

# References
- Kalafut & Visscher
- hmmlearn
