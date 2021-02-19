# sfHMM
Step finding based HMM.
sfHMM is composed of four steps:
- `step_finding()`
- `denoising()`
- `gmmfit()`
- `hmmfit()`
Other functions:
- `plot()` = visualize the result of sfHMM analysis.
- `do_all()` = conduct all the four steps.
- `tdp()` = show the results in pseude transition density plot.

![ex1](animation.gif)

# Basic Usage
## Single trajectory
Trial with simulated data.
```python
from sfHMM import sfHMM, hmm_sampling
data = hmm_sampling()
sf = sfHMM(data)
sf.do_all()
```

## Multiple trajectories
```python
from sfHMM import Multi_sfHMM, hmm_sampling
msf = Multi_sfHMM()
for i in range(10):
    data = hmm_sampling()
    msf.append(data)
msf.do_all()
```

# References
