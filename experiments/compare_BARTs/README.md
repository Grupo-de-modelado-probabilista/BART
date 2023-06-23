# Comparation PyMC-BART vs R-BART

This folder contains the notebooks used to compare [PyMC-BART](https://github.com/pymc-devs/pymc-bart) with [R-BART](https://CRAN.R-project.org/package=BART), this comparation is based on the dataset of [Boston housing values](https://bookdown.org/egarpor/PM-UC3M/lm-ii-lab-boston.html). The notebooks contains the following:

- [01_Boston_housing_values](01_Boston_housing_values.ipynb): Runs the models for PyMC-BART (with Gamma or Gaussian distributions) and R-BART (with standard configuration), using all variables or the two most important variables.
- [02_Compare_BARTs_variables](02_Compare_BARTs_variables.ipynb): Contains the analysis of the previous notebook, these are Convergence plots, Effective Sample Size (ESS), $\hat{R}$, LOO, the comparison between predicted vs observed, RMSD and, MAD for each case.
- [03_Boston_CV](03_Boston_CV.ipynb): Runs the Cross-validation tests for the two packages and analyzes the results comparing predicted vs observed (in-sample and out-of-sample), RMSD and, MAD.
