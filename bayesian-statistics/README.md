# Bayesian Statistics for Physics Models

Bayesian Methods offer a statistically-robust route to assessing how confident one should be in the predictions of a machine learning model.
Knowing how to work with Bayesian methods is a excellent when you have small amounts of data, a good sense of the model which could describe your data,
or need for meaningful error bars on the properties.

## Learning Objectives

The learning objectives for our model are:

- Learning how to define the prior and likelihood functions. *How do you include the model form in the likehood function?*
- Understanding MCMC sampling. *What is burn-in and how do I achieve it?*
- Comparing the evidence for different models. *What is a Bayes factor and what is a large Bayes factor?*
- Using the posterior to estimate error bars on predictions. *Can you produce a 95% confidence interval?*

## Useful Papers

A few papers to read to gain a better understanding of this field include:

- [Paulson et al. IJES (2019)](https://doi.org/10.1016/j.ijengsci.2019.05.011): Tutorial and demonstration of using Bayesian methods to predict thermoelectric properties of materials
- [de Schoot et al. Nature Reviews (2021)](https://www.nature.com/articles/s43586-020-00001-2): Great tutorial review of methods behind Bayesian models

## Installation

Our environment is simple and can be installed entirely with Anaconda on most operating systems:

```bash
conda env create --file environment.yml --force
```

Then activate with

```bash
conda activate bayes
```
