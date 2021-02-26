# Assignment: Learning the Thermodynamics of Liquid Hafnium

Our assignment is based on the work [Paulson et al. (2019)](https://linkinghub.elsevier.com/retrieve/pii/S0020722518314721) that learned thermodynamic models for the 
phases of Hafnium (Hf). 
The goal of this assignment is to learn models for the properties of liquid Hf and to explore a few aspects of parameter estimation that were not covered in the example notebooks:

- Enforcing consistancy between different model
- Making models less outlier sensitive
- Learning dataset weights

## Setup: Loading the the data

We provide two of the datasets from Paulson2019 to use in this assignment. Load them into memory using Pandas:

```python
h_data = pd.read_csv('data/Cag2008.csv', delim_whitespace=True)
cp_data = pd.read_csv('data/Kor2005.csv', delim_whitespace=True)
```

For reference, the unit of temperature $T$ is K, specific heat capacity $C_p$ is J mol<sup>-1</sup> K<sup>-1</sup>, and enthalpy $H$ is J mol<sup>-1</sup>.
The datasets contain the value of the property and an uncertainty for the measurement
in the column prefixed with `sigma_`.

## Problem 1: Building Prior Distributions

*HINT*: Try copy the text below including the LATEX expressions (enclosed by a pair of '$') and paste into a markdown block in your notebook, in order to render nice math formulas.

We are going to fit the parameters for a model that expresses the heat capacity ($C_p$) as a linear model:

$C_p(T) = c_1 + c_2 T$

Thermodyanmics dictates that the enthalpy ($H$) is defined by:

$H(T) = \int_0^T C_p(T) dT = c_0 + c_1 T + c_2 T^2/2$

### Prior Distributions for Model Parameters

The thermodynamic model has three parameters (denoted $c_0$, $c_1$ and $c_2$) with two shared between the proeprties ($c_1$, $c_2$).

Our first step is to define the prior distribution for these parameters. Start by fitting an estimate for 
$c_1$ and $c_2$ by fitting a linear model to the $C_p$ data using [robust regression from SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.siegelslopes.html).

- Make a plot of specific heat capacity $C_p$ against temperature $T$. Show both the data points and the fitted curve.

Next, compute an estimate for $c_0$ by subtracting the linear and quadratic terms from each point in `h_data` given your newly-fit $c_1$ and $c_2$ values.

```python
residual = h_data['H'] - c_1 * h_data['T'] - c_2 * h_data['T'] ** 2 / 2
```

Compute the median of `residual` to estimate $c_0$.

- Make a plot of enthalpy $H$ against temperature $T$. Show both the data points and the fitted curve.

Construct prior distributions using `scipy`'s [Normal distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html#scipy.stats.norm) 
setting the location to be the values of the parameter and the scale to be 50% of that parameter value.

- Make 3 plots showing the prior distribution (probability density) of 3 parameters.
- What is the **log** prior probability of $c_0$ = 5000, $c_1$ = 35, and $c_2$ = 0.005?

### Prior Distribution for Dataset Hyperparameters

Paulson et al. used a "dataset hyperparameter" that adjusts the uncertainty values of each dataset ($\alpha_i$) used in fitting.
We will use an "uniformed prior" that assumes this value can be anywhere between 0 and 10 with equal likelihood. 

Use a [uniform prior distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.uniform.html#scipy.stats.uniform) to define a 
dataset hyperparameter for the $C_p$ dataset ($\alpha_{C_p}$) and $H$ dataset ($\alpha_H$).

- What is the prior probability of $\alpha_{C_p} = 5$ and $\alpha_H = 11$?

## Problem 2: Defining the Likelihood Function

We now must define a function to compute the posterior probability for certain set of parameters given the prior distribution ($P(\theta|M)$) and the likelihood of the observed data ($P(D|\theta,M)$).

$P(\theta|D,M) = P(D|\theta,M)P(\theta|M)$

To recap, our parameters include 3 model parameters ($c_0$, $c_1$, $c_2$) and two dataset hyperparameters ($\alpha_{C_p}$ and $\alpha_H$).
We have defined prior distributions for each parameter and can use them to calculate the prior probability in the above equation.
Our task now is to define the likelihood of the data.

The likelihood of observing a single point can be computed from the uncertainity of the point ($\sigma_i$), the prediction from the model $M(T_i|\theta)$, and the weight of our uncertainty ($\alpha$):

$P(D_i|\theta,M) = \mathcal{N}(D_i|M(T_i|\theta), \sigma_i/\alpha_D)$

where $\mathcal{N}(x|y,\sigma)$ is the probability distribution function for the normal distribution with mean $y$ and standard deviation $\sigma$.

We assume that the probability of each data point is independent, so you can compute the probability of the whole dataset as:

$P(D|\theta,M) = \prod_i P(D_i|\theta,M) = \sum_i \log P(D_i|theta,M)$

Build the posterior function by completing the methods in this class:

```python
class Posterior:
    """Posterior probability of liquid H thermodynamic models
    
    Assumes parameters are in the order: c_0, c_1, c_2, alpha_cp, alpha_h
    """
    
    def __init__(self, cp_data, h_data, priors):
        """Initialize the class
        
        Args:
            cp_data: Heat capacity data
            h_data: Enthalpy data
            prior: Prior distributions for each parameter
        """
        self.cp_data = cp_data
        self.h_data = h_data
        self.priors = priors
        
    def logprob_cp_data(self, params):
        """Compute the log probability of observing the C_p data
        
        Args:
            params ([float]): Chosen parameters
        Returns:
            Log-likelihood of observing the data
        """
        
        ## FILL ME IN!
    
    def logprob_h_data(self, params):
        """Compute the log probability of observing the H data
        
        Args:
            params ([float]): Chosen parameters
        Returns:
            Log-likelihood of observing the data
        """
        
        ## FILL ME IN!
        
    def logprob_prior(self, params):
        """Compute the log probability of the parameters given the prior distributions
        
        Args:
            params ([float]): Chosen parameters
        Returns:
            Log-likelihood of the priors
        """
        
        ## FILL ME IN!
    
    def __call__(self, params):
        """Compute the log posterior probability given the paramers
        
        Args:
            params ([float]): Chosen parameters
        Returns:
            Log-likelihood of the posterior
        """
        
        return self.logprob_cp_data(params) + self.logprob_prior(params) + self.logprob_h_data(params)
```

As a consistency check, the log posterior of [5000, 35, 0.005, 1, 1] should be -1222.21.

**HINT**: Make the function and call it with the following lines
```python

ln_posterior = Posterior(cp_data, h_data, priors)
print(ln_posterior([5000, 35, 0.005, 1, 1]))
```

## Problem 3: Sampling the Posterior

Our next tasks is to make samples from the posterior.

First, sample 128 points from the prior distribution for each of the model parameters and store them as a `128 x 5` array.

Next, create a sampler using Kombine with 128 walkers using the posterior function you made in Problem 2. Then burn in the sampler using the initial points sampled from the prior as a starting point.

- Make a burn-in curve of acceptance probability against steps. 
- How many steps did it take for the sampler to converge? (Please do **NOT** simply read from the plot. Try to get more accurate statistics out of the sampler objects. Same for next question.)
- What was the acceptance probability of the last 5 steps? (If it is not ~0.5, re-run the burn in)

Run the convered sampler for 64 more steps and then draw all of the samples from the posterior. 

- What is the mean and the 95% confidence intervals of $c_2$ and $\alpha_H$? Compute the confidence intervals using the 2.5 and 97.5 percentiles with numpy's [percentile function](https://numpy.org/doc/stable/reference/generated/numpy.percentile.html).

- Plot the mean and 95% confidence intervals (*Hint*: `fill_between`) of the heat capacity ($C_p(T)$) as a function of temperature between 2500 and 5000K.

- Does the model consistently over- or underestimate the heat capacity at intermediate temperatures (3500-4000K)? Which data points do you think are leading to that systematic error?

## Problem 4: Outlier Sensitivity

As shown in Problem 3, our model fitting is currently sensitive to individual points. We want to make our model fitting more robust to outliers by reducing the penality for large errors.
We can do this by switching from using a normal distribution when defining the likelihood functions to a better alternative.

Repeat Problem 2 and 3, but changing the normal distribution used in the `logprob_cp_data` and `logprob_h_data` functions to [a Student t distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html#scipy.stats.t) with a `df` of 2.1.

-Re-measure the mean and confidence intervals of $c_2$ and re-plot the mean and confidence intervals of $C_p$ as a function of temperature

- Is $c_2$ larger or smaller?
- Does the model have the same systematic errors or are they smaller? Describe why the change has occured?
