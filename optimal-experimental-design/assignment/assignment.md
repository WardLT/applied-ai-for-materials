# Optimal Experimental Design for Conformer Search

Conformers define the different structures with the same bonding graph but different coordinates. 
Finding the lowest-energy conformation is a common task in molecular modeling, and one that often requires significant time to solve.
This homework is designed to explore the use of optimal experimental design techniques, which have [recently](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-019-0354-7) [emerged](https://pubs.acs.org/doi/full/10.1021/acs.jctc.0c00648) as a potential tool for accelerating the conformer search.

*Note*: Before submitting, please run your notebook without warnings being printed by running the following code in an early cell in your notebook.

```python
import warnings
warnings.simplefilter('ignore')
```

The Gaussian Process model we use produces many warnings that can be safely ignored.

## Problem 1: Active Learning for $n$-butane

We will start off with optimizing a simple molecule, $n$-butane, which has two dihedral angles of interest: $\phi_1$ - the bond in the center of the molecule and $\phi_2$ a bond for one of the alkyl groups at the end.

#### Setup: Loading Conformer Data

The [background notebook](./background.ipynb) shows how we can generate data for the conformers of a simple molecule, *n*-butane.

Load the data from that search with Pandas:

```python
butane = pd.read_csv('data/n-butane.csv')
```

Then, generate a subset of the data where we fix `phi_2` to be constant so that the global energy is minimized (as in Example 1).



### Part A: Learning a Potential Enery Surface

Our first ingredient is to learn a function that can predict energy as a function of dihedral angles.

Dihedral angles are periodic (e.g., 0 and 360 are the same) and we can create a kernel that reflects that by using a periodic kernel:

$K(x, y) = \sum_i \exp(-2 \frac{\sin^2(\pi (x_i - y_i)^2/p)}{l^2})$

We can use this custom kernel with Scikit-Learn's GuassianProcessRegressor by writing it a function to use with the [`PairwiseKernel` class](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.PairwiseKernel.html), 
which accepts custom kernels:

```python
def elementwise_expsine_kernel(x, y, gamma=10, p=360):
    """Compute the expoonential sine kernel
    
    Args:
        x, y: Coordinates to be compared
        gamma: Length scale of the kernel
        p: Periodicity of the kernel
    Returns:
        Kernel metric
    """
    
    # Compute the distances between the two points
    dists = np.subtract(x, y)
    
    # Compute the sine with a periodicity of p
    sine_dists = np.sin(np.pi * dists / p)
    
    # Return exponential of the squared kernel
    return np.sum(np.exp(-2 * np.power(sine_dists, 2) / gamma ** 2), axis=-1)

gpr = GaussianProcessRegressor(
    kernel=kernels.PairwiseKernel(metric=elementwise_expsine_kernel),
    # Starts using different guesses for parameters to achieve better fitting
    n_restarts_optimizer=4  
)
```

*Note*: This is different than the [`ExponentialSineKernel` class](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.ExpSineSquared.html) from scikit-learn, which sums the distances over all coordinates _first_ before computing the sine function. In contrast, our function computes the sine-distance over each point and then evaluates the sine function.

Take a random subset of 2 points from `butane_1d.` Use those points to fit a model with the Sine kernel introduced above and an RBF kernel (find how in scikit-learn documentation). Note that you should add ` n_restarts_optimizer=16` to the arguments for the RBF model to be able to fit our data well.

- Plot the mean and standard deviations of the predictions as a function of `phi_1`. Which function is a better approximator of the energy? (You may want to run the fitting multiple times to avoid basing your conclusion on haphazard, while you only need to show two plots in the notebook, one for each kernel, respectively).

Next, go back to the original 2D dataset with no constraint on `phi_2`. Train a series of models with each of the two kernels, with randomly-selected subsets of 3, 10, 30, and 100 points (in all there should be 2 x 4 models). Measure the error on the full dataset. 

- Plot the error in the Sine kernel model and the RBF kernel model as a function of training set size. Which model performs better? Explain why.

### Part B: Bayesian Optimization for Conformer Search

Now that we have a model, our next step is to use Bayesian optimization to guide our search for the conformers.
We will use [modAL's `BayesianOptimizer` tool](https://modal-python.readthedocs.io/en/latest/content/models/BayesianOptimizer.html).
As highlighted in the [example for Bayesian Optimization](https://modal-python.readthedocs.io/en/latest/content/examples/bayesian_optimization.html),
you use modAL by first defining an optimizer with an initial training set, GPR model and an acquisition function:

```python
initial_data = butane.sample(4)
input_cols = ['phi1', 'phi2']
optimizer = BayesianOptimizer(
    estimator=gpr,
    X_training=initial_data[input_cols].values,  # .values removes Panda's indices
    # negative because modAL is designed to maximize a function
    y_training=-initial_data['energy'].values,  
    query_strategy=max_EI
)
```

Once you have it a step of the process of "pick next point then add to training data" is

```python
chosen_inds, chosen_coords = optimizer.query(butane[input_cols].values)
optimizer.teach(
    butane[input_cols].iloc[chosen_inds].values, 
    -butane['energy'].iloc[chosen_inds].values
)
```

Run an active learning search where you start with 4 points that were determined randomly and then select 16 points using 
[maximum expected improvment (`max_EI`)](https://modal-python.readthedocs.io/en/latest/content/query_strategies/Acquisition-functions.html#expected-improvement)
and [uncertainty sampling `max_std_sampling`](https://modal-python.readthedocs.io/en/latest/content/query_strategies/Disagreement-sampling.html#standard-deviation-sampling).

- Plot the lowest-energy conformer as a function of step for each strategy. Which query strategy finds the lowest-energy conformer?
    - **HINT**: Use `np.minimum.accumulate` to quickly get the minimum at each step
- Measure the prediction performance of the model on the full butane dataset. Which query strategy creates the best model?
    - **HINT**: Use `optimizer.predict` to invoke the model used by an optimizer
- Explain the difference between the two query strategies and why their performances are different.


## Problem 2: Optimizing cysteine

Cysteine has 5 adjustable bondes, which makes it a much more difficult optimization problem. 
The issue with having many bonds is that sampling the conformer space becomes difficult.
For butane, we used 33 samples per axis which leads to $33^2 = 1089$ points to evaluate at each step, 
which is very easy to evaluate with a GPR. 
The same sampling density for cysteine would be $33^5 = 39135393$, which would be too time consuming.
Rather, we can use a global optimizer to suggest which points to sample.

#### Setup: Global optimization with for cysteine structure

Our [background notebook](./background.ipynb) and [utility functions](./confutils.py) define the data
and functions we need to quickly evaluate the energy of a cysteine molecule given the dihedral angles.

First, load them in using pickle

```python
import pickle as pkl

# Molecular structure of cysteine
with open('data/cysteine-atoms.pkl', 'rb') as fp:
    cysteine = pkl.load(fp)

# Definitions for the dihedral angles of cysteine
with open('data/dihedrals.pkl', 'rb') as fp:
    dihedrals = pkl.load(fp)
```

Now, define a function to evaluate the energy given new dihedral angles

```python
from confutils import set_dihedrals_and_relax

def evaluate_energy(angles):
    """Compute the energy of a cysteine molecule given dihedral angles
    
    Args:
        angles: List of dihedral angles
    Returns:
        energy of the structure
    """
    return set_dihedrals_and_relax(
        cysteine,
        zip(angles, dihedrals)
    )
```

We use a simple "generate random points" and minimize them strategy. 
This strategy generates a series of random points, uses a local optimizer ([Nelder-Mead](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method)) to find the nearby minimum in energy,
and then returns a list of all of the points sampled for the optimizer's consideration.
We use the optimizer's model rather than the `evaluate_energy` function so that this function is very fast.

```python
from scipy.optimize import minimize
import numpy as np

def get_search_space(optimizer: BayesianOptimizer, n_samples: int = 32):
    """Generate many samples by attempting to find the minima using a multi-start local optimizer
    
    Args: 
        optimizer: Optimizer being used to perform Bayesian optimization
        n_samples: Number of initial starts of the optimizer to use. 
            Will return all points sampled by the optimizer
    Returns:
        List of points to be considered
    """
    
    # Generate random starting points
    init_points = np.random.uniform(0, 360, size=(n_samples, 5))
    
    # Use local optimization to find the minima near these
    points_sampled = []  # Will hold all samples tested by the optimizer
    for init_point in init_points:
        minimize(
            # Define the function to be optimized
            #  The optimizer requires a 2D input and returns the negative energy
            #  We make our inputs 2D and compute the negative energy with a lambda function
            lambda x: -optimizer.predict([x]),  # Model predicts the negative energy and requires a 2D array,
            init_point,  # Initial guess
            method='nelder-mead',  # A derivative-free optimizer
             # Stores the points sampled by the optimizer at each step in "points_sampled"
            callback=points_sampled.append
        )
    
    # Combine the results from the optimizer with the initial points sampled
    all_points = np.vstack([
        init_points,
        *points_sampled
    ])
    return all_points
```

### Part A: Building a global optimization algorithm

Generate a set of 4 initial points where the values for each of the 5 dihedrals are chosen randomly from between 0 and 360 degrees (*HINT* use `np.random.uniform`).

Compute their energy and use them to create a Bayesian optimizer with the EI query strategy.

Write a loop that follows the following procedure:

1. Use the `get_search_space` function to create a series of points to be sampled
2. Select the best points to query with the optimizer
3. Evaluate the energy of the best points using `evaluate_energy` so that the results are more accurate, with reasonable sacrifice in efficiency
4. Add the best points and new energies to the training set of the optimizer

Run the loop 32 times to try to find the lowest energy conformer. This may take a half hour.

- Plot the energy of each sampled point as a function of step. Describe the energies of the conformers compared to that of the one produced by OpenBabel ($E = -19641.7084$ Ha)
