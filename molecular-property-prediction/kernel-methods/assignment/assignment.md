# Assignment: Fitting with a reduced set of features

[Browning et al.](http://pubs.acs.org/doi/10.1021/acs.jpclett.7b00038) have illustrated how you can get better Kernel Ridge Regression models by intelligently selecting which points from a large dataset to use for training.
We are going to recreate their work in this assignment by making an improved model for the Highest Occupied Molecular Orbital (HOMO) energy using only 100 training points.

## Problem 1: Fitting a Coulomb Matrix

Load in our QM9 dataset and compute the [Coulomb matrix](https://singroup.github.io/dscribe/latest/tutorials/coulomb_matrix.html) for each entry. Set maximum number of atoms to be 40. (QM9 dataset encompasses molecules with up to nine “heavy” atoms from the range C, O, N and F. So the molecule having most atoms would simply be Nonane, C<sub>9</sub>H<sub>20</sub>.)

Make a test set of 1000 entries.

Fit a model with 100 parameters to predict the HOMO energy (`'homo'`) using [KernelRidge](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html) regression with an RBF kernel. Make sure to fit the $\alpha$ and $\gamma$ parameters using [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).

*HINT*: Use parameters varying between $10^{-6}$ and $10^0$ with a logarithmic spacing of 16 steps.

Repeat the fitting process 16 times using different samples of 100 entries. Plot three [histrograms](https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.pyplot.hist.html) of the optimized $\alpha$, $\gamma$ parameters and the MAE on a separate test set, respectively.

- Do the optimized model parameters change with different subsets?
- How large of a variation do you observe in the hyperparameters ($\alpha$, $\gamma$)?
- Can we use the same set of parameters for all subsets of 100 entries?

## Problem 2: Plot a learning curve

Fit the Coulomb Matrix model using randomly-selected training sets of 10, 100, and 1000 entries, each training for 4 replicates.

- Plot how the averaged model accuracy on a test set, training time and inference times change as a function of training set size.

## Problem 3: Optimize the training set. 

We are going to use a genetic algorithm to determine an optimized training set with $100$ entries.

First, separate off a "validation set" of 1000 entries from the training set that we will use to assess the performance of our specially-chosen training sets.

Now, implement a function that will accept a list of points from the training set by their index and produce the score of that model on the validation set using MAE. 
This function will be used by the genetic algorithm to score each selection of points.
It should fulfill the following signature:

```python
def evaluate_subset(points: list, model, train_data: pd.DataFrame, test_data: pd.DataFrame) -> float:
    """Test a subset of points
    
    Args:
        points: Which points from the train_data to sample
        model: Model to use for testing
        train_data: All available training points
        test_data: Data used to test the model
    Returns:
        MAE on the test set
    """
```

Next, run the genetic algorithm code provided at the end of this document (also includes an explaination of GAs) and:

- Plot the best score in the population change as a function of generation. *Hint*: Convert the `all_options` to a DataFrame and use Panda's [aggregation functions](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.aggregate.html).
- Plot the performance of your optimized model (which would be a single dot in this case) with the learning curve from problem 2. How does it compare?

## Appendix: Genetic Algorithm Code

Here is some simple code for a genetic algorithm to use. 
It defines methods for mixing best-performing solutions (crossover) and applying small random changes to individual members (mutation).

```python
from random import sample

def mutate(points: set, total: int, fraction: float = 0.1) -> set:
    """Mutate a set of points
    
    Mutates from selecting points randomly from the dataset
    
    Args:
        points: Set of points to be mutated
        total: Total number of samples to choose from in dataset
        fraction: How many points to re-select
    """
    
    # Remove the desired amount of points
    n_to_remove = int(len(points) * fraction)
    to_remove = sample(points, k=n_to_remove)
    new_points = points.difference(to_remove)
    
    # Add more points to the set
    available_choices = set(range(total)).difference(new_points)
    new_points.update(sample(available_choices, n_to_remove))
    
    return new_points

def crossover(parent_a: set, parent_b: set):
    """Perform a crossover operation
    
    Randomly chooses points from both parents
    
    Args:
        parent_a: One choice of points
        parent_b: Another choice of points
    Returns:
        A new set that combines both parents
    """
    
    # Combine all points from each parents
    options = parent_a.union(parent_b)
    
    # Pick randomly from the combined set
    return set(sample(options, len(parent_a)))
```

Genetic algorithms work by applying crossover and mutation to the best-performing entries of each generation, and repeating the process over many generations. 
The idea is that best-performing traits ("genes") are present in the later generations.

```python
# Defining options
n_generations = 50
pop_size = 8
dataset_size = 100

# Array in which to store all results
all_options = []

# Make an initial population
#  Creates sets where each have different entries pull from the full dataset
population = np.array([set(sample(range(len(train_data)), k=100)) for i in range(pop_size)])

# Loop over the generations
for gen in tqdm(range(n_generations), desc='generation'):
    # Score each member of the population
    scores = [
        evaluate_subset(list(s), gs, train_data, valid_data) for s 
        in population
    ]
    
    # Store the results in the history
    for i, s in enumerate(population):
        all_options.append({
            'generation': gen,
            'points': s,
            'score': scores[i]
        })
        
    # Sort scores and pick the best quarter
    ranks = np.argsort(scores)
    best_members = population[ranks[:pop_size // 4]]
    
    # Create new members by crossover and mutation
    new_population = []
    for i in range(pop_size):
        # Pick two parents at random
        parent_a, parent_b = sample(best_members.tolist(), 2)
        
        # Form a new member by crossover
        new_member = crossover(parent_a, parent_b)
        
        # Mutate it for good measure
        new_population.append(
            mutate(new_member, total=len(train_data))
        )
    
    # Replace population with new population
    population = np.array(new_population)
```
