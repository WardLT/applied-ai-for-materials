# Optimal Experimental Design

Optimal Experimental Design methods help you design experiments as you perform them by recommending
which experiments will be the most valuable given previous results.
These notebooks introduce the fundamentals and tools used to perform optimizations.

## Learning Objectives

The learning objectives for this module are:

- Explaining the purpose and design of acquistion functions. *How can I vary degree of exploration with Upper Confidence Bound?*
- Identifying the best types of algorithms for different problems. *What approach should I choose for a high-throughput experiments?*
- Developing experiments to test active learning systems. *What do I use as a baseline?*

## Useful Papers

A few papers to read to gain a better understanding of this field include:

- [Lookman et al. *npj Comp Mat* (2019)](https://www.nature.com/articles/s41524-019-0153-8): A review on how active learning has been used for materials design
- [RocketSled](https://hackingmaterials.lbl.gov/rocketsled/): Software package for running [optimal experimental design on HPC](https://iopscience.iop.org/article/10.1088/2515-7639/ab0c3d)
- [Grizou et al. *Sci Adv* (2020)](https://advances.sciencemag.org/content/6/5/eaay4237): Demonstration of using robotics and curiosity-based experimental design methods
- [Seko et al. *PRB* (2014)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.89.054303): An early and clear example of Bayesian optimization ("kriging") used for materials design
- [Sivaraman et al. *npj Comp Mat* (2020)](https://www.nature.com/articles/s41524-020-00367-7): Fitting interatomic potentials with active learning
- [Jacobsen et al. *PRL* (2018)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.026102): Active learning for atomic structure optimization


## Installation

Our environment is simple and can be installed entirely with Anaconda on most operating systems:

```bash
conda env create --file environment.yml --force
```

Then activate with

```bash
conda activate oed
```
