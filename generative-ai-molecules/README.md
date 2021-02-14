# Generative Methods for Molecular Design

Generative methods are machine learning approaches that create data designed to fit a target distribution. 
For example, it could be to generate a set of molecules that have similar properties or a policy that generates
ideas for how to alter a molecule so that its properties improve.
This modules introduces a suite such approaches to generating new molecules with target properties.

## Learning Objectives

The learning objectives for this module are:

- Explaining the theory and practice behind reinforcement learning. *How do you define an environment? What is an agent? How do you learn policies with Policy Gradient?"
- [...] more to come!

## Useful Papers

A few papers to read to gain a better understanding of this field include:

- [You et al. NeurIPS (2018)](https://arxiv.org/abs/1806.02473): Using PPO and graph-convolution networks to design molecules
- [Zhou et al. Sci Rep (2019)](https://www.nature.com/articles/s41598-019-47148-x): Using Q-learning to produce molecules with target properties
- Chris Yoon's [blog](https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce-f887949bd63) [posts](https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f) on policy gradient: Simple illustrations on how policy gradient methods work

## Installation

Our environment is simple and can be installed entirely with Anaconda on most operating systems:

```bash
conda env create --file environment.yml --force
```

Then activate with

```bash
conda activate molgen
```
