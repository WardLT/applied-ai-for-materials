# Molecular Property Prediction

Machine learning for molecular properties is an area of machine learning that has seen [massive innovation in the past decade](https://www.nature.com/articles/s41467-020-18556-9). Learning about these topics is important for their impact on materials engineering and also for understanding how knowledge of the physics underlying a problem influence the design of machine learning algorithms. 

## Learning Objectives

The major goals of this module include:

- Highlighting roots of modern molecular machine learning in chemoinformatics. *What is QSAR and when should I use it?*
- Identifying an appropriate class of machine learning models for molecules. *When would you use graph convolution network for predicting molecular toxicity?*
- Explaining why (nearly) all neural networks for molecular properties are "message passing neural networks." *What sets MEGNet and Gilmer's MPNN apart?*
- Describing the key features, major methods and disadvantages of kernel-based machine learning. *What are the key features of FCHL?*
- Presenting history of machine learning for molecules. *How are SchNet and DTNN related?*

## Installation

We have a large Python environment for this particular module, which includes some dependencies that have installation difficulties.

### Linux

Before installing the environment, ensure you have installed a Fortran compiler and a LAPACK library. If running on Ubuntu, do so by calling:

```bash
sudo apt install gfortran liblapack-dev
```

### Mac OSX

You will need to install the GNU Compiler Collecton (GCC). The recommended route is to use [Homebrew]. Follow the installation instructions on [brew.sh](https://brew.sh/), then call

```bash
brew install gcc
```

It may also be possible to install GCC by adding ``- gcc`` to your ``environment.yml`` files.

### Windows

The recommended route to running these notebooks on Windows is to use Windows Subsystem for Linux (WSL). Follow the instructions from the [App Store](https://www.microsoft.com/en-us/p/ubuntu/9nblggh4msv6) to install Ubuntu via WSL, then install Anaconda and the required libraries from the Linux instructions.


### All

Once you have the Fortran compilers installed, build the environment using

```bash
conda env create --file environment.yml --force
```

Then activate with

```bash
conda activate molml
```
