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

We have a large Python environment for this particular module. It will require a system with a Fortran compiler and, so, we recommend using either Linux, Mac OS (following [these notes](http://www.qmlcode.org/installation.html#note-on-apple-mac-support)) or the Windows Subsystem for Linux.

Before installing the environment, ensure you have installed a Fortran compiler and a LAPACK library. If running on Ubuntu, do so by calling:

```bash
sudo apt install gfortran liblapack-dev
```

Install the environment using

```bash
conda env create --file environment.yml --force
```

Then activate with

```bash
conda activate molml
```
