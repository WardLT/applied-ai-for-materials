# Applied AI for Materials

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/WardLT/applied-ai-for-materials/HEAD)

This repository is a collection of notebooks and other materials used for the "Applied Artificial Intelligence for Materials Science and Engineering" course at The University of Chicago. It is very much a work in progress, so expect large changes in content and organization in the next few months.

## Using this Repository

Each subject area is organized into its own directory with notebooks, lecture notes and Python environment. They will generally be arranged as having multiple subfolders that focus on a specific subtopic. 

### Working from Binder

You can run all of the notebooks using [Binder](https://jupyter.org/binder). 
Simply click [this link](https://mybinder.org/v2/gh/WardLT/applied-ai-for-materials/HEAD) to launch a copy of the repository on cloud resources. 
**It will not save your changes**, but you can use it for exploring the notebooks and - if you download notebooks to your computer - completing the practical assignments.

### Local Installation

Either download the repository as a ZIP file or [clone it using git](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository). 
The computational environments needed for each module are described in the README for that module. 

If you install using git, you can update the course materials through calling `git pull` from within the directory. 
Otherwise, you will need to re-download the repository to receive updates.

## Course Layout

The course is broken out in to the following modules (some of which are TBD):

- Effectively using Python for data science: Working quickly and reproducibly with Anaconda and Jupyter 
  - Topics: Managing Python environments PyData Stack, Jupyter Notebooks
- The Materials Data Ecosystem: Infrastructure for finding, using and sharing data
  - Topics: Databases, laboratory information systems, image publication
- [Molecular property prediction](./molecular-property-prediction): How physics, chemistry and machine learning fit together
  - Topics: Kernel and graph methods, chemoinformatics
- [Supervised learning for inorganics](./ml-for-inorganic-materials): The importance of microstructure, composition and processing 
  - Topics: Representations for inorganic materials, coping with processing variation
- Generative methods for materials: Augmenting human creativity with AI
  - Topics: Generative Adversarial Networks, Reinforcement Learning, Autoencoders
- [Bayesian parameter estimation](./bayesian-statistics): Achieving greater certainty in using physics-based models
  - Topics: UQ for CALPHAD, model selection, learning from noisy data
- Computer vision and characterization: Better microscopy through intelligent software
  - Topics: Image segmentation, classification and noise reduction
- [Optimal experimental design](./optimal-experimental-design): Accelerate design optimization with software-assisted planning
  - Topics: Bayesian Optimization, active learning

### Module Layout

Each course module contains its own Python environment, instruction notebooks and assignments for evaluating comprehension.

The installation Python environments are described in the README for each module. 

The answer keys for the comprehension assignments are encrypted and can be decrypted with the [provided script](./bin/). Message me to get the passphrase or notebooks.

Some modules are broken into a few different subdirectories with their own notebooks and assignments.

## Related Resources

Further resources available for this course are available elsewhere:

- [Syllabus](https://1drv.ms/b/s!AswJEkleh18Ah5pEQM8zCT0uVf6Stg?e=B0ZGJU): Most recent syllabus for the associated course.
- [Slides](https://1drv.ms/u/s!AswJEkleh18Ah49dGc89htZMDm65cw?e=3GMRig): My working copy of the slides, available in PDF and PPTX format from OneDrive.
- [Lecture Recordings](https://www.youtube.com/watch?v=6ofUaBAIF0U&list=PLEjVJ0F11Nmn8Rc0OblMtzFfOGI1rAeIf): From Winter 21 are on [YouTube](https://www.youtube.com/watch?v=6ofUaBAIF0U&list=PLEjVJ0F11Nmn8Rc0OblMtzFfOGI1rAeIf)
