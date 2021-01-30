# Machine Learning for Inorganic Materials

Inorganic materials are often complicated. Materials can be composed of many chemical elements and form complex structures from the electronic up to the visible scales. Accordingly, there are a rich set of tools for building machine learning models that capture the subtleties of composition and structure. Here, we describe how to use them.

## Learning Objectives

The major goals of this module include:

- Compute feature based on composition and crystal structure using matminer. _What are featurizers and how do I use them to build an ML model?_
- Recognizing groups of similar materials within a dataset. _How do I perform clustering?_
- Incorporating or ignoring processing conditions or structural data from a dataset. _How can I include processing conditions into a model? Why would I knowingly ignore factors that effect material's behavior?_

## Installation

Build the environment with Anaconda using

```bash
conda env create --file environment.yml --force
```

Then activate with

```bash
conda activate matml
```

## Contributing

I am not an expert in learning from the microstructure of materials but believe we need a module on that. If you know how to work with microstructural data, please reach out!