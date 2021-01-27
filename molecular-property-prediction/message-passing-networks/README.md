# Message Passing Networks

This module explains how to build a Message-Passing Neural Network (MPNN) from scratch using Tensorflow.

The code in these libraries is close to what is used in early versions of [nfp](https://github.com/NREL/nfp).
If you are looking to do more state-of-the art deep learning on molecules, I would recommend learning a package
like [nfp](https://github.com/NREL/nfp), [megnet](https://github.com/materialsvirtuallab/megnet),
[schnetpack](https://schnetpack.readthedocs.io/en/stable/), or [deepchem](https://deepchem.io/).
The methods employed in our MPNN are simplified for learning purposes and intended to teach
what is going on in more sophisticated packages.

## Prerequistes

You should read the Keras documentation on [its functional API](https://keras.io/getting_started/intro_to_keras_for_engineers/#building-models-with-the-keras-functional-api)
before the "explain MPNN" notebook. We use Keras extensively there.

## Key Concepts

Through completing these modules, you should be able to:

- [ ] Convert materials training data into a form useful for 
- [ ] Understand what data loaders are and how they make training models efficient
- [ ] Build or modify a message-passing neural network
