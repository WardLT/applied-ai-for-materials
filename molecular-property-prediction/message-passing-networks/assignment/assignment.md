# Assignment: Message-Passing Neural Networks 

This assignment walks through how to build Message-Passing Neural Network models for two different molecular properties: atomization energy and band gap energy. We also will see how the design of a network can have a large influence on the accuracy of a machine learning model.

*Hint*: `Shift`+`Tab` in Jupyter brings up documentation for a function. You may be reading much documentation for functions during this exercise. You could also refer to the official documentation of deep learning modules on [TensorFlow](https://www.tensorflow.org/guide).

## Question 1: Making data loaders.

Our first step to create data loaders for the training, validation, and test data from the [datasets provided with this repository](../datasets).

Use the `make_loader` function the `mpnn` library to create a data loader for each, with `batch_size` of 32.
You can change the output property for each loader and a few other key settings:

- It's generally a rule of thumb to enable "shuffle" when creating the training data loader. Why is that? Which parameter do you set?
- Sample a few batches from the training data? What is the average of the output variable?

*Hint*: Investigate the documentation for `mpnn.data.make_data_loader` using Jupyter hotkey mentioned above or look into the source code in the local `mpnn` directory.

## Question 2: Training a network for real

*Step 1*: We need to make a few modifications to the `make_model` function from the [MPNN example notebook](../2_explain-message-passing-networks.ipynb) to build a network that will get noteworthy training accuracies:

1. Add a parameter that allows you change to which readout function is used for the `Readout` layer.

1. Add a [Dense](https://keras.io/api/layers/core_layers/dense/) layer with 'relu' activation and 32 units between the readout and the output layer.

1. Add "scaling layer" from `mpnn.layers` (i.e., `from mpnn.layers import Scaling`) after the current output layer, use the output of the scaling layer as the output of the model.
   Pass your Scaling layer a name of "scaling" (i.e., `Scaling(name='scaling')`)

*Step 2*: build a model with 64 features, 2 message passing layers and a "sum" readout function.

Once complete, your `model.summary()` should produce

```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
atom (InputLayer)               [(None, 1)]          0                                            
__________________________________________________________________________________________________
bond (InputLayer)               [(None, 1)]          0                                            
__________________________________________________________________________________________________
squeeze (Lambda)                (None,)              0           node_graph_indices[0][0]         
                                                                 atom[0][0]                       
                                                                 bond[0][0]                       
__________________________________________________________________________________________________
atom_embedding (Embedding)      (None, 64)           384         squeeze[1][0]                    
__________________________________________________________________________________________________
bond_embedding (Embedding)      (None, 64)           256         squeeze[2][0]                    
__________________________________________________________________________________________________
connectivity (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
message_passing (MessagePassing (None, 64)           0           atom_embedding[0][0]             
                                                                 bond_embedding[0][0]             
                                                                 connectivity[0][0]               
                                                                 message_passing[0][0]            
                                                                 bond_embedding[0][0]             
                                                                 connectivity[0][0]               
__________________________________________________________________________________________________
node_graph_indices (InputLayer) [(None, 1)]          0                                            
__________________________________________________________________________________________________
readout (Readout)               (None, 64)           0           message_passing[1][0]            
                                                                 squeeze[0][0]                    
__________________________________________________________________________________________________
dense (Dense)                   (None, 32)           2080        readout[0][0]                    
__________________________________________________________________________________________________
output (Dense)                  (None, 1)            33          dense[0][0]                      
__________________________________________________________________________________________________
scaling (Scaling)               (None, 1)            2           output[0][0]                     
==================================================================================================
Total params: 2,755
Trainable params: 2,755
Non-trainable params: 0
__________________________________________________________________________________________________
```

*Step 3*: Pre-seed the value of the scaling layer with the mean and value of a batch from the training dataset.

Set the weights for the layer by calling:

```python
scale = model.get_layer('scaling')
scale.mean = outputs.numpy().mean()
scale.std = outputs.numpy().std()
```

Completing Steps 1-3 will give your model more flexibility (with adding the additional dense layers) and ensure it predicts 
reasonable values for the output (with adding the scale layer)

*Step 4*: Fit the model using an [early stopping callback.](https://keras.io/api/callbacks/early_stopping/)

Use both the training and validation loaders. Run for 128 epochs with a batch size of 32 with an early stopping patience of 8 epochs. 
Make sure to use `restore_best_weights=True` in your callback.

Now, repeat steps 2-4 with 0, 1, 4 and 8 message passing layers (please use `verbose=False` when fitting, so the notebooks aren't huge):

- Plot the change in the best loss on the test set as a function of number of layers. Do you observe a continual increase with the number of layers?

## Question 3: Evaluate different readout layers

All of our previous questions used the atomization energy as an output. 
Atomization energy generally increases with the size of the molecule because there are more bonds to break.
So, our choice of summation for the readout function is a good one: more atoms means bigger sums and larger predicted values.

Band gaps, on the other hand, do not have such a scaling behavior.
They are instead due to a property of two specific atoms in the molecule: one that has a high-energy occupied orbital and
a second that has a low-energy unoccupied orbital.

In this problem, we will explore the effect of changing the "Readout function."

Train a total of 4 models each with 4 message passing steps and 64 features but varying whether we use `u0_atom` or `bandgap` to fit the model
and whether we use "sum" or "max" as a readout function:

- Explain how the max readout function produces features that do not scale with molecule size.
- Plot the predicted vs actual values for each of the networks on the test set. Which readout performs better for each output?
