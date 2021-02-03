
*Hint*: `Shift`+`Tab` in Jupyter brings up documentation for a function. You may be reading much documentation for functions during this exercise.

## Question 1: Making data loaders.

Our first step to create data loaders for the training, validation, and test data from the [datasets provided with this repository](../datasets).

Use the `make_loader` function the `mpnn` library to create a data loader for each.
You can change the output property for each loader and a few other key settings:

- Why should you use a "shuffle" value for the training data loader? Which parameter do you set?
- Sample a few batches from the training data? What is the average of the output variable?

## Question 2: Plotting atomic features

Build a model creation function that allows you to set the number of features, numbers of message layers and which readout function is used for the `Readout` layer.

Create a model with 0 message passing steps and 2 features. 

Train the model using the atomization energy (`u0_atom`) with a batch size of 64 for 8 epochs with the training data. (*Hint*: Investigate the documentation for `mpnn.data.make_data_loader`.)

Build a model to output the representation for each atom using second model using the trained model (here called `model`):

```python
# Get the readout layer
readout = model.get_layer('readout')

# Make a model that takes the molecule as inputs (same as the current model)
#  but outputs the inputs to the readout layer (which are the atomic features)
rep_model = Model(inputs=model.inputs, outputs=readout.input)
```

Use it to output the atomic representations for each model in the training set using the `predict` function of rep_model. 

Repeat the "train then output representation" process with networks that have 1 and 2 message passing layers:

- Make a scatter plot of the atomic features from all three models. What are the clusters? Why do they blur with more message passing layers?

## Question 3: Training a network for real

We now are going to train a network and actually care about how accurate it is.

*Step 1*: Add a [Dense](https://keras.io/api/layers/core_layers/dense/) layer with 'relu' activation and 32 units, and "scaling layer" from `mpnn.layers` (i.e., `from mpnn.layers import Scaling`) to the output of the network from your network in Question 1.

Take the output of the Readout layer and feed it as inputs into the Dense layer, then use the output of the Dense layer as inputs into the current output layer (`output`).
Next, take the output off the `output` layer as inputs to a scaling layer and use the scaling layer as the output of the network.

Pass your Scaling layer a name of "scaling" (i.e., `Scaling(name='scaling')`)

Build a model with 64 features and 2 message passing layers.

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
output (Dense)                  (None, 1)            65          readout[0][0]                    
__________________________________________________________________________________________________
scaling (Scaling)               (None, 1)            2           output[0][0]                     
==================================================================================================
Total params: 707
Trainable params: 707
Non-trainable params: 0
__________________________________________________________________________________________________
```

*Step 2*: Pre-seed the value of the scaling layer with the mean and value of a batch from the training dataset.

Set the weights for the layer by calling:

```python
scale = model.get_layer('scaling')
scale.mean = outputs.numpy().mean()
scale.std = outputs.numpy().std()
```

Completing Step 1 and 2 will give your model more flexibility (with adding the additional dense layers) and ensure it predicts 
reasonable values for the output (with adding the scale layer)

*Step 3*: Fit the model using an [early stopping callback.](https://keras.io/api/callbacks/early_stopping/)

Use both the training and validation loaders. Run for 128 epochs with a batch size of 32 with an early stopping patience of 8 epochs. 
Make sure to use `restore_best_weights=True` in your callback.

Now, repeat steps 2-3 with 0, 1, 2 and 4 message passing layers (please use `verbose=False` when fitting, so the notebooks aren't huge):

- Plot the change in the best loss on the test set as a function of number of layers. Do you observe a continual increase with the number of layers?

## Question 4: Explore readout layers

All of our previous questions used the atomization energy as an output. 
Atomization energy generally increases with the size of the molecule because there are more bonds to break.
So, our choice of summation for the readout function is a good one: more atoms means bigger sums and larger predicted values.

Band gaps, on the other hand, do not have such a scaling behavior.
They are instead due to a property of two specific atoms in the molecule: one that has a high-energy occupied orbital and
a second that has
In this problem, we will explore the effect of changing the "Readout function."

Train a total of 4 models each with 4 message passing steps and 64 features but varying whether we use `u0_atom` or `bandgap` to fit the model
and whether we use "sum" or "max" as a readout function:

- Explain how the max readout function produces features that do not scale with molecule size.
- Plot the predicted vs actual values for each of the networks on the test set. Which readout performs better for each output?
