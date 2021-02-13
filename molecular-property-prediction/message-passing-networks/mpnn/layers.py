"""Layers needed to create the MPNN model.

Taken from the ``tf2`` branch of the ``nfp`` code:

https://github.com/NREL/nfp/blob/tf2/examples/tf2_tests.ipynb
"""
from typing import List

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K


class MessagePassingLayer(layers.Layer):
    """Perform the message passing step"""

    def call(self, atom_features, bond_features, connectivity):
        """Perform the message passing steps.
        
        Takes the atom and bond features with the connectivity as inputs
        and produces a new set of atom features.
        """
        # Get the features for the atoms at each side of a bond
        source_atom = tf.gather(atom_features, connectivity[:, 0])
        target_atom = tf.gather(atom_features, connectivity[:, 1])

        # Make messages based on the "far side" of the bond and the bond type
        all_messages = tf.multiply(bond_features, target_atom)

        # Sum them up and add them to the original features
        messages = tf.math.segment_sum(all_messages, connectivity[:, 0])
        atom_features = atom_features + messages
        return atom_features

class Readout(layers.Layer):
    """Convert atomic to molecular features"""
    
    def __init__(self, reduce_function: str = "sum", **kwargs):
        """
        Args:
            reduce_function: Functon used to combine atomic features.
                Can be "sum," "mean", "max," "min," "prod" or "softmax"
        """
        super().__init__(**kwargs)
        self.reduce_function = reduce_function
    
    def call(self, atom_features, node_graph_indices):
        if self.reduce_function in ["sum", "mean", "max", "min", "prod"]:
            # Sum over all atoms in a mol to form a single fingerprint
            reduce_func = getattr(tf.math, f'segment_{self.reduce_function.lower()}')
            return reduce_func(atom_features, node_graph_indices)
        elif self.reduce_function == "softmax":
            # Compute the softmax for each feature
            atom_state_softmax = self._per_graph_softmax(atom_features, node_graph_indices)

            # Softmax gives a fraction of each feature to use when computing the "max"
            #  Dot product with the original values to get the a meaningful number
            atom_state_softmaxed = tf.multiply(atom_features, atom_state_softmax)

            # Sum over each molecule again
            return tf.math.segment_sum(atom_state_softmaxed, node_graph_indices)
    
    def _per_graph_softmax(self, atom_state, node_graph_indices):
        """Softmax where you compute the softmax for each graph in a batch separately
        
        Args:
            atom_state: Atomic state to be "softmax"ed. Shape is (N_atoms, N_features)
            node_graph_indices: Assignment of each node/atom to a graph. Shape (N_atoms,)
        Returns:
            Softmaxed version of atom_state. Shape (N_atoms, N_featurs)
        """
        
        # Compute the exponential of each atomic feature
        exp_atom_state = tf.exp(atom_state)
        
        # Compute the sum for each feature of each molecule
        atom_state_denom = tf.math.segment_sum(exp_atom_state, node_graph_indices)
        
        # Divide the atomic feature for each molecule by the summation for the whole molecule
        atom_state_denom_per_atom = tf.gather(atom_state_denom, node_graph_indices)
        atom_state_softmax = tf.divide(exp_atom_state, atom_state_denom_per_atom)
        
        # Each row of this array is the features for each atom
        #  The value of each feature sums to 1 for each molecule
        return atom_state_softmax

class Squeeze(layers.Layer):
    """Wrapper over the tf.squeeze operation"""

    def __init__(self, axis=1, **kwargs):
        """
        Args:
            axis (int): Which axis to squash
        """
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.squeeze(inputs, axis=self.axis)

    
class Scaling(layers.Layer):
    """Layer to scale the output layer to match the input data distribution"""

    def build(self, input_shape):
        self.mean = self.add_weight('mean', shape=(input_shape[-1],))
        self.std = self.add_weight('std', shape=(input_shape[-1],))

    def call(self, inputs, **kwargs):
        return inputs * self.std + self.mean
