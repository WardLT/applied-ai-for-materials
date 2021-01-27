"""Layers needed to create the MPNN model.

Taken from the ``tf2`` branch of the ``nfp`` code:

https://github.com/NREL/nfp/blob/tf2/examples/tf2_tests.ipynb
"""
from typing import List

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K


class MessageBlock(layers.Layer):
    """Message passing layer for MPNNs

    Takes the state of an atom and bond, and updates them by passing messages between nearby neighbors.

    Following the notation of Gilmer et al., the message function sums all of the atom states from
    the neighbors of each atom and then updates the node state by adding them to the previous state.
    """

    def __init__(self, atom_dimension, **kwargs):
        """
        Args:
             atom_dimension (int): Number of features to use to describe each atom
        """
        super(MessageBlock, self).__init__(**kwargs)
        self.atom_bn = layers.BatchNormalization()
        self.bond_bn = layers.BatchNormalization()
        self.bond_update_1 = layers.Dense(2 * atom_dimension, activation='sigmoid', use_bias=False)
        self.bond_update_2 = layers.Dense(atom_dimension)
        self.atom_update = layers.Dense(atom_dimension, activation='sigmoid', use_bias=False)
        self.atom_dimension = atom_dimension

    def call(self, inputs):
        original_atom_state, original_bond_state, connectivity = inputs

        # Batch norm on incoming layers
        atom_state = self.atom_bn(original_atom_state)
        bond_state = self.bond_bn(original_bond_state)

        # Gather atoms to bond dimension
        target_atom = tf.gather(atom_state, connectivity[:, 0])
        source_atom = tf.gather(atom_state, connectivity[:, 1])

        # Update bond states with source and target atom info
        new_bond_state = tf.concat([source_atom, target_atom, bond_state], 1)
        new_bond_state = self.bond_update_1(new_bond_state)
        new_bond_state = self.bond_update_2(new_bond_state)

        # Update atom states with neighboring bonds
        source_atom = self.atom_update(source_atom)
        messages = source_atom * new_bond_state
        messages = tf.math.segment_sum(messages, connectivity[:, 0])

        # Add new states to their incoming values (residual connection)
        bond_state = original_bond_state + new_bond_state
        atom_state = original_atom_state + messages

        return atom_state, bond_state

    def get_config(self):
        config = super().get_config()
        config.update({
            'atom_dimension': self.atom_dimension
        })
        return config


class GraphNetwork(layers.Layer):
    """Layer that implements an entire MPNN neural network

    Creates the message passing layers and also implements reducing the features of all nodes in
    a graph to a single output vector for a molecule.

    The "reduce" portion (also known as readout by Gilmer) can be configured a few different ways.
    One setting is whether the reduction occurs by combining the feature vectors for each atom
    and then using an MLP to determine the molecular properly or to first reduce the atomic feature
    vectors to a scalar with an MLP and then combining the result.
    You can also change how the reduction is performed, such as via a sum, average, or maximum.

    The reduction to a single feature for an entire molecule is produced by summing a single scalar value
    used to represent each atom. We chose this reduction approach under the assumption the energy of a molecule
    can be computed as a sum over atomic energies."""

    def __init__(self, atom_classes, bond_classes, atom_dimension, num_messages,
                 output_layer_sizes=None, atomic_contribution: bool = False,
                 attention_mlp_sizes: List[int] = None, n_outputs: int = 1,
                 reduce_function: str = 'sum',
                 return_atomic_features: bool = False, **kwargs):
        """
        Args:
             atom_classes (int): Number of possible types of nodes
             bond_classes (int): Number of possible types of edges
             atom_dimension (int): Number of features used to represent a node and bond
             num_messages (int): Number of message passing steps to perform
             output_layer_sizes ([int]): Number of nodes in hidden layers of output MLP
             attention_mlp_sizes ([int]): Number of nodes in MLP to map atomic features to attention weights
             n_outputs (int): Number of outputs for the network
             reduce_function (str): Which ``segment_*`` function to use to reduce atomic representations.
                Can also be "attention"
             dropout (float): Dropout rate
             return_atom_features (bool): Whether to return atomic features
        """
        super(GraphNetwork, self).__init__(**kwargs)
        self.atom_embedding = layers.Embedding(atom_classes, atom_dimension, name='atom_embedding')
        self.bond_embedding = layers.Embedding(bond_classes, atom_dimension, name='bond_embedding')
        self.message_layers = [MessageBlock(atom_dimension) for _ in range(num_messages)]
        self.atomic_contribution = atomic_contribution
        self.n_outputs = n_outputs
        self.atomic_features = return_atomic_features

        # Make the output MLP
        if output_layer_sizes is None:
            output_layer_sizes = []
        self.output_layers = [layers.Dense(s, activation='relu', name=f'dense_{i}')
                              for i, s in enumerate(output_layer_sizes)]
        self.output_layer_sizes = output_layer_sizes
        self.last_layer = layers.Dense(n_outputs, name='output')

        # Get the proper reduce function
        self.reduce_function = reduce_function.lower()
        if attention_mlp_sizes is None:
            attention_mlp_sizes = []
        self.attention_mlp_sizes = attention_mlp_sizes

        self.attention_layers = [layers.Dense(size, activation='relu', name=f'attn_{i}')
                                 for i, size in enumerate(self.attention_mlp_sizes)]
        self.attention_output = layers.Dense(1, activation='linear', name='attention_output')

    def call(self, inputs, **kwargs):
        atom_types, bond_types, node_graph_indices, connectivity = inputs

        # Initialize the atom and bond embedding vectors
        atom_state = self.atom_embedding(atom_types)
        bond_state = self.bond_embedding(bond_types)

        # Perform the message passing
        for message_layer in self.message_layers:
            atom_state, bond_state = message_layer([atom_state, bond_state, connectivity])

        # If desired, stop here and return the atomic features
        if self.atomic_features:
            return atom_state

        if self.atomic_contribution:
            # Represent the atom state as the state of the molecule
            mol_state = atom_state
        else:
            mol_state = self._readout(atom_state, node_graph_indices)

        # Apply the MLP layers
        for layer in self.output_layers:
            mol_state = layer(mol_state)

        # Reduce to a single prediction
        output = self.last_layer(mol_state)

        if self.atomic_contribution:
            # Sum up atomic contributions
            return self._readout(output, node_graph_indices, atom_state)
        else:
            # Return the value
            return output

    def _readout(self, atom_state, node_graph_indices, attention_input=None):
        """Perform the readout function for the graph

        Args:
            atom_state: State describe each atom. Shape is (N_atoms, N_features)
            node_graph_indices: Assignment of each node to a graph. Shape (N_atoms,)
            attention_input: State that describes each atom, used to compute attention weights. Shape (N_atoms, x)
        Returns:
            State of the molecule. Shape: (N_mols, N_features)
        """
        if self.reduce_function in ["sum", "mean", "max", "min", "prod"]:
            # Sum over all atoms in a mol to form a single fingerprint
            reduce_func = getattr(tf.math, f'segment_{self.reduce_function.lower()}')
            mol_state = reduce_func(atom_state, node_graph_indices)
        elif self.reduce_function == "softmax":
            # Compute the softmax for each feature
            self.atom_state_softmax = self._per_graph_softmax(atom_state, node_graph_indices)

            # Softmax gives a fraction of each feature to use when computing the "max"
            #  Dot product with the original values to get the a meaningful number
            atom_state_softmaxed = tf.multiply(atom_state, self.atom_state_softmax)

            # Sum over each molecule again
            mol_state = tf.math.segment_sum(atom_state_softmaxed, node_graph_indices)
        elif self.reduce_function == "attention":
            # Use the atomic state as inputs to the attention computer,
            if attention_input is None:
                attention_input = atom_state

            # Pass the attention input through an MLP
            for layer in self.attention_layers:
                attention_input = layer(attention_input)

            # Reduce the dimensionality to 1 and compute the softmax
            attention_weight = self.attention_output(attention_input)
            self.attention_values = self._per_graph_softmax(attention_weight, node_graph_indices)

            # Apply the attention weights to the atom_state to create the molecule state
            atom_state_weighted = tf.multiply(atom_state, self.attention_values)
            mol_state = tf.math.segment_sum(atom_state_weighted, node_graph_indices)
        else:
            raise ValueError(f'Operation not yet implemented: {self.reduce_func}')
        return mol_state

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
        return atom_state_softmax

    def get_config(self):
        config = super().get_config()
        config.update({
            'atom_classes': self.atom_embedding.input_dim,
            'bond_classes': self.bond_embedding.input_dim,
            'atom_dimension': self.atom_embedding.output_dim,
            'output_layer_sizes': self.output_layer_sizes,
            'num_messages': len(self.message_layers),
            'reduce_function': self.reduce_function,
            'n_outputs': self.n_outputs,
            'atomic_contribution': self.atomic_contribution,
            'attention_mlp_sizes': self.attention_mlp_sizes
        })
        return config


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

    def get_config(self):
        config = super().get_config()
        config['axis'] = self.axis
        return config


class CartesianProduct(layers.Layer):
    """Computes a Cartesian product of two 2D arrays

    Given 2D tensors of shapes [a, b] and [c, d], returns
    a tensor of [a, c, b + d]
    """

    def call(self, inputs):
        a, b = inputs

        # Expand both arrays to (M, N, x) arrays
        a_tiled = K.tile(K.expand_dims(a, 1), [1, K.shape(b)[0], 1])
        b_tiled = K.tile(K.expand_dims(b, 0), [K.shape(a)[0], 1, 1])

        return K.concatenate([a_tiled, b_tiled], axis=2)


class DenormalizeLayer(layers.Layer):
    """Layer to scale the output layer to match the input data distribution"""

    def build(self, input_shape):
        self.mean = self.add_weight('mean', shape=(input_shape[-1],))
        self.std = self.add_weight('std', shape=(input_shape[-1],))

    def call(self, inputs, **kwargs):
        return inputs * self.std + self.mean


custom_objects = {
    'GraphNetwork': GraphNetwork,
    'MessageBlock': MessageBlock,
    'Squeeze': Squeeze
}
