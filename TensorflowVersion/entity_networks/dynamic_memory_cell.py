from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

class DynamicMemoryCell(tf.nn.rnn_cell.RNNCell):
    """
    Implementation of a dynamic memory cell as a gated recurrent network.
    The cell's hidden state is divided into blocks and each block's weights are tied.
    """

    def __init__(self, num_blocks, num_units_per_block, a_keys, b_keys, initializer=None, activation=tf.nn.relu):
        self._num_blocks = num_blocks # M
        self._num_units_per_block = num_units_per_block # d
        self._a_keys = a_keys
        self._b_keys = b_keys
        self._activation = activation # \phi
        self._initializer = initializer

    @property
    def state_size(self):
        return self._num_blocks * self._num_units_per_block

    @property
    def output_size(self):
        return self._num_blocks * self._num_units_per_block

    def zero_state(self, batch_size, dtype):
        """
        We initialize the memory to the key values.
        """
        zero_state = tf.concat(1, [tf.expand_dims(key, 0) for key in self._keys])
        zero_state_batch = tf.tile(zero_state, tf.pack([batch_size, 1]))
        return zero_state_batch

    def get_gate(self, a_key_j, b_key_j, inputs):
        """
        Implements the gate (scalar for each block). Equation 2:

        g_j <- \sigma(s_t^T h_j + s_t^T w_j)
        """
        a = tf.reduce_sum(inputs * tf.expand_dims(a_key_j, 0), reduction_indices=[1]) # TODO: why there is expand_dims?
        b = tf.reduce_sum(inputs * tf.expand_dims(b_key_j, 0), reduction_indices=[1])
        return tf.sigmoid(a + b)

    def get_candidate(self, state_j, a_key_j, b_key_j, inputs, Wh, Wa, Wb, Ws):
        """
        Represents the new memory candidate that will be weighted by the
        gate value and combined with the existing memory. Equation 3:

        h_j^~ <- \phi(U h_j + V w_j + W s_t)
        """
        key_W = tf.matmul(tf.expand_dims(a_key_j, 0), Wa) + tf.matmul(tf.expand_dims(b_key_j, 0), Wb)
        state_W = tf.matmul(state_j, Wh)
        inputs_W = tf.matmul(inputs, Ws)
        return self._activation(state_W + key_W + inputs_W)

    def __call__(self, inputs, state, scope=None):
        # TODO: what is the shape of inputs
        with tf.variable_scope(scope or type(self).__name__, initializer=self._initializer):
            # Split the hidden state into blocks (each U, V, W are shared across blocks).
            # state is (_num_blocks, batch, hid__size), after split, state is a list of (batch, hid_size) tensors
            state = tf.split(1, self._num_blocks, state)

            # TODO: ortho init?
            Wh = tf.get_variable('Wh', [self._num_units_per_block, self._num_units_per_block])
            Wa = tf.get_variable('Wa', [self._num_units_per_block, self._num_units_per_block])
            Wb = tf.get_variable('Wb', [self._num_units_per_block, self._num_units_per_block])
            Ws = tf.get_variable('Ws', [self._num_units_per_block, self._num_units_per_block])

            # TODO: layer norm?

            next_states = []
            for j, state_j in enumerate(state): # Hidden State (j)
                a_key_j = self._a_keys[j]
                b_key_j = self._b_keys[j]
                gate_j = self.get_gate(a_key_j, b_key_j, inputs) # my eqution 1
                candidate_j = self.get_candidate(state_j, a_key_j, b_key_j, inputs, Wh, Wa, Wb, Ws) # my equation 2

                # Equation 4: h_j <- h_j + g_j * h_j^~
                # Perform an update of the hidden state (memory).
                state_j_next = (1 - tf.expand_dims(gate_j, -1)) * state_j + tf.expand_dims(gate_j, -1) * candidate_j

                # Equation 5: h_j <- h_j / \norm{h_j}
                # Forget previous memories by normalization.
                # state_j_next = tf.nn.l2_normalize(state_j_next, -1, epsilon=1e-7) # TODO: Is epsilon necessary?

                next_states.append(state_j_next)
            state_next = tf.concat(1, next_states)
        return state_next, state_next
