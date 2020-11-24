import os
import time
import json

import numpy as np
import tensorflow as tf

from data_util import read_sequential_target

lstm = tf.contrib.rnn.LSTMCell
fc = tf.contrib.layers.fully_connected

# Encoder for dRAE and aRAE
# x: input tensor, length: length of input, n_units: no. of nodes per LSTM layer, n_layers: no. of LSTM layers
def encoder(x, length, n_units=10, n_layers=1, scope="encoder"):
    with tf.variable_scope(scope, reuse=False):
        enc_cells = []
        for j in range(n_layers):         # Go through each layer
            with tf.variable_scope("layer_{}".format(j), reuse=False):
                enc_cell = lstm(num_units=n_units,      # define each LSTM layer
                                use_peepholes=True,
                                forget_bias=0.8)
                enc_cells.append(enc_cell)          # append each layer
        enc_cell = tf.contrib.rnn.MultiRNNCell(enc_cells)   # combine the LSTM layers to create the ultimate LSTM
        _, enc_states = tf.nn.dynamic_rnn(enc_cell,        # get the final encoded state (batch_size, no. of nodes)
                                          x,
                                          sequence_length=length,
                                          dtype=tf.float32,
                                          time_major=True)
        enc_states = tf.transpose(enc_states, (2,0,1,3))  # transpose to batchsize, n_layers, 2, n_units
        enc_states = tf.reshape(enc_states, (int(enc_states.shape[0]), -1))     # batch_size, 2 x no. of nodes
    return enc_states

# Decoder for dRAE
# init_in: y0 <BOS>, init_state: h0_dec, length: length of input, out_dim: output dimensionality (9),
# n_units: no. of nodes per LSTM layer, n_layers: no. of LSTM layers
def L_decoder(init_in, init_state, length, out_dim,
              n_units=10, n_layers=1, scope="L_decoder"):
    y = []
    # prepare the h0_dec for the LSTM
    init_state = tf.reshape(init_state,
                            (int(init_state.shape[0]), n_layers, 2, n_units))
    init_state = tf.transpose(init_state, (1,2,0,3))

    with tf.variable_scope(scope, reuse=False):
        dec_cells = []
        for i in range(length-1):              # The output length will equal input size minus one : four
            layer_in = init_in                # initial input
            dec_states = []
            if i == 0:          # at the initial timestep
                fc_reuse = False
                for j in range(n_layers):   # Go through each layer
                    with tf.variable_scope("layer_{}".format(j), reuse=False):
                        dec_state = (init_state[j][0], init_state[j][1])    # get h0_dec as a tuple
                        dec_cell = lstm(num_units=n_units,              # define the LSTM layer
                                        use_peepholes=True,
                                        forget_bias=0.8)
                        dec_cells.append(dec_cell)                  # stack the LSTM layers
                        h, dec_state = dec_cell(layer_in, dec_state)   # get h1_dec = DecCell(y0, h0_dec)
                        dec_states.append(dec_state)              # stack the decoder states
                        layer_in = h
            else:             # timesteps other than the initial
                layer_in = out          # y_(t-1)
                tf.get_variable_scope().reuse_variables()
                fc_reuse = True
                for j in range(n_layers):         # Go through each layer
                    with tf.variable_scope("layer_{}".format(j), reuse=False):
                        dec_cell = dec_cells[j]
                        dec_state = prev_dec_states[j]
                        h, dec_state = dec_cell(layer_in, dec_state)    # get ht_dec = DecCell(yt-1, ht-1_dec)
                        dec_states.append(dec_state)        # stack the decoder states
                        layer_in = h

            prev_dec_states = dec_states        # keep track of previous states
            out = fc(layer_in, out_dim,         # get the output y
                     activation_fn=tf.nn.softmax,
                     reuse=fc_reuse,
                     scope="L_decoder_fc")
            y.append(out)              # merge the outputs
    y = tf.stack(y)                 # stack the outputs
    return y           # return the outputs

# Decoder for aRAE
# V_in: visual input features, init_B_in: j1, init_state:h0_dec, length: length of input, out_dim: output dimensionality (10)
# n_units: no. of nodes per LSTM layer, n_layers: no. of LSTM layers
def VB_decoder(V_in, init_B_in, init_state, length, out_dim,
               n_units=100, n_layers=1, scope="VB_decoder"):
    y = []
    # prepare the h0_dec for the LSTM
    init_state = tf.reshape(init_state,
                            (int(init_state.shape[0]), n_layers, 2, n_units))
    init_state = tf.transpose(init_state, (1,2,0,3))
    
    with tf.variable_scope(scope, reuse=False):
        dec_cells = []
        for i in range(length-1):        # The output length will equal input size minus one
            current_V_in = V_in[i]       # current visual input features
            dec_states = []
            if i == 0:                 # at the initial timestep
                current_B_in = init_B_in     # first joint angles j1
                layer_in = tf.concat([current_V_in, current_B_in], axis=1)   # concatenate joint angles and visual features
                fc_reuse = False
                for j in range(n_layers):       # Go through each layer
                    with tf.variable_scope("layer_{}".format(j), reuse=False):
                        dec_state = (init_state[j][0], init_state[j][1])        # get h0_dec as a tuple
                        dec_cell = lstm(num_units=n_units,                # define the LSTM layer
                                        use_peepholes=True,
                                        forget_bias=0.8)
                        dec_cells.append(dec_cell)                   # stack the LSTM layers
                        h, dec_state = dec_cell(layer_in, dec_state)     # get h1_dec = DecCell(v1, j_hat1, h0_dec)
                        dec_states.append(dec_state)               # stack the decoder states
                        layer_in = h
            else:           # timesteps other than the initial
                current_B_in = out   # current action is the output of the previous timestep j_hat_t
                layer_in = tf.concat([current_V_in, current_B_in], axis=1)      # concatenate joint angles and visual features
                tf.get_variable_scope().reuse_variables()
                fc_reuse = True
                for j in range(n_layers):        # Go through each layer
                    with tf.variable_scope("layer_{}".format(j), reuse=False):
                        dec_cell = dec_cells[j]
                        dec_state = prev_dec_states[j]        # get the previous state
                        h, dec_state = dec_cell(layer_in, dec_state)       # get ht_dec = DecCell(vt, j_hat_t, h_t-1_dec)
                        dec_states.append(dec_state)              # stack the decoder states
                        layer_in = h

            prev_dec_states = dec_states               # keep track of previous states

            out = fc(layer_in, out_dim,              # get the output j_hat
                     activation_fn=tf.tanh,
                     reuse=fc_reuse,
                     scope="VB_decoder_fc")
            y.append(out)                          # merge the outputs
    y = tf.stack(y)               # stack the outputs
    return y               # return the outputs

# Create a placeholder for each description, action or visual tensor (fw: right order, bw: inverse order, bin: binary)
def make_placeholders(L_shape, B_shape, V_shape, batchsize):
    place_holders = {}
    place_holders["L_fw"] = tf.placeholder(tf.float32, [L_shape[0], batchsize, L_shape[2]], name="L_fw")
    place_holders["B_fw"] = tf.placeholder(tf.float32, [B_shape[0], batchsize, B_shape[2]], name="B_fw")
    place_holders["V_fw"] = tf.placeholder(tf.float32, [V_shape[0], batchsize, V_shape[2]], name="V_fw")
    place_holders["L_bw"] = tf.placeholder(tf.float32, [L_shape[0], batchsize, L_shape[2]], name="L_bw") 
    place_holders["B_bw"] = tf.placeholder(tf.float32, [B_shape[0], batchsize, B_shape[2]], name="B_bw")
    place_holders["V_bw"] = tf.placeholder(tf.float32, [V_shape[0], batchsize, V_shape[2]], name="V_bw")
    #place_holders["L_bin"] = tf.placeholder(tf.float32, [L_shape[0], batchsize, L_shape[2]], name="L_bin") 
    place_holders["B_bin"] = tf.placeholder(tf.float32, [B_shape[0], batchsize, B_shape[2]], name="B_bin")
    #place_holders["V_bin"] = tf.placeholder(tf.float32, [V_shape[0], batchsize, V_shape[2]], name="V_bin")
    place_holders["L_len"] = tf.placeholder(tf.float32, [batchsize], name="L_len")
    place_holders["B_len"] = tf.placeholder(tf.float32, [batchsize], name="B_len")
    place_holders["V_len"] = tf.placeholder(tf.float32, [batchsize], name="V_len")
    return place_holders

# Binding loss (X: description, Y: action)
def aligned_discriminative_loss(X, Y, margin=1.0):
    batchsize = int(X.shape[0])      # number of actions
    X_tile = tf.tile(X, (batchsize, 1))      # replicate the descriptions according to no. of actions
    Y_tile = tf.reshape(tf.tile(Y, (1, batchsize)),  # do the same for actions
                        (batchsize**2, -1))
    # Calculate the euclidean distance between paired descriptions and actions
    pair_loss = tf.sqrt(tf.reduce_sum(tf.square(X-Y), axis=1))
    all_pairs = tf.square(X_tile-Y_tile)
    loss_array = tf.reshape(tf.sqrt(tf.reduce_sum(all_pairs, axis=1)),
                            (batchsize, batchsize))
    # Make representation of an action be far from that of its unpaired description
    x_diff = tf.expand_dims(pair_loss, axis=0) - loss_array + margin
    y_diff = tf.expand_dims(pair_loss, axis=1) - loss_array + margin
    x_diff = tf.maximum(x_diff, 0)
    y_diff = tf.maximum(y_diff, 0)
    mask = 1.0 - tf.eye(batchsize)
    x_diff = x_diff * mask
    y_diff = y_diff * mask
    
    return tf.reduce_mean(x_diff) + tf.reduce_mean(y_diff) + tf.reduce_mean(pair_loss)  # returen the binding loss
