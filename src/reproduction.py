import tensorflow as tf
fc = tf.contrib.layers.fully_connected

from config import NetConfig, TrainConfig
from data_util import read_sequential_target, save_latent
from modules import *

def sampling(mean, log_sigma, scope="sampling"):
    with tf.variable_scope(scope, reuse=False):
        z_mean, z_log_sigma = mean, log_sigma
        epsilon = tf.keras.backend.random_normal(shape=(z_mean.shape[0].value, z_mean.shape[1].value),
                                                 mean=0.0, stddev=0.1)
    return z_mean + tf.exp(0.5*z_log_sigma) * epsilon

# Reproduce the actions without descriptions
def main():
    # get the network configuration (parameters such as number of layers and units)
    net_conf = NetConfig()
    net_conf.set_conf("../train/vae_conf.txt")
    L_num_units = net_conf.L_num_units
    L_num_layers = net_conf.L_num_layers
    VB_num_units = net_conf.VB_num_units
    VB_num_layers = net_conf.VB_num_layers
    LATENT_DIM = net_conf.S_dim

    # get the training configuration
    train_conf = TrainConfig()
    train_conf.set_conf("../train/train_conf.txt")
    batchsize = 1
    save_dir = train_conf.save_dir
    # get the dataset folders
    L_data_dir = train_conf.L_dir
    B_data_dir = train_conf.B_dir
    V_data_dir = train_conf.V_dir

    # get the descriptions
    L_fw, L_bw, L_bin, L_len, L_filenames = read_sequential_target(L_data_dir, True)
    #print(len(L_filenames))
    # get the joint angles for actions
    B_fw, B_bw, B_bin, B_len, B_filenames = read_sequential_target(B_data_dir, True)
    # get the visual features for action images
    V_fw, V_bw, V_bin, V_len = read_sequential_target(V_data_dir)
    # create variables for data shapes
    L_shape = L_fw.shape
    B_shape = B_fw.shape
    V_shape = V_fw.shape

    # go through every step for testing data
    if train_conf.test:
        L_data_dir_test = train_conf.L_dir_test
        B_data_dir_test = train_conf.B_dir_test
        V_data_dir_test = train_conf.V_dir_test
        L_fw_u, L_bw_u, L_bin_u, L_len_u, L_filenames_u = read_sequential_target(L_data_dir_test, True)
        print(len(L_filenames_u))
        B_fw_u, B_bw_u, B_bin_u, B_len_u, B_filenames_u = read_sequential_target(B_data_dir_test, True)
        V_fw_u, V_bw_u, V_bin_u, V_len_u = read_sequential_target(V_data_dir_test)
        L_shape_u = L_fw_u.shape
        B_shape_u = B_fw_u.shape
        V_shape_u = V_fw_u.shape

    # Create a placeholder dictionary for tensors
    placeholders = make_placeholders(L_shape, B_shape, V_shape, batchsize)

    # Encoding
    # Concatenate the joint angles with visual features
    VB_input = tf.concat([placeholders["V_bw"],
                          placeholders["B_bw"]],
                         axis=2)
    # Get the final action state by feeding the encoder with concatenated action input
    VB_enc_final_state = encoder(VB_input,
                                 placeholders["V_len"], 
                                 n_units=VB_num_units,
                                 n_layers=VB_num_layers,
                                 scope="VB_encoder")

    # Binding layer
    # Latent space: get z_mean and z_log_sigma and sample them
    VB_z_mean = fc(VB_enc_final_state, LATENT_DIM,
                   activation_fn=None, scope="VB_z_mean")
    VB_z_log_sigma = fc(VB_enc_final_state, LATENT_DIM,
                   activation_fn=None, scope="VB_z_log_sigma")
    VB_sampling = sampling(VB_z_mean, VB_z_log_sigma, scope='VB_sampling')
    VB_dec_init_state = fc(VB_sampling, VB_num_units*VB_num_layers*2,       # Initial decoder state for actions (h0_dec)
                           activation_fn=None, scope="VB_postshare")

    # Decoding
    B_output = VB_decoder(placeholders["V_fw"],              # Get the reconstructed action via the action decoder
                          placeholders["B_fw"][0],
                          VB_dec_init_state,
                          length=B_shape[0],
                          out_dim=B_shape[2],
                          n_units=VB_num_units,
                          n_layers=VB_num_layers,
                          scope="VB_decoder")

    # Use GPU
    gpuConfig = tf.ConfigProto(
        gpu_options=tf.GPUOptions(allow_growth=True,
                                  per_process_gpu_memory_fraction=train_conf.gpu_use_rate),
        device_count={'GPU': 1})

    # Launch the graph in a session
    sess = tf.Session(config=gpuConfig)
    sess.run(tf.global_variables_initializer())    # run the session
    saver = tf.train.Saver(tf.global_variables())  # create a saver for the model
    saver.restore(sess, save_dir)      # restore previously saved variables

    # Feed the dataset as input
    for i in range(B_shape[1]):
        feed_dict = {placeholders["L_fw"]: L_fw[:, i:i+1, :],
                     placeholders["B_fw"]: B_fw[:, i:i+1, :],
                     placeholders["V_fw"]: V_fw[:, i:i+1, :],
                     placeholders["L_bw"]: L_bw[:, i:i+1, :],
                     placeholders["B_bw"]: B_bw[:, i:i+1, :],
                     placeholders["V_bw"]: V_bw[:, i:i+1, :],
                     placeholders["B_bin"]: B_bin[:, i:i+1, :],
                     placeholders["L_len"]: L_len[i:i+1],
                     placeholders["V_len"]: V_len[i:i+1]}
        result = sess.run(B_output, feed_dict=feed_dict)
        save_latent(np.transpose(result, (1,0,2)), B_filenames[i], "reproduction")        # save the predicted actions

    # Do the same for the test set
    if train_conf.test:
        for i in range(B_shape_u[1]):
            feed_dict = {placeholders["L_fw"]: L_fw_u[:, i:i+1, :],
                         placeholders["B_fw"]: B_fw_u[:, i:i+1, :],
                         placeholders["V_fw"]: V_fw_u[:, i:i+1, :],
                         placeholders["L_bw"]: L_bw_u[:, i:i+1, :],
                         placeholders["B_bw"]: B_bw_u[:, i:i+1, :],
                         placeholders["V_bw"]: V_bw_u[:, i:i+1, :],
                         placeholders["B_bin"]: B_bin_u[:, i:i+1, :],
                         placeholders["L_len"]: L_len_u[i:i+1],
                         placeholders["V_len"]: V_len_u[i:i+1]}
            result = sess.run(B_output, feed_dict=feed_dict)
            save_latent(np.transpose(result, (1,0,2)), B_filenames_u[i], "reproduction") # save the predicted actions

if __name__ == "__main__":
    main()
