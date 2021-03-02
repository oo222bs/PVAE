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
    return z_mean + tf.exp(z_log_sigma) * epsilon


# Find the descriptions via given actions
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
    save_dir = '../train/checkpoints_vae_final/model'

    # get the dataset folders
    L_data_dir = "../target/language_train"
    B_data_dir = "../target/behavior_train"
    V_data_dir = "../target/vision_train"

    # get the descriptions
    L_fw, L_bw, L_bin, L_len, L_filenames = read_sequential_target(L_data_dir, True)
    # print(len(L_filenames))

    # get the joint angles for actions
    B_fw, B_bw, B_bin, B_len, B_filenames = read_sequential_target(B_data_dir, True)
    # normalise the joint angles between -1 and 1
    B_fw = 2 * ((B_fw - B_fw.min()) / (B_fw.max() - B_fw.min())) - 1
    B_bw = 2 * ((B_bw - B_bw.min()) / (B_bw.max() - B_bw.min())) - 1
    # get the visual features for action images
    V_fw, V_bw, V_bin, V_len = read_sequential_target(V_data_dir)
    # normalise the visual features between -1 and 1
    V_min = V_fw.min()
    V_fw = 2 * ((V_fw - V_fw.min()) / (V_fw.max() - V_fw.min())) - 1
    V_bw = 2 * ((V_bw - V_bw.min()) / (V_bw.max() - V_bw.min())) - 1
    for i in range(6):
        V_fw[50:, 54 + 108 * i: 108 + 108 * i, :] = 0
        V_bw[50:, 54 + 108 * i: 108 + 108 * i, :] = 0
        B_fw[50:, 54 + 108 * i: 108 + 108 * i, :] = 0
        B_bw[50:, 54 + 108 * i: 108 + 108 * i, :] = 0
    # create variables for data shapes
    L_shape = (L_fw.shape[0] // 8, L_fw.shape[1], L_fw.shape[2])
    B_shape = B_fw.shape
    V_shape = V_fw.shape

    # go through every step for testing data
    if train_conf.test:
        L_data_dir_test = "../target/language_test"
        B_data_dir_test = "../target/behavior_test"
        V_data_dir_test = "../target/vision_test"
        L_fw_u, L_bw_u, L_bin_u, L_len_u, L_filenames_u = read_sequential_target(L_data_dir_test, True)
        # print(len(L_filenames_u))
        B_fw_u, B_bw_u, B_bin_u, B_len_u, B_filenames_u = read_sequential_target(B_data_dir_test, True)
        # normalise the joint angles between -1 and 1
        B_fw_u = 2 * ((B_fw_u - B_fw_u.min()) / (B_fw_u.max() - B_fw_u.min())) - 1
        B_bw_u = 2 * ((B_bw_u - B_bw_u.min()) / (B_bw_u.max() - B_bw_u.min())) - 1
        V_fw_u, V_bw_u, V_bin_u, V_len_u = read_sequential_target(V_data_dir_test)
        # normalise the visual features between -1 and 1
        V_fw_u = 2 * ((V_fw_u - V_min) / (V_fw_u.max() - V_min)) - 1
        V_bw_u = 2 * ((V_bw_u - V_min) / (V_bw_u.max() - V_min)) - 1
        for i in range(6):
            V_fw_u[50:, 18 + 36 * i: 36 + 36 * i, :] = 0
            V_bw_u[50:, 18 + 36 * i: 36 + 36 * i, :] = 0
            B_fw_u[50:, 18 + 36 * i: 36 + 36 * i, :] = 0
            B_bw_u[50:, 18 + 36 * i: 36 + 36 * i, :] = 0
        L_shape_u = (L_fw_u.shape[0] // 8, L_fw_u.shape[1], L_fw_u.shape[2])
        B_shape_u = B_fw_u.shape
        V_shape_u = V_fw_u.shape

    tf.reset_default_graph()  # Clear the default graph stack and reset the global default graph

    placeholders = make_placeholders(L_shape, B_shape, V_shape,
                                     batchsize)  # Create a placeholder dictionary for tensors

    # Encoding
    # Get the final action state by feeding the encoder with concatenated action input
    VB_input = tf.concat([placeholders["V_bw"],
                          placeholders["B_bw"]],
                         axis=2)
    VB_enc_final_state = encoder(VB_input,
                                 placeholders["V_len"],
                                 n_units=VB_num_units,
                                 n_layers=VB_num_layers,
                                 scope="VB_encoder")

    # Latent space: get z_mean and z_log_sigma and sample them

    VB_z_mean = fc(VB_enc_final_state, LATENT_DIM,
                   activation_fn=None, scope="VB_z_mean")

    L_dec_init_state = fc(VB_z_mean, L_num_units * L_num_layers * 2,  # Initial decoder state for descriptions (h0_dec)
                          activation_fn=None, scope="L_postshare")

    # Decoding
    L_output = L_decoder(placeholders["L_fw"][0],  # Get the reconstructed description via the language decoder
                         L_dec_init_state,
                         length=L_shape[0],
                         out_dim=L_shape[2],
                         n_units=L_num_units,
                         n_layers=L_num_layers,
                         scope="L_decoder")
    L_output = tf.contrib.seq2seq.hardmax(L_output)  # Use softmax to choose the most likely words

    # Use GPU
    gpuConfig = tf.ConfigProto(
        gpu_options=tf.GPUOptions(allow_growth=True,
                                  per_process_gpu_memory_fraction=train_conf.gpu_use_rate),
        device_count={'GPU': 1})

    # Launch the graph in a session
    sess = tf.Session(config=gpuConfig)
    sess.run(tf.global_variables_initializer())  # run the session
    saver = tf.train.Saver(tf.global_variables())  # create a saver for the model
    saver.restore(sess, save_dir)  # restore previously saved variables

    # Feed the dataset as input
    for i in range(B_shape[1]):
        sentence_idx = np.random.randint(8)
        if sentence_idx == 0:
            L_fw_feed = L_fw[0:5, i:i + 1, :]
            L_bw_feed = L_bw[35:40, i:i + 1, :]
        elif sentence_idx == 1:
            L_fw_feed = L_fw[5:10, i:i + 1, :]
            L_bw_feed = L_bw[30:35, i:i + 1, :]
        elif sentence_idx == 2:
            L_fw_feed = L_fw[10:15, i:i + 1, :]
            L_bw_feed = L_bw[25:30, i:i + 1, :]
        elif sentence_idx == 3:
            L_fw_feed = L_fw[15:20, i:i + 1, :]
            L_bw_feed = L_bw[20:25, i:i + 1, :]
        elif sentence_idx == 4:
            L_fw_feed = L_fw[20:25, i:i + 1, :]
            L_bw_feed = L_bw[15:20, i:i + 1, :]
        elif sentence_idx == 5:
            L_fw_feed = L_fw[25:30, i:i + 1, :]
            L_bw_feed = L_bw[10:15, i:i + 1, :]
        elif sentence_idx == 6:
            L_fw_feed = L_fw[30:35, i:i + 1, :]
            L_bw_feed = L_bw[5:10, i:i + 1, :]
        else:
            L_fw_feed = L_fw[35:40, i:i + 1, :]
            L_bw_feed = L_bw[0:5, i:i + 1, :]

        feed_dict = {placeholders["L_fw"]: L_fw_feed,
                     placeholders["B_fw"]: B_fw[:, i:i + 1, :],
                     placeholders["V_fw"]: V_fw[:, i:i + 1, :],
                     placeholders["L_bw"]: L_bw_feed,
                     placeholders["B_bw"]: B_bw[:, i:i + 1, :],
                     placeholders["V_bw"]: V_bw[:, i:i + 1, :],
                     placeholders["B_bin"]: B_bin[:, i:i + 1, :],
                     placeholders["L_len"]: L_len[i:i + 1] / 8,
                     placeholders["V_len"]: V_len[i:i + 1]}

        result = sess.run([L_output], feed_dict=feed_dict)
        save_latent(np.transpose(result[0], (1, 0, 2)), L_filenames[i],
                    "recognition")  # save the predicted descriptions
        r = result[0][:, 0, :].argmax(axis=1)
        t = L_fw[1:5, i, :].argmax(axis=1)
        t_second = L_fw[6:10, i, :].argmax(axis=1)
        t_third = L_fw[11:15, i, :].argmax(axis=1)
        t_fourth = L_fw[16:20, i, :].argmax(axis=1)
        t_fifth = L_fw[21:25, i, :].argmax(axis=1)
        t_sixth = L_fw[26:30, i, :].argmax(axis=1)
        t_seventh = L_fw[31:35, i, :].argmax(axis=1)
        t_eighth = L_fw[36:40, i, :].argmax(axis=1)
        if (r == t).all() or (r == t_second).all() or (r == t_third).all() or (r == t_fourth).all() \
                or (r == t_fifth).all() or (r == t_sixth).all() or (r == t_seventh).all() or (r == t_eighth).all():
            print(True)  # Check if predicted descriptions match the original ones
        else:
            print(False)
    # Do the same for the test set
    if train_conf.test:
        print("test!")
        for i in range(B_shape_u[1]):
            sentence_idx = np.random.randint(8)
            if sentence_idx == 0:
                L_fw_feed_u = L_fw_u[0:5, i:i + 1, :]
                L_bw_feed_u = L_bw_u[35:40, i:i + 1, :]
            elif sentence_idx == 1:
                L_fw_feed_u = L_fw_u[5:10, i:i + 1, :]
                L_bw_feed_u = L_bw_u[30:35, i:i + 1, :]
            elif sentence_idx == 2:
                L_fw_feed_u = L_fw_u[10:15, i:i + 1, :]
                L_bw_feed_u = L_bw_u[25:30, i:i + 1, :]
            elif sentence_idx == 3:
                L_fw_feed_u = L_fw_u[15:20, i:i + 1, :]
                L_bw_feed_u = L_bw_u[20:25, i:i + 1, :]
            elif sentence_idx == 4:
                L_fw_feed_u = L_fw_u[20:25, i:i + 1, :]
                L_bw_feed_u = L_bw_u[15:20, i:i + 1, :]
            elif sentence_idx == 5:
                L_fw_feed_u = L_fw_u[25:30, i:i + 1, :]
                L_bw_feed_u = L_bw_u[10:15, i:i + 1, :]
            elif sentence_idx == 6:
                L_fw_feed_u = L_fw_u[30:35, i:i + 1, :]
                L_bw_feed_u = L_bw_u[5:10, i:i + 1, :]
            else:
                L_fw_feed_u = L_fw_u[35:40, i:i + 1, :]
                L_bw_feed_u = L_bw_u[0:5, i:i + 1, :]

            feed_dict = {placeholders["L_fw"]: L_fw_feed_u,
                         placeholders["B_fw"]: B_fw_u[:, i:i + 1, :],
                         placeholders["V_fw"]: V_fw_u[:, i:i + 1, :],
                         placeholders["L_bw"]: L_bw_feed_u,
                         placeholders["B_bw"]: B_bw_u[:, i:i + 1, :],
                         placeholders["V_bw"]: V_bw_u[:, i:i + 1, :],
                         placeholders["B_bin"]: B_bin_u[:, i:i + 1, :],
                         placeholders["L_len"]: L_len_u[i:i + 1] / 8,
                         placeholders["V_len"]: V_len_u[i:i + 1]}

            result = sess.run([L_output], feed_dict=feed_dict)
            save_latent(np.transpose(result[0], (1, 0, 2)), L_filenames_u[i], "recognition")
            result = result[0][:, 0, :].argmax(axis=1)
            target = L_fw_u[1:5, i, :].argmax(axis=1)
            target_second = L_fw_u[6:10, i, :].argmax(axis=1)
            target_third = L_fw_u[11:15, i, :].argmax(axis=1)
            target_fourth = L_fw_u[16:20, i, :].argmax(axis=1)
            target_fifth = L_fw_u[21:25, i, :].argmax(axis=1)
            target_sixth = L_fw_u[26:30, i, :].argmax(axis=1)
            target_seventh = L_fw_u[31:35, i, :].argmax(axis=1)
            target_eighth = L_fw_u[36:40, i, :].argmax(axis=1)
            if (result == target).all() or (result == target_second).all() or (result == target_third).all() or (
                    result == target_fourth).all() \
                    or (result == target_fifth).all() or (result == target_sixth).all() or (
                    result == target_seventh).all() or (result == target_eighth).all():
                print(True)  # Check if predicted descriptions match the original ones
            else:
                print(False)


if __name__ == "__main__":
    main()
