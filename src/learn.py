import tensorflow as tf

fc = tf.contrib.layers.fully_connected

from config import NetConfig, TrainConfig
from modules import *


def main():
    # get the network configuration (parameters such as number of layers and units)
    net_conf = NetConfig()
    net_conf.set_conf("../train/net_conf.txt")
    L_num_units = net_conf.L_num_units
    L_num_layers = net_conf.L_num_layers
    VB_num_units = net_conf.VB_num_units
    VB_num_layers = net_conf.VB_num_layers
    SHARE_DIM = net_conf.S_dim

    # get the training configuration (batch size, initialisation, num_of_iterations number, saving and loading directory)
    train_conf = TrainConfig()
    train_conf.set_conf("../train/train_conf.txt")
    seed = train_conf.seed
    batchsize = train_conf.batchsize
    num_of_iterations = train_conf.num_of_iterations
    save_dir = train_conf.save_dir
    if not os.path.exists(os.path.dirname(save_dir)):
        os.mkdir(os.path.dirname(save_dir))
    # get the dataset folders
    L_data_dir = train_conf.L_dir
    B_data_dir = train_conf.B_dir
    V_data_dir = train_conf.V_dir

    # get the descriptions
    L_fw, L_bw, L_bin, L_len, filenames = read_sequential_target(L_data_dir, True)
    print("Number of training patterns: ",
          len(filenames))  # print the number of training descriptions times six (actions recorded six times)

    # get the joint angles for actions
    B_fw, B_bw, B_bin, B_len = read_sequential_target(B_data_dir)
    # normalise the joint angles between -1 and 1
    B_fw = 2 * ((B_fw - B_fw.min()) / (B_fw.max() - B_fw.min())) - 1
    B_bw = 2 * ((B_bw - B_bw.min()) / (B_bw.max() - B_bw.min())) - 1
    # get the visual features for action images
    V_fw, V_bw, V_bin, V_len = read_sequential_target(V_data_dir)
    # normalise the visual features between -1 and 1
    V_fw = 2 * ((V_fw - V_fw.min()) / (V_fw.max() - V_fw.min())) - 1
    V_bw = 2 * ((V_bw - V_bw.min()) / (V_bw.max() - V_bw.min())) - 1
    # create variables for data shapes
    L_shape = L_fw.shape
    B_shape = B_fw.shape
    V_shape = V_fw.shape

    # go through every step for testing data
    if train_conf.test:
        L_data_dir_test = train_conf.L_dir_test  # get the folder for test language descriptions
        B_data_dir_test = train_conf.B_dir_test  # get the folder for test action joint angles
        V_data_dir_test = train_conf.V_dir_test  # get the folder for test visual features
        L_fw_u, L_bw_u, L_bin_u, L_len_u, filenames_u = read_sequential_target(L_data_dir_test,
                                                                               True)  # get test descriptions
        print("Number of testing patterns: ",
              len(filenames_u))  # print the number of test descriptions times six (actions recorded six times)
        B_fw_u, B_bw_u, B_bin_u, B_len_u = read_sequential_target(
            B_data_dir_test)  # get the joint angles for test actions
        # normalise the joint angles between -1 and 1
        B_fw_u = 2 * ((B_fw_u - B_fw_u.min()) / (B_fw_u.max() - B_fw_u.min())) - 1
        B_bw_u = 2 * ((B_bw_u - B_bw_u.min()) / (B_bw_u.max() - B_bw_u.min())) - 1
        V_fw_u, V_bw_u, V_bin_u, V_len_u = read_sequential_target(
            V_data_dir_test)  # get the visual features for test actions
        # normalise the visual features between -1 and 1
        V_fw_u = 2 * ((V_fw_u - V_fw_u.min()) / (V_fw_u.max() - V_fw_u.min())) - 1
        V_bw_u = 2 * ((V_bw_u - V_bw_u.min()) / (V_bw_u.max() - V_bw_u.min())) - 1
        L_shape_u = L_fw_u.shape
        B_shape_u = B_fw_u.shape
        V_shape_u = V_fw_u.shape

    # Random Initialisation
    np.random.seed(seed)

    tf.reset_default_graph()  # Clear the default graph stack and reset the global default graph
    tf.set_random_seed(seed)  # Set the graph-level random seed

    # Create a placeholder dictionary for tensors
    placeholders = make_placeholders(L_shape, B_shape, V_shape, batchsize)

    # Encoding
    # Get the final language state by feeding the encoder with descriptions (in the reverse order)
    # Reversing the input description introduces many short term dependencies between the source
    # and the target sentence which makes the optimisation problem easier (Sutskever et al. 2014)
    L_enc_final_state = encoder(placeholders["L_bw"],
                                placeholders["L_len"],
                                n_units=L_num_units,
                                n_layers=L_num_layers,
                                scope="L_encoder")

    # Concatenate the joint angles with visual features (reverse order)
    # the LSTM learns much better when the input is reversed
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
    L_shared = fc(L_enc_final_state, SHARE_DIM,
                  # Feed the final description state into a feedforward layer (output: z_dsc)
                  activation_fn=None, scope="L_share")
    VB_shared = fc(VB_enc_final_state, SHARE_DIM,
                   # Feed the final action state into a feedforward layer (output: z_act)
                   activation_fn=None, scope="VB_share")

    # Get the initial decoding states
    L_dec_init_state = fc(L_shared, L_num_units * L_num_layers * 2,  # Initial decoder state for descriptions (h0_dec)
                          activation_fn=None, scope="L_postshare")
    VB_dec_init_state = fc(VB_shared, VB_num_units * VB_num_layers * 2,  # Initial decoder state for actions (h0_dec)
                           activation_fn=None, scope="VB_postshare")

    # Decoding
    L_output = L_decoder(placeholders["L_fw"][0],  # Get the reconstructed description via the language decoder
                         L_dec_init_state,
                         length=L_shape[0],
                         out_dim=L_shape[2],
                         n_units=L_num_units,
                         n_layers=L_num_layers,
                         scope="L_decoder")
    B_output = VB_decoder(placeholders["V_fw"],  # Get the reconstructed action via the action decoder
                          placeholders["B_fw"][0],
                          VB_dec_init_state,
                          length=B_shape[0],
                          out_dim=B_shape[2],
                          n_units=VB_num_units,
                          n_layers=VB_num_layers,
                          scope="VB_decoder")

    # Calculate the losses
    with tf.name_scope('loss'):
        L_output = L_output  # no need to multiply binary array
        B_output = B_output * placeholders["B_bin"][1:]
        L_loss = tf.reduce_mean(-tf.reduce_sum(placeholders["L_fw"][1:] * tf.log(L_output),  # description loss
                                               reduction_indices=[2]))
        B_loss = tf.reduce_mean(tf.square(B_output - placeholders["B_fw"][1:]))  # action loss (MSE)
        share_loss = aligned_discriminative_loss(L_shared, VB_shared)  # binding loss
        loss = net_conf.L_weight * L_loss + net_conf.B_weight * B_loss + net_conf.S_weight * share_loss  # total loss

    loss_sum = tf.summary.scalar('Loss', loss)  # Loss summary to write to Tensorboard
    test_loss = tf.summary.scalar('Test Loss', loss)  # Test loss summary to write to Tensorboard
    # Graph for update operations
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(learning_rate=train_conf.learning_rate).minimize(loss)  # Use Adam Optimiser

    # Use GPU
    gpuConfig = tf.ConfigProto(
        gpu_options=tf.GPUOptions(allow_growth=True,
                                  per_process_gpu_memory_fraction=train_conf.gpu_use_rate),
        device_count={'GPU': 1})

    # Initialise the file writer for Tensorboard
    writer = tf.summary.FileWriter('.././logs')

    # Launch the graph in a session
    sess = tf.Session(config=gpuConfig)
    sess.run(tf.global_variables_initializer())  # run the session
    saver = tf.train.Saver(tf.global_variables())  # create a saver for the model

    # Training
    previous = time.time()  # time the training
    for step in range(num_of_iterations):
        batch_idx = np.random.permutation(B_shape[1])[:batchsize]
        feed_dict = {placeholders["L_fw"]: L_fw[:, batch_idx, :],
                     placeholders["B_fw"]: B_fw[:, batch_idx, :],
                     placeholders["V_fw"]: V_fw[:, batch_idx, :],
                     placeholders["L_bw"]: L_bw[:, batch_idx, :],
                     placeholders["B_bw"]: B_bw[:, batch_idx, :],
                     placeholders["V_bw"]: V_bw[:, batch_idx, :],
                     placeholders["B_bin"]: B_bin[:, batch_idx, :],
                     placeholders["L_len"]: L_len[batch_idx],
                     placeholders["V_len"]: V_len[batch_idx]}

        _, l, b, s, t, l_sum = sess.run([train_step,
                                         L_loss,
                                         B_loss,
                                         share_loss,
                                         loss, loss_sum],
                                        feed_dict=feed_dict)
        print("step:{} total:{}, language:{}, behavior:{}, share:{}".format(step, t, l, b, s))
        writer.add_summary(l_sum, step)
        # Do the same for the test set
        if train_conf.test and (step + 1) % train_conf.test_interval == 0:
            batch_idx = np.random.permutation(B_shape_u[1])[:batchsize]
            feed_dict = {placeholders["L_fw"]: L_fw_u[:, batch_idx, :],
                         placeholders["B_fw"]: B_fw_u[:, batch_idx, :],
                         placeholders["V_fw"]: V_fw_u[:, batch_idx, :],
                         placeholders["L_bw"]: L_bw_u[:, batch_idx, :],
                         placeholders["B_bw"]: B_bw_u[:, batch_idx, :],
                         placeholders["V_bw"]: V_bw_u[:, batch_idx, :],
                         placeholders["B_bin"]: B_bin_u[:, batch_idx, :],
                         placeholders["L_len"]: L_len_u[batch_idx],
                         placeholders["V_len"]: V_len_u[batch_idx]}

            l, b, s, t, t_loss = sess.run([L_loss, B_loss, share_loss, loss, test_loss],
                                          feed_dict=feed_dict)
            print("test")
            print("step:{} total:{}, language:{}, behavior:{}, share:{}".format(step, t, l, b, s))
            writer.add_summary(t_loss, step)
        if (step + 1) % train_conf.log_interval == 0:
            saver.save(sess, save_dir)
    writer.flush()
    writer.close()
    past = time.time()
    print(past - previous)  # print the elapsed time


if __name__ == "__main__":
    main()