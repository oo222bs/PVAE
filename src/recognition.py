import tensorflow as tf
fc = tf.contrib.layers.fully_connected


from src.config import NetConfig, TrainConfig
from src.data_util import read_sequential_target, save_latent
from src.modules import *

def main():
    net_conf = NetConfig()
    net_conf.set_conf("../train/net_conf.txt")

    L_num_units = net_conf.L_num_units
    L_num_layers = net_conf.L_num_layers
    VB_num_units = net_conf.VB_num_units
    VB_num_layers = net_conf.VB_num_layers
    SHARE_DIM = net_conf.S_dim
    
    train_conf = TrainConfig()
    train_conf.set_conf("../train/train_conf.txt")
    
    batchsize = 1
    save_dir = train_conf.save_dir
    
    L_data_dir = train_conf.L_dir
    B_data_dir = train_conf.B_dir
    V_data_dir = train_conf.V_dir
    L_fw, L_bw, L_bin, L_len, L_filenames = read_sequential_target(L_data_dir, True)
    print(len(L_filenames))
    B_fw, B_bw, B_bin, B_len, B_filenames = read_sequential_target(B_data_dir, True)
    V_fw, V_bw, V_bin, V_len = read_sequential_target(V_data_dir)
    L_shape = L_fw.shape
    B_shape = B_fw.shape
    V_shape = V_fw.shape


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
        
    tf.reset_default_graph()

    placeholders = make_placeholders(L_shape, B_shape, V_shape, batchsize)

    ##### encoding #####
    VB_input = tf.concat([placeholders["V_bw"],
                          placeholders["B_bw"]],
                         axis=2)
    VB_enc_final_state = encoder(VB_input,
                                 placeholders["V_len"], 
                                 n_units=VB_num_units,
                                 n_layers=VB_num_layers,
                                 scope="VB_encoder")

    ##### sharing #####
    VB_shared = fc(VB_enc_final_state, SHARE_DIM,
                   activation_fn=None, scope="VB_share")  
    L_dec_init_state = fc(VB_shared, L_num_units*L_num_layers*2,
                          activation_fn=None, scope="L_postshare")

    ##### decoding #####
    L_output = L_decoder(placeholders["L_fw"][0],
                         L_dec_init_state,
                         length=L_shape[0],
                         out_dim=L_shape[2],
                         n_units=L_num_units,
                         n_layers=L_num_layers,
                         scope="L_decoder")
    L_output = tf.contrib.seq2seq.hardmax(L_output)

    gpuConfig = tf.ConfigProto(
        gpu_options=tf.GPUOptions(allow_growth=True,
                                  per_process_gpu_memory_fraction=train_conf.gpu_use_rate),
        device_count={'GPU': 1})
        
    sess = tf.Session(config=gpuConfig)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, save_dir)    
    
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

        result = sess.run([L_output], feed_dict=feed_dict)
        save_latent(np.transpose(result[0], (1, 0, 2)), L_filenames[i], "recognition")
        r = result[0][:,0,:].argmax(axis=1)
        t = L_fw[1:,i,:].argmax(axis=1)
        print((r == t).all())


    if train_conf.test:
        print("test!!!!!!!!!!!!!!!!")
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
            result = sess.run([L_output], feed_dict=feed_dict)
            save_latent(np.transpose(result[0], (1, 0, 2)), L_filenames_u[i], "recognition")
            result = result[0][:,0,:].argmax(axis=1)
            target = L_fw_u[1:,i,:].argmax(axis=1)
            print((result == target).all())

if __name__ == "__main__":
    main()
