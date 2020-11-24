import tensorflow as tf
import numpy as np
from PIL import Image
from config import TrainConfig
import os, time
fc = tf.contrib.layers.fully_connected

# Read image and turn it into 120x160
def read_input_folder(images_path):
    all_file_list = []
    dir_list = []
    for (root, dirs, files) in os.walk(images_path):
        dir_list.append(root)
        file_list = []
        for file in files:
            file_list.append(os.path.join(root, file))
        file_list.sort()
        if(len(file_list)>0):
            all_file_list.append(file_list)

    all_file_list.sort()
    max_len = 1
    count = 0
    all_resized_images = np.zeros((len(all_file_list), max_len, 120, 160, 3))
    for file_list in all_file_list:
        num_of_images = len(file_list)
        file_list.sort()
        resized_images = []
        for i, image in enumerate(file_list):
            original_image = Image.open(image)
            resized_image = original_image.resize((160, 120))
            resized_image_arr = np.asarray(resized_image, np.float)
            resized_images.append(resized_image_arr)
        if num_of_images > max_len:
            max_len = num_of_images
            add_zeros = np.zeros((all_resized_images.shape[0],max_len-all_resized_images.shape[1], 120, 160, 3))
            all_resized_images = np.concatenate((all_resized_images, add_zeros), axis=1)
        all_resized_images[count] = np.asarray(resized_images)
        count += 1
    return all_resized_images

# Convolution
def conv_encoder(x, scope="conv_encoder"):
    with tf.variable_scope(scope, reuse=False):
        first_conv = tf.layers.conv2d(x, 8, 4, 2, padding='same')
        second_conv = tf.layers.conv2d(first_conv, 16, 4, 2, padding='same')
        third_conv = tf.layers.conv2d(second_conv, 32, 4, 2, padding='same')
        fourth_conv = tf.layers.conv2d(third_conv, 64, 8, 5, padding='same')
    return fourth_conv

# Fully connected layers
def dense_layers(enc):
    first_dense = fc(enc, 384, activation_fn=None)
    second_dense = fc(first_dense, 192, activation_fn=None)
    third_dense = fc(second_dense, 10, activation_fn=None)
    fourth_dense = fc(third_dense, 192, activation_fn=None)
    fifth_dense = fc(fourth_dense, 384, activation_fn=None)
    return third_dense, fifth_dense

# Deconvolution
def deconv_decoder(dense, scope="deconv_decoder"):
    with tf.variable_scope(scope, reuse=False):
        first_deconv = tf.layers.conv2d_transpose(dense, 32, 8, 5, padding='same')
        second_deconv = tf.layers.conv2d_transpose(first_deconv, 16, 4, 2, padding='same')
        third_deconv = tf.layers.conv2d_transpose(second_deconv, 8, 4, 2, padding='same')
        output = tf.layers.conv2d_transpose(third_deconv, 3, 4, 2, padding='same')
    return output


def main():
    # get the training configuration
    train_conf = TrainConfig()
    train_conf.set_conf("../train/train_conf.txt")
    seed = train_conf.seed
    batchsize = 32
    epoch = train_conf.epoch
    # set the save directory for features
    save_dir = train_conf.cae_save_dir
    if not os.path.exists(os.path.dirname(save_dir)):
        os.mkdir(os.path.dirname(save_dir))

    # get the dataset folder
    im_data_dir = train_conf.IM_dir
    resized_input = read_input_folder(im_data_dir)
    resized_input_batch = resized_input.reshape(resized_input.shape[0]*resized_input.shape[1],
                                     resized_input.shape[2], resized_input.shape[3], resized_input.shape[4])
    # get the dataset folder for testing
    if train_conf.test:
        im_data_dir_test = train_conf.IM_dir_test            # get the folder for test language descriptions
        resized_input_test = read_input_folder(im_data_dir_test)
        resized_input_test_batch = resized_input_test.reshape(resized_input_test.shape[0]*resized_input_test.shape[1],
                                     resized_input_test.shape[2], resized_input_test.shape[3], resized_input_test.shape[4])
    # Random Initialisation
    np.random.seed(seed)

    tf.reset_default_graph()            # Clear the default graph stack and reset the global default graph
    tf.set_random_seed(seed)            # Set the graph-level random seed

    placeholder = tf.placeholder(tf.float32, [batchsize, resized_input_batch.shape[1],
                                              resized_input_batch.shape[2], resized_input_batch.shape[3]], name="input_images")
    convolved = conv_encoder(placeholder)
    visual_features, dense = dense_layers(convolved)
    output = deconv_decoder(dense)

    # Calculate the loss
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(placeholder, output))

    # Graph for update operations
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)  # Use Adam Optimiser

    # Use GPU
    gpuConfig = tf.ConfigProto(
        gpu_options=tf.GPUOptions(allow_growth=True,
                                  per_process_gpu_memory_fraction=train_conf.gpu_use_rate),
        device_count={'GPU': 1})

    # Launch the graph in a session
    sess = tf.Session(config=gpuConfig)
    sess.run(tf.global_variables_initializer())  # run the session
    saver = tf.train.Saver(tf.global_variables())  # create a saver for the model

    # Training
    previous = time.time()  # time the training
    for step in range(epoch):
        batch_idx = np.random.permutation(resized_input_batch.shape[0])[:batchsize]
        feed_dict = {placeholder: resized_input_batch[batch_idx,:, :, :]}

        _, t = sess.run([train_step, loss],
                                 feed_dict=feed_dict)
        print("step:{} total:{}".format(step, t))

        if train_conf.test and (step + 1) % train_conf.test_interval == 0:
            batch_idx = np.random.permutation(resized_input_test_batch.shape[0])[:batchsize]
            feed_dict = {placeholder: resized_input_test_batch[batch_idx,:, :, :]}

            t = sess.run([loss],
                                  feed_dict=feed_dict)
            print("test")
            print("step:{} total:{}".format(step, t))

        if (step + 1) % train_conf.log_interval == 0:
            saver.save(sess, save_dir)
    past = time.time()
    print(past-previous)     # print the elapsed time

if __name__ == "__main__":
    main()