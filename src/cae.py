import tensorflow as tf
import numpy as np
from PIL import Image
from config import TrainConfig
import os, time


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
            resized_image_arr = np.asarray(resized_image, np.float) / 255.0
            resized_images.append(resized_image_arr)
        if num_of_images > max_len:
            max_len = num_of_images
            add_zeros = np.zeros((all_resized_images.shape[0],max_len-all_resized_images.shape[1], 120, 160, 3))
            all_resized_images = np.concatenate((all_resized_images, add_zeros), axis=1)
        all_resized_images[count][:len(resized_images)] = np.asarray(resized_images)
        count += 1
    return [all_resized_images, all_file_list]

# Convolution
def conv_encoder(x, scope="conv_encoder"):
    with tf.variable_scope(scope, reuse=False):
        first_conv = tf.layers.conv2d(x, 8, 4, 2, padding='same', activation=tf.nn.relu, name="enc_conv1")
        second_conv = tf.layers.conv2d(first_conv, 16, 4, 2, padding='same', activation=tf.nn.relu, name="enc_conv2")
        third_conv = tf.layers.conv2d(second_conv, 32, 4, 2, padding='same', activation=tf.nn.relu, name="enc_conv3")
        fourth_conv = tf.layers.conv2d(third_conv, 64, 8, 5, padding='same', activation=tf.nn.relu, name="enc_conv4")
    return fourth_conv

# Fully connected layers (Bottleneck)
def dense_layers(enc):
    flattened = tf.layers.flatten(enc)    # Flatten it before processing it through fully connected layers
    first_dense = tf.layers.dense(flattened, 384, activation=None, name="bneck_fc1")
    second_dense = tf.layers.dense(first_dense, 192, activation=None, name="bneck_fc2")
    third_dense = tf.layers.dense(second_dense, 10, activation=None,  name="bneck_fc3")     # Visual features with 10 dimensions (will be used later)
    fourth_dense = tf.layers.dense(third_dense, 192, activation=None, name="bneck_fc4")
    fifth_dense = tf.layers.dense(fourth_dense, 384, activation=None, name="bneck_fc5")
    return third_dense, fifth_dense

# Deconvolution
def deconv_decoder(dense, scope="deconv_decoder"):
    with tf.variable_scope(scope, reuse=False):
        reshaped = tf.reshape(dense, (dense.shape[0], 3, 4, -1))     # Reshape it back to 3D before deconvolution
        first_deconv = tf.layers.conv2d_transpose(reshaped, 32, 8, 5, padding='same', activation=tf.nn.relu, name="dec_conv1")
        second_deconv = tf.layers.conv2d_transpose(first_deconv, 16, 4, 2, padding='same', activation=tf.nn.relu, name="dec_conv2")
        third_deconv = tf.layers.conv2d_transpose(second_deconv, 8, 4, 2, padding='same', activation=tf.nn.relu, name="dec_conv3")
        output = tf.layers.conv2d_transpose(third_deconv, 3, 4, 2, padding='same', activation=tf.nn.sigmoid, name="dec_conv4_output")
    return output

# Extract 10 dimensional visual features from images
def extract_visual_features():
    # get the training configuration
    train_conf = TrainConfig()
    train_conf.set_conf("../train/train_conf.txt")

    im_data_dir = "../target/image_train"
    resized_input, filenames = read_input_folder(im_data_dir)
    batchsize = 1
    placeholder = tf.placeholder(tf.float32, [resized_input.shape[1], resized_input.shape[2],
                                              resized_input.shape[3], resized_input.shape[4]],
                                 name="input_images")
    # Model pipeline
    convolved = conv_encoder(placeholder)     # Encoder block with convolutions
    visual_features, _ = dense_layers(convolved)    # Bottleneck with visual features
    # Use GPU
    gpuConfig = tf.ConfigProto(
        gpu_options=tf.GPUOptions(allow_growth=True,
                                  per_process_gpu_memory_fraction=0.4),
        device_count={'GPU': 1})

    sess = tf.Session(config=gpuConfig)        # Launch the graph in a session
    sess.run(tf.global_variables_initializer())        # run the session
    saver = tf.train.Saver(tf.global_variables())      # create a saver for the model
    saver.restore(sess, train_conf.cae_save_dir)         # restore previously saved variables

    # Feed the dataset as input
    for i in range(resized_input.shape[0]):
        feed_dict = {placeholder: resized_input[i, :, :, :, :]}
        result = sess.run(visual_features, feed_dict=feed_dict)    # run the session
        filename = filenames[i][0].split(os.path.sep)[-2]
        name = "../train/visual_feature_extraction/171107/"+ filename + ".txt"
        dir_hierarchy = name.split("/")
        dir_hierarchy = filter(lambda z: z != "..", dir_hierarchy)
        save_name = os.path.join(*dir_hierarchy)
        #dirname = '../train/' + dirname
        save_name = os.path.join("..", save_name)
        if not os.path.exists(os.path.dirname(save_name)):
            os.makedirs(os.path.dirname(save_name))
        np.savetxt(save_name, result[:len(filenames[i])], fmt="%.6f")

        #save_latent(np.transpose(result, (1, 0, 2)), B_filenames[i], "visual_feature_extraction")  # save the features

# Reconstruct images
def reconstruct():
    # get the training configuration
    train_conf = TrainConfig()
    train_conf.set_conf("../train/train_conf.txt")

    im_data_dir = train_conf.IM_dir_test
    resized_input, filenames = read_input_folder(im_data_dir)
    batchsize = 1
    placeholder = tf.placeholder(tf.float32, [batchsize, resized_input.shape[2],
                                              resized_input.shape[3], resized_input.shape[4]],
                                 name="input_images")
    # Model pipeline
    convolved = conv_encoder(placeholder)     # Encoder block with convolutions
    _, dense = dense_layers(convolved)    # Bottleneck with fully connected layers
    output = deconv_decoder(dense)     # Decoder block with deconvolutions
    # Use GPU
    gpuConfig = tf.ConfigProto(
        gpu_options=tf.GPUOptions(allow_growth=True,
                                  per_process_gpu_memory_fraction=0.4),
        device_count={'GPU': 1})

    sess = tf.Session(config=gpuConfig)        # Launch the graph in a session
    sess.run(tf.global_variables_initializer())        # run the session
    saver = tf.train.Saver(tf.global_variables())      # create a saver for the model
    saver.restore(sess, train_conf.cae_save_dir)         # restore previously saved variables

    # Feed the dataset as input
    for i in range(resized_input.shape[1]):
        feed_dict = {placeholder: resized_input[:, i, :, :, :]}
        result = sess.run(output, feed_dict=feed_dict)   # run the session
        image_name = filenames[0][i].split(os.path.sep)[-1]
        name = "../train/20201123_Embodied_Language_Learning-Nico2Blocks_test/reconstructed/" + image_name
        dir_hierarchy = name.split("/")
        dir_hierarchy = filter(lambda z: z != "..", dir_hierarchy)
        save_name = os.path.join(*dir_hierarchy)
        save_name = os.path.join("..", save_name)
        if not os.path.exists(os.path.dirname(save_name)):
            os.makedirs(os.path.dirname(save_name))
        reconstructed_im = Image.fromarray(np.uint8(result[0]*255))
        reconstructed_im.save(save_name)

def main():
    # get the training configuration
    train_conf = TrainConfig()
    train_conf.set_conf("../train/train_conf.txt")
    seed = train_conf.seed
    batchsize = train_conf.batchsize
    num_of_iterations = train_conf.num_of_iterations
    # set the save directory for the model
    save_dir = train_conf.cae_save_dir
    if not os.path.exists(os.path.dirname(save_dir)):
        os.mkdir(os.path.dirname(save_dir))

    # get the dataset folder
    im_data_dir = train_conf.IM_dir
    resized_input, _ = read_input_folder(im_data_dir)
    resized_input_batch = resized_input.reshape(resized_input.shape[0]*resized_input.shape[1],
                                     resized_input.shape[2], resized_input.shape[3], resized_input.shape[4])
    # get the dataset folder for testing
    if train_conf.test:
        im_data_dir_test = train_conf.IM_dir_test            # get the folder for test language descriptions
        resized_input_test, _ = read_input_folder(im_data_dir_test)
        resized_input_test_batch = resized_input_test.reshape(resized_input_test.shape[0]*resized_input_test.shape[1],
                                     resized_input_test.shape[2], resized_input_test.shape[3], resized_input_test.shape[4])
    # Random Initialisation
    np.random.seed(seed)

    tf.reset_default_graph()            # Clear the default graph stack and reset the global default graph
    tf.set_random_seed(seed)            # Set the graph-level random seed

    # Define a placeholder for the input
    placeholder = tf.placeholder(tf.float32, [batchsize, resized_input_batch.shape[1],
                                              resized_input_batch.shape[2], resized_input_batch.shape[3]], name="input_images")
    # Model pipeline
    convolved = conv_encoder(placeholder)     # Encoder block with convolutions
    _, dense = dense_layers(convolved)    # Bottleneck with fully connected layers
    output = deconv_decoder(dense)     # Decoder block with deconvolutions

    # Calculate the loss
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(placeholder, output))  # Binary cross-entropy loss

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
    for step in range(num_of_iterations):
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
    #main()
    #reconstruct()
    extract_visual_features()
