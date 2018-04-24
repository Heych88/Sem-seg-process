import warnings
import sys
import getopt
import os
import pickle
import numpy as np
from pathlib import Path

import model2
import generator as gen
import tensorflow as tf
from tensorboard.plugins.beholder import Beholder

#import helper
from label_key import cityscapes_labels

# remove ROS system path link so we can use opencv
if("/opt/ros/kinetic/lib/python2.7/dist-packages" in sys.path):
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

def Placeholders(image_shape, num_classes):
    '''
    Input and label placeholders for the model
    :return: Model input, model label
    '''

    label_shape = (None,) + image_shape + (num_classes,)
    img_shape = (None,) + image_shape + (3,)

    with tf.name_scope('placholders'):
        x = tf.placeholder(tf.float32, img_shape, name='input_data')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        is_training = tf.placeholder(tf.bool, name='training')
        y = tf.placeholder(tf.int16, label_shape, name='label_data')

    with tf.name_scope('input_data_images'):
        # display some random sample images to verify the data is as we expect
        number_of_images = 12
        tf.summary.image('input', x, number_of_images)

    return x, learning_rate, is_training, y


def evaluate(logits, labels, num_classes):

    with tf.name_scope('mean_iou'):
        flat_soft = tf.reshape(tf.nn.softmax(logits=logits), [-1, num_classes])
        flat_labels = tf.reshape(labels, [-1, num_classes])

        return tf.metrics.mean_iou(flat_labels, flat_soft, num_classes)


def create_optimiser(nn_last_layer, y_label, learning_rate, num_classes):
    '''
    Creates the optimiser functions for training the network
    :param x: Model input placeholder
    :param correct_label: Model label placeholder
    :return: Network output, Tensorboard tensor, optimiser function, accuracy tensor
    '''

    with tf.name_scope('model_logits'):
        # run the model
        nn_last_layer = tf.identity(nn_last_layer, name='model_output')
        logits = tf.reshape(nn_last_layer, (-1, num_classes))
        # Name logits Tensor, so that is can be loaded from disk after training
        logits = tf.identity(logits, name='logits')

    with tf.name_scope('loss'):
        # get the final output of the model and find the loss
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=y_label)
        loss = tf.reduce_mean(cross_entropy, name='loss')
        #tf.summary.scalar('cross_entropy', cross_entropy)
        tf.summary.scalar('mean_loss', loss)

    with tf.name_scope('train'):
        # Update the model to predict better results based of the training loss
        optimiser = tf.train.AdamOptimizer(learning_rate)
        # train the model to minimise the loss
        optimiser = optimiser.minimize(loss, name='training')

    with tf.name_scope('mean_iou'):
        flat_soft = tf.reshape(tf.nn.softmax(logits=nn_last_layer), [-1, num_classes])
        flat_labels = tf.reshape(y_label, [-1, num_classes])

        mean_iou = tf.metrics.mean_iou(flat_labels, flat_soft, num_classes, name='mean_iou')

    # Merge all the summaries and write them to log_path location
    merged_summary = tf.identity(tf.summary.merge_all(), name='merged_summary')

    return logits, loss, optimiser, mean_iou, merged_summary


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, learning_rate, is_training, mean_iou, merged_summary, log_path, save_dir):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    # create tensorboard session at location log_path and save the graph there
    writer = tf.summary.FileWriter(log_path, graph=sess.graph)
    beholder = Beholder(log_path)

    saver = tf.train.Saver()

    images = []
    labels = []

    # Traing the model
    print("Training")
    for epoch in range(epochs):
        # train with ALL the training data per epoch, Training each pass with
        # batches of data with a batch_size count
        batch = 0
        for images, labels in get_batches_fn(batch_size):
            summary, _, loss = sess.run([merged_summary, train_op, cross_entropy_loss],
                                        feed_dict={input_image: images,
                                                   correct_label: labels,
                                                   is_training: True,
                                                   learning_rate: 0.01})
            batch += 1

            # add summaries to tensorboard
            writer.add_summary(summary, (epoch+1)*batch)

            print('Epoch {}, batch: {}, loss: {} '.format(epoch + 1, batch, loss))

        # check the accuracy of the model against the validation set
        # validation_accuracy = sess.run(accuracy, feed_dict={x: x_valid_reshape, y:one_hot_valid})
        iou = sess.run([mean_iou],
                       feed_dict={input_image: images, correct_label: labels, is_training: False})
        iou_sum = iou[0][0]

        # print out the models accuracies.
        # to print on the same line, add \r to start of string
        sys.stdout.write("EPOCH {}. IOU = {:.3f}\n".format(epoch + 1, iou_sum))

        beholder.update(session=sess)
        saver.save(sess, save_dir, epoch)

    saver_path = saver.save(sess, save_dir)
    print("Model saved in path: %s" % saver_path)

    writer.close()

def load_last_model(sess, restore_dir):
    '''
    Loads the last checkpoint data for continual training or predictions
    :param sess: Current tensorflow session
    :return: Model input, model label, tensorboard tensor, optimiser function, accuracy tensor
    '''

    # Load meta graph and restore weights
    saver = tf.train.import_meta_graph(restore_dir + 'model_final.meta')
    saver.restore(sess, tf.train.latest_checkpoint(restore_dir))

    # view all the graph tensor names
    #for op in sess.graph.get_operations():
    #    print(str(op.name), file=open("output.txt", "a"))

    # Get Tensors from loaded model
    graph = tf.get_default_graph()

    loaded_x = graph.get_tensor_by_name('placholders/input_data:0')
    loaded_y = graph.get_tensor_by_name('placholders/label_data:0')
    loaded_lr = graph.get_tensor_by_name('placholders/learning_rate:0')
    loaded_is_training = graph.get_tensor_by_name('placholders/training:0')
    loaded_merged_summary = graph.get_tensor_by_name('Merge/MergeSummary:0')
    model_output = graph.get_tensor_by_name('model/RS_out/RS_out_RU_conv_add:0')
    loaded_loss = graph.get_tensor_by_name('loss/loss:0')

    loaded_training = graph.get_operation_by_name("train/training")

    return loaded_x, loaded_y, loaded_lr, loaded_is_training, loaded_merged_summary, model_output, loaded_training, loaded_loss


def run(image_shape, train_list, num_classes):
    # Check for a GPU
    if not tf.test.gpu_device_name():
        warnings.warn('No GPU found. Please use a GPU to train your neural network.')
    else:
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    epochs = 1
    batch_size = 4

    runs_dir = './runs'
    restore_dir = "./model/checkpoints/"
    log_path = '/tmp/tensorboard/data/sem_seg'

    # Create function to get batches
    train_generator = gen.gen_batch_function(train_list, num_classes)

    if os.path.isfile(restore_dir + 'model_final.meta'):
        # continue training from the last checkpoint
        with tf.Session() as sess:
            x_input, y_label, learning_rate, is_training, merged_summary, model_output, optimiser, loss = load_last_model(sess, restore_dir)

            val_mean_iou = evaluate(model_output, y_label, num_classes)

            sess.run(tf.local_variables_initializer())

            train_nn(sess, epochs, batch_size, train_generator, optimiser, loss, x_input,
                     y_label, learning_rate, is_training, val_mean_iou, merged_summary, log_path,
                     restore_dir + 'model_final')

            #predict(logits)


    else:
        # create a new model and start training
        x_input, learning_rate, is_training, y_label = Placeholders(image_shape, num_classes)
        #nn_last_layer = model.networkModel(x_input, num_classes, 'model', is_training)
        nn_last_layer = model2.model(x_input, num_classes, is_training, name='model')
        logits, loss, optimiser, mean_iou, merged_summary = create_optimiser(nn_last_layer, y_label, learning_rate, num_classes)

        val_mean_iou = evaluate(nn_last_layer, y_label, num_classes)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            #train_model(sess, x, y, merged_summary, training, accuracy)
            train_nn(sess, epochs, batch_size, train_generator, optimiser, loss, x_input,
                     y_label, learning_rate, is_training, val_mean_iou, merged_summary, log_path, restore_dir+'model_final')

            #predict(logits)

            # Save inference data using helper.save_inference_samples
            #helper.save_inference_samples(runs_dir, sess, image_shape, logits, x_input, is_training)


    print("\nFinished")
    print("Run the command line:\n"
          "--> tensorboard --logdir={} "
          "\nThen open http://0.0.0.0:6006/ into your web browser"
          .format(log_path))


'''
    Prints error messages to screen and exits the program.

    msg <string> 
            Message to be printed to screen
    exit_code <int> 
            Exit code 
'''
def inputHelp(msg, exit_code):
    print(msg)
    print('main.py [-i --ifile] <data input pickle file> [-s --shape] <input image shape>')
    sys.exit(exit_code)


if __name__ == '__main__':

    input_dir = 'data/output/cityscape_list.p'
    num_classes = max([label.trainId for label in cityscapes_labels])+1
    image_shape = (256, 512)

    try:
        opts, args = getopt.getopt(sys.argv[1:],'hi:s',["ifile=","shape="])
    except getopt.GetoptError:
        inputHelp('Error collecting arguments.', 2)

    if(len(args) != 0):
        inputHelp('Error. Incorrect options and arguments supplied.', 2)

    for opt, arg in opts:
        if opt == '-h':
            inputHelp('', 0)
        elif opt in ("-i", "--ifile"):
            input_dir = arg
        elif opt in ("-s", "--shape"):
            image_shape = arg
        else:
            inputHelp('Unknown argument {}'.format(opt), 2)

    if (not os.path.exists(input_dir)):
        print("Directory [ {} ] does not exist.".format(input_dir))
        exit(1)

    try:
        if(len(image_shape) != 2):
            print("image_shape [ {} ] is not in the form (1,2).".format(image_shape))
            exit(1)
    except:
        print("image_shape [ {} ] is not in the form (1,2).".format(image_shape))
        exit(1)

    print('Input folder is {}'.format(input_dir))
    print('Image input shape is {}'.format(image_shape))

    # load the pickled data list
    print("Loading data from {}".format(input_dir))
    address_list = []
    try:
        address_list = pickle.load(open(input_dir, "rb"))
    except:
        print("{} is not a pickle data list.".format(input_dir))
        exit(1)

    print("Total data ", len(address_list))
    print("Total number of classes ", num_classes)

    # create the model checkpoint directory to save the models progress
    if (not os.path.exists('./model/checkpoints/')):
        os.makedirs('./model/checkpoints/')

    run(image_shape, address_list, num_classes)