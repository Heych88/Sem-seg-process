import os.path
import scipy.misc
import shutil
import time
import tensorflow as tf
from glob import glob
import cv2
import numpy as np

from label_key import cityscapes_labels


# function for colorizing a label image:
def testDataToColorImg(img, image_shape):

    color_palette = {label.id : label.color for label in cityscapes_labels}  # all the gt_image values observed in the image data

    img_height = image_shape[0]
    img_width = image_shape[1]

    unique, counts = np.unique(img, return_counts=True)

    out_image = np.zeros((img_height, img_width, 3))
    for row in range(img_height):
        for col in range(img_width):
            pixel_value = img[row, col]
            out_image[row, col] = np.array(color_palette[pixel_value])

    return out_image

def gen_test_output(sess, logits, image_pl, data_folder, image_shape, is_training):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep probability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(data_folder + '*.png'):

        image = cv2.resize(cv2.imread(image_file), (image_shape[1], image_shape[0]))

        logit = sess.run(logits, feed_dict={image_pl: [image], is_training: False})
        y = (tf.nn.softmax(logit)).eval()

        gray = np.reshape(y.argmax(axis=1), image_shape)
        colour = np.uint8(testDataToColorImg(gray, image_shape))
        street = cv2.addWeighted(image, 0.35, colour, 0.65, 0)

        yield os.path.basename(image_file), np.array(street)


def save_inference_samples(runs_dir, sess, image_shape, logits, input_image, is_training):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    print('Training Finished. Saving test images')
    # Run NN on test images and save them to HD
    image_outputs = gen_test_output(
        sess, logits, input_image, 'data/test/lindau/', image_shape, is_training)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)

    print('Images saved to: {}'.format(output_dir))
