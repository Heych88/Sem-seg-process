import scipy
import numpy as np
from sklearn.utils import shuffle


def gen_batch_function(img_data, num_classes):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """

        shuf_img = shuffle(img_data)
        for offset in range(0, len(shuf_img), batch_size):
            batch_samples = shuf_img[offset:offset + batch_size]
            images = []
            onehot_images = []

            for image_file in batch_samples:

                image = scipy.misc.imread(image_file[0])
                label_image = scipy.misc.imread(image_file[1])

                # convert the gt_image label to onehot encoding
                #https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy
                onehot = (np.arange(num_classes) == label_image[:, :, None] - 1).astype(np.uint8)

                images.append(image)
                onehot_images.append(onehot)

            yield shuffle(np.array(images), np.array(onehot_images))

    return get_batches_fn