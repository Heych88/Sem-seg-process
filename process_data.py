import sys
import getopt
import os
import pickle
import numpy as np
from label_key import cityscapes_labels

# remove ROS system path link so we can use opencv
if("/opt/ros/kinetic/lib/python2.7/dist-packages" in sys.path):
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2


'''
    Maps the label data to the number of classes required for training.

    data <numpy array> 
            Label image data to be mapped

    return <list>
            Mapped label data 
'''
def mapLabel(data):
    # all the possible values that are in the observed in the label data. Palette must be given in sorted order
    palette = [label.id for label in cityscapes_labels]
    # key gives the new values you wish palette to be mapped to.
    key = np.array([label.trainId for label in cityscapes_labels])

    index = np.digitize(data.ravel(), palette, right=True)

    return np.array(key[index].reshape(data.shape).astype(np.uint8))


'''
    Processes all the images and labels in the training directory and store their
    address to an array.

    label_file_name <string> 
            File name of the labeled camera images
    train_imgs_dir <string> 
            Directory to find the rgb camera images
    train_label_dir <string> 
            Directory to find the labeled camera data images 
    output_dir <string> 
            Directory to save the processed image data too
    image_shape <int tuple> 
            Size of the image data to be resized too
            
    return <list>
            Address list of saved image data locations 
'''
def processData(label_file_name, train_imgs_dir, train_label_dir, output_dir, image_shape):

    address_list = []

    name_split = label_file_name.split("_gtFine_")

    # open and resize the input images
    file_name = name_split[0] + '_leftImg8bit'
    img_path = train_imgs_dir + file_name
    img = cv2.resize(cv2.imread(img_path+'.png', -1), (image_shape[1], image_shape[0]))

    # open and resize the label images
    label_path = train_label_dir + label_file_name
    label_img = cv2.imread(label_path, -1)
    # map ground truths to training labels
    label_img = cv2.resize(mapLabel(label_img), (image_shape[1], image_shape[0]))

    # save the input image
    img_out_name = output_dir + file_name
    #address_list.append(img_out_name + '.png')
    cv2.imwrite(img_out_name + '.png', img)

    # save the label image
    label_out_name = img_out_name + '_label'
    #address_list.append(label_out_name + '.png')
    cv2.imwrite(label_out_name + '.png', label_img)

    address_list.append([img_out_name + '.png', label_out_name + '.png'])

    # save the horizontal flipped input image
    img_out_name = img_out_name + '_horz_flip'
    #address_list.append(img_out_name + '.png')
    cv2.imwrite(img_out_name + '.png', cv2.flip(img, 1))

    # save the horizontal flipped label image
    label_out_name = label_out_name + '_horz_flip'
    #address_list.append(label_out_name + '.png')
    cv2.imwrite(label_out_name + '.png', cv2.flip(label_img, 1))

    address_list.append([img_out_name + '.png', label_out_name + '.png'])

    return address_list

'''
    Locates all the images and labels in the training directory and store their
    address to an array.

    input_rgb_dir <string> 
            Directory to find the rgb camera images
    input_label_dir <string> 
            Directory to find the labeled camera data images 
    output_dir <string> 
            Directory to save the processed image data too
    image_shape <int tuple> 
            Size of the image data to be resized too
'''
def loadAllData(input_rgb_dir, input_label_dir, output_dir, image_shape):

    address_list = []
    folder_num = 1 # keep track of folder number for display to user

    # Iterate through each folder and save the data into the one folder
    for root, dirs, files in os.walk(input_rgb_dir, topdown=False):
        for name in dirs:

            imgs_dir = (os.path.join(input_rgb_dir, name) + '/')
            labs_dir = (os.path.join(input_label_dir, name) + '/')

            if(not os.path.isdir(labs_dir)):
                print("No label folder {} found in {}. Skipping folder.".format(name, input_label_dir))
                continue

            for file_num, label_file_name in enumerate(os.listdir(labs_dir)):
                name_split = label_file_name.split("_gtFine_")

                if (name_split[1] == 'labelIds.png'):
                    print("\rprocessing file {}/{} in folder {}/{}     ".format(1 + (file_num) // 4,
                                                                                len(os.listdir(labs_dir)) // 4,
                                                                                folder_num, len(dirs)), end=' ')

                    address_list.extend(processData(label_file_name, imgs_dir, labs_dir, output_dir, image_shape))

            print()
            folder_num += 1

    print("\nTotal data ", len(address_list))

    saved_file_dir = output_dir + "cityscape_list.p"
    pickle.dump(address_list, open(saved_file_dir, "wb+"))

    print("Saved address data to file {}".format(saved_file_dir))


'''
    Prints error messages to screen and exits the program.

    msg <string> 
            Message to be printed to screen
    exit_code <int> 
            Exit code 
'''
def inputHelp(msg, exit_code):
    print(msg)
    print('process_data.py [-i --ifolder] <network inputfolder> [-l --lfolder] <network labelfolder> [-o --ofolder] <outputfolder>')
    sys.exit(exit_code)


if __name__ == '__main__':

    input_rgb_dir = 'data/train/'
    input_label_dir = 'data/label/'
    output_dir = 'data/output/'

    image_shape = (256, 512)

    try:
        opts, args = getopt.getopt(sys.argv[1:],'hil:o',["ifolder=","lfolder=","ofolder="])
    except getopt.GetoptError:
        inputHelp('Error collecting arguments.', 2)

    if(len(args) != 0):
        inputHelp('Error. Incorrect options and arguments supplied.', 2)

    for opt, arg in opts:
        if opt == '-h':
            inputHelp('', 0)
        elif opt in ("-i", "--ifolder"):
             input_rgb_dir = arg
        elif opt in ("-l", "--lfolder"):
            input_label_dir = arg
        elif opt in ("-o", "--ofolder"):
            output_dir = arg
        else:
            inputHelp('Unknown argument {}'.format(opt), 2)

    print('Input folder is {}'.format(input_rgb_dir))
    print('Label folder is {}'.format(input_label_dir))
    print('Output folder is {}'.format(output_dir))

    if(not os.path.exists(input_rgb_dir)):
        print("Directory [ {} ] does not exist.".format(input_rgb_dir))
        exit(1)
    if(not os.path.exists(input_label_dir)):
        print("Directory [ {} ] does not exist.".format(input_label_dir))
        exit(1)
    if(not os.path.exists(output_dir)):
        os.makedirs(output_dir)

    loadAllData(input_rgb_dir, input_label_dir, output_dir, image_shape)
