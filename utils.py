import sys
import skimage
import random as rd
import numpy as np
import os
from PIL import Image
import cv2
import matplotlib

matplotlib.use('Agg')

# path to the directory with the DataAugmentationForObjectDetection
sys.path.append('/Users/dariatunina/mach-lerinig/DataAugmentationForObjectDetection')
# sys.path.append('/home.stud/tunindar/DataAugmentationForObjectDetection')

from data_aug.data_aug import *
from data_aug.bbox_util import *

blues = [
    [0, 0, 139],  # darkblue
    [0, 0, 128],  # navy
    [0, 0, 205],  # mediumblue
    # [0, 0, 255],  # blue
    [25, 25, 112],  # midnightblue
    # [89, 28, 212],  # bic blue pen
    [0, 15, 85]  # blue ink pen
]


def choose_blue_colour():
    """ Chooses the RGB colour from array of blue RGB colours

    :return: the chosen colour
    """
    return blues[rd.randint(0, len(blues) - 1)]


def rotate_img(img, bboxes):
    """ Randomly rotates an image

    :param img: image to rotate
    :param bboxes: bounding boxes of an objects
    :return: rotated image
    """
    return RandomRotate(10)(img.copy(), bboxes.copy())


def shear_img(img, bboxes):
    """ Randomly shears an image

    :param img: image to shear
    :param bboxes: bounding boxes of an objects
    :return: sheared image
    """
    return RandomShear(0.2)(img.copy(), bboxes.copy())


def rotate_and_shear_img(img, bboxes):
    """ Randomly rotates and shears an image

    :param img: image to rotate and shear
    :param bboxes: bounding boxes of an objects
    :return: rotated and sheared image
    """
    img, bboxes = rotate_img(img, bboxes)
    return shear_img(img, bboxes)


def save_image(image, path):
    """ Saves image to the given path

    :param image: image to save
    :param path: path to save image to
    :return:
    """
    sizes = np.shape(image)
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(width / height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(image)
    plt.savefig(path, dpi=height)
    plt.close()


def save_labels(borders, path_lbl, i, j, width, height):
    """ Saves labels about each object on the image to one txt-file

    :param borders: array of bounding boxes of objects
    :param path_lbl: path to save file to
    :param i: iteration number
    :param j: type of image
    :param width: width of the image
    :param height: height of the image
    :return:
    """
    file = open(path_lbl + str(i) + '_' + str(j) + '.txt', 'w+')
    for border in borders:
        obj_width = border[2] - border[0]
        obj_height = border[3] - border[1]
        center_x = (border[0] + (obj_width / 2))
        center_y = (border[1] + (obj_height / 2))
        file.write(str(int(border[4])) + ' ' + str(center_x / float(width)) + ' ' +
                   str(center_y / float(height)) + ' ' + str(obj_width / width) + ' ' + str(obj_height / height))
        file.write('\n')


def plot_noise(img, mode, path):
    """ Adds chosen noise to the image

    :param img: initial image
    :param mode: chosen noise
    :param path: path to save image to
    :return:
    """
    save_image(skimage.util.random_noise(img, mode=mode), path)


def add_noise(img, path):
    """ Adds gaussian, poisson, speckle or localvar noise to an image

    :param img: initial image
    :param path: path to save image to
    :return:
    """
    indx = rd.randint(0, 3)
    if indx == 0:
        plot_noise(img, "gaussian", path)
    elif indx == 1:
        plot_noise(img, "poisson", path)
    elif indx == 2:
        plot_noise(img, "speckle", path)
    elif indx == 3:
        plot_noise(img, "localvar", path)


def get_full_path(path, extension, num):
    """ Creates new path to the new image

    :param path: initial path
    :param extension:
    :param num: type of image
    :return: new path
    """
    return path + '_' + str(num) + extension


def delete_old_files(path):
    """ Clear the given folder

    :param path: path to the directory
    :return:
    """
    for the_file in os.listdir(path):
        file_path = os.path.join(path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


def change_size_img(img, font_height):
    """ Change the size of a digit image

    :param img: initial image
    :param font_height: height of the font
    :return: resized image
    """
    change_to = rd.randint(font_height - 5, font_height + 5)
    wpercent = (change_to / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    return img.resize((change_to, hsize), Image.ANTIALIAS)


def get_image_array(imgs):
    """

    :param imgs:
    :return:
    """
    return (np.reshape(-1 - imgs[rd.randint(0, len(imgs) - 1)], (28, 28))).astype(np.uint8)


def get_digit(imgs, offset, bg_img, font_height):
    """ choose and change handwritten digit

    :param imgs: array of handwritten images
    :param offset: x coordinate of a digit
    :param bg_img: background image
    :param font_height: height of the font
    :return: image of a handwritten digit
    """
    img_array = np.stack((np.array(change_size_img(Image.fromarray(get_image_array(imgs)), font_height)),) * 3, axis=-1)
    blue_colour = choose_blue_colour()
    bg_img_array = np.array(bg_img)
    for i in range(0, img_array.shape[0]):
        for j in range(0, img_array.shape[1]):
            if 0 <= np.sum(img_array[j][i]) <= 100 * 3:
                img_array[j][i] = blue_colour
            elif 230 * 3 <= np.sum(img_array[j][i]) <= 255 * 3:
                img_array[j][i] = bg_img_array[j + offset[1]][i + offset[0]]
    return Image.fromarray(img_array)
