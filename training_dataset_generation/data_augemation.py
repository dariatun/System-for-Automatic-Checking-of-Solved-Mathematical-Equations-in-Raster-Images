import sys
import numpy as np
# path to the directory with the DataAugmentationForObjectDetection
sys.path.append('/Users/dariatunina/mach-lerinig/DataAugmentationForObjectDetection')
# sys.path.append('/home.stud/tunindar/DataAugmentationForObjectDetection')

from data_aug.data_aug import *
from data_aug.bbox_util import *


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


def change_shear(img):
    bboxes = np.array([[0, 0, img.shape[0], img.shape[1], 0]])
    return Shear(0.1)(img.copy(), bboxes)
