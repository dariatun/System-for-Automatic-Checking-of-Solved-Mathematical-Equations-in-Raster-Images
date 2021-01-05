import cv2
import imutils
import numpy as np
import math
from scipy import ndimage


def get_best_shift(img):
    """
    from https://github.com/opensourcesblog/tensorflow-mnist
    :param img:
    :return:
    """
    cy, cx = ndimage.measurements.center_of_mass(img)

    rows, cols = img.shape
    shiftx = np.round(cols / 2.0 - cx).astype(int)
    shifty = np.round(rows / 2.0 - cy).astype(int)

    return shiftx, shifty


def shift(img, sx, sy):
    """
    from https://github.com/opensourcesblog/tensorflow-mnist
    :param img:
    :return:
    """
    rows, cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(img, M, (cols, rows))
    return shifted


def rbg_image_to_grey(image):
    """
    Converts image to greyscale
    :param image: image to convert
    :return: greyscale image
    """
    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    grey = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    return grey


def delete_all_black_lines(grey):
    """
    Removes all black lines at the sides of the image
    :param grey: the image to change
    :return: the changed image
    """
    while np.sum(grey[0]) == 0:
        grey = grey[1:]

    while np.sum(grey[:, 0]) == 0:
        grey = np.delete(grey, 0, 1)

    while np.sum(grey[-1]) == 0:
        grey = grey[:-1]

    while np.sum(grey[:, -1]) == 0:
        grey = np.delete(grey, -1, 1)
    return grey


def add_lines_back(grey):
    """
    Adds black lines to the image to make the digit be in the middle
    :param grey: the image to change
    :return: the changed image
    """
    rows, cols = grey.shape

    if rows > cols:
        factor = 20.0 / rows
        rows = 20
        cols = int(round(cols * factor))
        # first cols than rows
        grey = cv2.resize(grey, (cols, rows))
    else:
        factor = 20.0 / cols
        cols = 20
        rows = int(round(rows * factor))
        # first cols than rows
        grey = cv2.resize(grey, (cols, rows))

    colsPadding = (int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0)))
    rowsPadding = (int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0)))
    grey = np.lib.pad(grey, (rowsPadding, colsPadding), 'constant')
    return grey


def preprocessing_handwritten_image(image):
    """
    from https://github.com/opensourcesblog/tensorflow-mnist
    :param image:
    :return:
    """
    # rescale it
    grey = cv2.resize(255 - image, (28, 28))

    # better black and white version
    (thresh, grey) = cv2.threshold(grey, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    grey = cv2.GaussianBlur(grey, (3, 3), 3)
    kernel = np.ones((3, 3), np.uint8)

    grey = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 1)

    erode = cv2.erode(grey, kernel, iterations=5)
    grey = cv2.bitwise_or(grey, erode)
    grey = cv2.morphologyEx(grey, cv2.MORPH_OPEN, kernel)

    grey = delete_all_black_lines(grey)
    grey = add_lines_back(grey)

    shiftx, shifty = get_best_shift(grey)
    shifted = shift(grey, shiftx, shifty)
    grey = shifted

    return grey


def prepare_handwritten_image(image):
    """ Resize image and convert it to a greyscale image

    :param image: initial image
    :return: changed image
    """
    resized_image = cv2.resize(image, (28, 28))

    ret, thresh = cv2.threshold(resized_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    result = cv2.GaussianBlur(thresh, (5, 5), 5)

    return result





def cut_image(x, y, width, height, image):
    """ Cut the image by the border coordinates

    :param x: x coordinate of the border's left side
    :param y: y coordinate of the border's top side
    :param width: the width of the border
    :param height: the height of the border
    :param image: the image to cut digit from
    :return: image of a digit
    """
    return image[y:y+height, x:width+x]


def rotate_image(image, angle):
    """
    Rotates the image on the angle
    :param image: the image to rotate
    :param angle: the angle to ratate on
    :return: the rotated image
    """
    return imutils.rotate(image, angle)


def preprocess_equation_image(image):
    """
    Preprocesses image before detection
    :param image: the image to change
    :return: changed image
    """
    ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blured = cv2.GaussianBlur(thresh, (7, 7), 9)

    result = cv2.medianBlur(blured, 5)

    kernel = np.ones((5, 5), np.uint8)

    erode = cv2.erode(result, kernel, iterations=5)
    result = cv2.bitwise_or(result, erode)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)

    return result




