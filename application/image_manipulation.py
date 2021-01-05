import cv2
import numpy as np
import math
from scipy import ndimage

from application.handwritten_recogniser import detect_handwritten_digit
from application.recognise_text import recognise_text
from application.utils import cut_image

DEBUG = False
DEBUG_TEXT = False
INCREASE_BORDER_VALUE = 12

INCREASE_HANDWRITTEN_BORDER_VALUE = 20
MAX_ACCEPTABLE_BLACK_PIXEL_COUNT = 6
MAX_BLACK_VALUE = 120
MIN_WHITE_VALUE = 20

DIGIT_IMAGE_SIZE = 28


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
    if DEBUG and grey.size != 0:
        cv2.imshow("FULL IMAGE", grey)
        cv2.waitKey(0)
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
    if DEBUG and grey.size != 0:
        cv2.imshow("FULL IMAGE changed", grey)
        cv2.waitKey(0)
    return grey


def prepare_handwritten_image(image):
    """ Resize image and convert it to a greyscale image

    :param image: initial image
    :return: changed image
    """

    resized_image = cv2.resize(image, (28, 28))

    if DEBUG and resized_image.size != 0:
        cv2.imshow("original", resized_image)
        cv2.waitKey(0)

    ret, thresh = cv2.threshold(resized_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    if DEBUG and thresh.size != 0:
        cv2.imshow("FULL IMAGE", thresh)
        cv2.waitKey(0)

    blured = cv2.GaussianBlur(thresh, (5, 5), 5)

    if DEBUG and blured.size != 0:
        cv2.imshow("Blured", blured)
        cv2.waitKey(0)

    result = blured
    if DEBUG and result.size != 0:
        cv2.imshow("FULL IMAGE changed", result)
        cv2.waitKey(0)

    return result


def count_white_pixels_around(image):
    """
    Calculates the number of white pixels around the bounding box
    :param image: the digit image
    :return: the number of the white pixels to the left,
             to the right, above and below the bounding box
    """
    left_white, right_white, up_white, down_white = 0, 0, 0, 0
    for i in range(0, 28):
        if image[0][i] > MIN_WHITE_VALUE:
            up_white += 1
        if image[28 - 1][i] > MIN_WHITE_VALUE:
            down_white += 1
    for i in range(0, 28):
        if image[i][0] > MIN_WHITE_VALUE:
            left_white += 1
        if image[i][28 - 1] > MIN_WHITE_VALUE:
            right_white += 1
    if DEBUG or DEBUG_TEXT:
        print(left_white, right_white, up_white, down_white)
    return left_white, right_white, up_white, down_white


def get_new_coordinates(left_white, right_white, up_white, down_white, x, y, w, h, image):
    """
    Extends the bounding box if needed
    :param left_white: the number of the white pixels to the left of the bounding box
    :param right_white: the number of the white pixels to the right of the bounding box
    :param up_white: the number of the white pixels above the bounding box
    :param down_white: the number of the white pixels below the bounding box
    :param x: the x-coordinate of the bounding box
    :param y: the y-coordinate of the bounding box
    :param w: the width of the bounding box
    :param h: the height of the bounding box
    :param image: the image where the object is at
    :return: new bounding box coordinates
    """
    new_x, new_y, new_w, new_h = x, y, w, h
    if left_white > MAX_ACCEPTABLE_BLACK_PIXEL_COUNT:
        new_x = x - INCREASE_HANDWRITTEN_BORDER_VALUE
        if new_x < 0:
            new_x = 0
    if right_white > MAX_ACCEPTABLE_BLACK_PIXEL_COUNT:
        test_w = new_w + INCREASE_HANDWRITTEN_BORDER_VALUE
        if test_w > image.shape[0]:
            new_w = image.shape[0]
        else:
            new_w = test_w
    if up_white > MAX_ACCEPTABLE_BLACK_PIXEL_COUNT:
        new_y = y - INCREASE_HANDWRITTEN_BORDER_VALUE
        if new_y < 0:
            new_y = 0
        new_h += y - new_y
    if down_white > MAX_ACCEPTABLE_BLACK_PIXEL_COUNT:
        test_h = new_h + INCREASE_HANDWRITTEN_BORDER_VALUE
        if test_h > image.shape[1]:
            new_h = image.shape[1]
        else:
            new_h = test_h
    return new_x, new_y, new_w, new_h


def resize_image_if_needed(x, y, w, h, image, loop_number):
    """
    If there are white pixels around the digit image, will extend the bounding box to include them
    :param x: the x-coordinate of the bounding box
    :param y: the y-coordinate of the bounding box
    :param w: the width of the bounding box
    :param h: the height of the bounding box
    :param image: the image where the object is at
    :param loop_number: the times the bounding box was extended
    :return: image of the digit and its bounding box
    """
    digit_image = cut_image(x, y, w, h, image)

    changed_digit_image = prepare_handwritten_image(digit_image)

    if DEBUG and changed_digit_image.size != 0:
        cv2.imshow("Handwritten", changed_digit_image)
        cv2.waitKey(0)

    if loop_number > 4:
        return digit_image, [x, y, w, h]

    left_white, right_white, up_white, down_white = count_white_pixels_around(changed_digit_image)

    new_x, new_y, new_w, new_h = get_new_coordinates(left_white, right_white, up_white, down_white, x, y, w, h, image)

    if x != new_x or y != new_y or w != new_w or h != new_h:
        return resize_image_if_needed(new_x, new_y, new_w, new_h, image, loop_number + 1)

    return digit_image, [x, y, w, h]


def detect_an_object(image, x, y, w, h, class_id):
    """
    Detects the objects in the image that is cut by the bounding box
    :param image: the image to detect objects on
    :param x: the x-coordinate of the bounding box
    :param y: the y-coordinate of the bounding box
    :param w: the width of the bounding box
    :param h: the height of the bounding box
    :param class_id: the class id of the object to detect
    :return: prediction of the object, new bounding box, the legitimacy of the object
    """
    image = rbg_image_to_grey(image)
    if x < 0: x = 0
    if y < 0: y = 0
    box = [x, y, w, h]
    is_legitimate = True
    if image.size == 0:
        print('Leaving detect_an_object, size of image is 0.')
        return "", box
    if class_id == 0:
        prediction, box, is_legitimate = recognise_text(image, x, y, w, h, loop_number=0)
    else:
        digit_image, box = resize_image_if_needed(x, y, w, h, image, 0)
        digit_image = preprocessing_handwritten_image(digit_image)
        _, prediction = detect_handwritten_digit(digit_image)
        prediction = str(prediction[0])
        if DEBUG or DEBUG_TEXT:
            print(prediction)
        if (DEBUG or DEBUG_TEXT) and digit_image.size != 0:
            cv2.imshow("Handwritten", digit_image)
            cv2.waitKey(0)
    return prediction, box, is_legitimate


