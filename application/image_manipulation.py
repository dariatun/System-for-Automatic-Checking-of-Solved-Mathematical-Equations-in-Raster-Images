import cv2
import numpy as np
import math
from scipy import ndimage

from application.handwritten_recogniser import recognise_handwritten_image
from application.recognise_text import recognise_text
from utils.utils import cut_image

DEBUG = False
DEBUG_TEXT = False
INCREASE_BORDER_VALUE = 12

INCREASE_HANDWRITTEN_BORDER_VALUE = 20
MAX_ACCEPTABLE_BLACK_PIXEL_COUNT = 6
MAX_BLACK_VALUE = 120
MIN_WHITE_VALUE = 20

DIGIT_IMAGE_SIZE = 28


def getBestShift(img):
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


def value_is_in_range(val, length, img_length):
    if val - INCREASE_BORDER_VALUE >= 0 and val + length + 2*INCREASE_BORDER_VALUE < img_length:
        return val - INCREASE_BORDER_VALUE, length + 2*INCREASE_BORDER_VALUE
    return val, length


def rbg_image_to_grey(image):
    # image = Image.fromarray(image)
    # convert rgb image to greyscale image
    """
    cv2.imshow('Original', image)
    cv2.waitKey(0)

    tub_kernel = (2, 3)
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones(tub_kernel, np.uint8)
    grey = cv2.dilate(grey, kernel, iterations=1)
    grey = cv2.erode(grey, kernel, iterations=2)

    #  Apply threshold to get image with only black and white
    # images = cv2.adaptiveThreshold(images, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    _, grey = cv2.threshold(grey, 127, 255, cv2.THRESH_BINARY)
    # image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow("Grey", grey)
    cv2.waitKey(0)
    """

    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    grey = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)

    #ret, thresh = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #kernel = np.ones((5, 5), np.uint8)
    #erode = cv2.erode(thresh, kernel, iterations=2)
    #result = cv2.bitwise_or(grey, erode)
    # result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    #result[result > 170] = 255

    """
    if DEBUG and result.size != 0:
        img_to_show = cv2.resize(result, (800, 600))
        cv2.imshow("FULL IMAGE", img_to_show)
        cv2.waitKey(0)
    """
    return grey #image


def preprocessing_handwritten_image(image):
    """
    from https://github.com/opensourcesblog/tensorflow-mnist
    :param image:
    :return:
    """
    # rescale it
    gray = cv2.resize(255 - image, (28, 28))
    #gray = image
    if DEBUG and gray.size != 0:
        cv2.imshow("FULL IMAGE", gray)
        cv2.waitKey(0)
    # better black and white version
    (thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    gray = cv2.GaussianBlur(gray, (3, 3), 3)
    kernel = np.ones((3, 3), np.uint8)

    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 1)

    erode = cv2.erode(gray, kernel, iterations=5)
    gray = cv2.bitwise_or(gray, erode)
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    while np.sum(gray[0]) == 0:
        gray = gray[1:]

    while np.sum(gray[:, 0]) == 0:
        gray = np.delete(gray, 0, 1)

    while np.sum(gray[-1]) == 0:
        gray = gray[:-1]

    while np.sum(gray[:, -1]) == 0:
        gray = np.delete(gray, -1, 1)

    rows, cols = gray.shape

    if rows > cols:
        factor = 20.0 / rows
        rows = 20
        cols = int(round(cols * factor))
        # first cols than rows
        gray = cv2.resize(gray, (cols, rows))
    else:
        factor = 20.0 / cols
        cols = 20
        rows = int(round(rows * factor))
        # first cols than rows
        gray = cv2.resize(gray, (cols, rows))

    colsPadding = (int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0)))
    rowsPadding = (int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0)))
    gray = np.lib.pad(gray, (rowsPadding, colsPadding), 'constant')

    shiftx, shifty = getBestShift(gray)
    shifted = shift(gray, shiftx, shifty)
    gray = shifted
    if DEBUG and gray.size != 0:
        cv2.imshow("FULL IMAGE changed", gray)
        cv2.waitKey(0)
    return gray


def prepare_handwritten_image(image):
    """ Resize image and convert it to a greyscale image

    :param image: initial image
    :return: changed image
    """
    # cv2.imwrite(str(rd.randint(0, 2000)) + '.jpg', image)

    # image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)


    #mean_px = image.mean().astype(np.float32)
    #std_px = image.std().astype(np.float32)
    #image = (image - mean_px) / (std_px)

    #image[image < 0] = 0
    #image[image > 0] = 255

    resized_image = cv2.resize(image, (28, 28))
    #result = resized_image


    if DEBUG and resized_image.size != 0:
        cv2.imshow("original", resized_image)
        cv2.waitKey(0)

    ret, thresh = cv2.threshold(resized_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    if DEBUG and thresh.size != 0:
        cv2.imshow("FULL IMAGE", thresh)
        cv2.waitKey(0)
    

    """
    thresh = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 71, 0)

    if DEBUG and thresh.size != 0:
        cv2.imshow("addaptive", thresh)
        cv2.waitKey(0)
    """

    blured = cv2.GaussianBlur(thresh, (5, 5), 5)

    if DEBUG and blured.size != 0:
        cv2.imshow("Blured", blured)
        cv2.waitKey(0)

    #result = cv2.adaptiveThreshold(blured, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 71, 1)
    result = blured
    if DEBUG and result.size != 0:
        cv2.imshow("FULL IMAGE changed", result)
        cv2.waitKey(0)

    # convert image to 28x28 size
    #result = Image.fromarray(result)

    #change_to = 28
    #result = result.resize((change_to, change_to), Image.ANTIALIAS)


    # change size

    # image[image <= 100] = 0

    return result


def resize_image_if_needed(x, y, w, h, image, loop_number):
    digit_image = cut_image(x, y, w, h, image)

    changed_digit_image = prepare_handwritten_image(digit_image) #prepare_handwritten_image(digit_image)
    #digit_image[digit_image > 100] = 255
    if DEBUG and changed_digit_image.size != 0:
        cv2.imshow("Handwritten", changed_digit_image)
        cv2.waitKey(0)

    if loop_number > 4:
        return preprocessing_handwritten_image(digit_image), [x, y, w, h]

    left_black, right_black, up_black, down_black = 0, 0, 0, 0

    for i in range(0, 28):
        if changed_digit_image[0][i] > MIN_WHITE_VALUE:
            up_black += 1
        if changed_digit_image[28 - 1][i] > MIN_WHITE_VALUE:
            down_black += 1
    for i in range(0, 28):
        if changed_digit_image[i][0] > MIN_WHITE_VALUE:
            left_black += 1
        if changed_digit_image[i][28 - 1] > MIN_WHITE_VALUE:
            right_black += 1

    if DEBUG or DEBUG_TEXT:
        print(left_black, right_black, up_black, down_black)
    new_x, new_y, new_w, new_h = x, y, w, h
    if left_black > MAX_ACCEPTABLE_BLACK_PIXEL_COUNT:
        new_x = x - INCREASE_HANDWRITTEN_BORDER_VALUE
        if new_x < 0:
            new_x = 0
        new_w += x - new_x
    if right_black > MAX_ACCEPTABLE_BLACK_PIXEL_COUNT:
        test_w = new_w + INCREASE_HANDWRITTEN_BORDER_VALUE
        if test_w > image.shape[0]:
            new_w = image.shape[0]
        else:
            new_w = test_w
    if up_black > MAX_ACCEPTABLE_BLACK_PIXEL_COUNT:
        new_y = y - INCREASE_HANDWRITTEN_BORDER_VALUE
        if new_y < 0:
            new_y = 0
        new_h += y - new_y
    if down_black > MAX_ACCEPTABLE_BLACK_PIXEL_COUNT:
        test_h = new_h + INCREASE_HANDWRITTEN_BORDER_VALUE
        if test_h > image.shape[1]:
            new_h = image.shape[1]
        else:
            new_h = test_h
    if x != new_x or y != new_y or w != new_w or h != new_h:
        return resize_image_if_needed(new_x, new_y, new_w, new_h, image, loop_number + 1)

    return preprocessing_handwritten_image(digit_image), [x, y, w, h]


def recognise_object(image, x, y, w, h, class_id):
    # grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite('grey.jpg', grey)
    image = rbg_image_to_grey(image)
    w_img, h_img = image.shape
    #x, w = value_is_in_range(x, w, w_img)
    #y, h = value_is_in_range(y, h, h_img)
    # image = cut_image(xy[0] , xy[1] , w , h , init_img_arr)
    if x < 0: x = 0
    if y < 0: y = 0
    box = [x, y, w, h]
    is_legitimate = True
    if image.size == 0:
        print('Leaving recognise_object, size of image is 0.')
        return "", box
    if class_id == 0:
        prediction, box, is_legitimate = recognise_text(image, x, y, w, h, loop_number=0)
    else:
        digit_image, box = resize_image_if_needed(x, y, w, h, image, 0)
        #digit_image = prepare_handwritten_image(digit_image)
        _, prediction = recognise_handwritten_image(digit_image)
        prediction = str(prediction[0])
        if DEBUG or DEBUG_TEXT:
            print(prediction)
        if( DEBUG or DEBUG_TEXT) and digit_image.size != 0:
            cv2.imshow("Handwritten", digit_image)
            cv2.waitKey(0)
    return prediction, box, is_legitimate


