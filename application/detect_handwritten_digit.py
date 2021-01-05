import sys
import numpy as np
from keras.optimizers import SGD

# path to the directory with the CNN_Keras
from application.changeables import CNN_KERAS_PATH, SAVED_WEIGHTS_PATH
from application.constants import MIN_WHITE_VALUE, MAX_ACCEPTABLE_BLACK_PIXEL_COUNT, INCREASE_HANDWRITTEN_BORDER_VALUE

sys.path.append(CNN_KERAS_PATH)
from cnn.neural_network import CNN

from application.image_manipulation import preprocessing_handwritten_image, \
    prepare_handwritten_image, cut_image


def detect_handwritten_digit(image, x, y, w, h):
    """ Predict the image with the CNN model

    :param image: image of a digit
    :return: prediction
    """
    digit_image, box = resize_image_if_needed(x, y, w, h, image, 0)
    digit_image = preprocessing_handwritten_image(digit_image)

    change_to = 28
    digit_image = np.reshape(digit_image, (1, 1, change_to, change_to))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    clf = CNN.build(width=28, height=28, depth=1, total_classes=10,
                    Saved_Weights_Path=SAVED_WEIGHTS_PATH)
    clf.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    probs = clf.predict(digit_image)
    prediction = probs.argmax(axis=1)

    return prediction


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

    if loop_number > 4:
        return digit_image, [x, y, w, h]

    left_white, right_white, up_white, down_white = count_white_pixels_around(changed_digit_image)

    new_x, new_y, new_w, new_h = get_new_coordinates(left_white, right_white, up_white, down_white, x, y, w, h, image)

    if x != new_x or y != new_y or w != new_w or h != new_h:
        return resize_image_if_needed(new_x, new_y, new_w, new_h, image, loop_number + 1)

    return digit_image, [x, y, w, h]