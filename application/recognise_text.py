from PIL import Image
import pytesseract
import cv2
import os
import numpy as np

from application.constants import DEFAULT_SYMBOL_WIDTH_EQUATIONS, LEFT, RIGHT, BOTH, SYMBOL_WIDTH_MULTIPLIER, \
    ACCEPTABLE_LOOP_NUMBER
from application.utils import cut_image, add_to_dictionary, get_element_with_the_biggest_value, get_numbers_and_delimiter

DEBUG = False
DEBUG_WRITE = False


def if_is_symbol(c):
    return not (c == '+' or c == '-' or c == '=')


def is_digit(c):
    return '9' >= c >= '0'


def is_number(num):
    for c in num:
        if not is_digit(c): return False
    return True


def check_first_digit(el):
    if el[0] == '0' and len(el) > 1:
        el = '5' + el[1]
    return el


def add_to_dict(dict, el):
    if len(el) > 0:
        el = check_first_digit(el)
        add_to_dictionary(dict, el)


def find_first(num, dict):
    for i in range(0, len(num)):
        if i < 2:
            add_to_dict(dict, num[0:i + 1])
            if num[i] == '4' and i != 0:
                add_to_dict(dict, num[0:i - 1])
        else:
            add_to_dict(dict, num[i - 1] + num[i])
            if num[i] == '4':
                add_to_dict(dict, num[i - 2] + num[i - 1])


def find_second(num, dict, symbol):
    len_num = len(num)
    for i in reversed(range(0, len_num)):
        if len_num - i < 3:
            add_to_dict(dict, num[i:len_num])
            if num[i] == '4' and '-' not in symbol:
                add_to_dict(dict, num[i + 1:len_num])
        else:
            add_to_dict(dict, num[i] + num[i + 1])
            if num[i] == '4' and '-' not in symbol:
                add_to_dict(dict, num[i + 1] + num[i + 2])


def add_number(num, first, second, symbol):
    num_len = len(num)
    if num_len == 1 or num_len == 2:
        add_to_dict(first, num)
        add_to_dict(second, num)

    else:
        # TODO if there is a 4 in the middle it could be +
        find_first(num, first)
        find_second(num, second, symbol)
        # add_number(num, first, second)


def find_max_list(list):
    return max(list, key=len)


def get_lines_predictions(image):
    filename = "temporary/{}.png".format(os.getpid())
    cv2.imwrite(filename, image)

    line = pytesseract.image_to_string(Image.open(filename),
                                           config="-c tessedit_char_whitelist=0123456789+-= --psm 7")
    os.remove(filename)
    line = line.replace('\n', '')
    line = line.replace('\x0c', '')

    if DEBUG_WRITE or DEBUG:
        print(line)

    return line


def increase_size_of_an_image(image, x, y, w, h, prediction, loop_number, increase_to):
    pred_len = len(prediction)
    symbol_width = DEFAULT_SYMBOL_WIDTH_EQUATIONS
    symbol_width_multiplier = 1
    if not pred_len == 0:
        symbol_width = int(w / pred_len)
    left_x = x - symbol_width_multiplier * symbol_width
    if left_x < 0:
        left_x = 0
    if increase_to == LEFT:
        return recognise_text(image, left_x, y, w + x - left_x, h, loop_number + 1)
    elif increase_to == RIGHT:
        right_x = w + symbol_width * symbol_width_multiplier
        if right_x > image.shape[0]:
            right_x = image.shape[0]
        return recognise_text(image, x, y, right_x, h, loop_number + 1)
    elif increase_to == BOTH:
        right_x = w + symbol_width * symbol_width_multiplier * 2
        if right_x > image.shape[0]:
            right_x = image.shape[0]
        return recognise_text(image, left_x, y, right_x, h, loop_number + 1)


def add_delimiter_to_dict(dict, symbol):
    i = 0
    while i < len(symbol):
        c = symbol[i]
        i += 1
        if c == '=': continue
        add_to_dictionary(dict, c)


def preprocess_image(image):
    ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if DEBUG and thresh.size != 0:
        cv2.imshow("FULL IMAGE", thresh)
        cv2.waitKey(0)

    blured = cv2.GaussianBlur(thresh, (7, 7), 9)

    if DEBUG and blured.size != 0:
        cv2.imshow("Blured", blured)
        cv2.waitKey(0)

    blured = cv2.medianBlur(blured, 5)

    if DEBUG and blured.size != 0:
        cv2.imshow("more BLURED", blured)
        cv2.waitKey(0)
    result = blured
    kernel = np.ones((5, 5), np.uint8)

    erode = cv2.erode(result, kernel, iterations=5)
    result = cv2.bitwise_or(result, erode)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)

    return result


def recognise_text(image, x, y, w, h, loop_number):
    is_legitimate = True
    if loop_number == 0:
        left_x = x - SYMBOL_WIDTH_MULTIPLIER * DEFAULT_SYMBOL_WIDTH_EQUATIONS
        if left_x < 0:
            left_x = 0
        w += x - left_x
        x = left_x
    equation_image = cut_image(x, y, w, h, image)
    equation_image = preprocess_image(equation_image)
    if DEBUG and equation_image.size != 0:
        cv2.imshow("Equation", equation_image)
        cv2.waitKey(0)

    line = get_lines_predictions(equation_image)

    first = {}
    second = {}
    delimiter = {}

    if len(line) != 0:

        numbers, symbol = get_numbers_and_delimiter(line)

        if len(numbers) >= 2 and len(symbol) != 0:

            numbers[0] = check_first_digit(numbers[0])
            number_1 = int(numbers[0])
            numbers[1] = check_first_digit(numbers[1])
            number_2 = int(numbers[1])

            if number_1 < 100:
                add_to_dictionary(first, numbers[0])
            else:
                add_to_dictionary(first, numbers[0][:-1])
            if number_2 < 100:
                add_to_dictionary(second, numbers[1])
            else:
                add_to_dictionary(second, numbers[1][:-1])

            add_delimiter_to_dict(delimiter, symbol)

    if DEBUG or DEBUG_WRITE:
        print(first, delimiter, second)
    prediction = ''
    f = get_element_with_the_biggest_value(first)
    s = get_element_with_the_biggest_value(second)
    if len(first) > 0 and len(delimiter) > 0 and len(second) > 0:
        number_1 = find_max_list(f)
        number_2 = find_max_list(s)
        symbol = max(delimiter, key=delimiter.get)
        prediction = number_1 + symbol + number_2 + '='
        if symbol == '-' and int(number_1) - int(number_2) < 0 and loop_number < ACCEPTABLE_LOOP_NUMBER:
            if DEBUG:
                print('increase to the left')
            if not x == 0:
                prediction, [x, y, w, h], is_legitimate = increase_size_of_an_image(image, x, y, w, h, prediction, loop_number, increase_to=LEFT)
    elif loop_number < ACCEPTABLE_LOOP_NUMBER:
        if len(second) > 0:
            if DEBUG:
                print('increase to the left')
            if not x == 0:
                prediction, [x, y, w, h], is_legitimate = increase_size_of_an_image(image, x, y, w, h, prediction, loop_number, increase_to=LEFT)
        elif len(first) > 0:
            if DEBUG:
                print('increase to the right')
            if not y + h >= image.shape[1]:
                prediction, [x, y, w, h], is_legitimate = increase_size_of_an_image(image, x, y, w, h, prediction, loop_number, increase_to=RIGHT)
        else:
            if DEBUG:
                print('increase to the left and right')
            if not y + h >= image.shape[1]:
                if not x == 0:
                    prediction, [x, y, w, h], is_legitimate = increase_size_of_an_image(image, x, y, w, h, prediction, loop_number, increase_to=BOTH)
                else:
                    prediction, [x, y, w, h], is_legitimate = increase_size_of_an_image(image, x, y, w, h, prediction, loop_number, increase_to=RIGHT)
            elif not x == 0:
                prediction, [x, y, w, h], is_legitimate = increase_size_of_an_image(image, x, y, w, h, prediction, loop_number, increase_to=LEFT)

    if DEBUG or DEBUG_WRITE:
        print(prediction)
    if len(prediction) == 0:
        return line, [x, y, w, h], False

    return prediction, [x, y, w, h], is_legitimate\



