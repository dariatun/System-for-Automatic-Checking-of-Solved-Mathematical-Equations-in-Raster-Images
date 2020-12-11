from PIL import Image
import pytesseract
import cv2
import os
import re
import numpy as np
from utils.utils import cut_image, add_to_dictionary, get_element_with_the_biggest_value, get_numbers_and_delimeter

DEBUG = False
DEBUG_WRITE = False

DEFAULT_SYMBOL_WIDTH = 80
LEFT = 0
RIGHT = 1
BOTH = 2
SYMBOL_WIDTH_MULTIPLIER = 1
ACCEPTABLE_LOOP_NUMBER = 1


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
            # for j in range(0, i):
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
            # for j in range(i + 1, len_num):
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
    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, image)
    lines_size = 1
    lines = [None] * lines_size
    """
    lines[0] = pytesseract.image_to_string(Image.open(filename),
                                           config="-c tessedit_char_whitelist=0123456789+-= --psm 12 ")

    lines[1] = pytesseract.image_to_string(Image.open(filename),
                                           config="-c tessedit_char_whitelist=0123456789+-= --psm 11")

    lines[2] = pytesseract.image_to_string(Image.open(filename),
                                           config="-c tessedit_char_whitelist=0123456789+-= --oem 3--psm 6")
    """
    lines[0] = pytesseract.image_to_string(Image.open(filename),
                                           config="-c tessedit_char_whitelist=0123456789+-= --psm 7") #IM USING THIS ONE!!!!
    """
    lines[4] = pytesseract.image_to_string(Image.open(filename),
                                           config="-c tessedit_char_whitelist=0123456789+-= --psm 9")
    lines[5] = pytesseract.image_to_string(Image.open(filename),
                                           config="-c tessedit_char_whitelist=0123456789+-= --psm 10")
    lines[6] = pytesseract.image_to_string(Image.open(filename),
                                           config="-c tessedit_char_whitelist=0123456789+-= --psm 13")
    """
    os.remove(filename)
    if DEBUG_WRITE or DEBUG:
        print(lines)
    return lines, lines_size #[lines[3]], 1#lines_size


def increase_size_of_an_image(image, x, y, w, h, prediction, loop_number, increase_to):
    pred_len = len(prediction)
    symbol_width = DEFAULT_SYMBOL_WIDTH
    symbol_width_multiplier = 1
    if not pred_len == 0:
        symbol_width = int(w / pred_len)
    # print(symbol_width)
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
    """
    blured = thresh
    result = cv2.adaptiveThreshold(blured, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 1)
    
    if DEBUG and result.size != 0:
        cv2.imshow("FULL IMAGE changed", result)
        cv2.waitKey(0)
    """
    kernel = np.ones((5, 5), np.uint8)

    erode = cv2.erode(result, kernel, iterations=5)
    result = cv2.bitwise_or(result, erode)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)

    #result = cv2.adaptiveThreshold(result, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 1)

    return result


def recognise_text(image, x, y, w, h, loop_number):
    is_legitimate = True
    if loop_number == 0:
        left_x = x - SYMBOL_WIDTH_MULTIPLIER * DEFAULT_SYMBOL_WIDTH
        if left_x < 0:
            left_x = 0
        w += x - left_x
        x = left_x
    equation_image = cut_image(x, y, w, h, image)
    equation_image = preprocess_image(equation_image)
    # equation_image[equation_image < 80] = 0
    if DEBUG and equation_image.size != 0:
        cv2.imshow("Equation", equation_image)
        cv2.waitKey(0)

    lines, lines_size = get_lines_predictions(equation_image)

    first = {}
    second = {}
    delimiter = {}

    for i in range(0, lines_size):

        line = lines[i]
        line = line.replace('\n', '')
        line = line.replace('\x0c', '')

        if len(line) == 0: continue

        numbers, symbol = get_numbers_and_delimeter(line)

        if len(numbers) < 2: continue
        if len(symbol) == 0: continue

        numbers[0] = check_first_digit(numbers[0])
        number_1 = int(numbers[0])
        numbers[1] = check_first_digit(numbers[1])
        number_2 = int(numbers[1])

        if '+' in symbol and number_1 + number_2 >= 100:
            continue

        if number_1 < 100:
            add_to_dictionary(first, numbers[0])
        if number_2 < 100:
            add_to_dictionary(second, numbers[1])
        add_delimiter_to_dict(delimiter, symbol)

    if DEBUG or DEBUG_WRITE:
        print(first, delimiter, second)
    prediction = ''
    """
    if loop_number == 0:
        symbol_width = DEFAULT_SYMBOL_WIDTH
        if not len(line) == 0:
            symbol_width = int(w / len(line))
            # print(symbol_width)
        left_x = x - SYMBOL_WIDTH_MULTIPLIER * symbol_width
        if left_x < 0:
            left_x = 0
        return recognise_text(image, left_x, y, w + x - left_x, h, loop_number + 1)
    """
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



