from PIL import Image
import pytesseract
import cv2
import os

from application.constants import DEFAULT_SYMBOL_WIDTH_EQUATIONS, LEFT, RIGHT, BOTH, SYMBOL_WIDTH_MULTIPLIER, \
    ACCEPTABLE_LOOP_NUMBER
from application.image_manipulation import cut_image, preprocess_equation_image
from application.utils import add_to_dictionary, get_element_with_the_biggest_value, get_numbers_and_delimiter

DEBUG = False
DEBUG_WRITE = False


def check_first_digit(el):
    """
    If the first digit in the number is '0', it probably is a '5'
    :param el: the element to check
    :return: modified element
    """
    if el[0] == '0' and len(el) > 1:
        el = '5' + el[1]
    return el


def divide_number(number, first, second, delimiter):
    """
    If Tesseract recognise only one number with the number 4 in the middle,
        will divide the number on equation of addition
    :param number: the number to divide
    :param first: the dictionary of first number
    :param second: the dictionary of second number
    :param delimiter: the dictionary of symbols
    :return:
    """
    first_num = number[0]
    second_num = ''
    for i in range(1, len(number)):
        if number[i] == '4':
            second_num = number[i:-1]
            break
        else:
            first_num += number[i]
    if len(second_num) != 0:
        add_to_dictionary(first, first_num)
        add_to_dictionary(second, second_num)
        add_delimiter_to_dict(delimiter, '+')


def find_max_list(list):
    """
    Finds the longest string in the list
    :param list: the list
    :return: the longest element
    """
    return max(list, key=len)


def get_lines_predictions(image):
    """
    Get Tessetact prediction
    :param image: image to predict
    :return: predicted string
    """
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
    """
    Increases the size of the bounding box on the given size
    :param image: the image to increase
    :param x: the x-coordinate of the bounding box
    :param y: the y-coordinate of the bounding box
    :param w: the width of the bounding box
    :param h: the height of the bounding box
    :param prediction: the predicted string
    :param loop_number: the number of times this functions was called
    :param increase_to: the side the bounding box will be increased to
    :return: prediction and the bounding box
    """
    pred_len = len(prediction)
    symbol_width = DEFAULT_SYMBOL_WIDTH_EQUATIONS
    symbol_width_multiplier = 1
    if not pred_len == 0:
        symbol_width = int(w / pred_len)
    left_x = x - symbol_width_multiplier * symbol_width
    if left_x < 0:
        left_x = 0
    if increase_to == LEFT:
        return detect_mathematical_equation(image, left_x, y, w + x - left_x, h, loop_number + 1)
    elif increase_to == RIGHT:
        right_x = w + symbol_width * symbol_width_multiplier
        if right_x > image.shape[0]:
            right_x = image.shape[0]
        return detect_mathematical_equation(image, x, y, right_x, h, loop_number + 1)
    elif increase_to == BOTH:
        right_x = w + symbol_width * symbol_width_multiplier * 2
        if right_x > image.shape[0]:
            right_x = image.shape[0]
        return detect_mathematical_equation(image, left_x, y, right_x, h, loop_number + 1)


def add_delimiter_to_dict(dict, symbol):
    """
    Adds symbol to the dictionary
    :param dict: dictionary
    :param symbol:
    :return:
    """
    i = 0
    while i < len(symbol):
        c = symbol[i]
        i += 1
        if c == '=': continue
        add_to_dictionary(dict, c)

def detect_mathematical_equation(image, x, y, w, h, loop_number):
    """
    Detects mathematical equation
    :param image: the image to detect
    :param x: the x-coordinate of the bounding box
    :param y: the y-coordinate of the bounding box
    :param w: the width of the bounding box
    :param h: the height of the bounding box
    :param loop_number: the number of times this functions was called
    :return: prediction, bounding box
    """
    is_legitimate = True
    if loop_number == 0:
        left_x = x - SYMBOL_WIDTH_MULTIPLIER * DEFAULT_SYMBOL_WIDTH_EQUATIONS
        if left_x < 0:
            left_x = 0
        w += x - left_x
        x = left_x
    equation_image = cut_image(x, y, w, h, image)
    equation_image = preprocess_equation_image(equation_image)
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
        elif len(numbers) == 1 and len(symbol) == 0 and len(numbers[0]) > 0:
            divide_number(numbers[0])

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



