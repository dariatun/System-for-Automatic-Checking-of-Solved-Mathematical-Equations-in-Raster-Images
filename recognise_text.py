from PIL import Image
import pytesseract
import cv2
import os
import re


def if_is_symbol(c):
    return not (c == '+' or c == '-' or c == '=')


def is_digit(c):
    return '9' >= c >= '0'


def is_number(num):
    for c in num:
        if not is_digit(c): return False
    return True


def add_to_dict(dict, el):
    dict[el] = 1 + dict[el] if el in dict else 1


def find_first(num, dict):
    for i in range(0, len(num)):
        if i < 2:
            add_to_dict(dict, num[0:i+1])
            if num[i] == '4':
                add_to_dict(dict, num[0:i - 1])
        else:
            # for j in range(0, i):
            add_to_dict(dict, num[i - 1] + num[i])
            if num[i] == '4':
                add_to_dict(dict, num[i - 2] + num[i - 1])


def find_second(num, dict):
    len_num = len(num)
    for i in reversed(range(0, len_num)):
        if len_num - i < 3:
            add_to_dict(dict, num[i:len_num])
            if num[i] == '4':
                add_to_dict(dict, num[i + 1:len_num])
        else:
            # for j in range(i + 1, len_num):
            add_to_dict(dict, num[i] + num[i + 1])
            if num[i] == '4':
                add_to_dict(dict, num[i + 1] + num[i + 2])


def add_number(num, first, second):
    num_len = len(num)
    if num_len == 1 or num_len == 2:
        add_to_dict(first, num)
        add_to_dict(second, num)

    else:
        # TODO if there is a 4 in the middle it could be +
        find_first(num, first)
        find_second(num, second)
        # add_number(num, first, second)


def find_max_list(list):
    return max(list, key=len)


def recognise_text(image):
    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, image)

    lines = [None] * 7
    lines[0] = pytesseract.image_to_string(Image.open(filename),
                                           config="-c tessedit_char_whitelist=0123456789+-= --psm 12 ")
    lines[1] = pytesseract.image_to_string(Image.open(filename),
                                           config="-c tessedit_char_whitelist=0123456789+-= --psm 11")
    lines[2] = pytesseract.image_to_string(Image.open(filename),
                                           config="-c tessedit_char_whitelist=0123456789+-= --psm 6")
    lines[3] = pytesseract.image_to_string(Image.open(filename),
                                           config="-c tessedit_char_whitelist=0123456789+-= --psm 8")
    lines[4] = pytesseract.image_to_string(Image.open(filename),
                                           config="-c tessedit_char_whitelist=0123456789+-= --psm 9")
    lines[5] = pytesseract.image_to_string(Image.open(filename),
                                           config="-c tessedit_char_whitelist=0123456789+-= --psm 10")
    lines[6] = pytesseract.image_to_string(Image.open(filename),
                                           config="-c tessedit_char_whitelist=0123456789+-= --psm 13")
    os.remove(filename)
    first = {}
    second = {}
    delimiter = {}
    for i in range(0, 7):
        lines[i] = lines[i].replace('\n', '')
        numbers = re.split('[=+-]', lines[i])
        numbers = list(filter(None, numbers))
        size = len(numbers)
        if size == 0:
            continue
        elif size == 1:
            if is_digit(lines[i][0]):
                add_number(numbers[0], first, second)
            else:
                find_second(numbers[0], second)
        elif size >= 2:
            if is_digit(lines[i][0]):
                find_first(numbers[0], first)
                find_second(numbers[1], second)
            else:
                find_second(numbers[0], second)
        
        symbols = re.split('[0-9]', lines[i])
        symbols = list(filter(None, symbols))
        if '=' in symbols: symbols.remove('=')
        if '==' in symbols: symbols.remove('==')
        symbol = ''
        if 1 >= len(symbols) > 0:
            symbol = symbols[0]
        elif len(symbols) > 1:
            symbol = symbols[-1]

        i = 0
        while i < len(symbol):
            c = symbol[i]
            delimiter[c] = 1 + delimiter[c] if c in delimiter else 1
            i += 1
    print(lines)
    print(first, delimiter, second)
    text = ''
    f = [k for k, v in first.items() if v == max(first.values())]
    s = [k for k, v in second.items() if v == max(second.values())]
    if len(first) > 0 and len(delimiter) > 0 and len(second) > 0:
        text = find_max_list(f) + max(delimiter, key=delimiter.get) + find_max_list(s) + '='

    print(text)
    return text
