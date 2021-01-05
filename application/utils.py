import imutils
from PIL import Image
import re


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


def get_xy_wh(coordinates, size):
    """ Recalculate x, y coordinates

    :param coordinates: coordinates of center of the image
    :param size: size of a full image
    :return: recalculated coordinates
    """
    width = int(coordinates['width'] * size[1])
    height = int(coordinates['height'] * size[0])
    x = int(coordinates['center_x'] * size[1] - width / 2)
    y = int(coordinates['center_y'] * size[0] - height / 2)
    return (x, y), width, height


def rotate_img(img_path, angle):
    img = Image.open(img_path)
    img = img.rotate(angle, expand=1)
    img.save(img_path)


def rotate_img_opencv(image, angle):
    return imutils.rotate(image, angle)


def add_to_dictionary(dictionary, el):
    dictionary[el] = 1 + dictionary[el] if el in dictionary else 1


def get_element_with_the_biggest_value(dictionary):
    return [k for k, v in dictionary.items() if v == max(dictionary.values())]


def get_delimeter(line):
    symbols = re.split('[0-9 ]', line)
    symbols = list(filter(None, symbols))
    symbols = remove_equation_symbol(symbols)

    symbol = ''
    if len(symbols) == 1:
        symbol = symbols[0]
    elif len(symbols) > 1:
        symbol = symbols[1]
    return symbol


def remove_equation_symbol(symbols):
    """
    Removes equation symbols from list ogf symbols
    :param symbols: 
    :return:
    """
    if '=' in symbols: symbols.remove('=')
    if '==' in symbols: symbols.remove('==')
    return symbols


def get_numbers_and_delimiter(expression):
    """
    Divides expression ito numbers and a delimiter
    :param expression: the expression to divide
    :return: the numbers and the delimiter
    """
    numbers = re.split('[=+-/" /"]', expression)
    numbers = list(filter(None, numbers))
    symbol = get_delimeter(expression)
    return numbers, symbol
