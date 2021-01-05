import re


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


def add_to_dictionary(dictionary, el):
    """
    Adds element to the dictionary
    :param dictionary: the dictionary the element is added to
    :param el: the element that added to te dictionary
    :return:
    """
    dictionary[el] = 1 + dictionary[el] if el in dictionary else 1


def get_element_with_the_biggest_value(dictionary):
    """
    Get the element with the biggest value from the dictionary
    :param dictionary: the dictionary the value is taken from
    :return: the biggest value
    """
    return [k for k, v in dictionary.items() if v == max(dictionary.values())]


def get_delimiter(line):
    """
    Gets the delimiter from the equation line
    :param line: the equation line
    :return: the delimiter
    """
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
    :param symbols: the list of symbols
    :return: the modified list of symbols
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
    symbol = get_delimiter(expression)
    return numbers, symbol
