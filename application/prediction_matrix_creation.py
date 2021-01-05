from functools import cmp_to_key

from application.constants import PREDICTION, CLASS_ID, IS_LEGITIMATE, BOX, POSSIBLE_OVERLAP_WIDTH, X, W, \
    EQUATIONS, POSSIBLE_WIDTH_BETWEEN_CLOSE_OBJECTS, SAME_LINE_CHARACTER_SPACE


def divide_by_lines(boxes, predictions, class_ids, are_legitimate):
    """
    Divide list of object's characteristic into lines
    :param boxes: bounding boxes of predicted objects
    :param predictions: predictions of objects
    :param class_ids: class ids of predicted objects
    :param are_legitimate: list og booleans that tells if equations are correct
    :return: lines that the boxes are divided into
    """
    height_indxs, lines = [], []
    for i in range(0, len(boxes)):
        y = boxes[i][1]
        found_line = False
        for j in range(0, len(height_indxs)):
            if abs(height_indxs[j] - y) < SAME_LINE_CHARACTER_SPACE:
                lines[j].append([boxes[i], predictions[i], class_ids[i], are_legitimate[i]])
                found_line = True
        if not found_line:
            height_indxs.append(y)
            lines.append([[boxes[i], predictions[i], class_ids[i], are_legitimate[i]]])
    return lines


def sort_by_x_coordinate(lines):
    """
    Sorts objects in lines by the x-coordinate of their bounding box
    :param lines: the lines to sort
    :return: sorted lines
    """
    def cmp_function(a, b):
        """
        Sorts two objects
        :param a: first object
        :param b: second object
        :return: boolean, the order of sort
        """
        if a[CLASS_ID] == b[CLASS_ID]:
            return a[BOX][X] - b[BOX][X]
        elif a[CLASS_ID] == EQUATIONS:
            if a[BOX][X] < b[BOX][X]:
                return a[BOX][X] - b[BOX][X]
            return - (b[BOX][X] + POSSIBLE_OVERLAP_WIDTH) + (a[BOX][X] + a[BOX][W])
        else:
            if b[BOX][X] < a[BOX][X]:
                return - b[BOX][X] + a[BOX][X]
            return a[BOX][X] + POSSIBLE_OVERLAP_WIDTH - (b[BOX][X] + b[BOX][W])

    for i in range(0, len(lines)):
        lines[i] = sorted(lines[i], key=cmp_to_key(cmp_function))
    return lines


def sort_by_y_coordinate(lines):
    """
    Sorts lines by the y-coordinate of their objects bounding box
    :param lines: the lines to sort
    :return: sorted lines
    """
    return sorted(lines, key=lambda x: x[0][0][1])


def add_appropriate_amount_of_empty_strings(matrix_line, handwritten_digit_count):
    """
    Add empty space to the line of objects
    :param matrix_line: line of added objects
    :param handwritten_digit_count: count of handwritten digits in a row
    :return:
    """
    if handwritten_digit_count == 0:
        matrix_line.extend(["", ""])
    elif handwritten_digit_count == 1:
        matrix_line.append("")


def last_three_elements_is_empty(length, matrix, i):
    """
    Checks if last three elements are empty
    :param length: list's length
    :param matrix: matrix that contains prediction
    :param i: the x-coordinate of the matrix
    :return: boolean, returns True if the last three elements are empty,
                      returns False otherwise
    """
    return len(matrix[i][length - 1]) == 0 and len(matrix[i][length - 2]) == 0 and len(matrix[i][length - 3]) == 0


def add_handwritten_to_matrix(handwritten_digit_count, matrix_line, element, last_handwritten, last_equation):
    """

    :param handwritten_digit_count: count of handwritten digits in a row
    :param matrix_line: list of added objects
    :param element: handwritten digit element
    :param last_handwritten: the last handwritten element added to the list
    :param last_equation: the last equation element added to the list
    :return: updated list of objects, updated count of handwritten digits in a row
             updated the last handwritten element added to the list
    """
    if handwritten_digit_count == -1:
        matrix_line.append("")
        handwritten_digit_count = 0
    elif last_handwritten is not None and last_equation is not None and\
            element[BOX][X] - (last_handwritten[BOX][X] + last_handwritten[BOX][W]) >\
            POSSIBLE_WIDTH_BETWEEN_CLOSE_OBJECTS * 2 and \
            element[BOX][X] - POSSIBLE_WIDTH_BETWEEN_CLOSE_OBJECTS * 2 - (
            last_equation[BOX][X] + last_equation[BOX][W]) > 0:
        add_appropriate_amount_of_empty_strings(matrix_line, handwritten_digit_count)
        matrix_line.append("")
        handwritten_digit_count = 0
    elif handwritten_digit_count == 2:
        return matrix_line, handwritten_digit_count, last_handwritten
    matrix_line.append(element[PREDICTION])
    handwritten_digit_count += 1
    last_handwritten = element
    return matrix_line, handwritten_digit_count, last_handwritten


def max_element_in_line(matrix):
    """
    Find the longest line of non-empty objects
    :param matrix: matrix that contains object's prediction
    :return: the longest line of non-empty objects
    """
    max_el = [0, 0]
    for i in range(0, len(matrix)):
        count = 0
        length = len(matrix[i])
        for j in range(0, length):
            if len(matrix[i][j]) != 0:
                count += 1
        if max_el[0] < count:
            if last_three_elements_is_empty(length, matrix, i):
                max_el = [count, length - 3]
            else:
                max_el = [count, length]
        elif max_el[0] == count and max_el[1] < length:
            if last_three_elements_is_empty(length, matrix, i):
                max_el = [count, length - 3]
            else:
                max_el = [count, length]
    return max_el[1]


def make_lines_one_length(matrix):
    """
    Make matrix's rows the same length
    :param matrix: matrix that contains object's prediction
    :return: updated matrix
    """
    row_len = max_element_in_line(matrix)
    for i in range(0, len(matrix)):
        if len(matrix[i]) == row_len:
            continue
        elif len(matrix[i]) > row_len:
            matrix[i] = matrix[i][:row_len - len(matrix[i])]
        else:
            for j in range(0, row_len - len(matrix[i])):
                matrix[i].append("")
    return matrix


def create_matrix_from_lines(lines):
    """
    Create matrix similar to the position of the expressions in the image
    :param lines: lines of object's bounding boxes, class ids, predictions
    :return: matrix of object's prediction
    """
    matrix = []
    last_handwritten = None
    last_equation = None

    for line in lines:
        if len(line) <= 1:
            continue
        matrix_line = []
        handwritten_digit_count = -1
        for element in line:
            if element[CLASS_ID] == 0:
                if not element[IS_LEGITIMATE] or len(element[PREDICTION]) == 0:
                    continue
                add_appropriate_amount_of_empty_strings(matrix_line, handwritten_digit_count)
                matrix_line.append(element[PREDICTION])
                handwritten_digit_count = 0
                last_equation = element
            else:
                matrix_line, handwritten_digit_count, last_handwritten = add_handwritten_to_matrix(
                    handwritten_digit_count, matrix_line, element, last_handwritten, last_equation)
        add_appropriate_amount_of_empty_strings(matrix_line, handwritten_digit_count)

        matrix.append(matrix_line)

    return make_lines_one_length(matrix)

