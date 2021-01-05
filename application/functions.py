from functools import cmp_to_key

import numpy as np

from application.constants import CORRECT_ANSWER, UNDECIDED_ANSWER, INCORRECT_ANSWER, SAD, SMILE, SUCCESS_PERCENT, \
    NEUTRAL, PREDICTION, OUTPUT_PATH, TXT, CLASS_ID, IS_LEGITIMATE, BOX, CONFIDENCE, POSSIBLE_OVERLAP_WIDTH, X, W, \
    EQUATIONS, POSSIBLE_WIDTH_BETWEEN_CLOSE_OBJECTS, SAME_LINE_CHARACTER_SPACE
from application.detect_equations import detect_mathematical_equation
from application.detect_handwritten_digit import detect_handwritten_digit
from application.image_manipulation import rbg_image_to_grey
from application.utils import get_numbers_and_delimiter


def extract_boxes_confidences_classids(outputs, width, height):
    """
    Divide predicted objects on boxes, confidences and class ids
    :param outputs: predicted objects
    :param width: image's width
    :param height: image's height
    :return:
    """
    boxes = []
    confidences = []
    classIDs = []
    indx = 0
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            conf = scores[classID]

            if conf > CONFIDENCE:
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, w, h = box.astype('int')

                x = int(centerX - (w / 2))
                y = int(centerY - (h / 2))
                box = [x, y, int(w), int(h)]

                boxes.append(box)
                confidences.append(float(conf))
                classIDs.append(classID)

                indx += 1

    return boxes, confidences, classIDs


def remove_similar_elements(boxes, classIDs, idxs):
    """
    Remove objects with similar bounding boxes
    :param boxes: bounding boxes of predicted objects
    :param classIDs: class ids of predicted objects
    :param idxs: indexes of objects to "keep"
    :return:
    """
    new_boxes = []
    new_class_ids = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            new_boxes.append(boxes[i])
            new_class_ids.append(classIDs[i])
    return new_boxes, new_class_ids


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


def print_prediction_percent(approx_pred, all, file, string):
    """
    ----- EVALUATION -----
    Prints prediction accuracy percent

    :param approx_pred: count of correctly predicted predictions
    :param all: count of all of the prediction
    :param file: file that accuracy percent is added to
    :param string: type of the prediction
    :return: prediction accuracy percent
    """
    if all == 0:
        pred_percent = 0
    else:
        pred_percent = approx_pred / all
    print('Prediction string accuracy percent ' + string + ': ' + str(pred_percent))
    file.write(str(pred_percent))
    file.write('\n')
    return pred_percent


def print_overall_prediction_correctness(value, dividend, name):
    """
        ----- EVALUATION -----
    Prints overall prediction accuracy percent
    :param value: the count of correctly predicted objects
    :param dividend: the count of all objects
    :param name: type of the prediction
    :return:
    """
    print('overall prediction correctness' + name + ': ' + str(value / dividend))


def print_merged_answer(name, matrix):
    """
    Add predicted objects to the file
    :param name: type of the prediction
    :param matrix: matrix of objects predictions
    :return:
    """
    file = open(OUTPUT_PATH + name + '_answer' + TXT, 'w+')
    for i in range(0, len(matrix)):
        for j in range(0, len(matrix[0])):
            file.write(str(matrix[i][j]))
            file.write(' ')
        file.write('\n')
    file.close()


def get_handwritten_answer(handwritten_number_1, handwritten_number_2):
    """
    Calculates handwritten answer
    :param handwritten_number_1: first handwritten digit
    :param handwritten_number_2: second handwritten digit
    :return: calculated answer
    """
    if handwritten_number_1 == '' and handwritten_number_2 == '':
        return -1
    elif handwritten_number_2 == '':
        answer = int(handwritten_number_1)
    elif handwritten_number_1 == '':
        answer = int(handwritten_number_2)
    else:
        answer = int(handwritten_number_1) * 10 + int(handwritten_number_2)
    return answer


def get_correct_answer(equation):
    """
    Calculate the answer to the equation
    :param equation: equation to calculate
    :return: answer to the equation
    """
    numbers, symbol = get_numbers_and_delimiter(equation)

    if symbol == '+':
        correct_answer = int(numbers[0]) + int(numbers[1])
    else:
        correct_answer = int(numbers[0]) - int(numbers[1])

    return correct_answer


def check_answer_of_one_equation(equation, handwritten_number_1, handwritten_number_2):
    """
    Check if the given answer is correct answer for the equation
    :param equation: equation to check
    :param handwritten_number_1: first handwritten digit
    :param handwritten_number_2: second handwritten digit
    :return: the decided correctness of the answer
    """
    if len(equation) == 0:
        return UNDECIDED_ANSWER

    correct_answer = get_correct_answer(equation)
    answer = get_handwritten_answer(handwritten_number_1, handwritten_number_2)

    if answer == -1:
        return UNDECIDED_ANSWER
    elif correct_answer == answer:
        return CORRECT_ANSWER
    else:
        return INCORRECT_ANSWER


def confirm_results(prediction_matrix):
    """
    Check if the given answers are correct answers for all of the equation
    :param prediction_matrix: matrix of the predicted objects
    :return: list of the results
    """
    result_list = []
    for i in range(0, len(prediction_matrix)):
        for j in range(0, len(prediction_matrix[i]), 3):
            answer_result = check_answer_of_one_equation(prediction_matrix[i][j],
                                                         prediction_matrix[i][j + 1],
                                                         prediction_matrix[i][j + 2])
            result_list.append(answer_result)

    return result_list


def get_emotion(results):
    """
    Get emotion that will be shown based on the calculated results
    :param results: list of all of the expressions results
    :return: type of the emotion
             number of the correct answers
             number of the expressions with an answer
    """
    success_count = 0
    undefined_count = 0
    defined_count = 0
    for i in range(0, len(results)):
        if results[i] == CORRECT_ANSWER:
            success_count += 1
            defined_count += 1
        elif results[i] == UNDECIDED_ANSWER:
            undefined_count += 1
        else:
            defined_count += 1

    overall = len(results)
    if overall == 0 or defined_count == 0:
        return NEUTRAL, success_count, defined_count
    elif success_count / defined_count > SUCCESS_PERCENT:
        return SMILE, success_count, defined_count
    else:
        return SAD, success_count, defined_count


def get_text_from_result(result):
    """
    Get text that will be displayed near the expression
    :param result: expression's result
    :return: chosen text
    """
    if result == CORRECT_ANSWER:
        text = 'Answer is correct!'
    elif result == INCORRECT_ANSWER:
        text = 'Check this, one more time!'
    else:
        text = 'Keep going, you\'re doing great!'
    return text


def run_object_detection(image, boxes, class_ids):
    """
    Get the detection of the handwritten digits and mathematical equations
    :param image: image to detect objects on
    :param boxes: bounding boxes of localised objects
    :param class_ids: class ids of localised objets
    :return: predictions of the objects
             the correctness of the mathematical equations
    """
    predictions = [''] * len(boxes)
    are_legitimate = [True] * len(boxes)
    for indx in range(0, len(boxes)):
        box = boxes[indx]
        predictions[indx], boxes[indx], are_legitimate[indx] = detect_an_object(image, box[0], box[1], box[2],
                                                                                box[3], class_ids[indx])
    return predictions, are_legitimate


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
        prediction, box, is_legitimate = detect_mathematical_equation(image, x, y, w, h, loop_number=0)
    else:
        prediction = detect_handwritten_digit(image, x, y, w, h)
        prediction = str(prediction[0])
    return prediction, box, is_legitimate
