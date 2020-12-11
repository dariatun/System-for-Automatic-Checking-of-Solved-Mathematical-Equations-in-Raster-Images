import re
from functools import cmp_to_key

import numpy as np
from application.constants import *


def extract_boxes_confidences_classids(outputs, confidence, width, height):
    boxes = []
    confidences = []
    classIDs = []
    indx = 0
    for output in outputs:
        for detection in output:
            # Extract the scores, classid, and the confidence of the prediction
            scores = detection[5:]
            classID = np.argmax(scores)
            conf = scores[classID]

            # Consider only the predictions that are above the confidence threshold
            if conf > confidence:  # (classID == 1 and conf > confidence) or (classID == 0 and conf > 0.4):
                # Scale the bounding box back to t:
                # Scale the bounding box back to the size of the image
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, w, h = box.astype('int')

                # Use the center coordinates, width and height to get the coordinates of the top left corner
                x = int(centerX - (w / 2))
                y = int(centerY - (h / 2))
                box = [x, y, int(w), int(h)]

                boxes.append(box)
                confidences.append(float(conf))
                classIDs.append(classID)

                indx += 1

    return boxes, confidences, classIDs


def get_modified_by_indxes(boxes, confidences, classIDs, idxs):
    new_boxes = []
    new_confidences = []
    new_class_ids = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            new_boxes.append(boxes[i])
            new_confidences.append(confidences[i])
            new_class_ids.append(classIDs[i])
    return new_boxes, new_confidences, new_class_ids


def divide_by_lines(boxes, predictions, class_ids, are_legitimate):
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
    def cmp_function(a, b):
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
    return sorted(lines, key=lambda x: x[0][0][1])


def add_appropriate_amount_of_empty_strings(matrix_line, handwritten_digit_count):
    if handwritten_digit_count == 0:
        matrix_line.extend(['', ''])
    elif handwritten_digit_count == 1:
        matrix_line.append('')


def last_three_elements_is_empty(length, matrix, i):
    return len(matrix[i][length - 1]) == 0 and len(matrix[i][length - 2]) == 0 and len(matrix[i][length - 3]) == 0


def create_matrix_from_lines(lines):
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
                if handwritten_digit_count == -1:
                    matrix_line.append('')
                    handwritten_digit_count = 0
                elif last_handwritten is not None and last_equation is not None and element[BOX][X] - (
                        last_handwritten[BOX][X] + last_handwritten[BOX][
                    W]) > POSSIBLE_WIDTH_BETWEEN_CLOSE_OBJECTS * 2 and \
                        element[BOX][X] - POSSIBLE_WIDTH_BETWEEN_CLOSE_OBJECTS * 2 - (
                        last_equation[BOX][X] + last_equation[BOX][W]) > 0:
                    add_appropriate_amount_of_empty_strings(matrix_line, handwritten_digit_count)
                    matrix_line.append('')
                    handwritten_digit_count = 0
                elif handwritten_digit_count == 2:
                    continue
                matrix_line.append(element[PREDICTION])
                handwritten_digit_count += 1
                last_handwritten = element
        add_appropriate_amount_of_empty_strings(matrix_line, handwritten_digit_count)

        matrix.append(matrix_line)

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
    row_len = max_el[1]
    # row_len = math.ceil(row_len / 3) * 3
    for i in range(0, len(matrix)):
        if len(matrix[i]) == row_len:
            continue
        elif len(matrix[i]) > row_len:
            matrix[i] = matrix[i][:row_len - len(matrix[i])]
        else:
            for j in range(0, row_len - len(matrix[i])):
                matrix[i].append('')

    return matrix


def print_prediction_percent(approx_pred, all, file, string):
    pred_percent = approx_pred / all
    print('Prediction string comparacent percent ' + string + ': ' + str(pred_percent))
    file.write(str(pred_percent))
    file.write('\n')
    return pred_percent


def print_overall_prediction_correctness(value, dividend, name):
    print('overall prediction correctness' + name + ': ' + str(value / dividend))


def print_merged_answer(name, matrix):
    file = open(OUTPUT_PATH + name + '_answer' + TXT, 'w+')
    for i in range(0, len(matrix)):
        for j in range(0, len(matrix[0])):
            file.write(str(matrix[i][j]))
            file.write(' ')
        file.write('\n')
    file.close()


def get_handwritten_answer(handwritten_number_1, handwritten_number_2):
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
    numbers = re.split('[=+-/" /"]', equation)
    symbols = re.split('[0-9 ]', equation)

    if symbols[0] == '+':
        correct_answer = int(numbers[0]) + int(numbers[1])
    else:
        correct_answer = int(numbers[0]) - int(numbers[1])

    return correct_answer


def check_answer_of_one_equation(equation, handwritten_number_1, handwritten_number_2):
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
    result_matrix = []
    for i in range(0, len(prediction_matrix)):
        result_matrix.append([])
        for j in range(0, len(prediction_matrix[i]), 3):
            answer_result = check_answer_of_one_equation(prediction_matrix[i][j],
                                                         prediction_matrix[i][j + 1],
                                                         prediction_matrix[i][j + 2])
            result_matrix[i].append(answer_result)

    return result_matrix


def get_emotion(results):
    success_count = 0
    undefined_count = 0
    for i in range(0, len(results)):
        for j in range(0, len(results[i])):
            if results[i][j] == CORRECT_ANSWER:
                success_count += 1
            elif results[i][j] == UNDECIDED_ANSWER:
                undefined_count += 1

    overall = len(results) * len(results[0])
    if overall == 0:
        return NEUTRAL
    elif success_count / overall > SUCCESS_PROCENT:
        return SMILE
    elif undefined_count / overall > SUCCESS_PROCENT:
        return NEUTRAL
    else:
        return SAD

