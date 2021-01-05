import re
from application.constants import TXT, TEST_DATA_ANNOTATIONS_PATH, TEST_DATA_BOX_ANNOTATIONS_PATH, EQUATIONS, \
    HANDWRITTEN, BOX, CLASS_ID
from application.utils import get_numbers_and_delimiter


def compare_predictions(name, prediction_matrix, file):
    """
    Count correctly recognised objects.
    Compare predictions with the annotation file for the image
    :param name: name of the image
    :param prediction_matrix: matrix with the predicted objects
    :param file: file to write count of mistakes in
    :return: the number of the correctly recognised objects
    """
    right_count = [0, 0]
    corr_pred_file = open(TEST_DATA_ANNOTATIONS_PATH + name + TXT, 'r')
    lines = corr_pred_file.readlines()
    incorrect_symbol_count, incorrect_one_number_count, incorrect_two_numbers_count,\
        incorrect_number_and_symbol_count, incorrect_all_count, incorrect_digit_count = 0, 0, 0, 0, 0, 0

    for i in range(0, len(prediction_matrix)):
        if i == len(lines):
            break
        line = lines[i]
        elements = re.split('\|', line)
        for j in range(0, len(prediction_matrix[i])):
            class_id = get_class_id(j)
            if elements[j] == prediction_matrix[i][j]:
                right_count[class_id] += 1
            else:
                if class_id == EQUATIONS:
                    if len(prediction_matrix[i][j]) == 0:
                        incorrect_all_count += 1
                        break
                    symbol_is_correct, number_1_is_correct, number_2_is_correct = \
                        get_equation_correctness(elements[j],prediction_matrix[i][j])
                    if number_1_is_correct and number_2_is_correct:
                        incorrect_symbol_count += 1
                    elif number_1_is_correct and symbol_is_correct or number_2_is_correct and symbol_is_correct:
                        incorrect_one_number_count += 1
                    elif number_1_is_correct or number_2_is_correct:
                        incorrect_number_and_symbol_count += 1
                    elif symbol_is_correct:
                        incorrect_two_numbers_count += 1
                    else:
                        incorrect_all_count += 1
                else:
                    incorrect_digit_count += 1

    corr_pred_file.close()
    output_string = str(incorrect_symbol_count) + ' ' + str(incorrect_one_number_count) + ' ' + \
                    str(incorrect_two_numbers_count) + ' ' + str(incorrect_number_and_symbol_count) + ' ' + \
                    str(incorrect_all_count) + ' ' + str(incorrect_digit_count) + ' ' + \
                    str() + ' ' + str()
    print('Incorrect predictions: ' + output_string)
    file.write(output_string)
    return right_count


def get_class_id(indx):
    """
    Get the class id of the object
    :param indx: index in the matrix
    :return: class id of the object
    """
    if indx % 3 == 0:
        class_id = EQUATIONS
    else:
        class_id = HANDWRITTEN
    return class_id


def get_equation_correctness(corr_equation, predicted_equation):
    """
    Check the parts of equation correction
    :param corr_equation: the correct equation from the annotations file
    :param predicted_equation: the predicted equation
    :return: booleans, that tells whether delimiter, first number and second number
             are correctly predicted
    """
    corr_numbers, corr_symbol = get_numbers_and_delimiter(corr_equation)
    numbers, symbol = get_numbers_and_delimiter(predicted_equation)
    symbol_is_correct = symbol == corr_symbol
    number_1_is_correct = numbers[0] == corr_numbers[0]
    number_2_is_correct = numbers[1] == corr_numbers[1]
    return symbol_is_correct, number_1_is_correct, number_2_is_correct


def compare_box_predictions(boxes_classids, name):
    """
    Compares the correctness of predicted bounding boxes
    :param boxes_classids: the bounding boxes and class ids of predicted objects
    :param name: the name of the annotation file
    :return: the number of correctly predicted objects' boxes,
             the number of bounding boxes in the annotation file,
             the number of incorrectly predicted bounding boxes
    """
    annotation_boxes, annotation_classIDs = read_annotation_file(name)
    len_annotation_boxes = len(annotation_boxes)
    incorrect_box_positions_count = 0
    count = 0
    for i in range(0, len(boxes_classids)):
        element = boxes_classids[i]
        box = element[BOX]
        class_id = element[CLASS_ID]
        box_coordinates = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
        was_categorised = False
        for j in range(0, len(annotation_boxes)):
            annotation_box, annotation_class_id = annotation_boxes[j], annotation_classIDs[j]
            iou = get_iou(box_coordinates, annotation_box)
            if class_id == annotation_class_id and 1.0 > iou > 0.3:
                count += 1
                annotation_boxes.remove(annotation_box)
                annotation_classIDs.remove(annotation_class_id)
                was_categorised = True
                break
        if not was_categorised:
            incorrect_box_positions_count += 1

    return count, len_annotation_boxes, incorrect_box_positions_count


def get_iou(box_1, box_2):
    """
    Get IoU of two bounding boxes
    :param box_1: first bounding box
    :param box_2: second bounding box
    :return: the IoU value
    """
    x_left = max(box_1[0], box_2[0])
    y_top = max(box_1[1], box_2[1])
    x_right = min(box_1[2], box_2[2])
    y_bottom = min(box_1[3], box_2[3])

    if x_right - x_left < 0 or y_bottom - y_top < 0:
        return 0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    box_1_area = (box_1[2] - box_1[0]) * (box_1[3] - box_1[1])
    box_2_area = (box_2[2] - box_2[0]) * (box_2[3] - box_2[1])

    union_area = float(box_1_area + box_2_area - intersection_area)

    return intersection_area / union_area


def read_annotation_file(name):
    """
    Get bounding boxes and class ids from the annotation file
    :param name: the name of the annotation file
    :return: the lists of bounding boxes and class ids from the annotation file
    """
    annotations_file = open(TEST_DATA_BOX_ANNOTATIONS_PATH + name + TXT, 'r')
    lines = annotations_file.readlines()
    first = True
    annotation_boxes = []
    annotation_classIDs = []
    for line in lines:
        if first:
            first = False
            continue
        numbers = re.split(' ', line)
        annotation_box = numbers[0:-1]
        annotation_box = [int(annotation_box[0]), int(annotation_box[1]), int(annotation_box[2]),
                          int(annotation_box[3])]
        annotation_boxes.append(annotation_box)
        annotation_classIDs.append(int(numbers[-1]))

    return annotation_boxes, annotation_classIDs
