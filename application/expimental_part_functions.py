import re
from application.constants import TXT, TEST_DATA_ANNOTATIONS_PATH, TEST_DATA_BOX_ANNOTATIONS_PATH, EQUATIONS, HANDWRITTEN


def compare_predictions(name, prediction_matrix):
    right_count = [0, 0]
    file = open(TEST_DATA_ANNOTATIONS_PATH + name + TXT, 'r')
    lines = file.readlines()
    for i in range(0, len(prediction_matrix)):
        if i == len(lines):
            break
        line = lines[i]
        elements = re.split('\|', line)
        for j in range(0, len(prediction_matrix[i])):
            if j % 3 == 0:
                class_id = EQUATIONS
            else:
                class_id = HANDWRITTEN
            if elements[j] == prediction_matrix[i][j]:
                right_count[class_id] += 1
    file.close()
    return right_count


def compare_box_predictions(boxes, classIDs, name):
    annotation_boxes, annotation_classIDs = read_annotation_file(name)
    len_annotation_boxes = len(annotation_boxes)
    count = 0
    for i in range(0, len(boxes)):
        box = boxes[i]
        class_id = classIDs[i]
        box_coordinates = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
        # print(box_coordinates)
        for j in range(0, len(annotation_boxes)):
            annotation_box, annotation_class_id = annotation_boxes[j], annotation_classIDs[j]
            iou = get_iou(box_coordinates, annotation_box)
            # print(iou)

            if class_id == annotation_class_id and 1.0 > iou > 0.3:
                count += 1
                annotation_boxes.remove(annotation_box)
                annotation_classIDs.remove(annotation_class_id)
                break

    return count / len_annotation_boxes


def get_iou(box_1, box_2):
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
