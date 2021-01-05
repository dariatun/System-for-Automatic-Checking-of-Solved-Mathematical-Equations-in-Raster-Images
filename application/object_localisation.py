import cv2
import numpy as np

from application.constants import CONFIDENCE, THRESHOLD


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


def run_nms_algorithm(boxes, confidences, class_ids):
    """
    The NMS algorithm
    :param boxes: the list of bounding boxes
    :param confidences: the list of prediction confidences
    :param class_ids: the list of class ids
    :return: updated lists of bounding boxes, prediction confidences, class ids
    """
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)

    return remove_similar_elements(boxes, class_ids, indexes)

