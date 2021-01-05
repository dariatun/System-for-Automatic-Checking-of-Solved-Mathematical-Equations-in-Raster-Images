from application.detect_equations import detect_mathematical_equation
from application.detect_handwritten_digit import detect_handwritten_digit
from application.image_manipulation import rbg_image_to_grey


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