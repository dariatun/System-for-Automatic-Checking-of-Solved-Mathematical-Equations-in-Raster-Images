import json
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from keras.optimizers import SGD

from recognise_text import recognise_text
from utils import delete_old_files
from utils import get_xy_wh, cut_image

# path to the directory with the CNN_Keras
CNN_Keras_PATH = '/Users/dariatunina/mach-lerinig/Handwritten-Digit-Recognition-using-Deep-Learning/CNN_Keras'
sys.path.append(CNN_Keras_PATH)
from cnn.neural_network import CNN

SAVE_EACH_NUMBER = False
INPUTS_FROM_STDIN = False


def recognise_image(image):
    """ Predict the image with the CNN model

    :param image: image of a digit
    :return: image, prediction
    """
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    clf = CNN.build(width=28, height=28, depth=1, total_classes=10,
                    Saved_Weights_Path='cnn_weights.hdf5')
    clf.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    probs = clf.predict(image)
    prediction = probs.argmax(axis=1)
    return image, prediction


def prepare_image(image):
    #image = Image.fromarray(image)
    # convert rgb image to greyscale image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return image


def prepare_handwritten_image(image):
    """ Resize image and convert it to a greyscale image

    :param image: initial image
    :return: changed image
    """
    image = Image.fromarray(image)

    # convert image to 28x28 size
    change_to = 28
    image = image.resize((change_to, change_to), Image.ANTIALIAS)
    image = np.array(image)

    # change size
    image = np.reshape(image, (1, 1, change_to, change_to))

    image = -image

    # image[image <= 100] = 0

    return image


def plot_single_digit1(image, prediction):
    """ Plots image with the predicted label

    :param image:
    :param prediction:
    :return:
    """
    plt.title('Predicted Label: {0}'.format(prediction))
    plt.imshow(image, interpolation='nearest', cmap='gray')
    plt.show()


def plot_single_digit(image, prediction):
    """ Plots image with the predicted label

    :param image:
    :param prediction:
    :return:
    """
    two_d = (np.reshape(image, (28, 28))).astype(np.uint8)
    plt.title('Predicted Label: {0}'.format(prediction))
    plt.imshow(two_d, interpolation='nearest', cmap='gray')
    plt.show()


def plot_full_image(predictions, xy_coords, draw):
    """ Plots initial image with written on it labels and bounding boxes around objects

    :param predictions: predicted labels
    :param xy_coords: coordinates of the predictions
    :param draw: ImageDraw module, that allows putting label on an image
    :return:
    """
    for i in range(0, len(predictions)):
        font = ImageFont.truetype("fonts/arial.ttf", int(xy_coords[i][1] * 0.5))
        draw.text(xy=xy_coords[i][0], text=str(predictions[i]), fill=(255, 0, 0), font=font)


def recognise_one_image_at_a_time(objects, img):
    """ Prediction is done by one image at a time
    :param objects: array of objects in the image
    :param img: initial image
    :return:
    """
    for obj in objects:

        xy, w, h = get_xy_wh(obj['relative_coordinates'], img.shape)
        image = cut_image(xy[0], xy[1], w, h, img)
        image = prepare_image(image)
        if obj['class_id'] == 0:
            prediction = recognise_text(image)
            if len(prediction) == 0:
                continue
            plot_single_digit1(image, prediction)

        else:
            image, prediction = recognise_image(prepare_handwritten_image(image))
            plot_single_digit(image, prediction)


def recognise_object(image, x, y, w, h, class_id):
    image = cut_image(x - 12, y - 12, w + 24, h + 24, image)
    # image = cut_image(xy[0] , xy[1] , w , h , init_img_arr)
    if image.size == 0:
        return ""
    image = prepare_image(image)
    if class_id == 0:
        prediction = recognise_text(image)
        if len(prediction) == 0: return ""
    else:
        image = prepare_handwritten_image(image)
        _, prediction = recognise_image(image)
        prediction = str(prediction[0])
    return prediction


def recognises_all_digits(objects, init_img_arr, filename):
    """ Prediction is done by taking all of the objects from one image

    :param objects: array of objects in the image
    :param init_img_arr: initial image
    :param filename: name of the image file
    :return:
    """
    handwrttn_imgs = None
    handwrttn_xy_coords = []
    equations_xy_coords = []
    equations_predictions = []
    init_img = Image.fromarray(init_img_arr)
    #init_img = cv2.cvtColor(np.float32(init_img), cv2.COLOR_BGR2RGB)
    draw = ImageDraw.Draw(init_img)

    for obj in objects:
        if obj['confidence'] < 0.5:
            continue
        xy, w, h = get_xy_wh(obj['relative_coordinates'], init_img_arr.shape)
        image = cut_image(xy[0] - 12, xy[1] - 12, w + 24, h + 24, init_img_arr)
        #image = cut_image(xy[0] , xy[1] , w , h , init_img_arr)
        if image.size == 0:
            continue
        image = prepare_image(image)
        if obj['class_id'] == 0:
            prediction = recognise_text(image)
            if len(prediction) == 0: continue
            draw.rectangle([xy, (w + xy[0], h + xy[1])], outline="green")
            equations_xy_coords.append([xy, h, w])
            equations_predictions.append(prediction)
        else:
            draw.rectangle([xy, (w + xy[0], h + xy[1])], outline="blue")

            handwrttn_xy_coords.append([xy, h])
            image = prepare_handwritten_image(image)
            if handwrttn_imgs is None:
                handwrttn_imgs = image
            else:
                handwrttn_imgs = np.append(handwrttn_imgs, image, axis=0)
    predictions = None
    if handwrttn_imgs is not None:
        _, predictions = recognise_image(handwrttn_imgs)
        plot_full_image(predictions, handwrttn_xy_coords, draw)
    if len(equations_predictions) > 0:
        plot_full_image(equations_predictions, equations_xy_coords, draw)
    init_img.save('out/' + filename + '_rec.jpg')
    print('added ' + filename + '.jpg')
    return predictions, handwrttn_xy_coords, equations_predictions, equations_xy_coords


def handwritten_recogniser():
    if INPUTS_FROM_STDIN:
        path_to_image_folder = input('Enter path to the folder with images: ')
        path_to_json_file = input('Enter path to the json file: ')
    else:
        path_to_image_folder = '/Users/dariatunina/mach-lerinig/mLStuff/'
        path_to_json_file = '/Users/dariatunina/mach-lerinig/mLStuff/result.json'
    with open(path_to_json_file) as json_file:
        data = json.load(json_file)
        for file in data:
            filename = file['filename'].split('/')[-1]
            try:
                img = np.array(Image.open(path_to_image_folder + filename))
            except Exception as e:
                continue
            if SAVE_EACH_NUMBER:
                recognise_one_image_at_a_time(file['objects'], img)
            else:
                return recognises_all_digits(file['objects'], img, filename[0:-4])
    print('Finished')


if __name__ == "__main__":
    handwritten_recogniser()
