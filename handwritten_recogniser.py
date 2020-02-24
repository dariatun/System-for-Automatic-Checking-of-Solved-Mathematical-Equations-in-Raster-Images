import json
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from keras.optimizers import SGD

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


def cut_image(x, y, width, height, image):
    """ Cut the image by the border coordinates

    :param x: x coordinate of the border's left side
    :param y: y coordinate of the border's top side
    :param width: the width of the border
    :param height: the height of the border
    :param image: the image to cut digit from
    :return: image of a digit
    """
    return image[y:y+height, x:width+x, :]


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


def prepare_image(image):
    """ Resize image and convert it to a greyscale image

    :param image: initial image
    :return: changed image
    """
    image = Image.fromarray(image)

    # convert image to 28x28 size
    change_to = 28
    image = image.resize((change_to, change_to), Image.ANTIALIAS)
    image = np.array(image)

    # convert rgb image to greyscale image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # change size
    image = np.reshape(image, (1, 1, change_to, change_to))

    image = -image

    image[image <= 100] = 0

    return image


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
        if obj['class_id'] == 0:
            continue
        xy, w, h = get_xy_wh(obj['relative_coordinates'], img.shape)
        image = cut_image(xy[0], xy[1], w, h, img)
        image, prediction = recognise_image(prepare_image(image))
        plot_single_digit(image, prediction)


def recognises_all_digits(objects, init_img_arr, filename):
    """ Prediction is done by taking all of the objects from one image

    :param objects: array of objects in the image
    :param init_img_arr: initial image
    :param filename: name of the image file
    :return:
    """
    images = None
    xy_coords = []
    init_img = Image.fromarray(init_img_arr)
    draw = ImageDraw.Draw(init_img)

    for obj in objects:
        xy, w, h = get_xy_wh(obj['relative_coordinates'], img.shape)

        if obj['class_id'] == 0:
            draw.rectangle([xy, (w + xy[0], h + xy[1])], outline="green")
            continue
        draw.rectangle([xy, (w + xy[0], h + xy[1])], outline="blue")

        image = cut_image(xy[0], xy[1], w, h, init_img_arr)
        xy_coords.append([xy, h])
        image = prepare_image(image)
        if images is None:
            images = image
        else:
            images = np.append(images, image, axis=0)
    if images is not None:
        _, predictions = recognise_image(images)
        plot_full_image(predictions, xy_coords, draw)
    init_img.save('out/' + filename + '_rec.jpg')
    print('added ' + filename + '.jpg')


if __name__ == "__main__":
    if INPUTS_FROM_STDIN:
        path_to_image_folder = input('Enter path to the folder with images: ')
        path_to_json_file = input('Enter path to the json file: ')
    else:
        path_to_image_folder = '/Users/dariatunina/mach-lerinig/darknet/data/'
        path_to_json_file = '/Users/dariatunina/mach-lerinig/darknet/result.json'
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
                recognises_all_digits(file['objects'], img, filename[0:-4])
    print('Finished')
