import json
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from keras.optimizers import SGD

sys.path.append('/Users/dariatunina/mach-lerinig/Handwritten-Digit-Recognition-using-Deep-Learning/CNN_Keras')
from cnn.neural_network import CNN


def recognise_image(image):
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    clf = CNN.build(width=28, height=28, depth=1, total_classes=10,
                    Saved_Weights_Path='/Users/dariatunina/mach-lerinig/Handwritten-Digit-Recognition-using-Deep'
                                       '-Learning/CNN_Keras/cnn_weights.hdf5')
    clf.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    probs = clf.predict(image)
    prediction = probs.argmax(axis=1)
    return image, prediction


def cut_image(x, y, width, height, image):
    return image[y:y+height, x:width+x, :]


def get_xy_wh(coordinates, size):
    width = int(coordinates['width'] * size[1])
    height = int(coordinates['height'] * size[0])
    x = int(coordinates['center_x'] * size[1] - width / 2)
    y = int(coordinates['center_y'] * size[0] - height / 2)
    return (x, y), width, height


def prepare_image(image):
    image = Image.fromarray(image)

    # convert image to 28x28 size
    change_to = 28
    image = image.resize((change_to, change_to), Image.ANTIALIAS)
    image = np.array(image)

    # convert rgb image to grayscale image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = lit.rgb2gray(image)

    # change size
    image = np.reshape(image, (1, 1, change_to, change_to))

    image = -image

    image[image <= 100] = 0

    return image


def plot_single_digit(image, prediction):
    two_d = (np.reshape(image, (28, 28))).astype(np.uint8)
    plt.title('Predicted Label: {0}'.format(prediction))
    plt.imshow(two_d, interpolation='nearest', cmap='gray')
    plt.show()


def plot_full_image(predictions, xy_coords, draw):
    font = ImageFont.truetype("fonts/arial.ttf", 20)
    for i in range(0, len(predictions)):
        draw.text(xy=xy_coords[i], text=str(predictions[i]), fill=(255, 0, 0), font=font)


def recognise_one_image_at_a_time(objects, img):
    for obj in objects:
        if obj['class_id'] == 0:
            continue
        xy, w, h = get_xy_wh(obj['relative_coordinates'], img.shape)
        image = cut_image(xy[0], xy[1], w, h, img)
        image, prediction = recognise_image(prepare_image(image))
        plot_single_digit(image, prediction)


def recognises_all_digits(objects, init_img_arr, filename):
    images = None
    xy_coords = []
    init_img = Image.fromarray(init_img_arr)
    draw = ImageDraw.Draw(init_img)

    for obj in objects:
        xy, w, h = get_xy_wh(obj['relative_coordinates'], img.shape)
        draw.rectangle([xy, (w + xy[0], h + xy[1])], outline="green")

        if obj['class_id'] == 0:
            continue
        image = cut_image(xy[0], xy[1], w, h, init_img_arr)
        xy_coords.append(xy)
        image = prepare_image(image)
        if images is None:
            images = image
        else:
            images = np.append(images, image, axis=0)
    if images is not None:
        _, predictions = recognise_image(images)
        plot_full_image(predictions, xy_coords, draw)
    init_img.save('/Users/dariatunina/mach-lerinig/test-images/' + filename + '_rec.jpg')
    print('added ' + filename + '.jpg')


if __name__ == "__main__":
    path_to_image_folder = '/Users/dariatunina/mach-lerinig/test-images/'
    SAVE_EACH_NUMBER = False
    with open('result.json') as json_file:
        data = json.load(json_file)
        for file in data:
            filename = file['filename'].split('/')[-1]
            img = np.array(Image.open(path_to_image_folder + filename))
            if SAVE_EACH_NUMBER:
                recognise_one_image_at_a_time(file['objects'], img)
            else:
                recognises_all_digits(file['objects'], img, filename[0:-4])
    print('Finished')
