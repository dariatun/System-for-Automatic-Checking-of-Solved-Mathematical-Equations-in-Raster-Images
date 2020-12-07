import sys
import numpy as np
from keras.optimizers import SGD


# path to the directory with the CNN_Keras
CNN_Keras_PATH = '/Users/dariatunina/mach-lerinig/Handwritten-Digit-Recognition-using-Deep-Learning/CNN_Keras'
sys.path.append(CNN_Keras_PATH)
from cnn.neural_network import CNN

DEBUG = False


def recognise_handwritten_image(image):
    """ Predict the image with the CNN model

    :param image: image of a digit
    :return: image, prediction
    """
    change_to = 28
    image = np.reshape(image, (1, 1, change_to, change_to))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    clf = CNN.build(width=28, height=28, depth=1, total_classes=10,
                    Saved_Weights_Path='cnn_weights.hdf5')
    clf.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    probs = clf.predict(image)
    prediction = probs.argmax(axis=1)
    if DEBUG:
        print(prediction)
    return image, prediction


if __name__ == "__main__":
    print()
    # handwritten_recogniser()
