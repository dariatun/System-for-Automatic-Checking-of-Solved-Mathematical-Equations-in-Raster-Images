import sys
import numpy as np
from keras.optimizers import SGD
from application.constants import CNN_KERAS_PATH, SAVED_WEIGHTS_PATH

# path to the directory with the CNN_Keras
sys.path.append(CNN_KERAS_PATH)
from cnn.neural_network import CNN

DEBUG = False


def detect_handwritten_digit(image):
    """ Predict the image with the CNN model

    :param image: image of a digit
    :return: image, prediction
    """
    change_to = 28
    image = np.reshape(image, (1, 1, change_to, change_to))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    clf = CNN.build(width=28, height=28, depth=1, total_classes=10,
                    Saved_Weights_Path=SAVED_WEIGHTS_PATH)
    clf.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    probs = clf.predict(image)
    prediction = probs.argmax(axis=1)
    if DEBUG:
        print(prediction)
    return image, prediction


if __name__ == "__main__":
    print()
    # handwritten_recogniser()
