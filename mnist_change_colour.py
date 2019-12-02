import scipy.ndimage
from PIL import Image as PILImage

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

lena = PILImage.open('help_stuff/grey.jpg')


def get_mnist_batch(batch_size=256, change_colors=True):
    # Select random batch (WxHxC)
    idx = np.random.choice(x_train.shape[0], batch_size)
    batch_raw = x_train[idx, :, :, 0].reshape((batch_size, 28, 28, 1))

    # Resize (this is optional but results in a training set of larger images)
    batch_resized = np.asarray([scipy.ndimage.zoom(image, (2.3, 2.3, 1), order=1) for image in batch_raw])

    # Extend to RGB
    batch_rgb = np.concatenate([batch_resized, batch_resized, batch_resized], axis=3)

    # Convert the MNIST images to binary
    batch_binary = (batch_rgb > 0.5)

    # Create a new placeholder variable for our batch
    batch = np.zeros((batch_size, 64, 64, 3))

    for i in range(batch_size):
        # Take a random crop of the Lena image (background)
        x_c = np.random.randint(0, lena.size[0] - 64)
        y_c = np.random.randint(0, lena.size[1] - 64)
        image = lena.crop((x_c, y_c, x_c + 64, y_c + 64))
        # Conver the image to float between 0 and 1
        image = np.asarray(image) / 255.0

        if change_colors:
            # Change color distribution
            for j in range(3):
                image[:, :, j] = (image[:, :, j] + np.random.uniform(0, 1)) / 2.0

        # Invert the colors at the location of the number
        image[batch_binary[i]] = 1 - image[batch_binary[i]]

        batch[i] = image

    return batch


if __name__ == "__main__":
    # Read MNIST data
    x_train = input_data.read_data_sets("mnist", one_hot=True).train.images
    x_train = x_train.reshape(-1, 28, 28, 1).astype(np.float32)
    count = 20
    examples = get_mnist_batch(count)

    plt.figure(figsize=(15, 3))
    for i in range(count):
        plt.subplot(2, count // 2, i + 1)
        plt.imshow(examples[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()
