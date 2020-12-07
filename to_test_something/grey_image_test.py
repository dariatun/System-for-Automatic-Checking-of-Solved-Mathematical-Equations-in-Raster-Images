import cv2
import numpy as np
from application.scaner import scan
from application.constants import NUMBER_OF_IMAGES, NUMBER_IMAGES_PODTYPES, NUMBER_OF_IMAGE_TYPES


def segmentation_one_threshold(img, threshold):
    '''Segments image into black & white using one threshold

    Parameters
    ----------
    img : numpy.ndarray
        Input image.
    threshold : int
        Pixels with value lower than threshold are considered black, the others white.
    Returns
    -------
    Output image.
    '''
    _, dst = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return dst


def segmentation_auto_threshold(img):
    '''Segments image into black & white using automatic threshold

    Parameters
    ----------
    img : numpy.ndarray
        Input image.
    Returns
    -------
    Output image.
    '''
    _, dst = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return dst


def segmentation_two_thresholds(img, lower, higher):
    '''Segments image into black & white using two thresholds

    Parameters
    ----------
    img : numpy.ndarray
        Input image.
    lower : int
        Pixels with value lower than threshold are considered black, the others white.
    higher : int
        Pixels with value higher than threshold are considered black, the others white.
    Returns
    -------
    Output image.
    '''
    return cv2.inRange(img, min(lower, higher), max(lower, higher))


def segmentation_adaptive_threshold(img, size, constant=0):
    '''Segments image into black & white using calculated adaptive
    threshold using Gaussian function in pixel neighbourhood.

    Parameters
    ----------
    img : numpy.ndarray
        Input image.
    size : int
        Size of used gaussian. Lowest value is 3. Algorithm uses only odd numbers.
    constant : int
        Value that is added to calculated threshlod. It could be negative as well as zero as well as positive number.
    Returns
    -------
    Output binary image.
    '''
    if size < 3:
        size = 3
    elif size % 2 == 0:
        size -= 1
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, size, int(constant))


def process_image(file_name):
    image = scan(file_name, 'capture.jpg', to_crop=False)

    image = cv2.resize(image, (960, 540))
    # convert rgb image to greyscale image

    cv2.imshow('Original', image)
    cv2.waitKey(0)

    """
    tub_kernel = (1, 1)
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones(tub_kernel, np.uint8)
    grey = cv2.dilate(grey, kernel, iterations=1)
    grey = cv2.erode(grey, kernel, iterations=2)

    #  Apply threshold to get image with only black and white
    # images = cv2.adaptiveThreshold(images, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    _, grey = cv2.threshold(grey, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow("Grey", grey)
    cv2.waitKey(0)
    """
    """
    ret, thresh = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    erode = cv2.erode(thresh, kernel, iterations=1)
    result = cv2.bitwise_or(image, erode)
    """

    #image = rbg_image_to_grey(image)
    #result = image
    #

    # result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    # Remove noise
    # Gaussian
    no_noise = []
        #no_noise.append(blur)
    #image = blur
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 15)
    cv2.imshow('Blured', gray)
    cv2.waitKey(0)
    # Segmentation
    #thresh = cv2.GaussianBlur(gray, (3, 3), 2)

    kernel = np.ones((5, 5), np.uint8)

    erode = cv2.erode(gray, kernel, iterations=5)
    result = cv2.bitwise_or(gray, erode)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)

    #ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #thresh = cv2.GaussianBlur(gray, (9, 9), 0)

    #thresh = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    cv2.imshow('Segmented', result)
    cv2.waitKey(0)

    result = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 4)

    cv2.imshow('Result1', result)
    cv2.waitKey(0)

    # Further noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    cv2.imshow('Result', unknown)
    cv2.waitKey(0)


for i in range(4, NUMBER_OF_IMAGES):
    for j in range(0, NUMBER_IMAGES_PODTYPES):
        for k in range(0, NUMBER_OF_IMAGE_TYPES):
            filename = str(i) + '_' + str(j) + '_' + str(k)
            process_image('test_data/' + filename + '.jpg')
