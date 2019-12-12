import sys
import skimage
import random as rd
import numpy as np
import os
from PIL import Image
import cv2
import matplotlib
matplotlib.use('Agg')

sys.path.append('/home.stud/tunindar/DataAugmentationForObjectDetection')

from data_aug.data_aug import *
from data_aug.bbox_util import *

bg_imgs = ["help_stuff/plain-white-paper.jpg", "help_stuff/crumpled.jpg", #"help_stuff/crump_fold.jpg",
           #"help_stuff/crump_old.jpg", "help_stuff/folded.jpg",
           #"help_stuff/grey.jpg",
           "help_stuff/paper.jpg"]


def rotate_img(img, bboxes):
    # print(bboxes)
    return RandomRotate(20)(img.copy(), bboxes.copy())
    # plotted_img = draw_rect(img_, bboxes_)
    # plt.imshow(plotted_img)
    # plt.axis("off")


def shear_img(img, bboxes):
    return RandomShear(0.2)(img.copy(), bboxes.copy())


def rotate_or_shear_img(img, bboxes):
    rand = rd.randint(0, 1)
    if rand == 0:
        return rotate_img(img, bboxes)
    elif rand == 1:
        return shear_img(img, bboxes)
    return img, bboxes


def save_image(data, fn):
    sizes = np.shape(data)
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(width / height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(data)
    plt.savefig(fn, dpi=height)
    plt.close()


def get_bg_path():
    length = len(bg_imgs)
    return bg_imgs[rd.randint(0, length - 1)]


def save_labels(borders, path_lbl, i, j, width, height):
    file = open(path_lbl + str(i) + '_' + str(j) + '.txt', 'w+')
    for border in borders:
        obj_width = border[2] - border[0]
        obj_height = border[3] - border[1]
        center_x = (border[0] + (obj_width / 2))
        center_y = (border[1] + (obj_height / 2))
        # print(center_x, width, center_y, height)
        file.write(str(int(border[4])) + ' ' + str(center_x / float(width)) + ' ' +
                   str(center_y / float(height)) + ' ' + str(obj_width / width) + ' ' + str(obj_height / height))
        file.write('\n')


def plotnoise(img, mode, path):
    # plt.plot(r,c)
    save_image(skimage.util.random_noise(img, mode=mode), path)
    # plt.imshow(gimg)
    # plt.title(mode)
    # plt.axis("off")


def add_noise(img, path):
    indx = rd.randint(0, 3)
    if indx == 0:
        plotnoise(img, "gaussian", path)
    elif indx == 1:
        plotnoise(img, "poisson", path)
    elif indx == 2:
        plotnoise(img, "speckle", path)
    elif indx == 3:
        plotnoise(img, "localvar", path)


def get_full_path(path, extension, num):
    return path + '_' + str(num) + extension


def delete_old_files(path):
    for the_file in os.listdir(path):
        file_path = os.path.join(path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def get_handwritten_digit(imgs):
    return Image.fromarray((np.reshape(-1-imgs[rd.randint(0, len(imgs) - 1)], (28, 28))).astype(np.uint8))


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def change_sizeimg(img):
    change_to = rd.randint(img.size[0] - 3, img.size[0] + 10)
    wpercent = (change_to / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    return img.resize((change_to, hsize), Image.ANTIALIAS)
