import sys
import skimage
import random as rd
import numpy as np
import os
from PIL import Image
import cv2
import matplotlib

matplotlib.use('Agg')
from mask_generation import *

sys.path.append('/Users/dariatunina/mach-lerinig/DataAugmentationForObjectDetection')
# sys.path.append('/home.stud/tunindar/DataAugmentationForObjectDetection')

from data_aug.data_aug import *
from data_aug.bbox_util import *

blues = [
    [0, 0, 139],  # darkblue
    [0, 0, 128],  # navy
    [0, 0, 205],  # mediumblue
    # [0, 0, 255],  # blue
    [25, 25, 112],  # midnightblue
    # [89, 28, 212],  # bic blue pen
    [0, 15, 85]  # blue ink pen
]


def choose_blue_colour():
    return blues[rd.randint(0, len(blues) - 1)]


def rotate_img(img, bboxes):
    # print(bboxes)
    return RandomRotate(10)(img.copy(), bboxes.copy())
    # plotted_img = draw_rect(img_, bboxes_)
    # plt.imshow(plotted_img)
    # plt.axis("off")


def shear_img(img, bboxes):
    return RandomShear(0.2)(img.copy(), bboxes.copy())


def rotate_or_shear_img(img, bboxes):
    # rand = 1 #rd.randint(0, 1)
    # if rand == 0:
    img, bboxes = rotate_img(img, bboxes)
    # elif rand == 1:
    return shear_img(img, bboxes)
    # return img, bboxes


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


def get_handwritten_digit(imgs, font_height):
    return change_sizeimg(Image.fromarray(get_image_array(imgs)), font_height)


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def change_sizeimg(img, font_height):
    change_to = font_height + 5#rd.randint(font_height - 5, font_height + 5) #img.size[0] + 10#rd.randint(img.size[0] - 3, img.size[0] + 10)
    wpercent = (change_to / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    return img.resize((change_to, hsize), Image.ANTIALIAS)


def get_image_array(imgs):
    return (np.reshape(-1 - imgs[rd.randint(0, len(imgs) - 1)], (28, 28))).astype(np.uint8)


def get_digit(imgs, offset, bg_img, font_height):
    img_array = np.stack((np.array(get_handwritten_digit(imgs, font_height)),) * 3, axis=-1)
    blue_colour = choose_blue_colour()
    bg_img_array = np.array(bg_img)
    for i in range(0, img_array.shape[0]):
        for j in range(0, img_array.shape[1]):
            if 0 <= np.sum(img_array[j][i]) <= 100 * 3:
                img_array[j][i] = blue_colour
            elif 230 * 3 <= np.sum(img_array[j][i]) <= 255 * 3:
                img_array[j][i] = bg_img_array[j + offset[1]][i + offset[0]]
    return Image.fromarray(img_array)


def add_parallel_light(image, light_position=None, direction=None, max_brightness=255, min_brightness=0,
                       mode="gaussian", linear_decay_rate=None, transparency=None):
    """
    Add mask generated from parallel light to given image
    """
    if transparency is None:
        transparency = random.uniform(0.5, 0.85)
    frame = cv2.imread(image)
    height, width, _ = frame.shape
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = generate_parallel_light_mask(mask_size=(width, height),
                                        position=light_position,
                                        direction=direction,
                                        max_brightness=max_brightness,
                                        min_brightness=min_brightness,
                                        mode=mode,
                                        linear_decay_rate=linear_decay_rate)
    hsv[:, :, 2] = hsv[:, :, 2] * transparency + mask * (1 - transparency)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    frame[frame > 255] = 255
    frame = np.asarray(frame, dtype=np.uint8)
    return frame
