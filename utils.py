import sys
import skimage
import random as rd
sys.path.append('/Users/dariatunina/mach-lerinig/DataAugmentationForObjectDetection')

from data_aug.data_aug import *
from data_aug.bbox_util import *

bg_imgs = ["help_stuff/blank_page2.png", "help_stuff/crumpled.jpg", "help_stuff/lined.jpg", "help_stuff/squared.jpg"]


def rotate_img(img, bboxes):
    #print(bboxes)
    return RandomRotate(20)(img.copy(), bboxes.copy())
    #plotted_img = draw_rect(img_, bboxes_)
    #plt.imshow(plotted_img)
    #plt.axis("off")


def shear_img(img, bboxes):
    return RandomShear(0.2)(img.copy(), bboxes.copy())


def rotate_or_shear_img(img, bboxes):
    rand = rd.randint(0, 2)
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
    return bg_imgs[rd.randint(0, 3)]


def save_labels(borders, path_lbl, i, width, height):
    file = open(path_lbl + str(i) + '.txt', 'w+')
    obj_width = borders[2] - borders[0]
    obj_height = borders[3] - borders[1]
    center_x = int(borders[0] + (obj_width / 2))
    center_y = int(borders[1] + (obj_height / 2))

    file.write(str(int(borders[4])) + ' ' + str(center_x / width) + ' ' +
               str(center_y / height) + ' ' + str(obj_width / width) + ' ' + str(obj_height / height))
    file.write('\n')


def plotnoise(img, mode, path):
    #plt.plot(r,c)
    save_image(skimage.util.random_noise(img, mode=mode), path)
        #plt.imshow(gimg)
    #plt.title(mode)
    #plt.axis("off")


def add_noise(img, path):
    indx = rd.randint(0, 4)
    if indx == 0:
        plotnoise(img, "gaussian", path)
    elif indx == 1:
        plotnoise(img, "poisson", path)
    elif indx == 2:
        plotnoise(img, "speckle", path)
    elif indx == 3:
        plotnoise(img, "localvar", path)

