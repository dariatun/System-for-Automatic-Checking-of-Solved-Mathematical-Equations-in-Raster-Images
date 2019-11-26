import numpy as np
import cv2
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import os
import loremipsum as li
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skimage

from MNIST_Dataset_Loader.mnist_loader import MNIST
import random as rd

from os import path
import sys
sys.path.append('/Users/dariatunina/mach-lerinig/DataAugmentationForObjectDetection/')
from adjust_colour import adjust_gamma
"""
from adjust_colour import adjust_gamma

from utils import *

symbols = ['+', '-']
# numbers = [7, 9]
fonts = ["fonts/COMIC_SANS.ttf", "fonts/times-new-roman.ttf", "fonts/arial.ttf"]
EQUATION = True


def add_next_line_symbols(s, x, width, font):
    length = font.getsize(s)[0]
    len_s = len(s)
    ret_str = ''
    i = 0
    new_y = 0
    l = int((width - x * 2) / (length / len_s))
    while length >= l:
        add_str = s[i:i + l]
        ret_str += add_str + '\n'
        length -= font.getsize(add_str)[0]
        i += l
        new_y += font.getsize(add_str)[1]
    ret_str += s[i:]
    return ret_str, new_y


def generate_number(x, y, width, height, file, add_to_file, font):
    s = ''
    new_x = x
    for j in range(0, rd.randint(1, 2)):
        # num = rd.randint(0, 9) #numbers[rd.randint(0, 1)]
        if j == 0:
            num = rd.randint(1, 9)
        else:
            num = rd.randint(0, 9)
        s += str(num)
        # map[str(num)].append(indx)
        if add_to_file:
            size = font.getsize(str(num))
            center_x = int(new_x + (size[0] / 2))
            center_y = int(y + (size[1] / 2))

            file.write(str(num) + ' ' + str((center_x) / width) + ' ' +
                       str((center_y) / height) + ' ' + str(size[0] / width) + ' ' + str(size[1] / height))
            file.write('\n')
            new_x += size[0]  # font_size + 4
    # print(str(num) + ' ' + str((center_x)) + ' ' +
    #           str((center_y)) + ' ' + str(size[0]) + ' ' + str(size[1]))
    # indx += 1

    return s, new_x


def get_equation(x, y, width, height, file, font):
    equation = ''
    s, new_x = generate_number(x, y, width, height, file, add_to_file=False, font=font)
    equation += s + ' ' + symbols[rd.randint(0, 1)] + ' '
    # print('old: ' +  str(new_x))
    # print(font.getsize(symb))
    # print('new:' + str(new_x))
    s, new_x = generate_number(new_x, y, width, height, file, add_to_file=False, font=font)
    equation += s + ' ='

    size = font.getsize(equation)
    """
    center_x = int(x + (size[0] / 2))
    center_y = int(y + (size[1] / 2))
    
    file.write('0 ' + str((center_x) / width) + ' ' +
               str((center_y) / height) + ' ' + str(size[0] / width) + ' ' + str(size[1] / height))
    file.write('\n')
    """
    border = np.array([[x, y, x + size[0], y + size[1], 0]]).astype(np.float)
    return equation, new_x + size[0], border


def generate_expresion(x, width, path_lbl, i, y, height, font):
    # map = {'9': [], '7':[]}
    # map = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': []}
    # indx = 0
    # s = ''
    # new_y = 0
    #file = open(path_lbl + str(i) + '.txt', 'w+')
    file = None
    s, new_y = add_next_line_symbols(li.get_sentences(200, True)[2], x, width, font)
    # s += '\n'
    if EQUATION:
        s1, new_x, border = get_equation(x, new_y + y, width, height, file, font)
    else:
        s1, new_x = generate_number(x, new_y + y, width, height, file, add_to_file=True, font=font)
        s += s1 + ' '
        symb = symbols[rd.randint(0, 1)]
        s += symb
        s += ' '
        # print('old: ' +  str(new_x))
        new_x += font.getsize(' ')[0] * 2 + font.getsize(symb)[0]
        # print(font.getsize(symb))
        # print('new:' + str(new_x))
        s1, new_x = generate_number(new_x, new_y + y, width, height, file, add_to_file=True, font=font)
        s += s1 + ' ='
    s += s1 + '\n\n'
    # s += s1 + ' =\n\n'
    s1, new_y = add_next_line_symbols(li.get_sentences(1000, True)[2], x, width, font)
    s += s1
    # file.close()
    return s, border


if __name__ == "__main__":
    path = '/Users/dariatunina/mach-lerinig/numbers-eqs/'
    # print(font.getsize(' '))
    for the_file in os.listdir(path):
        file_path = os.path.join(path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)
    path_img = path  # + 'images/'
    path_lbl = path  # + 'labels/'
    for i in range(0, 5000):
        # print(str(i))
        bg_path = get_bg_path()
        img = Image.open(bg_path)
        draw = ImageDraw.Draw(img)
        # font = ImageFont.truetype(<font-file>, <font-size>)
        # draw.text((x, y),"Sample Text",(r,g,b))
        size = img.size
        x = rd.randint(0, 100)
        y = rd.randint(0, int(size[1]/2))
        font = ImageFont.truetype(fonts[rd.randint(0, 2)], rd.randint(40, 120))
        gen, border = generate_expresion(x=x, y=y, width=size[0], path_lbl=path_lbl, i=i, height=size[1],
                                         font=font)
        # print(str(y) + ' ' + str(font.getsize(gen)[1]) + ' ' + str(size[1]))
        #print(border[0][3])
        if border[0][3] > size[1]:
            change_on = border[0][3] - size[1] + rd.randint(10, 50)
            border[0][1] -= change_on
            border[0][3] -= change_on
            y -= change_on
            print('changing_y ' + str(i) + ' ' + str(change_on))
        color = rd.randint(0, 50)
        draw.text(xy=(x, y), text=gen, fill=(color, color, color), font=font)
        # for num in map:
        #    for j in map[num]:
        curr_img_path = path_img + str(i) + bg_path[-4:]
        # img = img.rotate(rd.randint(-25, 25), expand=1, fillcolor='white')
        # img.save(curr_img_path)
        # img = Image.open(curr_img_path)

        #img = img.point(lambda p: p * rd.uniform(0.1, 10.0))
        img.save(curr_img_path)
        # print('changed color')

        # img = mpimg.imread(curr_img_path)  # skimage.io.imread(curr_img_path) / 255.0
        # plt.figure(figsize=(size[0] / 255.0, size[1] / 255.0))
        # print('loaded')
        img = cv2.imread(curr_img_path, 1)
        img = adjust_gamma(img, gamma=rd.uniform(0.1, 2.5))
        save_image(img, curr_img_path)

        # cv2.imwrite(curr_img_path, img)
        img = cv2.imread(curr_img_path)[:, :, ::-1]
        add_noise(img, curr_img_path)
        # print('added noise')
        # save_image(img, curr_img_path)
        img = cv2.imread(curr_img_path)[:, :, ::-1]
        # print('loaded')
        img, border = rotate_or_shear_img(img, border)
        save_image(img, curr_img_path)
        # print(bg_path)
        save_labels(border[0], path_lbl, i, size[0], size[1])
        # print(str(i))
        """
        fig = plt.figure(frameon=False)
        fig.set_size_inches(1,1)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(img, aspect='normal')
        fig.savefig(curr_img_path, 24)
        """
        # img.save(curr_img_path)

        # img = mpimg.imread(curr_img_path)#skimage.io.imread(curr_img_path) / 255.0
        # plt.figure(figsize=(size[0]/255.0, size[1]/255.0))
        # add_noise(img, rd.randint(0, 5))
        # plt.savefig(curr_img_path)
        # img = cv2.imread(curr_img_path, 1)
        # img = adjust_gamma(img, gamma=rd.uniform(0.1, 2.5))
        # cv2.imwrite(curr_img_path, img)

    """
    print('\nLoading MNIST Data...')
    data = MNIST('./MNIST_Dataset_Loader/dataset/')
    
    print('\nLoading Training Data...')
    img_train, labels_train = data.load_training()
    train_img = np.array(img_train)
    train_labels = np.array(labels_train)
    x = train_img[0, :]
    y = np.reshape(x, (28, 28,))
    plt.imshow(y, cmap='gray')
    plt.show()
    """

    print('Finished')
