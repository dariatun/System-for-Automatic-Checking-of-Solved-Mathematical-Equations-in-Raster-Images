import numpy as np
import cv2
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import os
import loremipsum as li

from utils import *
import matplotlib
matplotlib.use('Agg')

sys.path.append('/home.stud/tunindar/bachelorWork/MNIST_Dataset_Loader')
from mnist_loader import MNIST

symbols = ['+', '-']
fonts = ["fonts/times-new-roman.ttf", "fonts/arial.ttf"]
MAX_NUMBER_OF_EQ_IN_COLUMN = 6
MIN_NUMBER_OF_EQ_IN_COLUMN = 2

def add_next_line_symbols(s, x, width, font):
    length = font.getsize(s)[0]
    len_s = len(s)
    ret_str = ''
    i = 0
    l = int((width - x * 2) / (length / len_s))
    while length >= l:
        add_str = s[i:i + l]
        ret_str += add_str + '\n'
        length -= font.getsize(add_str)[0]
        i += l
    ret_str += s[i:] + '\n'
    return ret_str


def generate_number_in_range(min_range, max_range):
    if min_range == max_range:
        return str(min_range)
    s = ''
    curr_num = 0
    length = rd.randint(1, 2)
    for j in range(0, length):
        if j == 0 and length != 1:
            num = rd.randint(1, 9)
            while num < min_range or num > max_range:
                num = rd.randint(1, 9)
            curr_num = num
        else:
            if curr_num * 10 > max_range:
                break
            num = rd.randint(0, 9)
            poss_num = curr_num * 10 + num
            while poss_num < min_range or poss_num > max_range:
                num = rd.randint(0, 9)
                poss_num = curr_num * 10 + num
            curr_num = poss_num
        s += str(num)
    return s


def generate_number():
    s = ''
    for j in range(0, rd.randint(1, 2)):
        if j == 0:
            num = rd.randint(1, 9)
        else:
            num = rd.randint(0, 9)
        s += str(num)
    return s


def get_equation(x, y, font, border, draw):
    s = generate_number()
    symbol = symbols[rd.randint(0, 1)]
    equation = s + ' ' + symbol + ' '
    s = generate_number_in_range(0, int(s) if symbol == '-' else 99 - int(s))
    equation += s + ' ='
    size = draw.textsize(equation, font)
    if border is None:
        border = np.array([[x, y, x + size[0], y + size[1], 0]])
    else:
        border = np.vstack([border, [x, y, x + size[0], y + size[1], 0]])
    # print(size[1], new_y)
    return equation, y + size[1], border


def generate_equation_column(x, y, font, border, height, draw, color, spacing):
    iterations = rd.randint(MIN_NUMBER_OF_EQ_IN_COLUMN, MAX_NUMBER_OF_EQ_IN_COLUMN)
    currIter = 0
    while currIter != iterations:
        # print(str(height) + ' ' + str(check))
        if height < y + draw.textsize('1', font)[1]:
            # print('leaving..')
            break
        s, new_y, border = get_equation(x, y, font, border, draw)
        draw.text(xy=(x, y), text=s, fill=(color, color, color), font=font)
        y = new_y + spacing
        currIter += 1
    return y, border


def add_equations(x, y, font, draw, color, width, height):
    num_of_columns = rd.randint(1, 2)
    border = None
    spacing = rd.randint(10, 30)
    new_y, border = generate_equation_column(x, y, font, border, height, draw, color, spacing)
    if num_of_columns == 1:
        return new_y, border.astype(np.float)
    new_x = int(width / 2 + x)
    new_y_1, border = generate_equation_column(new_x, y, font, border, height, draw, color, spacing)
    return max(new_y, new_y_1), border.astype(np.float)


def add_text(x, y, font, draw, color, width):
    if rd.randint(0, 1) == 0:
        s = add_next_line_symbols(li.get_sentences(200, True)[2], x, width, font)
        draw.text(xy=(x, y), text=s, fill=(color, color, color), font=font)
        y += draw.textsize(s, font)[1]
    return y


def generate_images(path, train_img):
    path_img = path  # + 'images/'
    path_lbl = path  # + 'labels/'
    for i in range(0, 5000):
        print(i)
        # initialise
        bg_path = get_bg_path()
        img = Image.open(bg_path)
        draw = ImageDraw.Draw(img)
        size = img.size
        width = size[0]
        height = size[1]
        x = rd.randint(0, 100)
        y = rd.randint(0, int(height / 2))
        font = ImageFont.truetype(fonts[rd.randint(0, 1)], rd.randint(20, 50))
        color = rd.randint(0, 50)

        # add start text
        y = add_text(x, y, font, draw, color, width)

        # add equations
        y, border = add_equations(x, rd.randint(y, y), font, draw, color, width, height)

        # add end text
        add_text(x, rd.randint(y, y + 300), font, draw, color, width)

        curr_img_path = path_img + str(i)
        curr_img_extension = bg_path[-4:]
        full_path = get_full_path(curr_img_path, curr_img_extension, 0)

        # save initial files
        img.save(full_path)
        save_labels(border, path_lbl, i, 0, size[0], size[1])

        # add handwritten
        length = len(border)
        add_digit_to = rd.randint(int(length / 2), length)
        count = 0
        equations_with_h_digits = []
        while count < add_digit_to:
            curr_indx = rd.randint(0, length - 1)
            while curr_indx in equations_with_h_digits:
                curr_indx = rd.randint(0, length - 1)
            equations_with_h_digits.append(curr_indx)
            h_x = 0
            for j in range(0, rd.randint(1,2)):
                h_digit_img = get_handwritten_digit(train_img)
                h_digit_img = change_sizeimg(h_digit_img)
                offset = (int(border[curr_indx][2] + 10 + h_x), int(border[curr_indx][1]))
                img.paste(h_digit_img, offset)
                h_x = h_digit_img.size[0]
            count += 1

        full_path = get_full_path(curr_img_path, curr_img_extension, 4)
        img.save(full_path)
        save_labels(border, path_lbl, i, 4, size[0], size[1])

        # change brightness
        img = cv2.imread(full_path, 1)
        img = adjust_gamma(img, gamma=rd.uniform(0.45, 2.5))
        full_path = get_full_path(curr_img_path, curr_img_extension, 1)
        save_image(img, full_path)
        save_labels(border, path_lbl, i, 1, size[0], size[1])

        # add noise
        img = cv2.imread(full_path)[:, :, ::-1]
        full_path = get_full_path(curr_img_path, curr_img_extension, 2)
        add_noise(img, full_path)
        save_labels(border, path_lbl, i, 2, size[0], size[1])

        # change rotation
        img = cv2.imread(full_path)[:, :, ::-1]
        img, border = rotate_or_shear_img(img, border)
        save_image(img, get_full_path(curr_img_path, curr_img_extension, 3))
        save_labels(border, path_lbl, i, 3, size[0], size[1])


if __name__ == "__main__":
    path = '/datagrid/personal/tunindar/numbers-eqs1/'

    data = MNIST('./MNIST_Dataset_Loader/dataset/')
    img_train, _ = data.load_testing()
    train_img = np.array(img_train)
    delete_old_files(path)

    # h_digit_img = get_handwritten_digit(train_img)
    # save_image(h_digit_img, path + 'digit.png')

    generate_images(path, train_img)

    print('Finished')
