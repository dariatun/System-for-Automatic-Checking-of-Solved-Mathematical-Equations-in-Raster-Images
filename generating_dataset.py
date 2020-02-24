import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFont, ImageDraw
import os
import loremipsum as li

from utils import *
import matplotlib
matplotlib.use('Agg')

LINUX = True
INPUTS_FROM_STDIN = False

<<<<<<< HEAD
#from MNIST_Dataset_Loader.mnist_loader import MNIST
sys.path.append('/home.stud/tunindar/bachelorWork/MNIST_Dataset_Loader')
=======
# path to the directory with the MNIST_Dataset_Loader
MNIST_PATH = '/Users/dariatunina/mach-lerinig/Handwritten-Digit-Recognition-using-Deep-Learning/CNN_Keras/MNIST_Dataset_Loader/'
sys.path.append(MNIST_PATH)

#sys.path.append('/home.stud/tunindar/bachelorWork/MNIST_Dataset_Loader')
>>>>>>> 9950f31cf58f224deb2dfa8b21570e7dffaf6ade
from mnist_loader import MNIST

symbols = ['+', '-']
fonts = ["fonts/times-new-roman.ttf", "fonts/arial.ttf"]
bg_imgs = []
MAX_NUMBER_OF_EQ_IN_COLUMN = 6
MIN_NUMBER_OF_EQ_IN_COLUMN = 3


def get_bg_path():
    """ Randomly chooses background image
    :return: path to background image
    """
    length = len(bg_imgs)
    return bg_imgs[rd.randint(0, length - 1)]


def get_bg_imgs(path):
    """ Creates array of possible background images

    :param path: path to the directory with background images
    :return:
    """
    for file_path in os.listdir(path):
        if file_path.endswith(".jpg") or file_path.endswith(".png"):
            bg_imgs.append(path + '/' + file_path)


def add_next_line_symbols(s, x, width, font):
    """ Adds next line symbols to a string,
     so the width of the text is smaller than width of the image

    :param s: text
    :param x: x coordinate of the equation starting position
    :param width: width of the image
    :param font: font and size of the text
    :return: text with added next line symbols
    """
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
    """ Generates number in a given range

    :param min_range:
    :param max_range:
    :return: generated number
    """
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
    """ Generates number from 0 to 99
    :return: generated number
    """
    s = ''
    for j in range(0, rd.randint(1, 2)):
        if j == 0:
            num = rd.randint(1, 9)
        else:
            num = rd.randint(0, 9)
        s += str(num)
    return s


def get_equation(x, y, font, border, draw):
    """ Generates equation

    :param x: x coordinate of the equation starting position
    :param y: y coordinate of the equation starting position
    :param font: font and size of the equation
    :param border: numpy array of borders of the equations
    :param draw: ImageDraw module, that allows putting equation on an image
    :return: equation string, last y coordinate of the equation, borders of the equations
    """
    s = generate_number()
    symbol = symbols[rd.randint(0, 1)]
    equation = s + ' ' + symbol + ' '
    s = generate_number_in_range(0, int(s) if symbol == '-' else 99 - int(s))
    equation += s + ' ='
    size = draw.textsize(equation, font)
    if border is None:
        border = np.array([[x - 3, y, x + size[0] + 6, y + size[1]+5, 0]])
    else:
        border = np.vstack([border, [x - 3, y, x + size[0] + 6, y + size[1]+5, 0]])
    return equation, y + size[1], border


def generate_equation_column(x, y, font, border, height, draw, color, spacing, iterations):
    """ Generate column of equations

    :param x: x coordinate of the equation starting position
    :param y: y coordinate of the equation starting position
    :param font: font and size of the equation
    :param border: numpy array of borders of the equations
    :param height: height of the image
    :param draw: ImageDraw module, that allows putting equation on an image
    :param color: colour of the equation
    :param spacing: distance between equations in a column
    :param iterations: number of equations in a column
    :return: last y coordinate of the equation, borders of the equations
    """
    currIter = 0
    while currIter != iterations:
        if height < y + draw.textsize('1', font)[1]:
            break
        s, new_y, border = get_equation(x, y, font, border, draw)
        draw.text(xy=(x, y), text=s, fill=(color, color, color), font=font)
        y = new_y + spacing
        currIter += 1
    return y, border


def add_equations(x, y, font, draw, color, width, height):
    """ Adds equations to the image on the specific coordinates

    :param x: x coordinate of the equation starting position
    :param y: y coordinate of the equation starting position
    :param font: font and size of the equation
    :param draw: ImageDraw module, that allows putting equation on an image
    :param color: colour of the equation
    :param width: width of the image
    :param height: height of the image
    :return: last y coordinate of the equation, borders of the equations
    """
    num_of_columns = rd.randint(1, 2)
    border = None
    spacing = rd.randint(10, 30)
    iterations = rd.randint(MIN_NUMBER_OF_EQ_IN_COLUMN, MAX_NUMBER_OF_EQ_IN_COLUMN)
    new_y, border = generate_equation_column(x, y, font, border, height, draw, color, spacing, iterations)
    if num_of_columns == 1:
        return new_y, border.astype(np.float)

    new_x = int(width / 2 + x - 10)
    new_y_1, border = generate_equation_column(new_x, y, font, border, height, draw, color, spacing, iterations)
    return max(new_y, new_y_1), border.astype(np.float)


def add_text(x, y, font, draw, color, width):
    """ Adds string of text to the image on the specific coordinates

    :param x: x coordinate of the text starting position
    :param y: y coordinate of the text starting position
    :param font: font and size of the text
    :param draw: ImageDraw module, that allows putting text on an image
    :param color: colour of the text
    :param width: width of the image
    :return: y coordinate of the text last position
    """
    if rd.randint(0, 1) == 0:
        s = add_next_line_symbols(li.get_sentences(200, True)[2], x, width, font)
        draw.text(xy=(x, y), text=s, fill=(color, color, color), font=font)
        y += draw.textsize(s, font)[1]
    return y


def generate_images(path, digits_array):
    """ Generates dataset to given directory with given dataset of handwritten digits

    :param path: where to save created dataset
    :param digits_array: dataset of handwritten digits
    :return:
    """
    num_of_iters = input('Enter number of images to generate:')
    for i in range(0, int(num_of_iters)):
        print(i)
        # initialise
        bg_path = get_bg_path()
        img = Image.open(bg_path)
        draw = ImageDraw.Draw(img)
        size = img.size
        width = size[0]
        height = size[1]
        x = rd.randint(20, 40)
        y = rd.randint(40, 100)
        font = ImageFont.truetype(fonts[rd.randint(0, 1)], rd.randint(35, 40))
        color = rd.randint(0, 50)

        # add start text
        y = add_text(x, y, font, draw, color, width)

        # add equations
        y, border = add_equations(x, rd.randint(y, y), font, draw, color, width, height)

        # add end text
        add_text(x, rd.randint(y, y + 300), font, draw, color, width)

        curr_img_path = path + str(i)
        curr_img_extension = bg_path[-4:]
        full_path = get_full_path(curr_img_path, curr_img_extension, 0)

        # save initial files
        img.save(full_path)
        save_labels(border, path, i, 0, size[0], size[1])

        # add handwritten digits
        length = len(border)
        add_digit_to = rd.randint(int(length / 2), length)
        count = 0
        equations_with_h_digits = []
        while count < add_digit_to:
            curr_indx = rd.randint(0, length - 1)
            while curr_indx in equations_with_h_digits:
                curr_indx = rd.randint(0, length - 1)
            equations_with_h_digits.append(curr_indx)
            st_x = border[curr_indx][2] + 10
            st_y = border[curr_indx][1]
            all_w = 0
            all_h = 0
            for j in range(0, rd.randint(1, 2)):
                offset = (int(st_x), int(st_y))
                if offset[0]+font.getsize('9')[0]+5 >= width or offset[1]+font.getsize('9')[1]+5 >= height:
                    break
                h_digit_img = get_digit(digits_array, offset, img, font.getsize('9')[1])
                img.paste(h_digit_img, offset)
                h_x = h_digit_img.size[0]
                all_w += h_x
                all_h = max(all_h, h_digit_img.size[0])
                border = np.vstack([border, [st_x, st_y, st_x + h_x, st_y + h_x, 1]])
                st_x += h_x
            count += 1

        full_path = get_full_path(curr_img_path, curr_img_extension, 4)
        img.save(full_path)
        save_labels(border, path, i, 4, size[0], size[1])

        # add noise
        img = cv2.imread(full_path)[:, :, ::-1]
        full_path = get_full_path(curr_img_path, curr_img_extension, 2)
        add_noise(img, full_path)
        save_labels(border, path, i, 2, size[0], size[1])

        # change rotation
        img = cv2.imread(full_path)[:, :, ::-1]
        img, border = rotate_and_shear_img(img, border)
        save_image(img, get_full_path(curr_img_path, curr_img_extension, 3))
        save_labels(border, path, i, 3, size[0], size[1])


if __name__ == "__main__":
    get_bg_imgs('bg_images')
    if INPUTS_FROM_STDIN:
        path = input('Enter path to save dataset at: ')
    else:
        if LINUX:
            path = '/datagrid/personal/tunindar/numbers-eqs/'
        else:
            path = '/Users/dariatunina/mach-lerinig/numbers-eqs/'
    data = MNIST(MNIST_PATH + 'dataset/')
    img_train, _ = data.load_testing()
    train_img = np.array(img_train)
    delete_old_files(path)
    generate_images(path, train_img)

    print('Finished')
