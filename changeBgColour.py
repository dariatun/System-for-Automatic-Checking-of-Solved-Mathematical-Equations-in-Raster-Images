import numpy as np
from PIL import Image

from utils import get_image_array, change_sizeimg, save_image, get_image_array1, get_bg_path, choose_blue_colour
from MNIST_Dataset_Loader.mnist_loader import MNIST


if __name__ == "__main__":
    bg_path = "help_stuff/crumpled.jpg"#get_bg_path()
    bgImg = Image.open(bg_path)
    bgImg_array = np.array(bgImg)
    data = MNIST('./MNIST_Dataset_Loader/dataset/')
    img_train, _ = data.load_testing()
    train_img_arr = np.array(img_train)
    img_array = np.array(get_image_array(train_img_arr))
    img_array = np.stack((img_array,)*3, axis=-1)
    img = change_sizeimg(Image.fromarray(img_array))
    offset = (1, 1)
    bgImg.paste(img, offset)
    save_image(bgImg, 'out/number1.jpg')
    blue_colour = choose_blue_colour()
    for i in range(0, img_array.shape[0]):
        for j in range(0, img_array.shape[1]):
            if 0 <= np.sum(img_array[i][j]) <= 100*3:
                img_array[i][j] = blue_colour
            elif 230*3 <= np.sum(img_array[i][j]) <= 255*3:
                img_array[i][j] = bgImg_array[i+offset[0]][j+offset[1]]
    bgImg = Image.open(bg_path)
    bgImg.paste(change_sizeimg(Image.fromarray(img_array)), offset)
    save_image(bgImg, 'out/number2.jpg')
    #save_image(change_sizeimg(Image.fromarray(img)), 'out/number2.jpg')
    print('Finished')
