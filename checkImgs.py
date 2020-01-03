import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import os.path
import numpy as np

path_to_save = '/Users/dariatunina/mach-lerinig/docs/'
path = '/Users/dariatunina/mach-lerinig/numbers-eqs/'
path_img = path #+ 'images/'
path_lbl = path #+ 'labels/'


def check_img(img_name):
    curr_path = path + img_name
    img = Image.open(curr_path)  # os.path.join(dirpath, filename))
    # lbl_file = open(os.path.join(dirpath, filename[:-3]) + 'txt', 'r')
    # with open(os.path.join(dirpath, filename[:-3]) + 'txt', 'r') as lbl_file:
    with open(curr_path[:-3] + 'txt', 'r') as lbl_file:
        #line1 = lbl_file.readline()
        #
        for line in lbl_file:

            line = lbl_file.readline()
            #lbl_file.close()
            arr = line.split(' ')
            fig1, ax1 = plt.subplots(1)
            ax1.imshow(img)
            # print(arr)
            # print((float(arr[1])*img.size[1], float(arr[2])*img.size[0]),float(arr[3])*img.size[0],float(arr[4])*img.size[1])
            w = float(arr[3]) * img.size[0]
            h = float(arr[4][0:-2]) * img.size[1]
            x = float(arr[1]) * img.size[0] - w/2
            y = float(arr[2]) * img.size[1] - h/2
            rect1 = Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='None')
            ax1.add_patch(rect1)
            #line2 = lbl_file.readline()
        # line = lbl_file.readline()
        lbl_file.close()
        """arr = line2.split(' ')
        fig2, ax2 = plt.subplots(1)
        ax2.imshow(img)
        # print(arr)
        # print((float(arr[1])*img.size[1], float(arr[2])*img.size[0]),float(arr[3])*img.size[0],float(arr[4])*img.size[1])
        rect2 = Rectangle((float(arr[1]) * img.size[0], float(arr[2]) * img.size[1]), float(arr[3]) * img.size[0],
                         float(arr[4][0:-2]) * img.size[1], linewidth=1, edgecolor='r', facecolor='None')
        ax2.add_patch(rect2)"""
    # plt.show()
    # Image.fromarray(initialMean2, mode='L').save(\"initial2_mean.png\")
    plt.savefig(path_to_save + img_name)
    print('saved in a ' + path_to_save + img_name)
    # img.save('out/' + filename)
    plt.close()


if __name__ == "__main__":
    #for dirpath, dirnames, filenames in os.walk(path):
        #for filename in [f for f in filenames if f.endswith(".jpg")]:
            # im = np.array(Image.open('stinkbug.png'), dtype=np.uint8)
    check_img('2_1.jpg')
    print('Finished')
