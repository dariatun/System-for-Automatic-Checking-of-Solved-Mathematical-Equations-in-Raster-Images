import json
import numpy as np
from PIL import Image
from utils import cut_image, get_xy_wh, delete_old_files

INPUTS_FROM_STDIN = False
LINUX = True

if __name__ == "__main__":
    delete_old_files('/datagrid/personal/tunindar/tensorflow/')
    print('have cleared folder')
    if INPUTS_FROM_STDIN:
        path_to_image_folder = input('Enter path to the folder with images: ')
        path_to_json_file = input('Enter path to the json file: ')
    else:
        if LINUX:
            path_to_image_folder = '/datagrid/personal/tunindar/numbers-eqs/'
            path_to_json_file = '/home.stud/tunindar/new/darknet/result.json'
        else:
            path_to_image_folder = '/Users/dariatunina/mach-lerinig/test-data/'
            path_to_json_file = '/Users/dariatunina/mach-lerinig/darknet/result.json'
    with open(path_to_json_file) as json_file:
        data = json.load(json_file)
        count = 0
        for file in data:
            filename = file['filename'].split('/')[-1]
            try:
                img = np.array(Image.open(path_to_image_folder + filename))
            except Exception as e:
                continue
            for obj in file['objects']:
                if obj['class_id'] == 1:
                    continue
                xy, w, h = get_xy_wh(obj['relative_coordinates'], img.shape)
                image = cut_image(xy[0], xy[1], w, h, img)
                image = Image.fromarray(image)
                image.save('/datagrid/personal/tunindar/tensorflow/' + filename[:-4] + '_' + str(count) + '.jpg')
                print('Saving ' + str(count))
                count += 1
            
            if count > 5000:
                break
    print('Finished')
