import os
from shutil import copyfile
from other.heic_to_jpg_converter import convert


def go_through_dir(directory, count_eq, count_hw):
    count = 0
    for filename in os.listdir(directory):
        pre, ext = os.path.splitext(filename)
        if ext == '.jpg' or ext == '.HEIC' or ext == '.JPG':
            newname = str(count_eq) + '_' + str(count_hw) + '_' + str(count) + '.jpg'
            #os.rename(directory + filename, directory + newname)
            if ext == '.HEIC':
                convert(directory + filename)
            copyfile(directory + pre + '.jpg', '/Users/dariatunina/mach-lerinig/mLStuff/test_data/' + newname)
            count += 1


if __name__ == "__main__":
    path = '/photos/'
    count_hw = 0
    for i in range(20, 40):
        count_eq, count_hw = int(i / 2), count_hw % 2
        go_through_dir(path + str(i) + '/', count_eq, count_hw)
        count_hw += 1