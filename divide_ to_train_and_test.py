import os.path

INPUTS_FROM_STDIN = True


def distribute_dataset():
    """ Distributes dataset into training and validation data

    :return:
    """
    if INPUTS_FROM_STDIN:
        path = input('Enter path to dataset: ')
        path_to_files = input('Enter path to save train and test files at: ')
        distribution = input('Enter distribution: ')
    else:
        path = '/datagrid/personal/tunindar/numbers-eqs/'
        path_to_files = '/home.stud/tunindar/new/darknet/data/'
        distribution = 10
    trainFile = open(path_to_files + 'train.txt', 'w+')
    testFile = open(path_to_files + 'test.txt', 'w+')
    count = 1
    trainCount = 0
    testCount = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in [f for f in filenames if f.endswith(".jpg") or f.endswith(".png")]:
            if count % distribution == 0:
                testFile.write(path + os.path.join(dirpath[-1], filename))
                testFile.write('\n')
                testCount += 1
            else:
                trainFile.write(path + os.path.join(dirpath[-1], filename))
                trainFile.write('\n')
                trainCount += 1
            count += 1
    print('Train: ' + str(trainCount) + ', Test: ' + str(testCount) + '\n')


if __name__ == "__main__":
    distribute_dataset()
    print('Finished')
