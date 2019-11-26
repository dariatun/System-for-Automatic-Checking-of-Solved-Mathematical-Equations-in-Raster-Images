import os
import os.path

path = '/Users/dariatunina/mach-lerinig/numbers-eqs/'
save_path = 'data/numbers-eqs'
trainFile = open('/Users/dariatunina/mach-lerinig/train-eq.txt', 'w+')
testFile = open('/Users/dariatunina/mach-lerinig/test-eq.txt', 'w+')
count = 1
trainCount = 0
testCount = 0
for dirpath, dirnames, filenames in os.walk(path):
    for filename in [f for f in filenames if f.endswith(".jpg") or f.endswith(".png")]:
        if count % 10 == 0:
            testFile.write(save_path + os.path.join(dirpath[-1], filename))
            testFile.write('\n')
            testCount += 1
        else:
            trainFile.write(save_path + os.path.join(dirpath[-1], filename))
            trainFile.write('\n')
            trainCount += 1
        count += 1
print('Train: ' + str(trainCount) + ', Test: ' + str(testCount) + '\n')
print('Finished')
