import os
import cv2

path = "/Users/dariatunina/Downloads/pytorch_custom_yolo_training-master/data/numbers/images"
count = 0
train_file = open(path[:-6] + "/" + "train.txt", "w", encoding='utf-8')
val_file = open(path[:-6] + "/" + "val.txt", "w", encoding='utf-8')

for img_name in os.listdir(path):
    if img_name.endswith(".jpg"):
        img_path = path + "/" + img_name
        img = cv2.imread(img_path)
        if count != 7:
            file = train_file
        else:
            file = val_file
        file.write(img_path)
        file.write('\n')
        count += 1

print("Finished")