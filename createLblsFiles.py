import os
import cv2

path = "/Users/dariatunina/mach-lerinig/basic_training_dataset/images"
for img_name in os.listdir(path):
    if img_name.endswith(".jpg"):
        img = cv2.imread(path + "/" + img_name)
        file = open(path[:-6] + "labels" + "/" + img_name[:-3] + "txt", "w", encoding='utf-8')
        file.write("1 0 0 ")
        file.write(str(img.shape[0]) + " ")
        file.write(str(img.shape[1]))
print("Finished")



