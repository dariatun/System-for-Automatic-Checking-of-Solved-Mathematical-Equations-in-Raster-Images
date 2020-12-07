from application.constants import NUMBER_OF_IMAGES, NUMBER_IMAGES_PODTYPES, TXT, TEST_DATA_ANNOTATIONS_PATH

for i in range(0, NUMBER_OF_IMAGES):
    for j in range(0, NUMBER_IMAGES_PODTYPES):
        filename = str(i) + '_' + str(j) + TXT
        file = open(TEST_DATA_ANNOTATIONS_PATH + filename, "w+")
        file.close()
