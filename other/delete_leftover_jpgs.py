import os

directory = '/Users/dariatunina/mach-lerinig/mLStuff/'
for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        if '0' <= filename[0] <= '9':
            os.remove(directory + filename)
