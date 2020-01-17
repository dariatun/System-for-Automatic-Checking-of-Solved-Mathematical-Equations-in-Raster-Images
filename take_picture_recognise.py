from subprocess import call
import os

if __name__ == "__main__":
    # take a picture
    #os.system("imagesnap -w 2 img.png")
    call("python check_prediction_correction.py", shell=True)

    print('Finished')