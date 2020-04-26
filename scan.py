from transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils

#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True, help="Path to the image to be scanned")
#args = vars(ap.parse_args())


def scan(file_name):
    image = cv2.imread(file_name)
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height=500)

    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grey = cv2.GaussianBlur(grey, (5, 5), 0)
    edged = cv2.Canny(grey, 75, 200)

    print('Step 1: Edge Detection')
    #cv2.imshow("Image", image)
    #cv2.imshow("Edged", edged)
    cv2.imwrite('edged.jpg', edged)
    #cv2.waitKey(0)
    cv2.destroyAllWindows()

    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)

        if len(approx) == 4:
            screenCnt = approx
            break
    if screenCnt is None:
        return 0
    print('Step 2: Find contours of paper')
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    #cv2.imshow("Outline", image)
    cv2.imwrite('countour.jpg', image)
    #cv2.waitKey(0)
    cv2.destroyAllWindows()

    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

    # warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # T = threshold_local(warped, 11, offset=10, method="gaussian")
    # warped = (warped > T).astype("uint8") * 255

    print('Step 3: Apply perspective transform')
    #cv2.imshow("Original", imutils.resize(orig, height=650))
    #cv2.imshow("Scanned", imutils.resize(warped, height=650))
    cv2.imwrite('capture.jpg', warped)
    #cv2.waitKey(0)
    return 1


if __name__ == "__main__":
    scan("IMG_8611.jpg")