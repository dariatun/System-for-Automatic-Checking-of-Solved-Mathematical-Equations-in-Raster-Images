import cv2
import numpy as np

QR_orig = cv2.imread('capture.jpg', 0)
QR = cv2.imread('capture.jpg', 0)  # read the QR code binary image as grayscale image to make sure only one layer
mask = np.zeros(QR.shape, np.uint8)  # mask image the final image without small pieces

# using findContours func to find the none-zero pieces
contours, hierarchy = cv2.findContours(QR, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# draw the white paper and eliminate the small pieces (less than 1000000 px). This px count is the same as the QR code dectection
for cnt in contours:
    if cv2.contourArea(cnt) > 1000000:
        cv2.drawContours(mask, [cnt], 0, 255,
                         -1)  # the [] around cnt and 3rd argument 0 mean only the particular contour is drawn

        # Build a ROI to crop the QR
        x, y, w, h = cv2.boundingRect(cnt)
        roi = mask[y:y + h, x:x + w]
        # crop the original QR based on the ROI
        QR_crop = QR_orig[y:y + h, x:x + w]
        # use cropped mask image (roi) to get rid of all small pieces
        QR_final = QR_crop * (roi / 255)

cv2.imwrite('capture1.jpg', QR_final)
