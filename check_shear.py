"""from data_augemation import change_shear
import cv2
from utils import save_image

img = cv2.imread("capture.jpg")[:, :, ::-1]
img, _ = change_shear(img)
save_image(img, 'res.jpg')
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('capture.jpg')
rows,cols,ch = img.shape

pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

M = cv2.getPerspectiveTransform(pts1,pts2)

dst = cv2.warpPerspective(img,M,(300,300))

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()
