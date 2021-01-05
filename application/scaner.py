import numpy as np
import cv2

# https://bretahajek.com/2017/01/scanning-documents-photos-opencv/

# Load image and convert it from BGR to RGB

DEBUG = False


def resize(img, height=800):
    """ Resize image to given height """
    rat = height / img.shape[0]
    return cv2.resize(img, (int(rat * img.shape[1]), height))


def get_edges(image):
    # Resize and convert to grayscale
    img = cv2.cvtColor(resize(image), cv2.COLOR_BGR2GRAY)
    # Bilateral filter preserv edges
    img = cv2.bilateralFilter(img, 9, 75, 75)
    if DEBUG:
        img_to_show = cv2.resize(img, (800, 600))
        cv2.imshow('BILATERAL_FILTER', img_to_show)
        cv2.waitKey(0)
    # Create black and white image based on adaptive threshold
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 71, -3)
    # Median filter clears small details
    img = cv2.medianBlur(img, 11)
    if DEBUG:
        img_to_show = cv2.resize(img, (800, 600))
        cv2.imshow('MEDIAN_BLUR', img_to_show)
        cv2.waitKey(0)
    # Add black border in case that page is touching an image border
    img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    edges = cv2.Canny(img, 200, 250)
    if DEBUG:
        img_to_show = cv2.resize(edges, (800, 600))
        cv2.imshow('EDGED', img_to_show)
        cv2.waitKey(0)
    return edges#cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 11)))


def fourCornersSort(pts):
    """ Sort corners: top-left, bot-left, bot-right, top-right """
    # Difference and sum of x and y value
    # Inspired by http://www.pyimagesearch.com
    diff = np.diff(pts, axis=1)
    summ = pts.sum(axis=1)
    # Top-left point has smallest sum...
    # np.argmin() returns INDEX of min
    return np.array([pts[np.argmin(summ)],
                     pts[np.argmax(diff)],
                     pts[np.argmax(summ)],
                     pts[np.argmin(diff)]])


def contourOffset(cnt, offset):
    """ Offset contour, by 5px border """
    # Matrix addition
    cnt += offset
    # if value < 0 => replace it by 0
    cnt[cnt < 0] = 0
    return cnt


def find_contour(contours, width, height):
    MAX_COUNTOUR_AREA = (width - 10) * (height - 10)
    maxAreaFound = MAX_COUNTOUR_AREA / 16

    # Saving page contour
    pageContour = np.array([[[0, 0]], [[0, height - 5]], [[width - 5, height - 5]], [[width - 5, 0]]])
    # Go through all contours
    for cnt in contours:
        # Simplify contour
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)
        # Page has 4 corners and it is convex
        # Page area must be bigger than maxAreaFound
        if len(approx) == 4 and cv2.isContourConvex(approx) and \
                maxAreaFound < cv2.contourArea(approx) < MAX_COUNTOUR_AREA:
            maxAreaFound = cv2.contourArea(approx)
            pageContour = approx
    return pageContour


def crop_image(img, up, down, left, right):
    img = img[int(up*img.shape[0]):int(img.shape[0]-down*img.shape[0]), int(left*img.shape[1]):int(img.shape[1]-right*img.shape[1]), :]
    return img


def scan(to_crop, image=None, file_name=''):
    if image is None:
        image = cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_BGR2RGB)
    if to_crop:
        image = crop_image(image, 0.2, 0.2, 0.1, 0.1)
    orig_image = image
    #return orig_image
    #cv2.imwrite('croped.jpg', image)
    edges = get_edges(image)
    # Finding contour of biggest rectangle
    # Otherwise return corners of original image
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    height = edges.shape[0]
    width = edges.shape[1]

    pageContour = find_contour(contours, width, height)
    # Sort and offset corners
    pageContour = fourCornersSort(pageContour[:, 0])

    pageContour = contourOffset(pageContour, (-5, -5))

    # cv2.imwrite('countour.jpg', cv2.drawContours(resize(image), [pageContour], -1, (0, 255, 0), 3))
    if DEBUG:
        img_to_show = cv2.resize(image, (800, 600))
        cv2.imshow("Outline", img_to_show)
        cv2.waitKey(0)
    #cv2.imwrite('out_images/countour.jpg', image)
    # Recalculate to original scale - start Points
    sPoints = pageContour.dot(image.shape[0] / 800)

    # Using Euclidean distance
    # Calculate maximum height (maximal length of vertical edges) and width
    height = max(np.linalg.norm(sPoints[0] - sPoints[1]),
                 np.linalg.norm(sPoints[2] - sPoints[3]))
    width = max(np.linalg.norm(sPoints[1] - sPoints[2]),
                np.linalg.norm(sPoints[3] - sPoints[0]))
    # Create target points
    tPoints = np.array([[0, 0],
                        [0, height],
                        [width, height],
                        [width, 0]], np.float32)
    # getPerspectiveTransform() needs float32
    if sPoints.dtype != np.float32:
        sPoints = sPoints.astype(np.float32)
    # Wraping perspective
    M = cv2.getPerspectiveTransform(sPoints, tPoints)
    newImage = cv2.warpPerspective(orig_image, M, (int(width), int(height)))

    newImage = cv2.cvtColor(newImage, cv2.COLOR_BGR2RGB)

    return newImage


if __name__ == "__main__":
    scan("out_images/capture.jpg", "out_images/res.jpg")
    print("Finished")
