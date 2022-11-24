import numpy as np
import cv2
from scipy.ndimage import rotate



# Skew Correction (projection profile)
def _find_score(arr, angle):
    data = rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score

def _find_angle(img, delta = 0.5,  limit = 10):
    angles = np.arange(-limit, limit+delta, delta)
    scores = []
    for angle in angles:
        hist, score = _find_score(img, angle)
        scores.append(score)
    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]
    print(f'Best angle: {best_angle}')
    return best_angle

def correct_skew(img):
    # correctskew
    best_angle =_find_angle(img)
    data = rotate(img, best_angle, reshape=False, order=0)
    return data


def ocr_img_preprocess(img):
    # grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #     #1. Thresholding
    #     img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #             cv2.THRESH_BINARY,35,35) #115,35

    # 2. Skew correct
    img = correct_skew(img)

    #     #3. Thinning and Skeletonization
    #     kernel = np.ones((3,3), np.uint8)
    #     img = cv2.erode(img, kernel, iterations=1)
    #     img = cv2.dilate(img, kernel, iterations=1)

    #     #4. De-noising
    #     # histogram equalization
    #     equ = cv2.equalizeHist(img)
    #     # Gaussian blur
    #     img = cv2.GaussianBlur(equ, (3, 3), 0) #Filter high frequency noise.

    #     img = cv2.fastNlMeansDenoising(img, None, 25, 7, 21)

    #     # # histogram equalization
    #     # equ = cv2.equalizeHist(img)
    #     # # Gaussian blur
    #     # img = cv2.GaussianBlur(equ, (5, 5), 1) #cv2.GaussianBlur(img, (7, 7), 0)

    #     kernel = np.ones((2,2), np.uint8)
    #     img = cv2.erode(img, kernel, iterations=1)
    #     img = cv2.dilate(img, kernel, iterations=1)

    #     img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #             cv2.THRESH_BINARY,145,10) #115,35

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img
