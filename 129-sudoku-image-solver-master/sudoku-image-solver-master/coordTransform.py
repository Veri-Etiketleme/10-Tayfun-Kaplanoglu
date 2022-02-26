import cv2
import argparse

import matplotlib.pyplot as plt
import numpy as np

from getPoints import getSquare

def coordTransform(src, srcPts, sz):
    # Transform image, maping 4 srouce points to square of size sz
    fromPts = np.float32(srcPts)
    toPts = np.float32(((0,0), (sz, 0), (0,sz), (sz, sz)))
    M = cv2.getPerspectiveTransform(fromPts, toPts)
    dst = cv2.warpPerspective(src, M, (sz, sz))
    
    # Blur and threshold image 
    grayed = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    blured = cv2.GaussianBlur(grayed, (5,5), 0)
    _, threshed = cv2.threshold(blured, 0, 255, cv2.THRESH_TOZERO)
    threshed = cv2.adaptiveThreshold(threshed,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    return dst, threshed

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", 
            required = True, 
            help = "Path to image")
    args = vars(ap.parse_args())

    sz, offset = 300, 5
    src = cv2.imread(args["image"]) 
    srcPts, src = getSquare(src, offset)
    dst = coordTransform(src, srcPts, sz + 2*offset)

    plt.subplot(121), plt.imshow(src), plt.title("Input")
    plt.subplot(122), plt.imshow(dst), plt.title("Output")
    plt.show()
