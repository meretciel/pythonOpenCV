
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from os import path

def displayImage(img, title="untitled"):
    cv.imshow(title,img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.waitKey(1)

def getHeight(img):
    return img.shape[0]

def getWidth(img):
    return img.shape[1]

def hDisplayMultiImages(images, top=30, bottom=50, left=20, right=20, additionalPadding=40):
    outputs = []
    height = getHeight(images[0])
    vLine = np.full((height + top + bottom, 1), 0, dtype=images[0].dtype)

    # Add vertical lines
    for image in images:
        outputs.append(vLine)
        outputs.append(cv.copyMakeBorder(image, top, bottom, left, right, cv.BORDER_CONSTANT, value=255))

    outputs.append(vLine)
    
    newImage = cv.hconcat(outputs)
    newImage[0, :] = 0
    newImage[-1, :] = 0

    newImage = cv.copyMakeBorder(newImage, additionalPadding, \
                additionalPadding, additionalPadding, additionalPadding, cv.BORDER_CONSTANT, value=255)

    displayImage(newImage)
    
def plotHistogram(image, lower = 0, upper = 255):
    plt.hist(image.ravel(), bins=255, range=[lower, upper])
    