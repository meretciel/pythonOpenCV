import scipy.signal
import numpy as np
import cv2 as cv
from imageprocessing.imgutils import getImageDepth


def shiftLeftAndFill(image, fillvalue=0):
    result = np.roll(image, shift=-1, axis=1)
    result[:, -1] = fillvalue
    return result

def shiftRightAndFill(image, fillvalue=0):
    result = np.roll(image, shift=1, axis=1)
    result[:, 0] = fillvalue
    return result


def shiftUpAndFill(image, fillvalue=0):
    result = np.roll(image, shift=-1, axis=0)
    result[-1, :] = fillvalue
    return result

def shiftDownAndFill(image, fillvalue=0):
    result = np.roll(image, shift=1, axis=0)
    result[0, :] = fillvalue
    return result

def zeroCrossing(image):
    kernel = np.full((2,2), 1)
    result = scipy.signal.convolve2d(image, kernel, mode='same')

    left = shiftLeftAndFill(result)
    up = shiftUpAndFill(result)
    diag = shiftLeftAndFill(shiftUpAndFill(result))

    listOfArrs = [result, left, up, diag]

    maxValue = np.maximum.reduce(listOfArrs)
    minValue = np.minimum.reduce(listOfArrs)
    isZeroCrossed = (maxValue > 0) & (minValue < 0)

    isZeroCrossed[0,:] = False
    isZeroCrossed[-1,:] = False
    isZeroCrossed[:, 0] = False
    isZeroCrossed[:, -1] = False
    return isZeroCrossed


def marr_hildreth(image, ksize=(5,5), sigmaX=0.1, sigmaY=0.1, loGSize=3, outputValue=255):
    blurred = cv.GaussianBlur(image, ksize=ksize, sigmaX=sigmaX, sigmaY=sigmaY)
    result = cv.Laplacian(blurred, cv.CV_64FC1, ksize=loGSize)
    isZeroCrossed = zeroCrossing(result)
    return (isZeroCrossed * outputValue).astype(image.dtype)

