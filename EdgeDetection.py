
import cv2 as cv
from os import path

from imageprocessing.imgutils import hDisplayMultiImages
from imageprocessing.MarrHildreth import marr_hildreth


if __name__ == '__main__':

    dataDir = r'/Users/Ruikun/workspace/PythonProjects/ComputerVision/Feature_Extraction_And_Image_Processing/data'

    imgFile = path.join(dataDir, 'ch_4_girl.png')
    image = cv.imread(imgFile, cv.IMREAD_GRAYSCALE)

    canny = cv.Canny(image, 20, 60)
    gaussianBlur = cv.GaussianBlur(image, ksize=(5, 5), sigmaX=0.1, sigmaY=0.1)
    sobel = cv.Sobel(gaussianBlur, cv.CV_8UC1, 1, 1, ksize=5)
    marrhildreth = marr_hildreth(image, ksize=(5, 5), sigmaX=5, sigmaY=5, loGSize=11)
    newImage = hDisplayMultiImages([image, canny, sobel, marrhildreth])

    cv.imwrite("/Users/Ruikun/workspace/tmp/fig-1-20200110.png", newImage)




