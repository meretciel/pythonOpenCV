
import cv2 as cv
from os import path
import numpy as np
from imageprocessing.imgutils import hDisplayMultiImages, displayImage, getHeight, getWidth


if __name__ == '__main__':

    dataDir = r'/Users/Ruikun/workspace/PythonProjects/ComputerVision/Feature_Extraction_And_Image_Processing/data'
    dataFile = 'rectangle.png'
    lineFile = r'straight-line.png'
    circleFile = r'circle.png'

    # Hough Transform on lines
    lineImage = cv.imread(path.join(dataDir, lineFile), cv.IMREAD_GRAYSCALE)
    linesDetected = cv.HoughLines(lineImage, 2, np.pi / 180, 1000)
    lineDetectionImage = cv.cvtColor(lineImage, cv.COLOR_GRAY2BGR)

    for item in linesDetected:
        try:
            rho, theta = item.flatten()
            x1 = rho / np.cos(theta)
            y1 = 0
            x2 = 0
            y2 = rho / np.sin(theta)
            cv.line(lineDetectionImage, (x1, y1), (x2, y2), (0,0,255))
        except:
            print("Error: rho={}, theta={}.".format(rho, theta))

    hDisplayMultiImages([cv.cvtColor(lineImage, cv.COLOR_GRAY2BGR), lineDetectionImage])

    # Hough Transform on Circle
    circleImage = cv.imread(path.join(dataDir, circleFile), cv.IMREAD_GRAYSCALE)
    circlesDetected = cv.HoughCircles(circleImage, cv.HOUGH_GRADIENT, 1, 30, param1=60, param2=30)
    circleDetectionImage = cv.cvtColor(circleImage, cv.COLOR_GRAY2BGR)

    for x, y, r in circlesDetected[0]:
        print("x = {}, y = {}, r = {}".format(x,y,r))
        cv.circle(circleDetectionImage, (x,y), r, (0,0,255))

    hDisplayMultiImages([cv.cvtColor(circleImage, cv.COLOR_GRAY2BGR), circleDetectionImage])


    ###########################################################################
    # Genralized Hough Transform
    ###########################################################################

    image = cv.imread(path.join(dataDir, dataFile), cv.IMREAD_GRAYSCALE)

    displayImage(image)


    # houghTransformer = cv.GeneralizedHoughGuil()
    # houghTransformer = cv.createGeneralizedHoughGuil()

    # This is working but without scaling.
    # houghTransformer = cv.createGeneralizedHoughBallard()

    # The Guil implementation takes longer to execute.
    houghTransformer = cv.createGeneralizedHoughGuil()
    template = np.full((180,326), 255, dtype=np.uint8)
    template[0, :] = 0
    template[-1, :] = 0
    template[:, 0] = 0
    template[:, -1] = 0

    houghTransformer.setTemplate(template)
    positions, votes = houghTransformer.detect(template)

    # Get rid of the first dimension
    listOfPosition = positions[0]

    outputImage = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    for x, y, scale, orientation in listOfPosition:
        halfHeight = getHeight(template) / 2. * scale
        halfWidth = getWidth(template) / 2. * scale
        p1 = (int(x - halfWidth), int(y - halfHeight))
        p2 = (int(x + halfWidth), int(y + halfHeight))
        print("x = {}, y = {}, scale = {}, orientation = {}, p1 = {}, p2 = {}".format(x, y, scale, orientation, p1, p2))
        cv.rectangle(outputImage, p1, p2, (0,0,255))

    hDisplayMultiImages([cv.cvtColor(image, cv.COLOR_GRAY2BGR), outputImage])






