#!/bin/env python

from math import sqrt, atan, degrees, tan, radians, fabs
import cv2
import numpy as np
import datetime

'''
Copyright Team 6036 2019 Vision Subteam
File Written by Mihir Borkar, Slack Username: mihirb
'''

'''
camera/target constants:
'''

IMAGE_WIDTH = 640  # width of camera image in px (needs to be measured beforehand)
IMAGE_HEIGHT = 480  # height of camera image in px (needs to be measured beforehand)
HEIGHT_TARGET = 31.5  # height of center of target area in inches (needs to be measured beforehand)
HEIGHT_CAM = 32.45  # height of camera in inches (needs to be measured beforehand)
HEIGHT_TARGET_TO_CAM = fabs(HEIGHT_TARGET - HEIGHT_CAM)  # height between target and camera on robot in ft
CENTER_PIXEL_X = IMAGE_WIDTH / 2 - 0.5  # x-coordinate of center pixel of camera image
CENTER_PIXEL_Y = IMAGE_HEIGHT / 2 - 0.5  # y-coordinate of center pixel of camera image
DFOV = radians(68.5)  # diagonal field of view in radians
HORIZ_FOV = atan(tan(DFOV / 2) * (
            IMAGE_WIDTH / sqrt((IMAGE_HEIGHT ** 2) + (IMAGE_WIDTH ** 2)))) * 2  # horizontal field of view in radians
VERT_FOV = atan(tan(DFOV / 2) * (
            IMAGE_HEIGHT / sqrt((IMAGE_HEIGHT ** 2) + (IMAGE_WIDTH ** 2)))) * 2  # vertical field of view in radians
HORIZ_FOCAL_LENGTH = IMAGE_WIDTH / (2 * tan(
    HORIZ_FOV / 2))  # horizontal focal length of camera in px, needs to be measured using camera calibration program or looked up
VERT_FOCAL_LENGTH = IMAGE_HEIGHT / (2 * tan(
    VERT_FOV / 2))  # vertical focal length of camera in px, needs to be measured using camera calibration program or looked up

'''
grayscale tuning constants:
grayscale image: black and white, image pixels have values from 0-255, 0 = darkest black, 255 = brightest white
'''

LOWER_GRAY = 160  # lower cutoff for thresholding grayscale image
UPPER_GRAY = 205  # upper cutoff for thresholding grayscale image

'''
target detection constants:
solidity is the ratio of the detected contour's area to the detected rectangle around the contour's area
angle is how much the detected rectangle is tilted in the image
'''

ANGLE_VALUE = -15  # detected rectangle is considered a valid target if |detected_tilt_angle - ANGLE_VALUE| < ANGLE_TOLERANCE
ANGLE_TOLERANCE = 10
SOLIDITY_VALUE = 1  # detected rectangle is considered a valid target if |detected_solidity - SOLIDITY_VALUE| < SOLIDITY_TOLERANCE
SOLIDITY_TOLERANCE = 0.2


def within(x, a, c):
    # tells if number x is within c units of number a

    if (fabs(x - a) < c):

        return True


    else:

        return False


def getMinYPoint(ar):
    # returns point with minimum y coordinate in an array of points that are of the form [x,y]

    minY = ar[0][1]
    pos = 0

    for i in range(0, len(ar)):

        if (ar[i][1] < minY):
            minY = ar[i][1]
            pos = i

    return ar[pos]


def isValidTarget(contour_area, rect_width, rect_height, rect_angle):
    # checks if detected contour/rectangle is a valid piece of vision tape
    # may need to change algorithm depending on what your current build season's game is

    rect_area = rect_width * rect_height  # area of rectangle around contour
    rect_solidity = 0  # ratio of contour's area to area of rectangle around contour

    if (rect_area != 0):
        rect_solidity = contour_area / rect_area

    rect_aspect = 0  # ratio of height to width of rectangle around contour

    if (rect_width != 0):
        rect_aspect = rect_height / rect_width

    if (within(rect_solidity, SOLIDITY_VALUE, SOLIDITY_TOLERANCE) and (
            within(rect_angle, ANGLE_VALUE, ANGLE_TOLERANCE) or within(rect_angle, -90 - ANGLE_VALUE,
                                                                       ANGLE_TOLERANCE))):

        return True


    else:

        return False


def getTargetPixelCoords(img):
    # returns the pixel coordinates of the target site in an image, as well as number of vision tapes detected
    # may need to change algorithm depending on what your current build season's game is (may or may not include numberOfVisionTapesDetected)
    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(CAMERA_MATRIX, DISTORTION_CONSTANTS, (IMAGE_WIDTH, IMAGE_HEIGHT), 1, (IMAGE_WIDTH, IMAGE_HEIGHT))
    # img = cv2.undistort(image, CAMERA_MATRIX, DISTORTION_CONSTANTS, None, newcameramtx)

    g = img.copy()  # separate green image channel
    g[:, :, 0] = 0
    g[:, :, 2] = 0

    r = img.copy()  # separate red image channel
    r[:, :, 0] = 0
    r[:, :, 1] = 0

    img = g - r  # create new image subtracting red channel from green channel

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    mask = cv2.inRange(gray, LOWER_GRAY, UPPER_GRAY)  # create grayscale mask
    res = cv2.bitwise_and(gray, gray, mask=mask)  # apply grayscale mask to image
    _, contours, __ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # find contours in the image


    targetCoords = []  # array to hold target coordinates
    points = []  # list of highest corners of rectangles
    numberOfVisionTapesDetected = 0

    for contour in contours:

        contour_area = cv2.contourArea(contour)
        rect = cv2.minAreaRect(contour)  # minimum area rectangle around contour
        rect_width = rect[1][0]  # height of rectangle around contour is tilted
        rect_height = rect[1][1]  # height of rectangle around contour is tilted
        rect_angle = rect[2]  # angle at which rectangle around contour is tilted

        if (isValidTarget(contour_area, rect_width, rect_height, rect_angle)):
            # if condition to check if detected rectangle is a piece of vision tape

            numberOfVisionTapesDetected = numberOfVisionTapesDetected + 1  # add one to number of vision tapes detected
            box = cv2.boxPoints(rect)  # find corners of vision tape
            box = np.int0(box)  # convert to int0 array
            points.append(getMinYPoint(box))  # add the highest corner of the detected vision tape to the list of points
            cv2.drawContours(res, [box], 0, (255), 4)  # draw box around detected vision tape in output image

    if (len(points) > 0):

        targetCoords = list(sum(points) / len(points))  # centroid of the highest corners of the detected vision tapes
        targetCoords.append(numberOfVisionTapesDetected)  # number of vision tapes detected
        cv2.circle(res, (int(targetCoords[0]), int(targetCoords[1])), 4, (255), -1)  # draw dot on target area in output image


    else:

        targetCoords.append(numberOfVisionTapesDetected)  # add number of vision tapes detected to result

    cv2.imwrite('/Users/yiyi/Desktop/cv/contours/' + datetime.datetime.now().strftime(
        "%Y_%m_%d_%H_%M_%S") + '.jpg',
                res)  # write output image with white dot on target area and white boxes around detected vision tapes
    return targetCoords  # result is the centroid of the highest corners of the detected vision tapes


def getHorizAngleToTarget(targetX):
    # finds horizontal angle to target in deg

    return degrees(atan(((targetX - CENTER_PIXEL_X) / HORIZ_FOCAL_LENGTH)))


# return degrees((targetX-CENTER_PIXEL_X)*HORIZ_RADIANS_PER_PX)


def getVertAngleToTarget(targetY):
    # finds vertical angle to target in deg

    return degrees(atan(((targetY - CENTER_PIXEL_Y) / VERT_FOCAL_LENGTH)))


# return fabs(degrees((targetY-CENTER_PIXEL_Y)*VERT_RADIANS_PER_PX))


def getAbsDistanceToTarget(targetY):
    # finds absolute distance to target in inches

    return HEIGHT_TARGET_TO_CAM / tan(radians(fabs(getVertAngleToTarget(targetY))))


def getTargetPosition(img):
    # may need to change algorithm depending on what your current build season's game is

    '''
    returns array of headings to target
    array[0]: horizontal angle to target in degrees
    array[1]: vertical angle to target in degrees
    array[2]: absolute distance to target in inches
    array[3]: number of targets detected
    if no targets are detected array will have one element of 0
    and will look like this: [0]
    '''
    targetPixelCoords = getTargetPixelCoords(img)
    print('target coords: ' + str(targetPixelCoords))

    if (len(targetPixelCoords) > 1):

        targetX = targetPixelCoords[0]
        targetY = targetPixelCoords[1]
        numberOfVisionTapesDetected = targetPixelCoords[2]
        horiz_angle = getHorizAngleToTarget(targetX)
        vert_angle = getVertAngleToTarget(targetY)
        abs_distance = getAbsDistanceToTarget(targetY)
        result = [horiz_angle, vert_angle, abs_distance, numberOfVisionTapesDetected]
        return result


    else:

        numberOfVisionTapesDetected = targetPixelCoords[0]
        result = [numberOfVisionTapesDetected]
        return result

def get_image():
    retval, im = camera.read()
    return im


camera = cv2.VideoCapture(0)
for i in range(1):
    temp = get_image()
print("Taking image...")
# Take the actual image we want to keep
camera_capture = get_image()
file = "/Users/yiyi/Desktop/cv/captures/"

cv2.imwrite(file + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.jpg', camera_capture)

# img = cv2.imread('/Users/yiyi/Desktop/input.jpg')

camera_capture = cv2.flip(camera_capture, 0)
print("targetpositions : " + str(getTargetPosition(camera_capture)))

del camera







