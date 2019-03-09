import cv2
import numpy as np
import imutils


camera = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX


# 3D model points.
model_points = np.array([
    (141, 57, 0),  # upper right
    (55, 91, 0),  # upper left
    (103, 224, 0),  # lower left
    (200, 194, 0),  # lower right
], dtype="double")


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom

    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    print (boundingBoxes)
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

def sort_points(contour):
    theta = []
    for point in contour:
        theta[point] = np.arctan2(point[0], point[1])



while(True):
    # Read Image
    ret, im = camera.read()
    size = im.shape

    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    mask = cv2.dilate(cv2.inRange(cv2.bilateralFilter(hsv,9,75,75), (30, 90, 150), (40, 255, 255)), None, iterations=0)
    cv2.imshow("mask", mask)

    _, contours, __ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # find contours in the image


    if len(contours) > 0:
        res = cv2.bitwise_and(im, im, mask=mask)
        #gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

        biggestContour = max(contours, key = cv2.contourArea)
        #cv2.drawContours(res, biggestContour, -1, 255, 10)


        epsilon = 0.07 * cv2.arcLength(biggestContour, True)
        approx = cv2.approxPolyDP(biggestContour, epsilon, True) #4 points, hopefully
        cv2.drawContours(res, approx, -1, 255, 10) # blue



        box = np.int0(cv2.boxPoints(cv2.minAreaRect(biggestContour)))
        cv2.drawContours(res, [box], -1, (0,255,255), 1)

        if len(approx) == 4: # found the 4 corners
            print("found")
            for index, point in enumerate(approx):
                x=approx[index][0][0]
                y=approx[index][0][1]
                cv2.putText(res, str(index), (x,y), font, 4, (255, 255, 255), 2, cv2.LINE_AA)




            cv2.imshow("res", res)





            #SOLVEPNP

            # input 2D image points. changes with the image. This data is right tape.
            image_points = np.array([
                (approx[0][0][0], approx[0][0][1]),  # upper right
                (approx[1][0][0], approx[1][0][1]),  # upper left
                (approx[2][0][0], approx[2][0][1]),  # lower left
                (approx[3][0][0], approx[3][0][1]),  # lower right
            ], dtype="double")

            # Camera internals
            focal_length = size[1]
            center = (size[1] / 2, size[0] / 2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype="double"
            )
            dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                          dist_coeffs)

            # print("Rotation Vector:\n {0}".format(rotation_vector))
            # print("Translation Vector:\n {0}".format(translation_vector))

            # Project a 3D point (0, 0, 1000.0) onto the image plane.
            # We use this to draw a line sticking out of the nose

            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                             translation_vector,
                                                             camera_matrix, dist_coeffs)

            for p in image_points:
                cv2.circle(im, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

            p1 = (int(image_points[0][0]), int(image_points[0][1]))
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

            cv2.line(im, p1, p2, (255, 0, 0), 2)





    # Display image
    cv2.imshow("Output", im)

    # Display the resulting frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

