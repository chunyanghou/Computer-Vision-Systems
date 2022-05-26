import os

import numpy as np
import cv2

imgs = os.listdir('../carla_images/rgb/000')

for file in imgs:

    # read video frame & show on screen
    img = cv2.imread('../carla_images/rgb/001/'+file)

    frame = cv2.medianBlur(img, 11)
    # print(frame.shape)

    # snip section of video frame of interest & show on screen
    snip = frame[:,:,:]
    cv2.imshow("Snip",snip)

    # create polygon (trapezoid) mask to select region of interest
    mask = np.zeros((snip.shape[0], snip.shape[1]), dtype="uint8")
    pts = np.array([[0, 310], [370, 340], [580, 340], [1023, 340],[1023,511],[0,511]], dtype=np.int32)

    pts1 = np.array([[256,384],[768,384],[768,511],[256,511]])

    cv2.fillConvexPoly(mask, pts, 255)
    # cv2.fillConvexPoly(mask, pts1, 0)
    cv2.imshow("Mask", mask)

    # apply mask and show masked image on screen
    masked = cv2.bitwise_and(snip, snip, mask=mask)
    cv2.imshow("Region of Interest", masked)

    # convert to grayscale then black/white to binary image
   #frame = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV).astype(np.uint8)
    # lower = np.array([0,0,150])
    # upper = np.array([180,50,255])
    # frame = cv2.inRange(frame, lower, upper)


    frame = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    frame = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

    # sobelx = cv2.Sobel(frame, cv2.CV_64F, 0, 1)
    # sobelx = cv2.convertScaleAbs(sobelx)

    cv2.imshow("Black/White", frame)

    # blur image to help with edge detection
    blurred = cv2.blur(frame, (3, 11), 0)
    # cv2.imshow("Blurred", blurred)

    # identify edges & show on screen
    edged = cv2.Canny(blurred, 50, 150)
    cv2.imshow("Edged", edged)

    # perform full Hough Transform to identify lane lines
    lines = cv2.HoughLines(edged, 1, np.pi / 180, 100,100,10)

    # define arrays for left and right lanes
    rho_left = []
    theta_left = []
    rho_right = []
    theta_right = []

    # ensure cv2.HoughLines found at least one line
    if lines is not None:

        # loop through all of the lines found by cv2.HoughLines
        for i in range(0, len(lines)):


            # evaluate each row of cv2.HoughLines output 'lines'
            for cc in lines[i]:

                rho, theta = cc
                # print(rho)

                # collect left lanes
                if theta < np.pi/100*44 and theta > np.pi/100*30:
                    rho_left.append(rho)
                    theta_left.append(theta)

                    # # plot all lane lines for DEMO PURPOSES ONLY

                # collect right lanes
                if theta > np.pi/100*60 and theta < np.pi/100*70:

                    rho_right.append(rho)
                    theta_right.append(theta)
                    # a = np.cos(theta)
                    # b = np.sin(theta)
                    # x0 = a * rho
                    # y0 = b * rho
                    # x1 = int(x0 - 800 * b)
                    # y1 = int(y0 + 800 * a)
                    # x2 = int(x0 + 1200 * b)
                    # y2 = int(y0 - 1200 * a)
                    # cv2.line(snip, (x1, y1), (x2, y2), (0, 0, 255), 1)

    # statistics to identify median lane dimensions
    left_rho = np.median(rho_left)
    left_theta = np.median(theta_left)
    right_rho = np.median(rho_right)
    right_theta = np.median(theta_right)

    # print(left_theta,right_theta)
    # plot median lane on top of scene snip
    if left_theta > np.pi/4:
        a = np.cos(left_theta); b = np.sin(left_theta)
        x0 = a * left_rho; y0 = b * left_rho
        offset1 = 400; offset2 = 800
        x1 = int(x0 - offset1 * (-b)); y1 = int(y0 - offset1 * (a))
        x2 = int(x0 + offset2 * (-b)); y2 = int(y0 + offset2 * (a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 6)

    if right_theta > np.pi/5:
        a = np.cos(right_theta); b = np.sin(right_theta)
        x0 = a * right_rho; y0 = b * right_rho
        offset1 = 600; offset2 = 1600

        x3 = int(x0 - offset1 * (-b))
        y3 = int(y0 - offset1 * (a))

        x4 = int(x0 - offset2 * (-b))
        y4 = int(y0 - offset2 * (a))

        x3,x4 = x4,x3
        y3,y4 = y4,y3
        cv2.line(img, (x3, y3), (x4, y4), (255, 0, 0), 6)



    # overlay semi-transparent lane outline on original
    if left_theta > np.pi/4 and right_theta > np.pi/5:
        pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int32)

        # (1) create a copy of the original:
        overlay = img.copy()
        # (2) draw shapes:
        cv2.fillConvexPoly(overlay, pts, (0, 255, 0))
        # (3) blend with the original:
        opacity = 0.4
        cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)

    cv2.imshow("Lined", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()