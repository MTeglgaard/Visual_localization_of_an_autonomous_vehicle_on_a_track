#!/usr/bin/python
# -*- coding: utf-8 -*-

# first line force to use python3

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from itertools import islice
from datetime import datetime

# from laneFunc import *

recordVideo = bool(False)
showImages = bool(True)


def canny(image):

    # print 'canny'

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 75, 0)
    return canny

def contour_labeling(image, image_canny):
    edges = image_canny
    (contours, hierarchy) = cv2.findContours(edges.copy(),
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours = sorted(contours, key=cv2.contourArea,
                             reverse=True)  # https://www.programiz.com/python-programming/methods/built-in/sorted
    cnt_len = 0
    cnt_road = 0
    for (i, c) in enumerate(sorted_contours):
        cnt_len_tmp = cv2.arcLength(c, closed=False)
        if cnt_len < cnt_len_tmp:
            cnt_len = cnt_len_tmp
            cnt_road = c

    return cnt_road  # c


def region_of_intrest2(image):

    height = image.shape[0]
    width = image.shape[1]

    polygons = np.array([[(0, height), (0, height * 3 / 8), (width,
                        height * 3 / 8), (width, height)]])  #    [(0,480), (width, 580), (600,300)]

    polygonsCar = np.array([[
        (width, 720),
        (width, 630),
        (960, 534),
        (805, 500),
        (700, 500),
        (530, 540),
        (465, 610),
        (255, 665),
        (240, height),
        ]])

    mask = np.zeros_like(image)
    mask_inv = np.invert(mask)
    cv2.fillPoly(mask_inv, polygonsCar, 0)

    masked_image = cv2.bitwise_and(image, mask_inv)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(masked_image, mask)
    return masked_image


def region_of_intrest(image):

    # print 'ROI'

    height = image.shape[0]
    width = image.shape[1]
    polygons = np.array([[(0, height), (0, 430), (600, 300), (width,
                        430), (width, height)]])  #    [(0,480), (width, 580), (600,300)]
    polygonsCar = np.array([[
        (width, 720),
        (width, 630),
        (960, 534),
        (805, 500),
        (700, 500),
        (530, 540),
        (465, 610),
        (255, 665),
        (240, height),
        ]])

    mask = np.zeros_like(image)
    mask_inv = np.invert(mask)
    cv2.fillPoly(mask_inv, polygonsCar, 0)

    masked_image = cv2.bitwise_and(image, mask_inv)

    return masked_image



def display_lines(image, lines):

    # print 'display lines'

    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            (x1, y1, x2, y2) = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


def make_coordinates(image, line_parameters, y_min):

    # print 'make coordinates'

    (slope, intercept) = line_parameters

    # print(image.shape)

    y1 = image.shape[0]
    y2 = int(y_min)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    if slope > -0.000001 and slope < 0:

        # print "left coordinates Error"

        y1 = 0
        y2 = 0  # int(y_min)
        x1 = 0  # int(intercept)
        x2 = 0  # int(intercept)
    elif slope < 0.000001 and slope > 0:

        # print "right coordinates 0"

        y1 = 0  # image.shape[0]
        y2 = 0  # int(y_min)
        x1 = 0  # int((y1-intercept)/slope)
        x2 = 0  # int((y2-intercept)/slope)

    return np.array([x1, y1, x2, y2])


def reject_outliers(line_parameters, m):

    # print 'reject'

    slope_mean = np.mean(np.array(line_parameters)[:, 0])
    slope_mstd = m * np.std(np.array(line_parameters)[:, 0])
    intercept_mean = np.mean(np.array(line_parameters)[:, 1])
    intercept_mstd = m * np.std(np.array(line_parameters)[:, 1])

    for parameter in line_parameters:
        (slope, intercept) = parameter

        if abs(slope - slope_mean) >= slope_mstd or abs(intercept
                - intercept_mean) >= intercept_mstd:
            line_parameters.remove(parameter)

    return line_parameters  # [slope,intercept]


def average_slope_intercept(image, lines):

    # print 'average'

    polyDegree = 1
    left_fit = []
    right_fit = []

    if lines is None:

        # print "no lines"

        return np.array([[0, 0, 0, 0], [0, 0, 0, 0]])

    for line in lines:
        (x1, y1, x2, y2) = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), polyDegree)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    if len(left_fit) < 2 or len(right_fit) < 2:

        # print len(left_fit)

        return np.array([[0, 0, 0, 0], [0, 0, 0, 0]])

        # print 'rejected frame'

        pass

    left_fit_inliers = reject_outliers(left_fit, 1.5)
    right_fit_inliers = reject_outliers(right_fit, 1.5)

    if len(left_fit_inliers) < 2 or len(right_fit_inliers) < 2:

        # print len(left_fit_inliers)

        return np.array([[0, 0, 0, 0], [0, 0, 0, 0]])

        # print 'rejected frame'

        pass

    left_fit_average = np.average(left_fit_inliers, axis=0)
    right_fit_avarage = np.average(right_fit_inliers, axis=0)  # print("left:")

    (a1, b1) = left_fit_average
    (a2, b2) = right_fit_avarage

    y_min = (a1 * b2 - a2 * b1) / (a1 - a2)  # (left_fit_average(0)*right_fit_avarage(1) -right_fit_avarage(0)* left_fit_average(1))/(left_fit_average(0)-right_fit_avarage(0))

    left_line = make_coordinates(image, left_fit_average, y_min)
    right_line = make_coordinates(image, right_fit_avarage, y_min)
    return np.array([left_line, right_line])


def combine(
    img1,
    img2,
    img3,
    img4,
    img5,
    img6,
    ):
    if len(img1.shape) != 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) != 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    if len(img3.shape) != 3:
        img3 = cv2.cvtColor(img3, cv2.COLOR_GRAY2BGR)
    if len(img4.shape) != 3:
        img4 = cv2.cvtColor(img4, cv2.COLOR_GRAY2BGR)
    if len(img5.shape) != 3:
        img5 = cv2.cvtColor(img5, cv2.COLOR_GRAY2BGR)
    if len(img6.shape) != 3:
        img6 = cv2.cvtColor(img6, cv2.COLOR_GRAY2BGR)

    # frame_cropped = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR)

    visTop = np.concatenate((img1, img2), axis=1)
    visBottom = np.concatenate((img3, img4), axis=1)
    vis = np.concatenate((visTop, visBottom), axis=0)
    visMittle = np.concatenate((img5, img6), axis=0)
    vis = np.concatenate((vis, visMittle), axis=1)
    return vis


####### main

## sequence capture

print 'Named explicitly:'

image = \
    cv2.imread('/home/michael/Documents/div_Programming/laneLinesPython/testImg/left005731.png'
               )
(height, width, layers) = image.shape
size = (width, height)
if recordVideo:
    now = datetime.now()
    outVidName = \
        now.strftime('/home/michael/Documents/div_Programming/laneLinesPython/results/Contour_Poly6img%Y%m%d%H%MVideo.avi'
                     )
    outVidNameGray = \
        now.strftime('/home/michael/Documents/div_Programming/laneLinesPython/results/%Y%m%d%H%MGrayVideo.avi'
                     )
    print outVidName
    fps = 15
    out = cv2.VideoWriter(outVidName, cv2.VideoWriter_fourcc(*'DIVX'),
                          fps, size)  # ('/home/michael/Documents/div_Programming/laneLinesPython/results/wholeTrackROI.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
img_array = []

# mask for grabCut
newmask = cv2.imread('mask2_3.png', 0)
rect = (0, 300, 1280, 720)

i = 0

filenames = \
    sorted(glob.glob('/media/michael/My Book/roskilde2020june3/forthRecording/LR/left/*.png'
           ))
filenames_iter = iter(filenames)
for filename in filenames_iter:  


	#skip the irst frames
    if i == 0:
        next(islice(filenames_iter,4000,4001)) #print("ehj")
        i=i+1

    image = cv2.imread(filename)

    # ################## start morph line

    img_org = image.copy()  # cv.imread('/home/michael/Documents/div_Programming/laneLinesPython/testImg/left005000.png')#'left005460.png')

    img = img_org.copy()

    img_canny = canny(img)
    img_RIO = region_of_intrest(img_canny)

    kernel = np.ones((50, 50), np.uint8)
    closing = cv2.morphologyEx(img_RIO, cv2.MORPH_CLOSE, kernel)

    closing = np.invert(closing)
    closing_color = cv2.cvtColor(closing, cv2.COLOR_GRAY2RGB)
    blur = cv2.GaussianBlur(closing_color, (1, 1), 0)
    canny2 = cv2.Canny(blur, 0, 0)
    rio_canny2 = region_of_intrest2(canny2)

    contour_road= contour_labeling(rio_canny2, rio_canny2)
    img_con = cv2.drawContours(np.zeros(img.shape[:2],np.uint8), contour_road, -1, 255, 3)

    nonZarray = cv2.findNonZero(img_con)

    (x, y) = nonZarray.transpose()  
    x = x.reshape(-1)  # https://www.w3resource.com/numpy/manipulation/reshape.php
    y = y.reshape(-1)

    mymodel = np.poly1d(np.polyfit(x, y, 6))  # https://www.w3schools.com/python/python_ml_polynomial_regression.asp

    # print(mymodel)

    myline = np.linspace(np.amin(x), np.amax(x), endpoint=False)  # https://numpy.org/doc/stable/reference/generated/numpy.linspace.html

    # img_closing_canny=canny(closing_color)

    # masked_image = cv2.drawContours(img_org, contour_road, -1, 255, 3) #img_org, contour_road, -1, (0,255,0), 3)
    # masked_image = cv2.bitwise_and(img_org, closing_color)

    # e2 = cv2.getTickCount()
    # time = (e2 - e1) / cv2.getTickFrequency()
    # print ("Ticks: ", e2-e1, "computation time: ", time )

    win_name2 = 'rio Canny 2'
    win_name3 = 'Contour'

    fig = plt.figure()
    ax = plt.axes([0.0, 0.0, 1.0, 1.0])  # leaves no white space around the axes
    plt.imshow(img_org)  # cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB))#rio_canny2,cv2.COLOR_GRAY2RGB)) #
    plt.plot(myline, mymodel(myline), color='red')
    plt.axis('off')

    # If we haven't already shown or saved the plot, then we need to
    # draw the figure first

    fig.canvas.draw()

    # plt.savefig('algMorphImg/model_3order_left005460.png')

    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                         sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    (ht, wd, cc) = data.shape

    # print("data h: ", ht)
    # print("data w: ", wd)

    (hh, ww, cc) = img_org.shape

    # print("img h: ", hh)
    # print("img w: ", ww)
    # ww = 300
    # hh = 300

    color = (0, 0, 0)
    data_resize = np.full((hh, ww, cc), color, dtype=np.uint8)

    # compute center offset

    xx = (ww - wd) // 2
    yy = (hh - ht) // 2

    # copy img image into center of result image

    data_resize[yy:yy + ht, xx:xx + wd] = data

    (ht, wd, cc) = data_resize.shape

    # print("data h: ", ht)
    # print("data w: ", wd)

    # cv2.imshow("result", data)
    # cv2.waitKey(0)
    # plt.show()

    # cv2.destroyAllWindows()

    # #####################

    if showImages:

        combined_M_image= combine(image, img_con, img_RIO, np.invert(closing), rio_canny2,  data_resize)# , line_image
        
        imS = cv2.resize(combined_M_image, size)
        cv2.imshow('result', imS)

    if recordVideo:

        # frame = imS

        frame = data

        # frame = gC_img

        out.write(frame)

        #print "video is recording"

    if showImages:
        key = cv2.waitKey(1)
        plt.close('all')

        # print(key)

        if key == 113:  # or (filename == "/home/michael/Documents/div_Programming/laneLinesPython/testImg/left005330.png")
            break
    else:
        print ('i: %d\n', i)
        i = i + 1
        if i > 5100:# 5000:
            break


if recordVideo:

    out.release()
    print (outVidName, ' video released')
