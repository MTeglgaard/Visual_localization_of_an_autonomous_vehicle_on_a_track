#laneFunc.py

import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(image):
    print 'canny'
    gray =cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5),0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_intrest(image):
    print 'ROI'
    height = image.shape[0]
    width = image.shape[1]
    polygons = np.array([
    [(0,height),(0,430), (600,300), (width, 430),(width, height) ]#    [(0,480), (width, 580), (600,300)]
    ])
    polygonsCar = np.array([
    [(width,720), (width, 630), (960,534), (805,500), (700,500), (530,540), (465,610),(255,665),(240, height) ]
    ])

    mask = np.zeros_like(image)
    mask_inv = np.invert(mask)
    cv2.fillPoly(mask_inv, polygonsCar, 0)
    masked_image = cv2.bitwise_and(image, mask_inv)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(masked_image, mask)
    return masked_image

def display_lines(image, lines):
    print 'display lines'
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1,y1),(x2,y2), (255,0,0), 10)
    return line_image


def make_coordinates(image, line_parameters, y_min):
    print 'make coordinates'
    slope, intercept = line_parameters

    # print(image.shape)
    y1= image.shape[0]
    y2= int(y_min)
    x1= int((y1-intercept)/slope)
    x2= int((y2-intercept)/slope)

    if slope > -0.000001 and slope <0:
        print "left coordinates Error"
        y1= 0
        y2= 0#int(y_min)
        x1= 0#int(intercept)
        x2= 0#int(intercept)
    elif slope < 0.000001 and slope >0:
        print "right coordinates 0"
        y1= 0#image.shape[0]
        y2= 0#int(y_min)
        x1= 0#int((y1-intercept)/slope)
        x2= 0#int((y2-intercept)/slope)

    return np.array([x1,y1,x2, y2])

def reject_outliers(line_parameters, m):
    print 'reject'
    slope_mean = np.mean(np.array(line_parameters)[:,0])
    slope_mstd = m * np.std(np.array(line_parameters)[:,0])
    intercept_mean = np.mean(np.array(line_parameters)[:,1])
    intercept_mstd = m * np.std(np.array(line_parameters)[:,1])

    for parameter in line_parameters:
        slope, intercept =parameter

        if ((abs(slope - slope_mean) >= slope_mstd) or (abs(intercept - intercept_mean) >= intercept_mstd)):
            line_parameters.remove(parameter)

    return line_parameters#[slope,intercept]




def average_slope_intercept(image, lines):
    print 'average'
    polyDegree = 1
    left_fit = []
    right_fit =[]

    if lines is None:
        print "no lines"
        return np.array([[0,0,0, 0],[0,0,0, 0]])

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2), polyDegree)
        slope =parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))

    if (len(left_fit)<2) or (len(right_fit)<2):
        # print len(left_fit)
        return np.array([[0,0,0, 0],[0,0,0, 0]])
        print 'rejected frame'
        pass

    left_fit_inliers =    reject_outliers(left_fit, 1.5)
    right_fit_inliers =    reject_outliers(right_fit, 1.5)

    if (len(left_fit_inliers)<2) or (len(right_fit_inliers)<2):
        # print len(left_fit_inliers)
        return np.array([[0,0,0, 0],[0,0,0, 0]])
        print 'rejected frame'
        pass

    left_fit_average =np.average(left_fit_inliers, axis=0)
    right_fit_avarage = np.average(right_fit_inliers, axis=0)# print("left:")

    a1, b1 = left_fit_average
    a2, b2 = right_fit_avarage

    y_min = (a1*b2-a2*b1)/(a1-a2)#(left_fit_average(0)*right_fit_avarage(1) -right_fit_avarage(0)* left_fit_average(1))/(left_fit_average(0)-right_fit_avarage(0))

    left_line = make_coordinates(image, left_fit_average, y_min)
    right_line = make_coordinates(image, right_fit_avarage, y_min)
    return np.array([left_line,right_line])
