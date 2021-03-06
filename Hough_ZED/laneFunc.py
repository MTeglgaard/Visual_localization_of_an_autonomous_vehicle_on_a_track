#laneFunc.py

import cv2
import numpy as np
import matplotlib.pyplot as plt


def canny(image):
    # print 'canny'
    gray =cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5),0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_intrest(image):
    # print 'ROI'
    height = image.shape[0]
    width = image.shape[1]
    # height2 = image.get_height()
    # width2 = image.get_width()
    # print height
    # print width
    scaleH = height/ 720
    scaleW = width/ 1280
    # print (height)
    # # print (height2)
    # print (width)
    # print (width2)
    
    # CarPoint1 = (width,          height)
    # CarPoint2 = (width,          int(630*scaleH))
    # CarPoint3 = (int(960*scaleW),int(534*scaleH))
    # CarPoint4 = (int(805*scaleW),int(500*scaleH))
    # CarPoint5 = (int(700*scaleW),int(500*scaleH))
    # CarPoint6 = (int(530*scaleW),int(540*scaleH))
    # CarPoint7 = (int(465*scaleW),int(610*scaleH))
    # CarPoint8 = (int(255*scaleW),int(665*scaleH))
    # CarPoint9 = (int(240*scaleW),height)

    # RIO1 = (0               ,height)
    # RIO2 = (0               ,int(430*scaleH))
    # RIO3 = (int(600*scaleW) ,int(300*scaleH))
    # RIO4 = (width           ,int(430*scaleH))
    # RIO5 = (width           ,height)


    CarPoint1 = (width,          height)
    CarPoint2 = (width,          int((53/62)*height))
    CarPoint3 = (int((3/4)*width),int((5/7)*height))
    CarPoint4 = (int((12/19)*width),int((9/13)*height))
    CarPoint5 = (int((5/9)*width),int((9/13)*height))
    CarPoint6 = (int((7/17)*width),int((3/4)*height))
    CarPoint7 = (int((7/22)*width),int((11/13)*height))
    CarPoint8 = (int((5/24)*width),int((44/51)*height))
    CarPoint9 = (int((3/22)*width),height)

    RIO1 = (0               ,height)
    RIO2 = (0               ,int((14/27)*height))
    RIO3 = (int((1/2)*width),int(1/3*height))
    RIO4 = (width           ,int((14/27)*height))
    RIO5 = (width           ,height)



    polygons = np.array([
    [CarPoint1, CarPoint2, CarPoint3, CarPoint4, CarPoint5, CarPoint6, CarPoint7, CarPoint8, CarPoint9, RIO1,RIO2,RIO3,RIO4,RIO5] 
    ],dtype=np.int32)
    # print("p int:", polygons)

    # polygons = np.array([
    # [CarPoint1,CarPoint2, (600*scaleW,300*scaleH), (width, 430*scaleH),(width, height*scaleH) ]#    [(0,480), (width, 580), (600,300)]
    # ])
    # polygonsCar = np.array([
    # [(width,720), (width, 630), (960,534), (805,500), (700,500), (530,540), (465,610),(255,665),(240, height) ]
    # ])
    polygonsCar2 = np.array([
    [(width,720*scaleH), (width, 630*scaleH), (960*scaleW,534*scaleH), (805*scaleW,500*scaleH), (700*scaleW,500*scaleH), (530*scaleW,540*scaleH), (465*scaleW,610*scaleH),(255*scaleW,665*scaleH),(240*scaleW, height) , (0,height),(0,430*scaleH), (600*scaleW,300*scaleH), (width, 430*scaleH),(width, height)]
    ],dtype=np.int32)
    # print("polygonsCar2 :", polygonsCar2)
    mask = np.zeros_like(image)
    # mask_inv = np.invert(mask)
    # cv2.fillPoly(mask_inv, polygonsCar, 0)
    # cv2.imshow('mask_inv', mask_inv)
    # cv2.waitKey(0)
    # masked_image = cv2.bitwise_and(image, mask_inv)
    # cv2.fillPoly(mask, polygons, 255)
    white = 255
    if len(image.shape)==3:
        #polygonsCar2=polygonsCar3
        white = (255, 255, 255)

    cv2.fillPoly(mask, polygons, white)
    #
    # cv2.imshow('mask',mask)
    # cv2.waitKey(0)
    masked_image = cv2.bitwise_and(image, mask)
    # cv2.imshow('masked_image',masked_image)
    # cv2.waitKey(0)
    return masked_image

def display_lines(image, lines):
    # print 'display lines'
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1,y1),(x2,y2), (255,0,0), 10)
    return line_image

def make_coordinates(image, line_parameters, y_min):
    # print 'make coordinates'
    slope, intercept = line_parameters

    # print(image.shape)
    y1= image.shape[0]
    y2= int(y_min)
    x1= int((y1-intercept)/slope)
    x2= int((y2-intercept)/slope)

    if slope > -0.000001 and slope <0:
        # print "left coordinates Error"
        y1= 0
        y2= 0#int(y_min)
        x1= 0#int(intercept)
        x2= 0#int(intercept)
    elif slope < 0.000001 and slope >0:
        # print "right coordinates 0"
        y1= 0#image.shape[0]
        y2= 0#int(y_min)
        x1= 0#int((y1-intercept)/slope)
        x2= 0#int((y2-intercept)/slope)

    return np.array([x1,y1,x2, y2])

def reject_outliers(line_parameters, m):
    # print 'reject'
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
    # print 'average'
    polyDegree = 1
    left_fit = []
    right_fit =[]

    if lines is None:
        # print "no lines"
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
        # print 'rejected frame'
        pass

    left_fit_inliers =    reject_outliers(left_fit, 1.5)
    right_fit_inliers =    reject_outliers(right_fit, 1.5)

    if (len(left_fit_inliers)<2) or (len(right_fit_inliers)<2):
        # print len(left_fit_inliers)
        return np.array([[0,0,0, 0],[0,0,0, 0]])
        # print 'rejected frame'
        pass

    left_fit_average =np.average(left_fit_inliers, axis=0)
    right_fit_avarage = np.average(right_fit_inliers, axis=0)# print("left:")

    a1, b1 = left_fit_average
    a2, b2 = right_fit_avarage

    y_min = (a1*b2-a2*b1)/(a1-a2)#(left_fit_average(0)*right_fit_avarage(1) -right_fit_avarage(0)* left_fit_average(1))/(left_fit_average(0)-right_fit_avarage(0))

    left_line = make_coordinates(image, left_fit_average, y_min)
    right_line = make_coordinates(image, right_fit_avarage, y_min)
    return np.array([left_line,right_line])

def combine(img1, img2, img3, img4, img5, img6):
    if len(img1.shape)!=3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape)!=3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    if len(img3.shape)!=3:
        img3 = cv2.cvtColor(img3, cv2.COLOR_GRAY2BGR)
    if len(img4.shape)!=3:
        img4 = cv2.cvtColor(img4, cv2.COLOR_GRAY2BGR)
    if len(img5.shape)!=3:
        img5 = cv2.cvtColor(img5, cv2.COLOR_GRAY2BGR)
    if len(img6.shape)!=3:
        img6 = cv2.cvtColor(img6, cv2.COLOR_GRAY2BGR)

    visTop = np.concatenate((img1, img2), axis=1)
    visBottom = np.concatenate((img3, img4), axis=1)
    vis = np.concatenate((visTop, visBottom), axis=0)
    visMittle = np.concatenate((img5, img6), axis=0)
    vis = np.concatenate((vis, visMittle), axis=1)
    return vis
