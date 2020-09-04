#!/usr/bin/env python3
# first line force to use python3
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from datetime import datetime
from laneFunc import *

recordVideo = bool(False)
showImages = bool(True)

## sequence capture
print('Named explicitly:')
# for name in glob.glob('home/michael/Documents/div_Programming/laneLinesPython/*.png'):
#     print(name)
image = cv2.imread('/media/3643E31F14F73E0C/marts_4_Left/left000001.png')#'/home/michael/Documents/div_Programming/laneLinesPython/testImg/left005731.png')
height, width, layers = image.shape
size = (width,height)
if recordVideo:
    now = datetime.now()
    outVidName = now.strftime("/home/Desktop/laneLinesPython/results/%Y%m%d%H%MVideo.avi")
    outVidNameGray = now.strftime("/home/Desktop/laneLinesPython/results/%Y%m%d%H%MGrayVideo.avi")
    print(outVidName)
    fps = 15
    out     = cv2.VideoWriter(outVidName,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)#('/home/michael/Documents/div_Programming/laneLinesPython/results/wholeTrackROI.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

img_array = []
for filename in sorted(glob.glob('/media/3643E31F14F73E0C/marts_4_Left/*.png')):#'/home/michael/Documents/div_Programming/laneLinesPython/testImg/*.png')):#'/home/michael/Documents/div_Programming/laneLinesPython/testImg/*.png')): #sorted(glob.glob('/media/michael/My Book/roskilde2020june3/forthRecording/LR/left/*.png')): #
    # print filename
    image = cv2.imread(filename)
    lane_image = np.copy(image)
    canny_image = canny(lane_image)
    cropped_image = region_of_intrest(canny_image)
    cropped_lane_image = region_of_intrest(image)

    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) #resolution of theta= 1 degree (CV_PI/180)
    Hline_image = display_lines(lane_image, lines)
    averaged_lines = average_slope_intercept(lane_image, lines)
    line_image = display_lines(lane_image, averaged_lines)
    combo_image = cv2.addWeighted(lane_image,0.8, line_image,1,1)
    if showImages:

        combined_image= combine(image, cropped_lane_image ,canny_image, cropped_image, combo_image  , Hline_image )# , line_image
        cv2.namedWindow("combined_image", cv2.WINDOW_NORMAL)
        com_height, com_width, com_layers = combined_image.shape
        imS = cv2.resize(combined_image, size)
        cv2.imshow("combined_image",imS)

    if recordVideo:
        frame = imS
        out.write(frame)

        # print ("video is recording")

    if showImages:
        key = cv2.waitKey(1)
        if (key == 113) :
            break

if recordVideo:
    out.release()
    print(outVidName," video released")
