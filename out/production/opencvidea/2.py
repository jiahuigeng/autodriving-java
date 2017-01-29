#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import random
import numpy as np
import numpy.polynomial.polynomial as poly
from matplotlib import pyplot as plt

def region_of_interest(img, vertices):

    mask = np.zeros_like(img)


    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(frame,img,n_frame, lines, color=[255, 0, 0], thickness=2,):
    """
    Separate the detected lines into left and right based on slope. Then run a linear regression
    Obtain the coefficients and draw the llane lines for using a defined y and a calculated x:
    x=(y-b)/m
    """
    if lines == None:
        return
    #Use global variable to retain line end points between frames
    global x1r, x2r, y1r, y2r, x1l, x2l, y1l,y2l
    global prior_x1r, prior_x2r, prior_y1r, prior_y2r, prior_x1l, prior_x2l, prior_y1l,prior_y2l
    print(n_frame)
    #Initialize slope and intercept
    l_line=[]
    r_line=[]

    Yright=[]
    Yleft=[]
    Xright=[]
    Xleft=[]
    #Separate into lefy and right lane
    for line in lines:
        for x1,y1,x2,y2 in line:
            if (float(x2)-float(x1))!=0:
                slope=(float(y2)-float(y1))/(float(x2)-float(x1))
            if (slope >= 0.2 and x2 > 0.5*img.shape[1]):
                Yright.extend((y1,y2))
                Xright.extend((x1,x2))
            elif (slope <= -0.2 and x2 < 0.5*img.shape[1]):
                Yleft.extend((y1,y2))
                Xleft.extend((x1,x2))

    x1r=x2r=0
    y1r=(2.0*img.shape[0]/5.0)
    y2r=(img.shape[0])

    if (len(Xright)>0 and len(Yright)>0):
        r_coefficients = poly.polyfit(Xright, Yright, 1)
        x2r=((y2r-r_coefficients[0])/r_coefficients[1])
        x1r=(x2r-(y2r-y1r)/r_coefficients[1])
        prior_x1r=int(x1r)
        prior_x2r=int(x2r)
        prior_y1r=int(y1r)
        prior_y2r=int(y2r)
        cv2.line(img, (prior_x1r, prior_y1r), (prior_x2r, prior_y2r), [0,255,0], 10)
    elif (len(Xright)==0 and len(Yright)==0):
        x1r=int(prior_x1r)
        x2r=int(prior_x2r)
        y1r=int(prior_y1r)
        y2r=int(prior_y2r)
        cv2.line(img, (x1r, y1r), (x2r, y2r), [0,255,0], 10)



    x1l=x2l=0
    y1l=(2.0*img.shape[0]/5.0)
    y2l=(img.shape[0])

    if (len(Xleft)>0 and len(Yleft)>0):
        l_coefficients = poly.polyfit(Xleft, Yleft, 1)
        x2l=((y2l-l_coefficients[0])/l_coefficients[1])
        x1l=(x2l-(y2l-y1l)/l_coefficients[1])

        prior_x1l=int(x1l)
        prior_x2l=int(x2l)
        prior_y1l=int(y1l)
        prior_y2l=int(y2l)

        cv2.line(img, (prior_x1l, prior_y1l), (prior_x2l, prior_y2l), [0,0,255], 10)
    elif (len(Xleft)==0 and len(Yleft)==0):
        x1l=int(prior_x1l)
        x2l=int(prior_x2l)
        y1l=int(prior_y1l)
        y2l=int(prior_y2l )
        cv2.line(img, (x1l, y1l), (x2l, y2l), [0,0,255], 10)
    img_final=weighted_img(img, frame, alpha=0.8, beta=1., gamma=0.)
    cv2.imshow('frame',img_final)
#    Coefficients=[r_coefficients,l_coefficients]
#    return Coefficients



def hough_lines(frame,img_gaussian,n_frame, rho, theta, threshold, min_line_len, max_line_gap,):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """


    lines = cv2.HoughLinesP(img_gaussian, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img_gaussian.shape[0], img_gaussian.shape[1], 3), dtype=np.uint8)
    draw_lines(frame,line_img,n_frame, lines)


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, alpha=0.8, beta=1., gamma=0.):

    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)

def process_img(frame,n_frame):
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_canny = cv2.Canny(img_gray, 255, 255)
    masked_canny_image = region_of_interest(img_canny, vertices)
    img_gaussian = cv2.GaussianBlur(masked_canny_image, (5, 5), 0)
    hough_lines(frame,img_gaussian, n_frame,rho = 1, theta = np.pi / 180, threshold = 15,    min_line_len = 30, max_line_gap = 2)
#    cv2.imshow("frame",img_gaussian)




cap = cv2.VideoCapture('1.mp4')
vertices = np.array([[(750, 324), (429,324), (0,493), (0,604),(1278,604),(1278,493)]])
D_r=np.mat([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
sigma_d_r=np.eye(4)*1.25
M_r=np.mat([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
sigma_m_r=np.eye(4)*1.25
n_frame=0
while(cap.isOpened()):
    ret, frame = cap.read()
    n_frame=n_frame+1
    print(n_frame)
    process_img(frame,n_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
''' '''
