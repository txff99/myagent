#!/usr/bin/env python
# coding: utf-8
# created by hevlhayt@foxmail.com 
# Date: 2016/1/15 
# Time: 19:20
#
import os
import cv2
import numpy as np

class light(object):
    def __init__(self) -> None:
        self.lower_red1 = np.array([0,150,150])
        self.upper_red1 = np.array([10,255,255])
        self.lower_red2 = np.array([160,150,150])
        self.upper_red2 = np.array([180,255,255])
        self.lower_green = np.array([40,150,150])
        self.upper_green = np.array([90,255,255])
        # lower_yellow = np.array([15,100,100])
        # upper_yellow = np.array([35,255,255])
        self.lower_yellow = np.array([15,170,170])
        self.upper_yellow = np.array([35,255,255])
        self.light_position = []
        
    def light_detect(self, img):
        
        self.light_position = []

        font = cv2.FONT_HERSHEY_SIMPLEX
        # img = cv2.imread(filepath+file)
        cimg = img
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # color range
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        maskg = cv2.inRange(hsv, self.lower_green, self.upper_green)
        masky = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        maskr = cv2.add(mask1, mask2)

        size = img.shape
        # print size
        cv2.imshow('maskg',maskr)
        cv2.waitKey(50)
        # hough circle detect
        r_circles = cv2.HoughCircles(maskr, cv2.HOUGH_GRADIENT, 1, 50,
                                param1=100, param2=13, minRadius=2, maxRadius=10)

        g_circles = cv2.HoughCircles(maskg, cv2.HOUGH_GRADIENT, 1, 50,
                                    param1=100, param2=13, minRadius=2, maxRadius=10)

        y_circles = cv2.HoughCircles(masky, cv2.HOUGH_GRADIENT, 1, 30,
                                    param1=50, param2=13, minRadius=2, maxRadius=10)

        cimg = np.ascontiguousarray(cimg)
        # traffic light detect
        r = 5
        bound = 0.2#4.0 / 10
        if r_circles is not None:
            r_circles = np.uint16(np.around(r_circles))
            for i in r_circles[0, :]:
                # print(i[0])
                # print(i[1])
                # if i[0] > size[1] or i[1] > size[0]or i[1] > size[0]*bound:
                #     continue
                # print(1)

                h, s = 0.0, 0.0
                for m in range(-r, r):
                    for n in range(-r, r):

                        if (i[1]+m) >= size[0] or (i[0]+n) >= size[1]:
                            continue
                        h += maskr[i[1]+m, i[0]+n]
                        s += 1
                self.light_position.append((i[0],1/i[2],'red')) # location and distance
                if h / s > 50:
                    # print(i[2])#i[2] is radius
                    cv2.circle(cimg, (i[0], i[1]), i[2]+10, (0, 255, 0), 2)
                    cv2.circle(maskr, (i[0], i[1]), i[2]+30, (255, 255, 255), 2)
                    cv2.putText(cimg,'RED',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)
                    cv2.putText(cimg,f'd={int(100/i[2])}',(i[0], i[1]-30), font, 1,(255,0,0),2,cv2.LINE_AA) 

        if g_circles is not None:
            g_circles = np.uint16(np.around(g_circles))

            for i in g_circles[0, :]:
                # if i[0] > size[1] or i[1] > size[0] or i[1] > size[0]*bound:
                #     continue

                h, s = 0.0, 0.0
                for m in range(-r, r):
                    for n in range(-r, r):

                        if (i[1]+m) >= size[0] or (i[0]+n) >= size[1]:
                            continue
                        h += maskg[i[1]+m, i[0]+n]
                        s += 1
                self.light_position.append((i[0],1/i[2],'green'))
                if h / s > 50:
                    # print(i[2])
                    cv2.circle(cimg, (i[0], i[1]), i[2]+10, (0, 255, 0), 2)
                    cv2.circle(maskg, (i[0], i[1]), i[2]+30, (255, 255, 255), 2)
                    cv2.putText(cimg,'GREEN',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)
                    cv2.putText(cimg,f'd={int(100/i[2])}',(i[0], i[1]-30), font, 1,(255,0,0),2,cv2.LINE_AA)

        if y_circles is not None:
            y_circles = np.uint16(np.around(y_circles))

            for i in y_circles[0, :]:
                # if i[0] > size[1] or i[1] > size[0] or i[1] > size[0]*bound:
                #     continue

                h, s = 0.0, 0.0
                for m in range(-r, r):
                    for n in range(-r, r):

                        if (i[1]+m) >= size[0] or (i[0]+n) >= size[1]:
                            continue
                        h += masky[i[1]+m, i[0]+n]
                        s += 1
                self.light_position.append((i[0],1/i[2],'yellow'))
                if h / s > 50:
                    # print(i[2])
                    cv2.circle(cimg, (i[0], i[1]), i[2]+10, (0, 255, 0), 2)
                    cv2.circle(masky, (i[0], i[1]), i[2]+30, (255, 255, 255), 2)
                    cv2.putText(cimg,'YELLOW',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)
                    cv2.putText(cimg,f'd={int(100/i[2])}',(i[0], i[1]-30), font, 1,(255,0,0),2,cv2.LINE_AA)
        return cimg
        
        

        #,red,yellow,green
        # cv2.imshow('detected results', cimg)
        # cv2.imwrite(path+'//result//'+file, cimg)
        # # cv2.imshow('maskr', maskr)
        # # cv2.imshow('maskg', maskg)
        # # cv2.imshow('masky', masky)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    # if __name__ == '__main__':

    #     path = os.path.abspath('..')+'//light//'
    #     for f in os.listdir(path):
    #         print (f)
    #         if f.endswith('.jpg') or f.endswith('.JPG') or f.endswith('.png') or f.endswith('.PNG'):
    #             detect(path, f)

