# encoding:utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import math


MIN_CENTER_DIST = 10
Dist_MAX = 500
circlesFinal = []
concentricCircles = []
# concentricCircles：输出参数，用于存储选择出的与其他圆最接近的圆。
# targetCenter：输出参数，用于存储计算出的中心点。
def selectCircles(circlesFinal):
    center_x_temp= 0
    center_y_temp= 0
    centerDist = Dist_MAX
    l=len(circlesFinal)
    for i in range(len(circlesFinal)):
        for j in range(i+1 , len(circlesFinal)):
            dist = math.sqrt( (circlesFinal[i][0] - circlesFinal[j][0])**2 + (circlesFinal[i][1] - circlesFinal[j][1])**2 )
            if dist < centerDist :
                centerDist = dist
                center_x_temp = (circlesFinal[i][0] + circlesFinal[j][0]) / 2
                center_y_temp = (circlesFinal[i][1] + circlesFinal[j][1]) / 2

    for i in range(len(circlesFinal)):
        if (math.sqrt( (circlesFinal[i][0] - center_x_temp)**2 + (circlesFinal[i][1] - center_y_temp)**2 )< MIN_CENTER_DIST) :
            concentricCircles.append(circlesFinal[i])

    center_x = center_x_temp
    center_y = center_y_temp

    return concentricCircles, center_x , center_y

def disjunctionScore(path):

    #图像预处理
    defect_img = cv2.imread(path)
    #  均值飘逸
    mean_filter_img = cv2.pyrMeanShiftFiltering(defect_img, 20, 30)
    # 灰度
    gray = cv2.cvtColor(mean_filter_img, cv2.COLOR_RGB2GRAY)
    # 高斯去噪
    img_gaussian = cv2.GaussianBlur(gray, (5, 5), 0)

    # cv2.imshow('img_gaussian', img_gaussian)
    # cv2.waitKey(0)

    #边缘检测
    threshold_min = 50
    threshold_max = 150
    img_edges = cv2.Canny(img_gaussian, threshold1=threshold_min, threshold2= threshold_max, apertureSize=3)


    # cv2.imshow('img_edges', img_edges)
    # cv2.waitKey(0)

    #同心圆检测
    #max_Radius = int( img_edges.shape[1] / 2)
    max_Radius = img_edges.shape[1]
    min_Radius = 40
    step = 10

    #circles = cv2.HoughCircles(img_edges, cv2.HOUGH_GRADIENT, 1, 10, param1=threshold_max, param2=40, minRadius=min_Radius,maxRadius=min_Radius +step)

    for i in range(min_Radius , max_Radius ,step):
        circles = cv2.HoughCircles(img_edges, cv2.HOUGH_GRADIENT, 1, 10, param1=threshold_max, param2=40, minRadius=i,maxRadius=i + step)
        if circles is None:
            continue
        else:
            circlesFinal.extend(circles[0])
            circles = 0

    #选取同心圆
    Dist_MAX = img_edges.shape[1]
    # circlesFinal = []
    # concentricCircles = []

    concentricCircles , center_x , center_y = selectCircles(circlesFinal)

    finalCenter_x = []
    finalCenter_y = []

    # #标记同心圆
    # for i in range(len(concentricCircles)):
    #     finalCenter_x.append(round(concentricCircles[i][0]))
    #     finalCenter_y.append(round(concentricCircles[i][1]))
    #     radius = round(concentricCircles[i][2])
    #     cv2.circle(defect_img ,(finalCenter_x[i],finalCenter_y[i]), radius , color=(255, 0, 0), thickness=5)
    #
    # #计算同心圆 圆心位置
    #
    # if len(concentricCircles)>0 :
    #     finalCenter_x = sum(finalCenter_x) / len(concentricCircles)
    #     finalCenter_y = sum(finalCenter_y) / len(concentricCircles)
    #     cv2.circle(defect_img, (int(center_x) ,int(center_y)), 3,color=(255,0,0),thickness= 1)
    #
    # # plt.imshow(defect_img)
    # plt.savefig(r"C:\Users\HLP\Desktop\test.jpg")

    #脱节
    #计算脱节最大距离
    circlespoints_between_distance = math.sqrt((concentricCircles[0][0] - concentricCircles[1][0])**2 + (concentricCircles[0][1] - concentricCircles[1][1])**2)
    disjoint_distance = round(circlespoints_between_distance + max(concentricCircles[0][2],concentricCircles[1][2]) - min(concentricCircles[0][2],concentricCircles[1][2]) )
    return disjoint_distance

# 339f6faacd7f3bdbeffe7ac1734d7ee.png
path = r"C:\Users\zero\Pictures\pipeline\TJ\TJ-test.jpg"
dist = disjunctionScore(path)
print(dist)
