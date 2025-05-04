# encoding:utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import math

#障碍物
#评估等级：用病害的对角线 / 管道直径
# 画面宽度

defect_list=[]
# img_w = defect_list[5]
# img_h = defect_list[6]
# # 病害宽高
defect_w = defect_list[3]
defect_h = defect_list[4]

def auto_canny(image, sigma=0.33):
	# 计算单通道像素强度的中位数
	v = np.median(image)
	v1 = np.percentile(image, [25])

	# 选择合适的lower和upper值，然后应用它们
	lower = int(max(0, (1.0 - sigma) * v1))
	upper = int(min(255, (1.0 + sigma) * v1))
	edged = cv2.Canny(image, lower, upper)

	return edged

# defect_img = cv2.imread(r"C:\Users\HLP\Desktop\7.JPG")
defect_img = cv2.imread(r"C:\Users\zero\Pictures\pipeline\ZAW\001588.jpg")
# 检查图像是否为空
if defect_img is None:
    print("Error: Failed to read image")
    exit()

#  均值飘逸
mean_filter_img = cv2.pyrMeanShiftFiltering(defect_img, 20, 30)
# 灰度
gray = cv2.cvtColor(mean_filter_img, cv2.COLOR_RGB2GRAY)
# 高斯去噪
img_gaussian = cv2.GaussianBlur(gray, (5, 5), 0)

# cv2.imshow('img_gaussian', img_gaussian)
# cv2.waitKey(0)

# 边缘检测
threshold_min = 40
threshold_max = 200
img_edges = auto_canny(img_gaussian)

# cv2.imshow('img_edges', img_edges)
# cv2.waitKey(0)
max_Radius = img_edges.shape[1]
min_Radius = 40
circles = cv2.HoughCircles(img_edges, cv2.HOUGH_GRADIENT, 1, 10, param1=threshold_max, param2=50, minRadius=min_Radius,maxRadius=max_Radius)
pipeline_diameter = np.around(circles[0][0][2]) *2
# for i in range(len(circles)):
#     cv2.circle(defect_img ,(np.round(circles[i][0][0]),np.round(circles[i][0][1])), np.round(circles[i][0][2]) , color=(255, 0, 0), thickness=5)
defect_diagonal = math.sqrt(defect_w ** 2 + defect_h ** 2 )

diagonal_diameter_ratio = defect_diagonal / pipeline_diameter
print(diagonal_diameter_ratio)