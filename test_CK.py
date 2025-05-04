import cv2
import numpy as np

# 读取图片
image = cv2.imread(r'C:\Users\zero\Pictures\pipeline\CK\001272.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 边缘检测
edges = cv2.Canny(gray, 50, 150)

# 圆弧检测
circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # 设定允许的圆心位置范围
        min_x, max_x = 250, 300
        min_y, max_y = 150, 300
        # 设定允许的半径范围
        min_radius, max_radius = 90, 150
        if min_x < i[0] < max_x and min_y < i[1] < max_y and min_radius < i[2] < max_radius:
            # 画出检测到的圆
            cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # 画出圆心
            cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
            # 输出圆心坐标和半径
            print("圆心坐标: ({}, {})， 半径: {}".format(i[0], i[1], i[2]))
# 显示结果
cv2.imshow('detected circles', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
#
# if circles is not None:
#     circles = np.uint16(np.around(circles))
#
#     # 初始化变量来保存最大和最小的两个圆
#     max_y_circle = None
#     min_y_circle = None
#
#     for i in circles[0, :]:
#         # 设定允许的圆心位置范围
#         min_x, max_x = 300, 450
#         min_y, max_y = 150, 300
#         # 设定允许的半径范围
#         min_radius, max_radius = 90, 150
#         if min_x < i[0] < max_x and min_y < i[1] < max_y and min_radius < i[2] < max_radius:
#             # 输出圆心坐标和半径
#             print("圆心坐标: ({}, {})， 半径: {}".format(i[0], i[1], i[2]))
#
#             # 更新最大和最小的两个圆
#             if max_y_circle is None or i[1] > max_y_circle[1]:
#                 max_y_circle = i
#             if min_y_circle is None or i[1] < min_y_circle[1]:
#                 min_y_circle = i
#
#     # 画出最大和最小的两个圆
#     if max_y_circle is not None:
#         cv2.circle(image, (max_y_circle[0], max_y_circle[1]), max_y_circle[2], (0, 255, 0), 2)
#         cv2.circle(image, (max_y_circle[0], max_y_circle[1]), 2, (0, 0, 255), 3)
#     if min_y_circle is not None:
#         cv2.circle(image, (min_y_circle[0], min_y_circle[1]), min_y_circle[2], (0, 255, 0), 2)
#         cv2.circle(image, (min_y_circle[0], min_y_circle[1]), 2, (0, 0, 255), 3)
#
# # 显示结果
# cv2.imshow('detected circles', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


