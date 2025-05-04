# -*- coding : utf-8 -*-

import os
import xlwt
import math
from tqdm import tqdm
import time
import cv2
import numpy as np
# TEXT_PATH = "runs/detect/exp31/labels/"
# TEXT_PATH = "inference/output/exp55/labels/"
PL_3_PROPORTION = 0.1
PL_4_PROPORTION = 0.2
# 3及变形，15-25%的形变
BX_3_PROPORTION = 15
# 4及变形，25%以上的形变
BX_4_PROPORTION = 25
# pai
PI = 3.14
# 直径
DIAMETER = 1000


# 返回一个字典，key为文件名，value为文件里面每一行构成的一个二维list
def get_files_dict(TEXT_PATH):
    all_files = []
    all_content_dict = dict()
    # 所有文件list 带后缀
    files = os.listdir(TEXT_PATH)

    for i in files:
        img_name = i.split('.')[0]
        file_path = os.path.join(TEXT_PATH, i)
        # 每个文件的每一行 组成的一个list，二维的
        content_list = []
        print("file_path", file_path)
        with open(file_path, "r") as f:
            while True:
                line = f.readline()
                if line:
                    #  去除换行符
                    line = line.strip("\n")
                    # 字符串切割为list，转为数字
                    # i[0] 类别，1-4 是四个坐标，5-8是图片尺寸，9置信度
                    text_list = list(map(float, line.split(' ', 7)))

                    # # 计算占比 ，直接加入textlist
                    # # 面积占比
                    # # region_size = text_list[3] * text_list[4] 缺陷区域大小
                    # # img_size = text_list[5] * text_list[6]  图像大小
                    #
                    # # 候选框最长边和图像最长边 比例 /2.5
                    # # proportion = max(text_list[3], text_list[4]) / max(text_list[5], text_list[6]) * text_list[9] / 1.5
                    #
                    # # 对角线占比  不太行，近景会放大
                    # # region_diagonal = math.sqrt(text_list[3] * text_list[4])
                    # # img_diagonal = math.sqrt(text_list[5] * text_list[6])
                    # # proportion = region_diagonal/img_diagonal/1.5
                    #
                    # # 对角线比 画面的长边 再跟置信度扯上关系
                    # region_diagonal = math.sqrt(text_list[3] * text_list[4])
                    # # proportion = region_diagonal / max(text_list[5], text_list[6]) * text_list[9] * text_list[9]
                    # # 改进一下，小故障的候选框 必然 有一边 会特别小，而预测的时候 会导致 候选框相比故障 会大的明显，所以引入了 候选框的较小的一边，参与计算
                    # proportion = region_diagonal / max(text_list[5], text_list[6]) * text_list[9] * text_list[9] * (min(text_list[3], text_list[4])/100)

                    # 最基本的计算占比
                    proportion = (text_list[3] * text_list[4]) / (text_list[5] * text_list[6])

                    text_list.append(proportion)

                    content_list.append(text_list)
                else:
                    break
        all_content_dict.update({img_name: content_list})
    # print(all_content_dict)
    return all_content_dict


# 输入一个字典，找到同类型中占比最大的哪一个
# 返回一个list，各个类型需要写入excel的记录
def find_max_proportion_lists(all_content_dict,defect_img=None):
    # class,xywh,whwh,credit,proportion
    # [['0', '942.5', '384', '399', '672', '1280', '720', '1280', '720', '0.879861',0.22222],[],[]]
    # PL0 ，BX1， TJ2 ， CK3 ， ZAW4 ， CJ5
    # 所有文件的pl bx tj ck各自集合的二维list， 其内容为max_PL_list 加上文件名,返回值
    files_max_PL_list = []
    files_max_BX_list = []
    files_max_TJ_list = []
    files_max_CK_list = []
    files_max_ZAW_list = []
    files_max_CJ_list = []

    for k, v in all_content_dict.items():
        # 一张图片里面各个病害类型的各自的list
        PL_list, BX_list, TJ_list, CK_list, ZAW_list, CJ_list = make_class_list(v)
        # 找出其中各自占比最大得
        max_PL_list = get_max_proportion(PL_list)
        max_BX_list = get_max_proportion(BX_list)
        max_TJ_list = get_max_proportion(TJ_list)
        max_CK_list = get_max_proportion(CK_list)
        max_ZAW_list = get_max_proportion(ZAW_list)
        max_CJ_list = get_max_proportion(CJ_list)

        if len(max_PL_list) != 0:
            # 提取PL得属性
            attribute_list = export_attribute_formula(defect_type=0, defect_list=max_PL_list)
            # 把 k.img，就是文件名加入list
            max_PL_list.append(str(k) + ".img")
            # 把属性加入list
            # attribute_list[ind_arclength, ind_ruptureArea]
            max_PL_list.extend(attribute_list)
            # 构建files_max_xx_list
            files_max_PL_list.append(max_PL_list)

        if len(max_BX_list) != 0:
            attribute_list = export_attribute_formula(defect_type=1, defect_list=max_BX_list)
            max_BX_list.append(str(k) + ".img")
            max_BX_list.extend(attribute_list)
            # print(max_BX_list)
            files_max_BX_list.append(max_BX_list)

        if len(max_TJ_list) != 0:
            defect_img = cv2.imread(defect_img, -1)
            attribute_list = export_attribute_formula(defect_type=2, defect_list=max_TJ_list, defect_img=defect_img)
            max_TJ_list.append(str(k) + ".img")
            max_TJ_list.extend(attribute_list)
            files_max_TJ_list.append(max_TJ_list)

        if len(max_CK_list) != 0:
            defect_img = cv2.imread(defect_img, -1)
            attribute_list = export_attribute_formula(defect_type=3, defect_list=max_CK_list, defect_img=defect_img)
            max_CK_list.append(str(k) + ".img")
            max_CK_list.extend(attribute_list)
            files_max_CK_list.append(max_CK_list)

        if len(max_CJ_list) != 0:
            attribute_list = export_attribute_formula(defect_type=4, defect_list=max_CJ_list)
            max_CJ_list.append(str(k) + ".img")
            max_CJ_list.extend(attribute_list)
            files_max_CJ_list.append(max_CJ_list)

        if len(max_ZAW_list) != 0:
            attribute_list = export_attribute_formula(defect_type=5, defect_list=max_ZAW_list)
            max_ZAW_list.append(str(k) + ".img")
            max_ZAW_list.extend(attribute_list)
            files_max_ZAW_list.append(max_ZAW_list)



    return files_max_PL_list, files_max_BX_list, files_max_TJ_list, files_max_CK_list, files_max_ZAW_list, files_max_CJ_list


# 6种病害的提取属性的计算公式
def export_attribute_formula(defect_type, defect_list, defect_img=None):
    # defect_list
    # class,xywh,whwh,credit,proportion
    # ['0', '942.5', '384', '399', '672', '1280', '720', '1280', '720', '0.879861',0.22222]

    # 属性list，每种病害有不同的list构成，在下面的if里面自己定制
    attribute_list = []

    # PL0 ，BX1， TJ2 ， CK3 ， CJ4 ， ZAW5
    # PL 得提取公式
    if defect_type == 0:
        # 置信度 defect_list[9]
        # 画面宽度
        img_w = defect_list[5]
        img_h = defect_list[6]
        # 病害宽高
        defect_w = defect_list[3]
        defect_h = defect_list[4]
        # 画面中心点
        img_center = [defect_list[5] / 2, -defect_list[6] / 2]
        # 病害中心
        defect_center = [defect_list[1], -defect_list[2]]
        # 病害上下左右边界坐标值
        defect_left_bound = defect_center[0] - defect_w / 2
        defect_right_bound = defect_center[0] + defect_w / 2
        defect_top_bound = defect_center[1] + defect_list[4] / 2  # 这里是y坐标的形式，就是负的
        defect_buttom_bound = defect_center[1] - defect_list[4] / 2
        # min_bound 就是求defect四边 距离 img的四边 最小距离
        min_bound = min(defect_left_bound, defect_right_bound, abs(defect_top_bound), abs(img_h + defect_buttom_bound))
        # 病害对角线像素长度
        defect_diagonal = math.sqrt(defect_list[3] ** 2 + defect_list[4] ** 2)
        # 图像中心和病害中心的长度
        distance_imgcenter2defectcenter = math.hypot(img_center[0] - defect_center[0], img_center[1] - defect_center[1])

        # 画面中心和病害中心，连线作为透视线y=kx+b，计算比例
        k = (img_center[1] - defect_center[1]) / (img_center[0] - defect_center[0])
        b = img_center[1] - k * img_center[0]

        # 经过图片中心的对角线的斜率，k1，k2
        k1 = img_center[1] / img_center[0]
        k2 = (img_center[1]) / (img_center[0] - img_w)
        #  这边做上下左右的判断
        # 跟y=0交点在图片之外，根据斜率判断
        if k1 <= k <= k2:
            # defect框更靠近左边界
            if min(defect_left_bound, defect_right_bound) == defect_left_bound:
                # 计算中心连线跟x=0的交点
                x = 0
            # defect框更靠近右边界
            else:
                x = img_w
            y = k * x + b
            # 交点坐标（x，y）
            distance_imgcenter2bouding = math.hypot(img_center[0] - x, img_center[1] - y)
        # 跟y=0交点在图片边上
        else:
            # 更靠经上边界
            if min(abs(defect_top_bound), abs(img_h + defect_buttom_bound)) == abs(defect_top_bound):
                y = 0
            # 更靠近下边界
            else:
                y = -img_h
            x = (y - b) / k
            distance_imgcenter2bouding = math.hypot(img_center[0] - x, img_center[1] - y)

        # 如果图像中心和病害中心 过于靠近，需要判断到底是病害太远太小，还是病害不远而且大(也不是特别大)，导致的病害的中心接近图片的中心
        # 归为大病害框一类,用diagonal_center_distance_ratio 衡量
        diagonal_center_distance_ratio = defect_diagonal / distance_imgcenter2defectcenter
        # 病害宽高 大于 图片宽高的二分之一，确定为大病害框，用不同的算法预估大小
        if defect_w > img_w * 3 / 5 or defect_h > img_h * 3 / 5 or diagonal_center_distance_ratio > 10:
            # 主要是看大病害靠近img边界，h比较大 就比较中心到竖边的比例计算， w比较大，就是中心到横边的比例计算
            if defect_w >= defect_h:
                # 更靠上面
                if abs(defect_top_bound) <= (img_h + defect_buttom_bound):
                    distance_imgcenter2bouding = abs(img_center[1])
                    distance_imgcenter2defectcenter = abs(defect_top_bound - img_center[1])
                # 更靠下面
                else:
                    distance_imgcenter2bouding = abs(img_center[1])
                    distance_imgcenter2defectcenter = abs(img_center[1] - defect_buttom_bound)
            else:
                # 更靠左边
                if (defect_left_bound - 0) < (img_w - defect_right_bound):
                    distance_imgcenter2bouding = img_center[0]  # 选择最靠近的距离作为imgcenter2bouding
                    distance_imgcenter2defectcenter = abs(img_center[0] - defect_left_bound)
                # 更靠右边
                else:
                    distance_imgcenter2bouding = img_center[0]
                    distance_imgcenter2defectcenter = abs(defect_right_bound - img_center[0])

        convert_defect_diagonal = distance_imgcenter2bouding * defect_diagonal / distance_imgcenter2defectcenter

        # r就是defect的上边界（下边界）到bound的距离 占 bound到中心的距离 比例
        # r就是个系数，当图像过于靠近上下or左右 用来弥补
        if min_bound == defect_left_bound:
            rate = 1 - (defect_left_bound / img_center[0])
        elif min_bound == defect_right_bound:
            rate = (defect_right_bound - img_center[0]) / (img_center[0])
        elif min_bound == abs(defect_top_bound):
            rate = 1 - (defect_top_bound / img_center[1])
        else:
            rate = (defect_buttom_bound - img_center[1]) / (img_center[1])
        # rate>0.8 表示框 过分靠近图片边界
        # diagonal_center_distance_ratio > 10 表示 病害框中心过分靠近图片中心
        if rate >= 0.8 or diagonal_center_distance_ratio > 10:
            rate = 1
        # 计算后的真实长度
        real_defect_diagonal = convert_defect_diagonal * DIAMETER / (img_w) * rate
        real_defect_angle = real_defect_diagonal * 180 / (PI * DIAMETER / 2)

        # 写入excel的属性，角度和弧度
        ind_arclength = real_defect_diagonal
        ind_ruptureArea = real_defect_angle
        attribute_list.append(ind_arclength)
        attribute_list.append(ind_ruptureArea)
        return attribute_list
    # BX 得提取公式
    elif defect_type == 1:
        # region 对角线长度
        region_diagonal = math.sqrt(defect_list[3] * defect_list[4])
        # 改进一下，小故障的候选框 必然 有一边 会特别小，而预测的时候 会导致 候选框相比故障 会大的明显，所以引入了 候选框的较小的一边，参与计算
        deform_ratio = region_diagonal / max(defect_list[5], defect_list[6]) * defect_list[7] * defect_list[7] * (
                min(defect_list[3], defect_list[4]) / 100)
        attribute_list.append(deform_ratio)
        return attribute_list
    # TJ 得提取公式
    elif defect_type == 2:
        # 病害宽高
        defect_w = defect_list[3]
        defect_h = defect_list[4]
        # 病害中心
        defect_center = [defect_list[1], -defect_list[2]]
        # 病害上下左右边界坐标值
        defect_left_bound = defect_center[0] - defect_w / 2
        defect_right_bound = defect_center[0] + defect_w / 2
        defect_top_bound = defect_center[1] + defect_list[4] / 2  # 这里是y坐标的形式，就是负的
        defect_buttom_bound = defect_center[1] - defect_list[4] / 2

        #  均值飘逸
        mean_filter_img = cv2.pyrMeanShiftFiltering(defect_img, 20, 30)
        # 灰度
        gray = cv2.cvtColor(mean_filter_img, cv2.COLOR_RGB2GRAY)
        # 高斯去噪
        gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
        # canny边缘检测
        edges = cv2.Canny(gaussian, 60, 100, apertureSize=3)

        # 霍夫曼圆检测
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 500, param1=300, param2=40, minRadius=40,
                                   maxRadius=int(defect_w / 2 - 5))  # 这里有个-10 .浮动误差
        # 检测出来的圆的 数组
        # [[[254 208 144]]]  圆心，半径
        if circles is not None:
            circles = np.uint16(np.around(circles))
            print(circles)
            for i in circles[0, :]:
                cv2.circle(defect_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(defect_img, (i[0], i[1]), 2, (0, 0, 255), 3)

        # circles1 = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 500, param1=300, param2=31, minRadius=40,
        #                            maxRadius=int(defect_w/2)-10) # 这里有个-10 .浮动误差
        # # 检测出来的圆的 数组
        # # [[[254 208 144]]]  圆心，半径
        # if circles1 is not None:
        #     circles1 = np.uint16(np.around(circles1))
        #     print(circles1)
        #     for i in circles1[0, :]:
        #         cv2.circle(defect_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        #         cv2.circle(defect_img, (i[0], i[1]), 2, (0, 0, 255), 3)
        # plt.imshow(defect_img)
        # plt.show()

        # 选取圆心和圆 完全落在boundingbox的圆，做为脱节的
        defect_tj_circle = []
        if circles is not None:
            # 检测出来的圆的 数组
            # [[[254 208 144]]]  圆心，半径
            circles = np.uint16(np.around(circles))

            for circle in circles[0, :]:
                # 圆心在框里面
                if defect_left_bound < int(circle[0]) < defect_right_bound and defect_buttom_bound < -int(
                        circle[1]) < defect_top_bound:
                    defect_tj_circle = [int(circle[0]), int(circle[1]), int(circle[2])]
        if len(defect_tj_circle) != 0:
            tj_circle_radius = defect_tj_circle[2]
            disjoint_distance = max(defect_w / 2 - tj_circle_radius, defect_h / 2 - tj_circle_radius)
            print(disjoint_distance)
            if 12 <= disjoint_distance:
                disjoint_distance = disjoint_distance / 2.2
            elif 6 < disjoint_distance < 12:
                disjoint_distance = disjoint_distance / 3
            else:
                disjoint_distance = disjoint_distance / 3.4
        else:
            disjoint_distance = 0
        attribute_list.append(disjoint_distance)
        return attribute_list
    # CK 得提取公式
    elif defect_type == 3:
        # 病害宽高
        defect_w = defect_list[3]
        defect_h = defect_list[4]
        # 病害中心
        defect_center = [defect_list[1], -defect_list[2]]
        # 病害上下左右边界坐标值
        defect_left_bound = defect_center[0] - defect_w / 2
        defect_right_bound = defect_center[0] + defect_w / 2
        defect_top_bound = defect_center[1] + defect_list[4] / 2  # 这里是y坐标的形式，就是负的
        defect_buttom_bound = defect_center[1] - defect_list[4] / 2

        #  均值飘逸
        mean_filter_img = cv2.pyrMeanShiftFiltering(defect_img, 20, 30)
        # 灰度
        gray = cv2.cvtColor(mean_filter_img, cv2.COLOR_RGB2GRAY)
        # 高斯去噪
        gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
        # canny边缘检测
        edges = cv2.Canny(gaussian, 60, 100, apertureSize=3)

        # 霍夫曼圆检测
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 500, param1=300, param2=40, minRadius=40,
                                   maxRadius=int(defect_w / 2 - 5))  # 这里有个-10 .浮动误差
        # 检测出来的圆的 数组
        # [[[254 208 144]]]  圆心，半径
        if circles is not None:
            circles = np.uint16(np.around(circles))
            print(circles)
            for i in circles[0, :]:
                cv2.circle(defect_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(defect_img, (i[0], i[1]), 2, (0, 0, 255), 3)

        # 选取圆心和圆 完全落在boundingbox的圆，做为脱节的
        defect_ck_circle = []
        if circles is not None:
            # 检测出来的圆的 数组
            # [[[254 208 144]]]  圆心，半径
            circles = np.uint16(np.around(circles))

            for circle in circles[0, :]:
                # 圆心在框里面
                if defect_left_bound < int(circle[0]) < defect_right_bound and defect_buttom_bound < -int(
                        circle[1]) < defect_top_bound:
                    defect_ck_circle = [int(circle[0]), int(circle[1]), int(circle[2])]
        if len(defect_ck_circle) != 0:
            ck_circle_radius = defect_ck_circle[2]
            misalignment_distance = max(defect_w / 2 - ck_circle_radius, defect_h / 2 - ck_circle_radius)
            print(misalignment_distance)
            if 12 <= misalignment_distance:
                misalignment_distance = misalignment_distance / 2.2
            elif 6 < misalignment_distance < 12:
                misalignment_distance = misalignment_distance / 3
            else:
                misalignment_distance = misalignment_distance / 3.4
        else:
            misalignment_distance = 0
        attribute_list.append(misalignment_distance)
        thicknessOfTubewall=((defect_list[3]+defect_list[4])/2)*0.09/2
        attribute_list.append(thicknessOfTubewall)
        return attribute_list
    # CJ 的提取公式
    elif defect_type == 4:
        # 置信度 defect_list[9]
        # 画面宽度
        img_w = defect_list[5]
        img_h = defect_list[6]
        # 病害宽高
        defect_w = defect_list[3]
        defect_h = defect_list[4]
        # 画面中心点
        img_center = [defect_list[5] / 2, defect_list[6] / 2]
        # 病害中心
        defect_center = [defect_list[1], defect_list[2]]
        # 管径设置成图片长宽的最大值
        pipe_diameter = max(img_w, img_h)
        # 管道圆心距离沉积水面的距离
        cj2center_height = math.sqrt((pipe_diameter / 2) ** 2 - (defect_w / 2) ** 2)
        # 如果病害中心在画面中心点上部，则表明沉积超过一半管径，病害越窄，程度越严重
        if defect_center[1] < img_center[1]:
            deposition_ratio = (pipe_diameter / 2 + cj2center_height) / pipe_diameter
        # 如果病害中心在画面中心点下部，则表明沉积并未超过一半管径，病害越宽，程度越严重
        else:
            deposition_ratio = (pipe_diameter / 2 - cj2center_height) / pipe_diameter

        attribute_list.append(deposition_ratio)
        return attribute_list
    # ZAW 的提取公式
    elif defect_type == 5:
        print("defect_list",defect_list)
        # 病害宽高
        defect_w = defect_list[3]
        defect_h = defect_list[4]
        credit = defect_list[7]
        # 画面宽度
        img_w = defect_list[5]
        img_h = defect_list[6]
        # 评估等级：用病害的面积 / 图片面积
        crossSectionLoss = (defect_w / img_w) * (defect_h / img_h) * credit
        attribute_list.append(crossSectionLoss)
        return attribute_list

    else:
        print("病害类型错误")
        return attribute_list


# 输入一个四种类别混合的二维数组生成，四种类别各自的二维列表
def make_class_list(mix_class_list):
    # 0
    PL_list = []
    # 1
    BX_list = []
    # 2
    TJ_list = []
    # 3
    CK_list = []
    # 4
    ZAW_list = []
    # 5
    CJ_list = []
    # [['0', '942.5', '384', '399', '672', '1280', '720', '1280', '720', '0.879861',0.22222],[],[]]
    for i in mix_class_list:
        if i[0] == 0:
            PL_list.append(i)
        elif i[0] == 1:
            BX_list.append(i)
        elif i[0] == 2:
            TJ_list.append(i)
        elif i[0] == 3:
            CK_list.append(i)
        elif i[0] == 5:
            ZAW_list.append(i)
        elif i[0] == 4:
            CJ_list.append(i)
    return PL_list, BX_list, TJ_list, CK_list, ZAW_list, CJ_list


# 用于获得PL_list 等里面最大占比的框的那条记录
# 返回一维list
def get_max_proportion(list):
    # i[10] 就是占比
    max_proportion = max((i[8] for i in list), default=0)
    max_proportion_list = []
    for i in list:
        if i[8] == max_proportion:
            max_proportion_list = i
        else:
            continue
    return max_proportion_list


def delete_excel():
    path = os.getcwd()
    excel_dir = os.path.join(path, 'excel')
    excel_files = os.listdir(excel_dir)
    # 文件已存在  就删除
    if len(excel_files) != 0:
        for i in tqdm(excel_files, desc="删除excel文件ing"):
            excel_file_path = os.path.join(excel_dir, i)
            os.remove(excel_file_path)
            time.sleep(1)
    else:
        time.sleep(1)
    print("===============旧excel文件删除完毕！=====================")
    time.sleep(2)


def write_excel(list, excel_path):
    path = os.getcwd()
    # excel_dir = os.path.join(path, '.\excel')
    excel_dir = os.path.join(path, excel_path)
    # PL
    if list[0][0] == 0:  # labels.txt第一个字符表示管道缺陷类别
        workbook = xlwt.Workbook()
        worksheet = workbook.add_sheet("PL worksheet")
        worksheet.write(0, 0, 'Class')
        worksheet.write(0, 1, 'Instance')
        worksheet.write(0, 2, 'ind_arclength')
        worksheet.write(0, 3, 'ind_ruptureArea')
        for index, i in enumerate(list):
            # i[0] 类别，1-4 是四个坐标，5-6是图片尺寸，7置信度，8占比，9文件名, 10往后是故障属性
            # ['0', '942.5', '384', '399', '672', '1280', '720','0.879861',0.22222, 01.img, ind_arclength, ind_ruptureArea]
            worksheet.write(index + 1, 0, "Pipe")
            # i[11] 是文件名
            worksheet.write(index + 1, 1, i[9])
            # i[12]  ind_arclength
            worksheet.write(index + 1, 2, i[10])
            # i[13]   ind_ruptureArea
            worksheet.write(index + 1, 3, i[11])
        workbook.save(os.path.join(excel_dir, "PL.xls"))
        return os.path.join(excel_dir, "PL.xls")
    # BX
    elif list[0][0] == 1:
        workbook = xlwt.Workbook()
        worksheet = workbook.add_sheet("BX worksheet")
        worksheet.write(0, 0, 'Class')
        worksheet.write(0, 1, 'Instance')
        worksheet.write(0, 2, 'Diameter(mm)')
        worksheet.write(0, 3, 'DeformRatio')
        worksheet.write(0, 4, 'DeformArea(mm)')
        for index, i in enumerate(list):
            print("BX", list)
            # i[0] 类别，1-4 是四个坐标，5-6是图片尺寸，7置信度，8占比，9文件名, 10往后是故障属性
            # ['0', '942.5', '384', '399', '672', '1280', '720','0.879861',0.22222, 01.img, ATTRIBUTEs]
            worksheet.write(index + 1, 0, "Pipe")
            # i[11] 是文件名
            worksheet.write(index + 1, 1, i[9])
            # diameter 直径
            worksheet.write(index + 1, 2, DIAMETER)
            # deformRatio  这里暂时写入i[12] 只有一个属性
            worksheet.write(index + 1, 3, i[10])
            # deformArea   比率*1000
            worksheet.write(index + 1, 4, i[10] * 1000)
        workbook.save(os.path.join(excel_dir, "BX.xls"))
        return os.path.join(excel_dir, "BX.xls")
    # TJ
    elif list[0][0] == 2:
        workbook = xlwt.Workbook()
        worksheet = workbook.add_sheet("TJ worksheet")
        worksheet.write(0, 0, 'Class')
        worksheet.write(0, 1, 'Instance')
        worksheet.write(0, 2, 'DisjointDistance')

        for index, i in enumerate(list):
            # i[0] 类别，1-4 是四个坐标，5-8是图片尺寸，9置信度，10占比，11文件名, 12往后是故障属性
            # ['0', '942.5', '384', '399', '672', '1280', '720', '1280', '720',
            #       '0.879861',0.22222, 01.img, iDisjointDistance]
            # 以上为旧版，以下为新版，删除了重复的图片尺寸
            # 2.0, 357.5, 266.5, 413.0, 401.0, 720.0, 576.0, 0.972846, 0.39933690200617283, '339f6faacd7f3bdbeffe7ac1734d7ee.img', 6.136363636363636
            worksheet.write(index + 1, 0, "Pipe")
            # i[11] 是文件名
            worksheet.write(index + 1, 1, i[9])
            # i[12]  DisjointDistance
            worksheet.write(index + 1, 2, i[10])

        workbook.save(os.path.join(excel_dir, "TJ.xls"))
        return os.path.join(excel_dir, "TJ.xls")
    # CK
    elif list[0][0] == 3:
        workbook = xlwt.Workbook()
        worksheet = workbook.add_sheet("CK worksheet")
        worksheet.write(0, 0, 'Class')
        worksheet.write(0, 1, 'Instance')
        worksheet.write(0, 2, 'Misalignment_distance')
        worksheet.write(0, 3, 'thicknessOfTubewall')
        #
        #
        #
        for index, i in enumerate(list):
            print(list)
            # i[0] 类别，1-4 是四个坐标，5-6是图片尺寸，7置信度，8占比，9文件名, 10往后是故障属性
            # ['0', '942.5', '384', '399', '672', '1280', '720','0.879861',0.22222, 01.img, ind_arclength, ind_ruptureArea]
            worksheet.write(index + 1, 0, "Pipe")
            # i[9] 是文件名
            worksheet.write(index + 1, 1, i[9])
            # i[10] Misalignment_distance
            worksheet.write(index + 1, 2, i[10])
            # i[11] thicknessOfTubewall
            worksheet.write(index + 1, 3, i[11])

        workbook.save(os.path.join(excel_dir, "CK.xls"))
        return os.path.join(excel_dir, "CK.xls")
    elif list[0][0] == 4:
        workbook = xlwt.Workbook()
        worksheet = workbook.add_sheet("CJ worksheet")
        worksheet.write(0, 0, 'Class')
        worksheet.write(0, 1, 'Instance')
        worksheet.write(0, 2, 'deposition_ratio')
        #
        #
        #
        for index, i in enumerate(list):
            # i[0] 类别，1-4 是四个坐标，5-6是图片尺寸，7置信度，8占比，9文件名, 10往后是故障属性
            # ['0', '942.5', '384', '399', '672', '1280', '720','0.879861',0.22222, 01.img, ind_arclength, ind_ruptureArea]
            worksheet.write(index + 1, 0, "Pipe")
            # i[11] 是文件名
            worksheet.write(index + 1, 1, i[9])
            # i[12] deposition_ratio
            worksheet.write(index + 1, 2, i[10])

        workbook.save(os.path.join(excel_dir, "CJ.xls"))
        return os.path.join(excel_dir, "CJ.xls")
    elif list[0][0] == 5:
        workbook = xlwt.Workbook()
        worksheet = workbook.add_sheet("ZAW worksheet")
        worksheet.write(0, 0, 'Class')
        worksheet.write(0, 1, 'Instance')
        worksheet.write(0, 2, 'crossSectionLoss')
        #
        #
        #
        for index, i in enumerate(list):
            # i[0] 类别，1-4 是四个坐标，5-6是图片尺寸，7置信度，8占比，9文件名, 10往后是故障属性
            # ['0', '942.5', '384', '399', '672', '1280', '720','0.879861',0.22222, 01.img, ind_arclength, ind_ruptureArea]
            worksheet.write(index + 1, 0, "Pipe")
            # i[11] 是文件名
            worksheet.write(index + 1, 1, i[9])
            # i[12] crossSectionLoss
            worksheet.write(index + 1, 2, i[10])

        workbook.save(os.path.join(excel_dir, "ZAW.xls"))
        return os.path.join(excel_dir, "ZAW.xls")
    


if __name__ == '__main__':

    delete_excel()
    all_content_dict = get_files_dict()
    files_max_PL_list, files_max_BX_list, files_max_TJ_list, files_max_CK_list = find_max_proportion_lists(
        all_content_dict)
    for i in tqdm(all_content_dict, desc="生成新文件ing"):
        if len(files_max_PL_list) != 0:
            write_excel(files_max_PL_list)
        if len(files_max_BX_list) != 0:
            write_excel(files_max_BX_list)
        if len(files_max_TJ_list) != 0:
            write_excel(files_max_TJ_list)
        if len(files_max_CK_list) != 0:
            write_excel(files_max_CK_list)
        if len(files_max_ZAW_list) != 0:
            write_excel(files_max_ZAW_list)
        if len(files_max_CJ_list) != 0:
            write_excel(files_max_CJ_list)
            
        time.sleep(0.05)
    print("===============新文件生成完毕！=====================")
