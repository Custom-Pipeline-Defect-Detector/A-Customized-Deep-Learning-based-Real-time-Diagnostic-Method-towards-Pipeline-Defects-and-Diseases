import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def scale(img, xScale, yScale):
    res = cv2.resize(img, None, fx=xScale, fy=yScale, interpolation=cv2.INTER_AREA)
    return res


def crop(infile, height, width):
    im = Image.open(infile)
    imgwidth, imgheight = im.size
    for i in range(imgheight // height):
        for j in range(imgwidth // width):
            box = (j * width, i * height, (j + 1) * width, (i + 1) * height)
            yield im.crop(box)


def averagePixels(path):
    r, g, b = 0, 0, 0
    count = 0
    pic = Image.open(path)
    for x in range(pic.size[0]):
        for y in range(pic.size[1]):
            imgData = pic.load()
            tempr, tempg, tempb = imgData[x, y]
            r += tempr
            g += tempg
            b += tempb
            count += 1
    return (r / count), (g / count), (b / count), count

def convert_frame_to_grayscale(frame):
    gray_frame = None
    gray_frame_blur = None
    if frame is not None:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #灰度化
        # gray = scale(gray, 1, 1)
        gray_frame = scale(gray_frame, 1, 1)
        gray_frame_blur = cv2.GaussianBlur(gray_frame, (9, 9), 0.0)
    return gray_frame_blur

def prepare_dirs(keyframePath):
    if not os.path.exists(keyframePath):
        os.makedirs(keyframePath)


def plot_metrics(indices, y,lstfrm, lstdiffMag):
    plt.plot(indices, y[indices], "x")
    # plt.plot(lstfrm, lstdiffMag, 'r-')
    plt.plot(lstfrm, lstdiffMag, 'y')
    plt.xlabel('frames')
    plt.ylabel('pixel difference')
    plt.title("Pixel value differences from frame to frame and the peak values")
    plt.show()
    f = plt.gcf()  # 获取当前图像
    f.savefig(r'./static/keyFrames/frameDifferencesAndPeak.jpg')
    # f.savefig(r'D:\Pycharm_Projects\pipeline_detection_system\static\keyFrames\frameDifferencesAndPeak.jpg')
    f.clear()  # 释放内存
