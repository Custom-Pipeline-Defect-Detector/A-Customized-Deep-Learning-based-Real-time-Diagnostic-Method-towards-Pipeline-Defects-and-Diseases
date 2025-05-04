import os
import cv2
import csv
import numpy as np
import time
import peakutils
from keyFrame.utils import convert_frame_to_grayscale, prepare_dirs, plot_metrics


def keyframeDetection(source, dest, Thres=0.3,Min_dist=30,Deg=2, plotMetrics=True, verbose=False):

    # 存放关键帧的路径
    keyframePath = dest+'/keyFrames'
    # imageGridsPath = dest+'/imageGrids'
    #csv文件显示了生成了多少关键帧并且花费了多少时间
    # csvPath = dest+'/csvFile'
    # path2file = csvPath + '/output.csv'
    # prepare_dirs(keyframePath, imageGridsPath, csvPath)
    prepare_dirs(keyframePath)  #创建相应的文件夹

    # 读取视频，并且获得视频总共帧数
    cap = cv2.VideoCapture(source)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #帧数
    if (cap.isOpened()== False):
        print("Error opening video file")

    lstfrm = []
    lstdiffMag = []     #帧差
    # timeSpans = []
    # images = []     #存储灰度化的帧
    full_color = []     #存储rgb格式的帧
    lastFrame = None
    # Start_time = time.process_time()
    
    # Read until video is completed
    for i in range(count):
        ret, frame = cap.read() #返回索引和图像,ret的值为True,未被使用，没有用处;frame为三通道的图像
        blur_gray = convert_frame_to_grayscale(frame)    #对帧进行灰度化处理，并进行高斯模糊处理
        # print(grayframe.shape)
        # print(blur_gray.shape)
        # cv2.imshow("grayframe",grayframe)
        # cv2.imwrite("C://Users//50563//Videos//grayframe.jpg",grayframe)
        # cv2.imshow("blur_gray",blur_gray)
        # cv2.imwrite("C://Users//50563//Videos//blur_gray.jpg", blur_gray)
        # break
        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1 #获取帧的位置
        lstfrm.append(frame_number)
        # images.append(grayframe)
        full_color.append(frame)
        if frame_number == 0:
            lastFrame = blur_gray

        diff = cv2.subtract(blur_gray, lastFrame)   #将上一帧的高斯模糊和当前帧的高斯模糊进行差分计算
        diffMag = cv2.countNonZero(diff)    #统计非0值
        lstdiffMag.append(diffMag)
        # stop_time = time.process_time()
        # time_Span = stop_time-Start_time
        # timeSpans.append(time_Span)
        lastFrame = blur_gray

    cap.release()
    y = np.array(lstdiffMag)
    base = peakutils.baseline(y, deg=Deg) #计算所给数据的基线
    indices = peakutils.indexes(y-base, Thres, min_dist=Min_dist) #找峰值，indices图像关键帧的坐标
    
    ##plot to monitor the selected keyframe
    if (plotMetrics):
        plot_metrics(indices,y, lstfrm, lstdiffMag)

    keyFrames=[]
    cnt = 1
    for x in indices:
        #将关键帧保存
        cv2.imwrite(os.path.join(keyframePath , 'keyframe'+ str(cnt) +'.jpg'), full_color[x])
        keyFrames.append(os.path.join(keyframePath , 'keyframe'+ str(cnt) +'.jpg'))
        cnt +=1
        '''
        # 记录每个key frame处理所花费的时间
        log_message = 'keyframe ' + str(cnt) + ' happened at ' + str(timeSpans[x]) + ' sec.'
        if(verbose):
            print(log_message)
        with open(path2file, 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(log_message)
            csvFile.close()
        '''

    cv2.destroyAllWindows()
    # Finish_time=time.process_time()
    # print("process need time:",Finish_time-Start_time)
    return keyFrames,indices

if __name__ == '__main__':
    kfs,i=keyframeDetection(r"D:\python\yolov5_flask-main\static\uploads\test.mp4",r"D:\python\yolov5_flask-main\static",Thres=0.3,plotMetrics=True)