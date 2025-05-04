# -*- coding = utf-8 -*-
# @Time: 2023/1/11 16:34
# @Author: Zero_Right
# @File: model_detect.py
# @Software: PyCharm

# yolov5-5.0
'''
import torch
import sys
from pathlib import Path
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
import torch.backends.cudnn as cudnn
'''
# from excel_export_job import get_files_dict, find_max_proportion_lists, write_excel
from excel_export_job_zhx import get_files_dict, find_max_proportion_lists, write_excel
from tqdm import tqdm

# YOLOv5-7.0
import platform

from YOLOv5.models.common import DetectMultiBackend
from YOLOv5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from YOLOv5.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, colorstr, cv2,
                                  increment_path, non_max_suppression, scale_boxes, strip_optimizer, xyxy2xywh)
from YOLOv5.utils.plots import Annotator, colors, save_one_box
from YOLOv5.utils.torch_utils import select_device
import torch.backends.cudnn as cudnn
from numpy import random

# faster r-cnn
import numpy as np
from PIL import Image
from fasterRCNN.frcnn import FRCNN

# YOLOX 0.3.0
import os
import time

import cv2

import torch

from loguru import logger

from YOLOX.yolox.data.data_augment import ValTransform
from YOLOX.yolox.data.datasets import COCO_CLASSES
from YOLOX.yolox.exp import get_exp
from YOLOX.yolox.utils import fuse_model, get_model_info, postprocess, vis

from pathlib import Path
import sys

from ultralytics import YOLO

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# yolov5 yolov7
def run_yolov_5_7(
        # weights=ROOT / 'runs/train/exp/weights/yolov5_best_latest_20230224.pt',  # model path or triton URL
        weights=ROOT / 'runs/train/exp/weights/yolov7_best_weight_20230327.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/my.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device="0",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=True,  # save results to *.txt
        save_conf=True,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'static/output',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        yolo_version=7
):
    if weights == "":
        weights = ROOT / 'runs/train/exp/weights/yolov7_best_weight_20230327.pt'
    save_path = ""

    if yolo_version == 5:
        weights = ROOT / 'runs/train/exp/weights/yolov5_best_latest_20230224.pt'
    else:
        weights = ROOT / 'runs/train/exp/weights/yolov7_best_weight_20230327.pt'

    source = str(source)
    print("source:",source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            gn = torch.tensor(im0.shape)[[1, 0]]
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # 不要正则化
                        line = (cls, *xywh, *gn, conf) if save_conf else (cls, *xywh, *gn)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        if hide_labels == "False":
                            hide_labels = False
                        elif hide_labels == "True":
                            hide_labels = True
                        if hide_conf == "False":
                            hide_conf = False
                        elif hide_conf == "True":
                            hide_conf = True
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

    # 额外加入，（1）记录检测结果的图像路径（2）对labels文本信息进行处理
    # 将检测结果图像的路径保存到txt文件中
    with open('./inference/output/identify_image.txt', "w", encoding='UTF-8') as f:
        f.write(save_path)
    f.close()
    # 将label文件夹下的txt文件生成相应的excel文件
    print("===============generate excel process is starting!=====================")
    TEXT_PATH = str(str(save_dir / 'labels'))
    all_content_dict = get_files_dict(TEXT_PATH=TEXT_PATH)
    files_max_PL_list, files_max_BX_list, files_max_TJ_list, files_max_CK_list, files_max_ZAW_list, files_max_CJ_list = find_max_proportion_lists(
        all_content_dict,defect_img=source)
    for i in tqdm(all_content_dict, desc="生成新excel文件ing"):
        if len(files_max_BX_list) != 0:
            write_excel(files_max_BX_list, save_dir)
        if len(files_max_PL_list) != 0:
            write_excel(files_max_PL_list, save_dir)
        if len(files_max_TJ_list) != 0:
            write_excel(files_max_TJ_list, save_dir)
        if len(files_max_CK_list) != 0:
            write_excel(files_max_CK_list, save_dir)
        if len(files_max_CJ_list) != 0:
            write_excel(files_max_CJ_list, save_dir)
        if len(files_max_ZAW_list) != 0:
            write_excel(files_max_ZAW_list, save_dir)
            print("save_dir",save_dir)
        # time.sleep(0.05)
    print("===============excel file generation completed！=====================")

# faster_rcnn
def run_faster_rcnn(source, confidence=0.5, nms_iou=0.3, thickness=3):
    frcnn = FRCNN()
    # ----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'           表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'             表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'               表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'       表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    # ----------------------------------------------------------------------------------------------------------#
    mode = "predict"
    # -------------------------------------------------------------------------#
    #   crop                指定了是否在单张图片预测后对目标进行截取
    #   count               指定了是否进行目标的计数
    #   crop、count仅在mode='predict'时有效
    # -------------------------------------------------------------------------#
    crop = False
    count = False
    # ----------------------------------------------------------------------------------------------------------#
    #   video_path          用于指定视频的路径，当video_path=0时表示检测摄像头
    #                       想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   video_save_path     表示视频保存的路径，当video_save_path=""时表示不保存
    #                       想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
    #   video_fps           用于保存的视频的fps
    #
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    # ----------------------------------------------------------------------------------------------------------#
    video_path = 0
    video_save_path = ""
    video_fps = 25.0
    # ----------------------------------------------------------------------------------------------------------#
    #   test_interval       用于指定测量fps的时候，图片检测的次数。理论上test_interval越大，fps越准确。
    #   fps_image_path      用于指定测试的fps图片
    #
    #   test_interval和fps_image_path仅在mode='fps'有效
    # ----------------------------------------------------------------------------------------------------------#
    test_interval = 100
    # fps_image_path = "img/street.jpg"
    fps_image_path = source
    # -------------------------------------------------------------------------#
    #   dir_origin_path     指定了用于检测的图片的文件夹路径
    #   dir_save_path       指定了检测完图片的保存路径
    #
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    # -------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path = "img_out/"

    if mode == "predict":
        '''
        1、该代码无法直接进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
        具体流程可以参考get_dr_txt.py，在get_dr_txt.py即实现了遍历还实现了目标信息的保存。
        2、如果想要进行检测完的图片的保存，利用r_image.save("img.jpg")即可保存，直接在predict.py里进行修改即可。
        3、如果想要获得预测框的坐标，可以进入frcnn.detect_image函数，在绘图部分读取top，left，bottom，right这四个值。
        4、如果想要利用预测框截取下目标，可以进入frcnn.detect_image函数，在绘图部分利用获取到的top，left，bottom，right这四个值
        在原图上利用矩阵的方式进行截取。
        5、如果想要在预测图上写额外的字，比如检测到的特定目标的数量，可以进入frcnn.detect_image函数，在绘图部分对predicted_class进行判断，
        比如判断if predicted_class == 'car': 即可判断当前目标是否为车，然后记录数量即可。利用draw.text即可写字。
        '''
        img = source
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            # continue
        else:
            t0 = time.time()
            r_image, defect_info = frcnn.detect_image(image, crop=crop, count=count, nms_iou_set=nms_iou,
                                                      confidence_set=confidence, thickness_set=thickness)
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
            # r_image.show()
            project = ROOT / 'static/output'
            save_dir = increment_path(Path(project) / "exp", exist_ok=False)  # increment run
            os.makedirs(save_dir)
            save_path = str(save_dir / os.path.basename(source))
            r_image.save(save_path)
            # 额外加入，（1）记录检测结果的图像路径（2）对labels文本信息进行处理
            # 将检测结果图像的路径保存到txt文件中
            with open('./inference/output/identify_image.txt', "w", encoding='UTF-8') as f:
                f.write(save_path)
            f.close()
            # save defects information
            image_name_prefix = os.path.basename(source).split(".")[0]
            os.makedirs(os.path.join(save_dir, "label"))
            print("defect_info", defect_info)
            with open(os.path.join(save_dir, "label/" + image_name_prefix + ".txt"), mode="w") as f:
                for d_i in defect_info:
                    for i in range(len(d_i)):
                        f.write(str(d_i[i]) + " ")
                    f.write("\n")
            f.close()
            # 将label文件夹下的txt文件生成相应的excel文件
            print("===============generate excel process is starting!=====================")
            TEXT_PATH = str(str(save_dir / 'labels'))
            all_content_dict = get_files_dict(TEXT_PATH=TEXT_PATH)
            files_max_PL_list, files_max_BX_list, files_max_TJ_list, files_max_CK_list, files_max_ZAW_list, files_max_CJ_list = find_max_proportion_lists(
                all_content_dict, defect_img=source)
            for i in tqdm(all_content_dict, desc="生成新excel文件ing"):
                if len(files_max_BX_list) != 0:
                    write_excel(files_max_BX_list, save_dir)
                if len(files_max_PL_list) != 0:
                    write_excel(files_max_PL_list, save_dir)
                if len(files_max_TJ_list) != 0:
                    write_excel(files_max_TJ_list, save_dir)
                if len(files_max_CK_list) != 0:
                    write_excel(files_max_CK_list, save_dir)
                if len(files_max_CJ_list) != 0:
                    write_excel(files_max_CJ_list, save_dir)
                if len(files_max_ZAW_list) != 0:
                    write_excel(files_max_ZAW_list, save_dir)
                    print("save_dir", save_dir)
                # time.sleep(0.05)
            print("===============excel file generation completed！=====================")

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        fps = 0.0
        while (True):
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            frame = np.array(frcnn.detect_image(frame))
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %.2f" % (fps))
            frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if video_save_path != "":
                out.write(frame)

            if c == 27:
                capture.release()
                break
        capture.release()
        out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = frcnn.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        # import os

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)
                r_image = frcnn.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95,
                             subsampling=0)

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")

# yolox
IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def run_yolox(demo="image",
              experiment_name="",
              name="yolox-m",
              path="",
              camid=0,
              save_result=True,
              exp_file="YOLOX/yolox_voc_s.py",
              ckpt=sys.path[0] + "/YOLOX/best_yolox_20230227.pth",
              # ckpt=sys.path[0] + "/fasterRCNN/model_data/best_epoch_weights_fasterrcnn_resnet50.pth",
              device="cpu",
              conf=0.3,
              nms=0.3,
              tsize=640,
              fp16=False,
              legacy=False,
              fuse=False,
              trt=False):
    exp = get_exp(exp_file, name)
    main(exp, demo, experiment_name, path, camid, save_result, ckpt, device, conf, nms, tsize, fp16, legacy, fuse, trt)

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

class Predictor(object):
    def __init__(
            self,
            model,
            exp,
            cls_names=COCO_CLASSES,
            trt_file=None,
            decoder=None,
            device="cpu",
            fp16=False,
            legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]
        # preprocessing: resize
        bboxes /= ratio  # 原始图像中缺陷bounding box的左上坐标和右下坐标
        cls = output[:, 6]  # 缺陷类别
        scores = output[:, 4] * output[:, 5]  # 置信度
        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        defect_info = []
        for o in output:
            d_i = []
            x = (o[0] + o[2]) / 2
            y = (o[1] + o[3]) / 2
            w = o[2] - o[0]
            h = o[3] - o[1]
            d_i.append(int(o[6]))
            d_i.append(int(x))
            d_i.append(int(y))
            d_i.append(int(w))
            d_i.append(int(h))
            d_i.append(int(img_info["width"]))
            d_i.append(int(img_info["height"]))
            d_i.append(float(o[4] * o[5]))
            defect_info.append(d_i)
        return vis_res, defect_info

def image_demo(predictor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    for image_name in files:
        outputs, img_info = predictor.inference(image_name)
        result_image, defect_info = predictor.visual(outputs[0], img_info, predictor.confthre)
        if save_result:
            '''
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            ),klk
            os.makedirs(save_folder, exist_ok=True)
            '''
            # 修改检测结果保存路径
            # save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            project = ROOT / 'static/output'
            save_dir = increment_path(Path(project) / "exp", exist_ok=False)  # increment run
            os.makedirs(save_dir)
            save_path = str(save_dir / os.path.basename(image_name))  # im.jpg
            # save_file_name=os.path.join("static/output/yolox",os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_path))
            cv2.imwrite(save_path, result_image)

            # save defects information
            image_name_prefix = os.path.basename(image_name).split(".")[0]
            os.makedirs(os.path.join(save_dir, "label"))
            with open(os.path.join(save_dir, "label/" + image_name_prefix + ".txt"), mode="w") as f:
                for d_i in defect_info:
                    for i in range(len(d_i)):
                        f.write(str(d_i[i]) + " ")
                    f.write("\n")
            f.close()
            # 将label文件夹下的txt文件生成相应的excel文件
            print("===============generate excel process is starting!=====================")
            TEXT_PATH = str(str(save_dir / 'labels'))
            all_content_dict = get_files_dict(TEXT_PATH=TEXT_PATH)
            files_max_PL_list, files_max_BX_list, files_max_TJ_list, files_max_CK_list, files_max_ZAW_list, files_max_CJ_list = find_max_proportion_lists(
                all_content_dict, defect_img=path)
            for i in tqdm(all_content_dict, desc="生成新excel文件ing"):
                if len(files_max_BX_list) != 0:
                    write_excel(files_max_BX_list, save_dir)
                if len(files_max_PL_list) != 0:
                    write_excel(files_max_PL_list, save_dir)
                if len(files_max_TJ_list) != 0:
                    write_excel(files_max_TJ_list, save_dir)
                if len(files_max_CK_list) != 0:
                    write_excel(files_max_CK_list, save_dir)
                if len(files_max_CJ_list) != 0:
                    write_excel(files_max_CJ_list, save_dir)
                if len(files_max_ZAW_list) != 0:
                    write_excel(files_max_ZAW_list, save_dir)
                    print("save_dir", save_dir)
                # time.sleep(0.05)
            print("===============excel file generation completed！=====================")

            # 额外加入，（1）记录检测结果的图像路径（2）对labels文本信息进行处理
            # 将检测结果图像的路径保存到txt文件中
            with open('./inference/output/identify_image.txt', "w", encoding='UTF-8') as f:
                f.write(save_path)
            f.close()
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

# video webcam
def imageflow_demo(predictor, vis_folder, current_time, path, demo, camid, save_result):
    cap = cv2.VideoCapture(path if demo == "video" else camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    if save_result:
        save_folder = os.path.join(
            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
        os.makedirs(save_folder, exist_ok=True)
        if demo == "video":
            save_path = os.path.join(save_folder, os.path.basename(path))
        else:
            save_path = os.path.join(save_folder, "camera.mp4")
        logger.info(f"video save_path is {save_path}")
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame)
            result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            if save_result:
                vid_writer.write(result_frame)
            else:
                cv2.namedWindow("yolox", cv2.WINDOW_NORMAL)
                cv2.imshow("yolox", result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break

def main(exp, demo, experiment_name, path, camid, save_result, ckpt, device, conf, nms, tsize, fp16, legacy, fuse, trt):
    if not experiment_name:
        experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, experiment_name)
    # os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    '''
    if save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)
    '''

    if trt:
        device = "gpu"

    # logger.info("Args: {}".format(args))

    if conf is not None:
        exp.test_conf = conf
    if nms is not None:
        exp.nmsthre = nms
    if tsize is not None:
        exp.test_size = (tsize, tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if device == "gpu":
        model.cuda()
        if fp16:
            model.half()  # to FP16
    model.eval()

    if not trt:
        if ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if trt:
        assert not fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(
        model, exp, COCO_CLASSES, trt_file, decoder,
        device, fp16, legacy,
    )
    current_time = time.localtime()
    if demo == "image":
        image_demo(predictor, vis_folder, path, current_time, save_result)
    elif demo == "video" or demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, path, demo, camid, save_result)

def load_model(model_type, weights_path=None):
    if model_type == 'yolov5' or model_type == 'yolov7':
        # Load your YOLOv5 or YOLOv7 model here
        model = ...  # Replace with your YOLOv5/v7 model loading code
    elif model_type == 'yolov8':
        if weights_path:
            model = YOLO(weights_path)
        else:
            raise ValueError("Weights path must be provided for YOLOv8 model")
    elif model_type == 'fasterrcnn':
        model = FRCNN()
    elif model_type == 'yolox':
        exp = get_exp("YOLOX/yolox_voc_s.py", "yolox-m")
        model = exp.get_model()
        ckpt = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model.eval()
    else:
        raise ValueError("Unsupported model type")
    return model

def detect_objects(model, image):
    # Convert the image to RGB if it has an alpha channel
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    if isinstance(model, YOLO):
        # Convert PIL image to numpy array
        image_np = np.array(image)

        # Perform detection
        results = model(image_np)

        # Process results
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = box.conf[0].item()
                class_id = box.cls[0].item()
                detections.append({
                    'class_id': class_id,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2]
                })

        # Convert the numpy array back to a PIL image
        result_image = Image.fromarray(image_np)
        return result_image, detections

    elif isinstance(model, FRCNN):
        # Perform detection with Faster R-CNN model
        r_image, defect_info = model.detect_image(image)
        detections = []
        for info in defect_info:
            class_id, x_center, y_center, width, height, img_width, img_height, confidence = info
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            detections.append({
                'class_id': class_id,
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2]
            })
        return r_image, detections

    elif isinstance(model, torch.nn.Module):  # YOLOX model
        # Perform detection with YOLOX model
        predictor = Predictor(model, get_exp("YOLOX/yolox_voc_s.py", "yolox-m"), COCO_CLASSES)
        outputs, img_info = predictor.inference(np.array(image))
        result_image, defect_info = predictor.visual(outputs[0], img_info, predictor.confthre)
        detections = []
        for info in defect_info:
            class_id, x_center, y_center, width, height, img_width, img_height, confidence = info
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            detections.append({
                'class_id': class_id,
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2]
            })
        return result_image, detections
    else:
        # Perform detection with EfficientNet model
        # Replace with your EfficientNet detection code
        results = ...
        return results