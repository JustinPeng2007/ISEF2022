import pyttsx3
engine = pyttsx3.init()
import os, sys
import cv2
import pyrealsense2 as rs
import numpy as np
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())

import socket
UDP_IP = "192.168.1.205"
UDP_PORT = 10000
cond = True
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP

from http.server import BaseHTTPRequestHandler, HTTPServer
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync

@torch.no_grad()
def run():
    engine.setProperty("rate", 500)
    lastoutput = ' '
    global thread1, thread2
    weights = 'yolov5s.pt'  # model.pt path(s)
    imgsz = 640  # inference size (pixels)
    conf_thres = 0.25  # confidence threshold
    iou_thres = 0.45  # NMS IOU threshold
    max_det = 10  # maximum detections per image
    classes = None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms = False  # class-agnostic NMS
    augment = False  # augmented inference
    visualize = False  # visualize features
    line_thickness = 3  # bounding box thickness (pixels)
    hide_labels = False  # hide labels
    hide_conf = False  # hide confidences
    half = False  # use FP16 half-precision inference
    stride = 32
    device_num = ''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img = False  # show results
    save_crop = False  # save cropped prediction boxes
    nosave = False  # do not save images/videos
    update = False  # update all models
    name = 'exp'  # save results to project/name

    set_logging()
    device = select_device(device_num)
    half &= device.type != 'cpu'

    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)
    names = model.module.names if hasattr(model, 'module') else model.names
    if half:
        model.half()

    classify = False
    if classify:
        modelc = load_classifier(name='resnet50', n=2)
        modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()

    view_img = check_imshow()
    cudnn.benchmark = True

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)
    while (True):
        start = time.time()
        x = 0
        doublecheck = []
        while x < 2:
            t0 = time.time()

            frames = pipeline.wait_for_frames()

            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not depth_frame or not color_frame:
                continue

            img = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            s = np.stack([letterbox(x, imgsz, stride=stride)[0].shape for x in img], 0)  # shapes
            rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
            if not rect:
                print(
                    'WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

            img0 = img.copy()
            img = img[np.newaxis, :, :, :]
            img = np.stack(img, 0)
            img = img[..., ::-1].transpose((0, 3, 1, 2))
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            t1 = time_sync()
            pred = model(img, augment=augment,
                         visualize=increment_path(save_dir / 'features', mkdir=True) if visualize else False)[0]

            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            t2 = time_sync()

            if classify:
                pred = apply_classifier(pred, modelc, img, img0)

            for i, det in enumerate(pred):  # detections per image
                s = f'{i}: '

                obj = []
                annotator = Annotator(img0, line_width=line_thickness, example=str(names))
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                        center1 = round((c1[0] + c2[0]) / 2)
                        center2 = round((c1[1] + c2[1]) / 2)
                        center_point = center1, center2
                        circle = cv2.circle(img0, center_point, 5, (0, 255, 0), 2)
                        test_coord = cv2.putText(img0, str(center_point), center_point, cv2.FONT_HERSHEY_PLAIN, 2,
                                                 (0, 0, 255))
                        obj.append(center_point)
                    numofObjects = 0
                    counter = 0
                    for c in det[:, -1]:
                        n = (det[:, -1] == c).sum()
                        dist = depth_frame.get_distance(obj[numofObjects][0], obj[numofObjects][1])
                        dist = int(dist * 100)
                        s += f"{obj[numofObjects]}d{dist} ."
                        numofObjects = numofObjects + 1

                    h = 0
                    xcounter = []
                    depthleftcounter = []
                    depthrightcounter = []
                    cutstring = s.split(".")
                    ObjDis = 0
                    pattern = ";(.*?)."
                    direction = []
                    for y in range(numofObjects):
                        str1 = cutstring[y]
                        d1 = str1.find("d") + 1
                        d2 = str1.find(".")
                        ObjDis = str1[d1:d2]
                        ObjDis = int(ObjDis)
                        b1 = str1.find("(") + len("(")
                        b2 = str1.find(")")
                        strxy = str1[b1:b2]
                        y1 = strxy.find(",")
                        strx = strxy[:y1]
                        xvalue = int(strx)
                        if (ObjDis < 65 and ObjDis != 0):
                            direction.append("stop")
                            break
                        else:
                            if (xvalue < 400 and xvalue > 240):
                                leftcounter = 0
                                rightcounter = 0
                                for z in range(numofObjects):
                                    str1 = cutstring[z]
                                    b3 = str1.find("(") + len(")")
                                    b4 = str1.find(")")
                                    strxy1 = str1[b3:b4]
                                    y2 = strxy1.find(",")
                                    strx1 = strxy1[:y2]
                                    xvalue = int(strx1)
                                    d3 = str1.find("d") + 1
                                    d4 = str1.find(".")
                                    depthvalue = str1[d3:d4]
                                    depthvalue = int(depthvalue)
                                    if (xvalue < 320):
                                        leftcounter = leftcounter + 1
                                        depthleftcounter.append(int(depthvalue))
                                    if (xvalue > 320):
                                        rightcounter = rightcounter + 1
                                        depthrightcounter.append(int(depthvalue))

                                if (leftcounter > rightcounter):
                                    direction.append("right")
                                if (rightcounter > leftcounter):
                                    direction.append("left")
                                if (leftcounter == rightcounter):
                                    minleft = 0
                                    minright = 0
                                    for r in range(0, len(depthleftcounter)):
                                        if (depthleftcounter[r] < minleft):
                                            minleft = depthleftcounter[r]
                                    for p in range(0, len(depthrightcounter)):
                                        if (depthrightcounter[p] < minright):
                                            minright = depthrightcounter[p]
                                    if (minleft < minright):
                                        direction.append("right")
                                    if (minright < minleft):
                                        direction.append("left")
                            else:
                                direction.append("forward")
                        h = h + 1

                    stop = 0
                    left = 0
                    right = 0
                    forward = 0

                    num = 0
                    for u in range(len(direction)):
                        if (direction[num] == 'stop'):
                            stop += 1
                        else:
                            if (direction[num] == 'left'):
                                left += 1
                            else:
                                if (direction[num] == 'right'):
                                    right += 1
                                else:
                                    forward += 1
                        num += 1
                    if (stop > 0):
                        doublecheck.append("stop")
                    else:
                        if (left > 0):
                            doublecheck.append("left")
                        else:
                            if (right > 0):
                                doublecheck.append("right")
                            else:
                                doublecheck.append("forward")
                    c = int(cls)
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    x += 1

        num1 = 0
        stop1 = 0
        left1 = 0
        right1 = 0
        forward1 = 0
        for p in range(len(doublecheck)):
            if (doublecheck[num1] == 'stop'):
                stop1 += 1
            else:
                if (doublecheck[num1] == 'left'):
                    left1 += 1
                else:
                    if (doublecheck[num1] == 'right'):
                        right1 += 1
                    else:
                        forward1 += 1
            num1 += 1
        if (stop1 == 2):
            print("stop")
            bstop = bytes("Y0:X" + str(0), encoding='utf-8')
            sock.sendto(bstop, (UDP_IP, UDP_PORT))
            engine.say("stop")
            engine.runAndWait()
        else:
            if (left1 == 2):
                print("left")
                bstop = bytes("Y0:X" + str(80), encoding='utf-8')
                sock.sendto(bstop, (UDP_IP, UDP_PORT))
                engine.say("left")
                engine.runAndWait()
            else:
                if (right1 == 2):
                    print("right")
                    bstop = bytes("Y0:X" + str(-80), encoding='utf-8')
                    sock.sendto(bstop, (UDP_IP, UDP_PORT))
                    engine.say("right")
                    engine.runAndWait()
                else:
                    if (forward1 == 2):
                        print("clear")
                        bstop = bytes("Y0:X" + str(0), encoding='utf-8')
                        sock.sendto(bstop, (UDP_IP, UDP_PORT))
        end = time.time()
        totaltime = end - start
        roundedtime = round(totaltime, 2)
        print(roundedtime)

        x == 0

        cv2.imshow("IMAGE", img0)

        global stop_threads
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    run()
