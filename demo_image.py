#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import argparse
import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
from matplotlib import pyplot as plt

warnings.filterwarnings('ignore')

def main(yolo,read_type):

    print(args.test_datasets)

    bgr_image = cv2.imread(args.test_datasets)

    # from matplotlib import pyplot as plt
    tmp_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(tmp_image)
    # plt.show()

    image = Image.fromarray(tmp_image)
    # image = Image.fromarray(bgr_image)

    time3=time.time()
    boxs = yolo.detect_image(image)
    time4=time.time()
    print('detect cost is',time4-time3)
    print("box_num",len(boxs))


    plt.figure(figsize=(10, 10))
    plt.imshow(tmp_image)
    # plt.show()

    plt.figure(figsize=(10, 10))
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    plt.imshow(tmp_image)  # plot the image for matplotlib
    currentAxis = plt.gca()

    for i in range(len(boxs)):
        display_txt = 'person: (1.1)'
        coords = (int(boxs[i][0]), int(boxs[i][1])), int(boxs[i][2]), int(boxs[i][3])
        # coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
        color = colors[i]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        # currentAxis.text(int(boxs[i][0]), int(boxs[i][1]), display_txt, bbox={'facecolor': color, 'alpha': 0.5})

    img_name = (args.test_datasets).split('/')[-1]
    aaa = os.path.join('/home/ydwu/project4/deep_sort_yolov3/output/', img_name)
    plt.savefig(aaa)

    # plt.imshow(tmp_image, 'brg')


############################# error  why???
    # for i in range(len(boxs)):
    #     cv2.rectangle(tmp_image,(int(boxs[i][0]), int(boxs[i][1])), (int(boxs[i][2]), int(boxs[i][3])),(255,0,0), 2)
        # cv2.rectangle(bgr_image, (int(boxs[i][0]), int(boxs[i][1])), (int(boxs[i][2]), int(boxs[i][3])), (255, 0, 0), 2)
        # print(boxs[i][0])
        # print(boxs[i][1])
        # print(boxs[i][2])
        # print(boxs[i][3])


    # cv2.imshow('tmp_image', tmp_image)
    # cv2.imshow('bgr_image', bgr_image)

    # cv2.waitKey(0)





######################paraters######################
def parse_args():
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--read_type", help="camera or video",
        default='camera', required=False)
    parser.add_argument('--test_datasets', default='/home/ydwu/datasets/00-pictures/bank_datasets/177159764.jpg',
                        type=str, help='test image file path')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(YOLO(),args.read_type)
