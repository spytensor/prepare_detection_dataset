# -*- coding: utf-8 -*-
# Created on 9月-01-21 13:38
# @site: https://github.com/moyans
# @author: moyan
import os
import cv2
import sys
import numpy as np
import moyan
from tqdm import tqdm
from tools import drawRect, parse_rec

def main():

    VOCDir = r'D:\Dataset\VOC\VOCdevkit\VOC2007'
    JPEGImagesDir = os.path.join(VOCDir, 'JPEGImages')
    AnnotationsDir = os.path.join(VOCDir, 'Annotations')
    SAVE_VIS_DIR = os.path.join(VOCDir, 'vis_box')
    moyan.pathExit(SAVE_VIS_DIR)
    
    # select sample or read Main/txt get list
    vis_name_list = os.listdir(AnnotationsDir)
    vis_name_list = vis_name_list[:10]
    vis_name_list = [os.path.splitext(names)[0] for names in vis_name_list]

    for idx, name in tqdm(enumerate(vis_name_list)):
        img_names = name + '.jpg'
        img_path = os.path.join(JPEGImagesDir, img_names)
        xml_path = os.path.join(AnnotationsDir, name+'.xml')
        assert os.path.exists(img_path)
        assert os.path.exists(xml_path)
        bbox_list = parse_rec(xml_path)
        im = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)
        im_vis = drawRect(im, bbox_list, score=False)
        save_path = os.path.join(SAVE_VIS_DIR, img_names)
        cv2.imwrite(save_path, im_vis)

if __name__ == '__main__':
    main()