# -*- coding: utf-8 -*-
# Created on 8月-31-21 10:21
# @site: https://github.com/moyans
# @author: moyan
import glob
import os
import moyan
import pickle
import shutil
from tqdm import tqdm
import xml.etree.ElementTree as ET
from os import listdir, getcwd
from os.path import join

class VOC2YOLO:
    def __init__(self, classes=['person', 'car']):
        self.classes = classes

    def convert(self, size, box):
        dw = 1./(size[0])
        dh = 1./(size[1])
        x = (box[0] + box[1])/2.0 - 1
        y = (box[2] + box[3])/2.0 - 1
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x*dw
        w = w*dw
        y = y*dh
        h = h*dh
        return (x,y,w,h)

    def convert_annotation(self, xml_file, save_txt_path):

        out_file = open(save_txt_path, 'w')
        tree = ET.parse(xml_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in self.classes or int(difficult)==1:
                continue
            cls_id = self.classes.index(cls)
            xmlbox = obj.find('bndbox')
            xmin = float(xmlbox.find('xmin').text) 
            xmax = float(xmlbox.find('xmax').text) 
            ymin = float(xmlbox.find('ymin').text)
            ymax = float(xmlbox.find('ymax').text)
            if (xmin >= xmax) or (ymin >= ymax):
                print("Warning {} , bbox: xmin_{}, ymin_{}, xmax_{}, ymax_{}".format(
                    os.path.basename(xml_file), xmin, ymin, xmax, ymax
                ))
                continue
            b = (xmin, xmax, ymin, ymax)
            
            bb = self.convert((w,h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(round(a, 5)) for a in bb]) + '\n')

def main():

    classes = ['aeroplane', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant',
                'sheep', 'sofa', 'train', 'tvmonitor']
    voc2yolo = VOC2YOLO(classes=classes)
    
    VOCDir = r'D:\Dataset\VOC\VOCdevkit\VOC2007'
    JPEGImagesDir = os.path.join(VOCDir, 'JPEGImages')
    AnnotationsDir = os.path.join(VOCDir, 'Annotations')
    MainTXTDir = os.path.join(VOCDir, 'ImageSets', 'Main')

    SAVE_DIR = os.path.join(VOCDir, 'yolo_style')
    moyan.pathExit(SAVE_DIR)
    
    # select sample or read Main/txt get list
    sp_list = ['trainval', 'test']

    for sp_name in sp_list:

        print('prepare convert ', sp_name)
        save_images_dir = os.path.join(SAVE_DIR, sp_name, "images")
        save_labels_dir = os.path.join(SAVE_DIR, sp_name, "labels")
        moyan.pathExit(save_images_dir)
        moyan.pathExit(save_labels_dir)
        
        txt_path = os.path.join(MainTXTDir, sp_name+'.txt')
        assert os.path.exists(txt_path)
        img_name_list = moyan.readTxt2Lines(txt_path)
        for idx, name in tqdm(enumerate(img_name_list)):
            img_names = name + '.jpg'
            img_path = os.path.join(JPEGImagesDir, img_names)
            xml_path = os.path.join(AnnotationsDir, name+'.xml')
            assert os.path.exists(img_path)
            assert os.path.exists(xml_path)
            output_label_path = os.path.join(save_labels_dir, name+'.txt')
            output_image_path = os.path.join(save_images_dir, img_names)
            shutil.copyfile(img_path, output_image_path)
            voc2yolo.convert_annotation(xml_path, output_label_path)

if __name__ == '__main__':
    main()