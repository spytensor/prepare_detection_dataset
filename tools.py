import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    
    # 判断是否OpenCV图片类型
    if (isinstance(img, np.ndarray)):  
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)

    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8"
    )
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)

    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def drawRect(im, bbox_lists, score=True):
    c = (0, 0, 255)
    for bb in bbox_lists:
        cls_name = bb['name']
        txt = '{}'.format(cls_name)
        if score:
            cls_score = float(str(bb['score']))
            txt = '{}_{:.5f}'.format(cls_name, cls_score)
        bbox = np.array(bb['bbox'], dtype=np.int32)
        xmin, ymin, xmax, ymax = bbox
        assert xmax > xmin
        assert ymax > ymin
        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), c, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
        cv2.rectangle(im, (bbox[0], bbox[1] - cat_size[1] - 2),
                (bbox[0] + cat_size[0], bbox[1] - 2), 
                c, -1)
        im = cv2ImgAddText(im, txt, bbox[0]+2, bbox[1]-cat_size[1] -2, (0, 0, 0), 10)
    return im


def parse_rec(filename, score=False):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        if score:
            obj_struct['score'] = obj.find('score').text
        obj_struct['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(float(bbox.find('xmin').text))
        ymin = int(float(bbox.find('ymin').text))
        xmax = int(float(bbox.find('xmax').text))
        ymax = int(float(bbox.find('ymax').text))
        # assert xmin >= xmax
        # assert ymin >= ymax
        if (xmin >= xmax) or (ymin >= ymax):
            print("Warning {} , bbox: xmin_{}, ymin_{}, xmax_{}, ymax_{}".format(
                os.path.basename(filename), xmin, ymin, xmax, ymax
            ))
        obj_struct['bbox'] = [xmin, ymin, xmax, ymax]
        objects.append(obj_struct)
    return objects

def pathExit(path):
    if isinstance(path, list):
        for ipath in path:
            if not os.path.exists(ipath):
                os.makedirs(ipath)
    else:
        if not os.path.exists(path):
            print("create new folder: {}".format(path))
            os.makedirs(path)