import colorsys
import copy
import math
import os
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torch.nn import functional as F

from nets.frcnn import FasterRCNN
from utils.utils import DecodeBox, get_new_img_size, loc2bbox, nms

class FRCNN(object):
    _defaults = {
        "model_path"    : 'logs/RESNet101+ERCA.pth',
        "classes_path"  : 'voc_classes.txt',
        "confidence"    : 0.5,
        "iou"           : 0.3,
        "backbone"      : "resnet101",#resnet101,resnet50,resnet34
        "cuda"          : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.generate()

        self.mean = torch.Tensor([0, 0, 0, 0]).repeat(self.num_classes+1)[None]
        self.std = torch.Tensor([0.1, 0.1, 0.2, 0.2]).repeat(self.num_classes+1)[None]
        if self.cuda:
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()
            
        self.decodebox = DecodeBox(self.std, self.mean, self.num_classes)

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names


    def generate(self):

        self.num_classes = len(self.class_names)


        self.model = FasterRCNN(self.num_classes,"predict",backbone=self.backbone).eval()
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(self.model_path, map_location=device)
        self.model.load_state_dict(state_dict)
        
        if self.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"]="0" 
            # self.model = nn.DataParallel(self.model)
            self.model = self.model.cuda()

        print('{} model, anchors, and classes loaded.'.format(self.model_path))


        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
    


    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        old_width, old_height = image_shape[1], image_shape[0]
        old_image = copy.deepcopy(image)

        width,height = get_new_img_size(old_width, old_height)
        image = image.resize([width,height], Image.BICUBIC)

        photo = np.transpose(np.array(image,dtype = np.float32)/255, (2, 0, 1))

        with torch.no_grad():
            images = torch.from_numpy(np.asarray([photo]))
            if self.cuda:
                images = images.cuda()

            roi_cls_locs, roi_scores, rois, _ = self.model(images)

            outputs = self.decodebox.forward(roi_cls_locs[0], roi_scores[0], rois, height = height, width = width, nms_iou = self.iou, score_thresh = self.confidence)

            if len(outputs)==0:
                return None
            outputs = np.array(outputs)
            bbox = outputs[:,:4]
            label = outputs[:, 4]
            conf = outputs[:, 5]

            bbox[:, 0::2] = (bbox[:, 0::2]) / width * old_width
            bbox[:, 1::2] = (bbox[:, 1::2]) / height * old_height

        font = ImageFont.truetype(font='model_data/simhei.ttf',size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

        thickness = max((np.shape(old_image)[0] + np.shape(old_image)[1]) // old_width * 2, 1)
                
        image = old_image
        bboxlist=[]
        for i, c in enumerate(label):
            predicted_class = self.class_names[int(c)]
            score = conf[i]
            Y=0
            if score>0.81:
                left, top, right, bottom = bbox[i]
                top = top - 5
                left = left - 5
                bottom = bottom + 5
                right = right + 5

                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
                right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))
                bboxlist.append(np.array([left,top,right,bottom]))


                label = '{} {:.2f}'.format(predicted_class, score)
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)
                label = label.encode('utf-8')
                print(label, top, left, bottom, right)

                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                for i in range(thickness):
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline=self.colors[int(c)])
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=self.colors[int(c)])
                draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
                Y = float(label.split()[1])
                del draw
        if max(conf)>0.5:
            return image
        else:
            return  None
            # ---------------------------------------------------#