# -------------------------------------#
#       创建YOLO类
# -------------------------------------#
import colorsys
import os

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable

from nets.yolo3 import YoloBody
from utils.config import Config
from utils.utils import (DecodeBox, bbox_iou, letterbox_image,
                         non_max_suppression, yolo_correct_boxes)


# --------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和classes_path都需要修改！
#   如果出现shape不匹配，一定要注意
#   训练时的model_path和classes_path参数的修改
# --------------------------------------------#
class YOLO(object):
    _defaults = {
        "model_path": 'logs/Epoch100-Total_Loss25.2327-Val_Loss27.3005.pth',
        "classes_path": 'model_data/uod_classes.txt',
        "model_image_size": [416, 416, 3],
        "confidence": 0.5,
        "iou": 0.3,
        "cuda": True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化YOLO
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # CTK 使用本行后，_defaults中所包含的内容，直接使用self.【属性】即可进行调用
        self.class_names = self._get_class()  # 获得所有类别的名字
        self.config = Config
        #self.generate()     CTK 更改generate()方法调用位置

    # 这个函数主要实现generate()方法
    def change(self):
        self.generate()

    # ---------------------------------------------------#
    #   CTK  获得所有的分类 名字
    # ---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    # ---------------------------------------------------#
    #   生成模型
    # ---------------------------------------------------#
    def generate(self):
        # 因为 config里面虽然设定了类别的数量，但是有时候我们在classes_path那里更改了类别的数量，这里先进行更新类别的数量
        self.config["yolo"]["classes"] = len(self.class_names)  # 获得共有类别的数量 20类
        # ---------------------------------------------------#
        #   建立yolov3模型
        # ---------------------------------------------------#
        self.net = YoloBody(self.config)

        # ---------------------------------------------------#
        #   载入yolov3模型的权重
        # ---------------------------------------------------#
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(self.model_path, map_location=device)
        self.net.load_state_dict(state_dict)
        self.net = self.net.eval()

        if self.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.net = nn.DataParallel(self.net)  # CTK 采用数据并行的方式来处理，可以设置多卡并行处理
            self.net = self.net.cuda()

        # ---------------------------------------------------#
        #   建立三个特征层解码用的工具
        # ---------------------------------------------------#
        self.yolo_decodes = []
        for i in range(3):
            # yolo_decodes[0]的形式为（b,3*13*13, 4+1+20），对应最大的anchors
            # yolo_decodes[1]的形式为（b,3*26*26, 4+1+20），对应中等的anchors
            # yolo_decodes[2]的形式为（b,3*52*52, 4+1+20），对应最小的anchors
            self.yolo_decodes.append(DecodeBox(self.config["yolo"]["anchors"][i], self.config["yolo"]["classes"],
                                               (self.model_image_size[1], self.model_image_size[0])))

        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])

        # ---------------------------------------------------------#
        #   CTK  给图像增加灰条，实现不失真的resize
        # ---------------------------------------------------------#
        crop_img = np.array(letterbox_image(image, (self.model_image_size[1], self.model_image_size[0])))
        photo = np.array(crop_img, dtype=np.float32) / 255.0  # CTK 除以255代表归一化操作
        photo = np.transpose(photo, (2, 0, 1))
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        images = [photo]

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(images))  # 将图片从numpy转为tensor形式，才能够网络使用
            if self.cuda:
                images = images.cuda()

            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs = self.net(images)
            output_list = []
            for i in range(3):
                # output_list[0]的形式为（bs,3*13*13, 4+1+20）
                # output_list[1]的形式为（bs,3*26*26, 4+1+20）
                # output_list[2]的形式为（bs,3*52*52, 4+1+20）
                output_list.append(self.yolo_decodes[i](outputs[i]))

            # ---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            # ---------------------------------------------------------#
            output = torch.cat(output_list, 1)  # output的形状为 （b,3*13*13 + 3*26*26 + 3*52*52, 4+1+20）
            batch_detections = non_max_suppression(output, self.config["yolo"]["classes"],
                                                   conf_thres=self.confidence,
                                                   nms_thres=self.iou)

            # ---------------------------------------------------------#
            #   如果没有检测出物体，返回原图
            # ---------------------------------------------------------#
            try:
                batch_detections = batch_detections[0].cpu().numpy()
            except:
                return image

            # ---------------------------------------------------------#
            #   对预测框进行得分筛选
            # ---------------------------------------------------------#
            top_index = batch_detections[:, 4] * batch_detections[:, 5] > self.confidence
            top_conf = batch_detections[top_index, 4] * batch_detections[top_index, 5]
            top_label = np.array(batch_detections[top_index, -1], np.int32)
            top_bboxes = np.array(batch_detections[top_index, :4])
            top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:, 0], -1), np.expand_dims(
                top_bboxes[:, 1], -1), np.expand_dims(top_bboxes[:, 2], -1), np.expand_dims(top_bboxes[:, 3], -1)

            # -----------------------------------------------------------------#
            #   在图像传入网络预测前会进行letterbox_image给图像周围添加灰条
            #   因此生成的top_bboxes是相对于有灰条的图像的
            #   我们需要对其进行修改，去除灰条的部分。
            # -----------------------------------------------------------------#
            # CTK去掉灰条
            boxes = yolo_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                       np.array([self.model_image_size[0], self.model_image_size[1]]), image_shape)

        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.model_image_size[0], 1)

        # 将类别、得分画在图片上面
        for i, c in enumerate(top_label):

            predicted_class = self.class_names[c]
            score = top_conf[i]

            top, left, bottom, right = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            # 画框框
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
                    outline=self.colors[self.class_names.index(predicted_class)])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[self.class_names.index(predicted_class)])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image

