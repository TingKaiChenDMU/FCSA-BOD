#-------------------------------------#
#       创建YOLO类
#-------------------------------------#
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
import os
os.environ['DISPLAY'] = ':0'

#--------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和classes_path都需要修改！
#   如果出现shape不匹配，一定要注意
#   训练时的model_path和classes_path参数的修改
#--------------------------------------------#
class YOLO(object):
    print('进入YOLO啦')
    _defaults = {
        #"model_path"        : 'logs/Epoch100-Total_Loss25.2327-Val_Loss27.3005.pth',
        "model_path"        : 'logs/yolo_weights.pth',
        #"classes_path"      : 'model_data/uod_classes.txt',
        "classes_path"      : 'model_data/coco_classes.txt',
        "model_image_size"  : (416, 416, 3),
        "confidence"        : 0.5,
        "iou"               : 0.3,
        "cuda"              : True
    }
    print(_defaults)
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化YOLO
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        print('进入YOLO的init函数啦')
        self.__dict__.update(self._defaults)  # CTK 使用本行后，_defaults中所包含的内容，直接使用self.【属性】即可进行调用
        self.class_names = self._get_class()   # 获得所有类别的名字
        self.config = Config
        self.generate()


    #---------------------------------------------------#
    #   CTK  获得所有的分类 名字
    #---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    #---------------------------------------------------#
    #   生成模型
    #---------------------------------------------------#
    def generate(self):
        # 因为 config里面虽然设定了类别的数量，但是有时候我们在classes_path那里更改了类别的数量，这里先进行更新类别的数量
        self.config["yolo"]["classes"] = len(self.class_names)  # 获得共有类别的数量 20类
        #---------------------------------------------------#
        #   建立yolov3模型
        #---------------------------------------------------#
        self.net = YoloBody(self.config)

        #---------------------------------------------------#
        #   载入yolov3模型的权重
        #---------------------------------------------------#
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(self.model_path, map_location=device)
        self.net.load_state_dict(state_dict)
        self.net = self.net.eval()

        if self.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.net = nn.DataParallel(self.net)  # CTK 采用数据并行的方式来处理，可以设置多卡并行处理
            self.net = self.net.cuda()

        #---------------------------------------------------#
        #   建立三个特征层解码用的工具
        #---------------------------------------------------#
        self.yolo_decodes = []
        for i in range(3):
            # yolo_decodes[0]的形式为（b,3*13*13, 4+1+20），对应最大的anchors
            # yolo_decodes[1]的形式为（b,3*26*26, 4+1+20），对应中等的anchors
            # yolo_decodes[2]的形式为（b,3*52*52, 4+1+20），对应最小的anchors
            self.yolo_decodes.append(DecodeBox(self.config["yolo"]["anchors"][i], self.config["yolo"]["classes"], (self.model_image_size[1], self.model_image_size[0])))

        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])

        #---------------------------------------------------------#
        #   CTK  给图像增加灰条，实现不失真的resize
        #---------------------------------------------------------#
        crop_img = np.array(letterbox_image(image, (self.model_image_size[1],self.model_image_size[0])))
        photo = np.array(crop_img,dtype = np.float32) / 255.0  # CTK 除以255代表归一化操作
        photo = np.transpose(photo, (2, 0, 1))
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        images = [photo]

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(images))  # 将图片从numpy转为tensor形式，才能够网络使用
            if self.cuda:
                images = images.cuda()

            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            output_list = []
            for i in range(3):
                # output_list[0]的形式为（bs,3*13*13, 4+1+20）
                # output_list[1]的形式为（bs,3*26*26, 4+1+20）
                # output_list[2]的形式为（bs,3*52*52, 4+1+20）
                output_list.append(self.yolo_decodes[i](outputs[i]))
        return output_list



def show_CAM(image_path, feature_maps, class_id, all_ids=85,show_one_layer=True):
    '''
    feaure_maps: this is a list [tensor, tensor, tensor], tensor shape is [1,3,N, N, all_ids]
    '''
    SHOW_NAME = ['score', 'class', 'class_score']
    img_ori = cv2.imread(image_path)
    print(feature_maps[0].shape)
    layers0 = feature_maps[0].reshape([-1, all_ids])
    layers1 = feature_maps[1].reshape([-1, all_ids])
    layers2 = feature_maps[2].reshape([-1, all_ids])
    layers = torch.cat([layers0, layers1, layers2],0)
    score_max_v = layers[:,4].max()
    score_min_v = layers[:,4].min()
    class_max_v = layers[:, 5 + class_id].max()
    class_min_v = layers[:, 5 + class_id].min()
    all_ret = [[],[],[]]
    for j in range(3): #layers
        layer_one = feature_maps[j]

        anchors_score_max = layer_one[0, ..., 4].max(0)[0]

        anchors_class_max = layer_one[0, ..., 5 + class_id].max(0)[0]

        scores = ((anchors_score_max - score_min_v)/
                  (score_max_v - score_min_v))

        classes = ((anchors_class_max - class_min_v) /
                   (class_max_v - class_min_v))

        layer_one_list = []
        layer_one_list.append(scores)
        layer_one_list.append(classes)
        layer_one_list.append(scores*classes)

        for idx, one in enumerate(layer_one_list):
            layer_one = one.cpu().numpy()
            ret = ((layer_one - layer_one.min()) / (layer_one.max() - layer_one.min())) * 255
            ret = ret.astype(np.uint8)
            gray = ret[:,:, None]
            ret = cv2.applyColorMap(gray,cv2.COLORMAP_JET)

            if not show_one_layer:
                all_ret[j].append(cv2.resize(ret,(img_ori.shape[1],img_ori.shape[0])).copy())

            else:
                ret = cv2.resize(ret, (img_ori.shape[1], img_ori.shape[0]))
                show = ret * 0.8 + img_ori * 0.2
                show = show.astype(np.uint8)
                #cv2.imshow(f"one_{SHOW_NAME[idx]}",show)
                print('here 1')
                cv2.imwrite('./cam_result/head' + str(j)+'layer'+str(idx)+SHOW_NAME[idx]+'.jpg', show)
                print('here 2')
            # if show_one_layer:
            #     cv2.waitKey(0)
        if not show_one_layer:
            for idx, one_type in enumerate(all_ret):
                print( len(one_type))
                map_show = one_type[0] /3 + one_type[1]/3 +one_type[2]/3
                show = map_show * 0.8 + img_ori *0.2
                show = show.astype(np.uint8)
                map_show = map_show.astype(np.uint8)
                #cv2.imshow(f"all_{SHOW_NAME[idx]}",show)

                cv2.imwrite('./cam_results/head_cont'+str(idx)+SHOW_NAME[idx]+'.jpg',show)
            cv2.waitKey(0)

ret = []
stride = [13,26,52]
yolo = YOLO()
path = 'img/street.jpg'
image = Image.open(path)
output_list = yolo.detect_image(image)
#print(output_list)
for i,f in enumerate(output_list):
    #ret.append(f.reshape(1,3,stride[i],stride[i],10))
    ret.append(f.reshape(1,3,stride[i],stride[i],85))
show_CAM(path,ret,1)

