import os
import argparse
import xml.etree.ElementTree as ET

# num1 = convert_voc_annotation(data_path = "/home/yang/test/VOC/train/VOCdevkit/VOC2007', data_type = 'trainval', anno_path = "./data/dataset/voc_train.txt", use_difficult_bbox = False)
# arg_param
        # data_path: jpeg 数据路径
        # data_type：train_val类型或者是test类型
        # anno_path：将xml文件转为txt文件，所保存的位置
def convert_voc_annotation(data_path, data_type, anno_path, use_difficult_bbox=True):

    # classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    #            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
    #            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
    #            'train', 'tvmonitor']
    classes = ['echinus','holothurian','starfish','scallop','waterweeds']

    #classes = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
    #        'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

    # 得到训练图片的id(这里用六位数字进行表示，img_inds_file是一个txt文件)
    img_inds_file = os.path.join(data_path, 'ImageSets', 'Main', data_type + '.txt')

    with open(img_inds_file, 'r') as f:
        txt = f.readlines()
        # strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）
        image_inds = [line.strip() for line in txt]

    with open(anno_path, 'a') as f:
        for image_ind in image_inds:
            # 图片的绝对路径
            image_path = os.path.join(data_path, 'JPEGImages', image_ind + '.jpg')
            annotation = image_path
            # annotation的绝对路径
            label_path = os.path.join(data_path, 'Annotations', image_ind + '.xml')
            root = ET.parse(label_path).getroot()
            # 找到带有‘object’的物体
            objects = root.findall('object')
            for obj in objects:

                # ichen 2020_04_29 山东青岛 注释下面三行

                #difficult = obj.find('difficult').text.strip()
                #if (not use_difficult_bbox) and(int(difficult) == 1):
                #   continue

                bbox = obj.find('bndbox')

                # 依据上面的classes列表，找到物体类别所对应的编号（class_ind）
                class_ind = classes.index(obj.find('name').text.lower().strip())
                xmin = bbox.find('xmin').text.strip()
                xmax = bbox.find('xmax').text.strip()
                ymin = bbox.find('ymin').text.strip()
                ymax = bbox.find('ymax').text.strip()
                # str().join()表示：join是联合函数，将（）内按指定字符连接（这里用','进行连接）
                annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_ind)])
            print(annotation)
            f.write(annotation + "\n")
    return len(image_inds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    parser.add_argument("--train_annotation", default="/home/tingkai/1workspace/laptop/4_01_yolov3_pytorch/dataset/train.txt")  # 将所有图片读取到一个txt的位置（保存的位置）
    parser.add_argument("--test_annotation",  default="/home/tingkai/1workspace/laptop/4_01_yolov3_pytorch/dataset/test.txt")   # 同上
    flags = parser.parse_args()

    if os.path.exists(flags.train_annotation):os.remove(flags.train_annotation)   # 如果之前存在，则删除之前生成的文件（voc_train.txt）
    if os.path.exists(flags.test_annotation):os.remove(flags.test_annotation)     # 如果之前存在，则删除之前生成的文件（voc_test.txt）

    num1 = convert_voc_annotation('/home/tingkai/1workspace/common/data_uod/jpg_and_annotation_difficult/train', 'trainval', flags.train_annotation, False)  # trainval图片索引位置
    num2 = convert_voc_annotation('/home/tingkai/1workspace/common/data_uod/jpg_and_annotation_difficult/test',  'test', flags.test_annotation, False)       # test 图片索引位置
    print('=> The number of image for train is: %d\tThe number of image for test is:%d' %(num1, num2))


