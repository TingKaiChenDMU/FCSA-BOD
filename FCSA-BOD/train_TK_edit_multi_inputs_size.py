#-------------------------------------#
#       对数据集进行训练
#-------------------------------------#
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from nets.yolo3 import YoloBody
from nets.yolo_training import Generator, YOLOLoss
from utils.config import Config
from utils.dataloader import YoloDataset, yolo_dataset_collate
import random
from datetime import datetime
from tensorboardX import SummaryWriter

# 准备数量为batch_size、不同分辨率图片
def prepare_train_data(Batch_size, image_width, image_height):
    # 建立loss函数

    yolo_losses = []
    for i in range(3):
        # np.reshape(Config["yolo"]["anchors"],[-1,2])将所有的anchor boxes转换为形状（9，2）
        # Config["yolo"]["classes"]为 5
        # (image_width, image_height)为（416,416）
        # Cuda是否为启用加速
        # normalize是否对loss的输出值进行归一化，也就是说是否除以batch。
        yolo_losses.append(YOLOLoss(np.reshape(Config["yolo"]["anchors"], [-1, 2]),
                                    Config["yolo"]["classes"],
                                    (image_width, image_height),
                                    Cuda,
                                    normalize))

    # 准备数据
    # 利用pytorch中DataLoader可以帮助我们生成指定Batch大小的数据，并提供打乱数据，快速加载数据的功能。
    if Use_Data_Loader:
        train_dataset = YoloDataset(lines[:num_train], (image_height, image_width), True)
        val_dataset = YoloDataset(lines[num_train:], (image_height, image_width), False)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=8, pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=8, pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate)

    # 通过手动设置的方式，进行数据扩充，主要是包括图像翻转、图像裁剪等。
    else:
        gen = Generator(Batch_size, lines[:num_train],
                        (image_height, image_width)).generate(True)
        gen_val = Generator(Batch_size, lines[num_train:],
                            (image_height, image_width)).generate(False)

    # 将一次epoch分为几次训练 和 验证
    epoch_size = num_train // Batch_size
    epoch_size_val = num_val // Batch_size

    # 返回损失、遍历完一次“训练数据”所需的batch、遍历完一次“验证数据”所需的batch、训练数据，验证数据
    return yolo_losses, epoch_size, epoch_size_val, gen, gen_val



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_ont_epoch(net, yolo_losses, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, cuda):
    # CTK  参数含义：
    # net： 网络架构
    # yolo_losses： 训练损失
    # epoch：当前是第几次迭代
    # epoch_size：将一次epoch分为几次训练
    # epoch_size_val：将一次epoch分为几次验证
    # gen：训练类型数据集
    # gen_val：验证类型数据集
    # Freeze_Epoch：一共有几个epoch
    # Cuda：是否使用cuda进行加速

    total_loss = 0
    val_loss = 0

    # 训练的时候设置为train模式，启用 BatchNormalization 和 Dropout
    net.train()
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break

            # CTK 2021-07-20 images的shape是（bs, 3, 416, 416）
            # CTK 2021-07-20 targets是一个batch的目标位置信息（每张图片中所包含物体 左上角 和 右下角 坐标信息依照原始图片大小（随便举个例子，例如，325*856，总之，这里并不是416*416）被归一化0-1之间
            images, targets = batch[0], batch[1]


            with torch.no_grad():
                if cuda:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
                else:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]

            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   前向传播
            #----------------------#
            outputs = net(images)
            losses = []
            num_pos_all = 0
            #----------------------#
            #   计算损失
            #----------------------#
            for i in range(3):
                loss_item, num_pos = yolo_losses[i](outputs[i], targets)
                losses.append(loss_item)
                num_pos_all += num_pos

            loss = sum(losses) / num_pos_all
            #----------------------#
            #   反向传播
            #----------------------#
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), # 这里求得平均loss
                                'lr'        : get_lr(optimizer)})

            pbar.update(1)

            ##############################################################
            # CTK 保存event文件（损失函数）的文件（在前面定义了writer的相关属性）
            writer.add_scalar('loss/total_loss', total_loss / (iteration + 1), iteration)


    net.eval() # 评估的时候设置为eval模式，不启用 BatchNormalization 和 Dropout
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images_val, targets_val = batch[0], batch[1]

            with torch.no_grad():
                if cuda:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor)).cuda()
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                else:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor))
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                # ----------------------#
                #  CTK 确定 清零梯度
                # ----------------------#
                optimizer.zero_grad()
                outputs = net(images_val)
                losses = []
                num_pos_all = 0
                for i in range(3):
                    loss_item, num_pos = yolo_losses[i](outputs[i], targets_val)
                    losses.append(loss_item)
                    num_pos_all += num_pos
                loss = sum(losses) / num_pos_all
                val_loss += loss.item()
            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1)})
            pbar.update(1)
    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))

    print('Saving state, iter:', str(epoch+1))
    torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch+1),total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))

#----------------------------------------------------#
#   检测精度mAP和pr曲线计算参考视频
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#
if __name__ == "__main__":

    #----------------------------------------------------#
    #   获得图片路径和标签,需要强调的是：2007_test.txt和2007_val.txt里面没有内容是正常的。训练不会使用到。
    #----------------------------------------------------#
    # annotation_path = '2007_train.txt'
    annotation_path = 'model_data/train_DUO.txt'

    # 是否使用GPU
    Cuda = True

    #是否使用Dataloder
    Use_Data_Loader = True

    #是否对损失进行归一化，用于改变loss的大小；换句话说，用于决定计算最终loss是除上batch_size还是除上正样本数量
    normalize = False

    #   创建yolo模型， 训练前一定要修改Config里面的classes参数
    model = YoloBody(Config)

    # 加载预训练权重=======================================================================================================
    model_path = "model_data/yolo_weights.pth"
    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_dict = model.state_dict() # CTK 以字典的形式列出当前model的所有{key, value}
    #print('打印模型当中存在的变量名称\n\n\n', [k for k, v in model_dict.items() ])

    pretrained_dict = torch.load(model_path, map_location=device) # CTK 以字典的形式的加载预训练权重中所有{key, value}
    #print('打印权重文件中存在的变量名称\n\n\n', [k for k, v in pretrained_dict.items() ])

    #------------------------------------------------------#
    # CTK测试自己设计权重加载，成功！！！
    pretrained_dict = {k_w: v_w for k_w, v_w in pretrained_dict.items() if (k_w in [k_m for k_m, v_m in model_dict.items()]) & (np.shape(model_dict[k_w]) ==  np.shape(v_w) )}
    #print('model_dict 和 pretrained_dict字典中共同拥有名称且shape一致的变量名称\n\n\n ',[k for k, v in pretrained_dict.items()])

    # 官方给定权重加载方式，其实下面这行代码有一定的缺陷（pretrained_dict里面所包含的key值一定能在model_dict中找到），也就是说，pretrained_dict所包含的key数量小于等于model_dict所包含key数量
    #pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}  # 其中的 k 代表名称； 其中的 v 代表数据
    #------------------------------------------------------#

    # update() 方法用于修改当前集合，可以添加新的元素或集合到当前集合中，如果添加的元素在集合中已存在，则该元素只会出现一次，重复的会忽略。
    # https://www.runoob.com/python3/ref-set-update.html
    # x = {"apple": 1, "banana": 2, "cherry": 10}
    # y = {"google": 2, "runoob": 3, "apple": 2, 'tingkai': 99}
    #     x.update(y)
    # print(x) 得到 {'apple': 2, 'banana': 2, 'cherry': 10, 'google': 2, 'runoob': 3, 'tingkai': 99}
    # CTK 也就是说以y来更新x，假定x中没有而y中有的key:value，则添加到x中；x和y均有的key:value值，则以y的为最终输出；另外通常而言，模型（model_dict）所包含的key比预训练（pretrained_dict）多

    print('将原先 model_dict 中变量的权重更新为 pretrained_dict 中提前预训练好的权重')
    model_dict.update(pretrained_dict)


    # 模型加载权重
    model.load_state_dict(model_dict)
    print('权重加载完毕')
    # 加载预训练权重=======================================================================================================

    # 模型加载完权重之后应该立即设置是train还是eval模式
    net = model.train()

    if Cuda:
        net = torch.nn.DataParallel(model)  # CTK 采用数据并行的方式来处理，可以设置多卡并行处理
        cudnn.benchmark = True
        net = net.cuda()  # 将网络移植到cuda上面计算



    #划分训练集和验证集的比例，当前划分方式下，验证集和训练集的比例为1:9
    val_split = 0.01
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines) # CTK 每次shuffle后数据都被打乱，这个方法可以在机器学习训练#的时候在每个epoch结束后将数据重新洗牌进入下一个epoch的学习
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    # 定义保存文件的 writer
    writer = SummaryWriter('logs/tmp')

    # 冻结部分参数设计
    if True:

        lr = 1e-4
        Batch_size = 6
        Init_Epoch = 0
        Freeze_Epoch = 25

        optimizer = optim.Adam(net.parameters(), lr)
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda epoch: 1/(epoch+1))


        # 将model.backbone部分的参数进行冻结
        for param in model.backbone.parameters():
            param.requires_grad = False

        for epoch in range(Init_Epoch , Freeze_Epoch):

            # 随机选择本次的训练图片的分辨率
            #image_height = random.choice([320, 352, 384, 416, 448, 480, 512, 544, 576, 608])
            image_height = random.choice([576])
            image_width = image_height
            print('本次训练图片的宽度: ',image_width)

            #===============输入值部分============
            # Batch_size = 16；
            # image_width = 416;
            # image_height = 416;

            # ===============输出值部分============
            # 返回损失框架、将一次epoch分为几次“训练”、将一次epoch分为几次“验证”、训练数据，验证数据
            yolo_losses, epoch_size, epoch_size_val, gen, gen_val = prepare_train_data(Batch_size, image_width, image_height)

            # 迭代训练
            now = datetime.now()
            print('现在的时间为：', now)

            # net： 网络架构
            # yolo_losses： 训练损失
            # epoch：当前是第几次迭代
            # epoch_size：将一次epoch分为几次训练
            # epoch_size_val：将一次epoch分为几次验证
            # gen：训练类型数据集
            # gen_val：验证类型数据集
            # Freeze_Epoch：一共有几个epoch
            # Cuda：是否使用cuda进行加速
            fit_ont_epoch(net, yolo_losses,epoch,epoch_size, epoch_size_val, gen, gen_val, Freeze_Epoch, Cuda)
            lr_scheduler.step()

    # 全训练学习参数设计
    if True:

        lr = 1e-6
        Batch_size = 6
        Freeze_Epoch = 25
        Unfreeze_Epoch = 50

        optimizer = optim.Adam(net.parameters(),lr)
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda epoch: 1/(epoch+1))

        for param in model.backbone.parameters():
            param.requires_grad = True

        for epoch in range(Freeze_Epoch , Unfreeze_Epoch):

            # 随机选择本次的训练图片的分辨率
            #image_height = random.choice([320, 352, 384, 416, 448, 480, 512, 544, 576, 608])
            image_height = random.choice([576])
            image_width = image_height
            print('本次训练图片的宽度: ',image_width)
            # net： 网络架构
            # epoch：迭代次数
            # yolo_losses： 训练损失
            # epoch_size：将一次epoch分为几次训练
            # epoch_size_val：将一次epoch分为几次验证
            # gen：训练类型数据集
            # gen_val：验证类型数据集
            # Freeze_Epoch：一共有几个epoch
            # Cuda：是否使用cuda进行加速
            yolo_losses,epoch_size,epoch_size_val,gen,gen_val = prepare_train_data(Batch_size, image_width, image_height)

            # 迭代训练
            fit_ont_epoch(net, yolo_losses,epoch,epoch_size,epoch_size_val,gen,gen_val,Unfreeze_Epoch,Cuda)
            lr_scheduler.step()



