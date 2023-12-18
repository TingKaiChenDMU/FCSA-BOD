import math
from collections import OrderedDict

import torch
import torch.nn as nn

# import fuzzy_module_channel

from nets.fuzzy_module_channel import fuzzy_channel_inference
from nets.fuzzy_module_spatial import  fuzzy_spatial_inference
import numpy as np

import os
os.environ['DISPLAY'] = ':0'

#---------------------------------------------------------------------#
#   残差结构
#   利用一个1x1卷积下降通道数，然后利用一个3x3卷积提取特征并且上升通道数
#   最后接上一个残差边
#---------------------------------------------------------------------#
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes): # inplanes=1024, planes=[512, 1024]
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0])
        self.relu1 = nn.LeakyReLU(0.1)
        
        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1])
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += residual
        return out


class DarkNet(nn.Module):
    def __init__(self, layers):
        super(DarkNet, self).__init__()
        self.inplanes = 32
        # 416,416,3 -> 416,416,32
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu1 = nn.LeakyReLU(0.1)

        # 416,416,32 -> 208,208,64
        self.layer1 = self._make_layer([32, 64], layers[0])    #layers[0] = 1
        # 208,208,64 -> 104,104,128
        self.layer2 = self._make_layer([64, 128], layers[1])   #layers[1] = 2
        # 104,104,128 -> 52,52,256
        self.layer3 = self._make_layer([128, 256], layers[2])  #layers[2] = 8
        # 52,52,256 -> 26,26,512
        self.layer4 = self._make_layer([256, 512], layers[3])  #layers[3] = 8
        # 26,26,512 -> 13,13,1024
        self.layer5 = self._make_layer([512, 1024], layers[4]) #layers[4] = 4

        self.layers_out_filters = [64, 128, 256, 512, 1024]

        #CTK 下面这几行代码，CTK屏蔽掉之后并没有什么影响
        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

     # CTK  定义
        self.max_pool=nn.AdaptiveMaxPool2d(output_size=1)
        self.avg_pool=nn.AdaptiveAvgPool2d(output_size=1)

        self.mlp_1=nn.Sequential(
        nn.Linear(in_features=64,out_features=64,bias=False),
        nn.ReLU(),
        nn.Linear(in_features=64,out_features=64,bias=False))

        self.mlp_2=nn.Sequential(
        nn.Linear(in_features=128, out_features=128, bias=False),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=128, bias=False))

        self.mlp_out3=nn.Sequential(
        nn.Linear(in_features=256, out_features=256, bias=False),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=256, bias=False))

        self.mlp_out4=nn.Sequential(
        nn.Linear(in_features=512, out_features=512, bias=False),
        nn.ReLU(),
        nn.Linear(in_features=512, out_features=512, bias=False))

        self.mlp_out5=nn.Sequential(
        nn.Linear(in_features=1024, out_features=1024, bias=False),
        nn.ReLU(),
        nn.Linear(in_features=1024, out_features=1024, bias=False))


        self.sigmoid=nn.Sigmoid()

        #空间注意力机制
        self.conv3x3 = nn.Conv2d(in_channels=2,out_channels=1,kernel_size=3, dilation=1, stride=1, padding=3//2, bias=False)
        self.conv5x5 = nn.Conv2d(in_channels=2,out_channels=1,kernel_size=3, dilation=2, stride=1, padding=5//2, bias=False)
        self.conv7x7 = nn.Conv2d(in_channels=2,out_channels=1,kernel_size=3, dilation=3, stride=1, padding=7//2, bias=False)

    #---------------------------------------------------------------------#
    #   在每一个layer里面，首先利用一个步长为2的3x3卷积进行下采样
    #   然后进行残差结构的堆叠
    #---------------------------------------------------------------------#
    def _make_layer(self, planes, blocks):  # 对于残差4这个单元 self._make_layer([512, 1024], layers[4]) #layers[4] = 4
        layers = []
        # 下采样，步长为2，卷积核大小为3
        layers.append(("ds_conv", nn.Conv2d(self.inplanes, planes[1], kernel_size=3, stride=2, padding=1, bias=False)))  # self.inplanes=512， planes[1] = 1024
        layers.append(("ds_bn", nn.BatchNorm2d(planes[1])))
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))

        # 加入残差结构
        self.inplanes = planes[1]
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i), BasicBlock(self.inplanes, planes)))  # inplanes=1024, planes=[512, 1024]
        # 这里使用orderedDict是因为上面利用layer这一个list进行收集相关的层，经过查阅资料可以得知，在使用list收集
        # 相关层的时候，list并不会按照顺序进行保存，是一种乱序的模式。因此使用OrdererDict来是它按照收集的顺序保存。
        # 然后使用nn.sequential来嵌套OrderedDict里面的内容。
        return nn.Sequential(OrderedDict(layers))
    



    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # 416,416,32 -> 208,208,64=======================================================================================
        x = self.layer1(x)
        maxout=self.max_pool(x)
        maxout=self.mlp_1(maxout.view(maxout.size(0),-1))
        maxout = self.sigmoid(maxout)

        avgout=self.avg_pool(x)
        avgout=self.mlp_1(avgout.view(avgout.size(0),-1))
        avgout = self.sigmoid(avgout)
        # 将数据都转为1维
        maxout = maxout.reshape(-1)
        avgout = avgout.reshape(-1)
        # 将数据从tensor变为numpy
        maxout = maxout.cpu().detach().numpy()
        avgout = avgout.cpu().detach().numpy()
        # 定义通道注意力的容器
        container_channel = []
        for index in range(avgout.shape[0]):
            # 逐个完成模糊推理
            container_channel.append(fuzzy_channel_inference(maxout[index], avgout[index]))
        # 将list转为numpy，进而转为tensor
        channel_attention = torch.tensor(np.array(container_channel)).float()
        # 将维度还原
        channel_attention = channel_attention.view(x.size(0), x.size(1), 1, 1)
        # 计算通道注意力的输出
        channel_out = channel_attention.cuda() * x

                ##-----模糊空间注意力机制---------
        # 等号左侧max_out和mean_out的shape均为([6, 1, 8, 8])
        max_out,_=torch.max(channel_out,dim=1,keepdim=True)
        mean_out=torch.mean(channel_out,dim=1,keepdim=True)
        # 通道维度拼接max_out和mean_out
        out = torch.cat((max_out, mean_out), dim=1)
        # 等号左侧out3x3、out5x5和out7x7的shape均为([6, 1, 8, 8])
        out3x3 = self.sigmoid(self.conv3x3(out))
        out5x5 = self.sigmoid(self.conv5x5(out))
        out7x7 = self.sigmoid(self.conv7x7(out))
        # 将所有的数据均变换为1维
        out3x3 = out3x3.reshape(-1)
        out5x5 = out5x5.reshape(-1)
        out7x7 = out7x7.reshape(-1)
        # 将数据从tensor变为numpy
        out3x3 = out3x3.cpu().detach().numpy()
        out5x5 = out5x5.cpu().detach().numpy()
        out7x7 = out7x7.cpu().detach().numpy()
        # 定义空间注意力的容器
        container_spatial = []
        for index in range(out3x3.shape[0]):
            # 逐个完成模糊推理
            container_spatial.append(fuzzy_spatial_inference(out3x3[index], out5x5[index], out7x7[index]))
        # 将list转为numpy，进而转为tensor
        spatial_attention = torch.tensor(np.array(container_spatial)).float()
        # 将维度还原
        spatial_attention = spatial_attention.view(x.size(0), 1, x.size()[2], x.size()[3])
        # 计算空间注意力输出
        spatial_out = spatial_attention.cuda() * channel_out
        x = spatial_out
        
         # 208,208,64 -> 104,104,128=======================================================================================
        x = self.layer2(x)
        maxout=self.max_pool(x)
        maxout=self.mlp_2(maxout.view(maxout.size(0),-1))
        maxout = self.sigmoid(maxout)

        avgout=self.avg_pool(x)
        avgout=self.mlp_2(avgout.view(avgout.size(0),-1))
        avgout = self.sigmoid(avgout)
        # 将数据都转为1维
        maxout = maxout.reshape(-1)
        avgout = avgout.reshape(-1)
        # 将数据从tensor变为numpy
        maxout = maxout.cpu().detach().numpy()
        avgout = avgout.cpu().detach().numpy()
        # 定义通道注意力的容器
        container_channel = []
        for index in range(avgout.shape[0]):
            # 逐个完成模糊推理
            container_channel.append(fuzzy_channel_inference(maxout[index], avgout[index]))
        # 将list转为numpy，进而转为tensor
        channel_attention = torch.tensor(np.array(container_channel)).float()
        # 将维度还原
        channel_attention = channel_attention.view(x.size(0), x.size(1), 1, 1)
        # 计算通道注意力的输出
        channel_out = channel_attention.cuda() * x

                ##-----模糊空间注意力机制---------
        # 等号左侧max_out和mean_out的shape均为([6, 1, 8, 8])
        max_out,_=torch.max(channel_out,dim=1,keepdim=True)
        mean_out=torch.mean(channel_out,dim=1,keepdim=True)
        # 通道维度拼接max_out和mean_out
        out = torch.cat((max_out, mean_out), dim=1)
        # 等号左侧out3x3、out5x5和out7x7的shape均为([6, 1, 8, 8])
        out3x3 = self.sigmoid(self.conv3x3(out))
        out5x5 = self.sigmoid(self.conv5x5(out))
        out7x7 = self.sigmoid(self.conv7x7(out))
        # 将所有的数据均变换为1维
        out3x3 = out3x3.reshape(-1)
        out5x5 = out5x5.reshape(-1)
        out7x7 = out7x7.reshape(-1)
        # 将数据从tensor变为numpy
        out3x3 = out3x3.cpu().detach().numpy()
        out5x5 = out5x5.cpu().detach().numpy()
        out7x7 = out7x7.cpu().detach().numpy()
        # 定义空间注意力的容器
        container_spatial = []
        for index in range(out3x3.shape[0]):
            # 逐个完成模糊推理
            container_spatial.append(fuzzy_spatial_inference(out3x3[index], out5x5[index], out7x7[index]))
        # 将list转为numpy，进而转为tensor
        spatial_attention = torch.tensor(np.array(container_spatial)).float()
        # 将维度还原
        spatial_attention = spatial_attention.view(x.size(0), 1, x.size()[2], x.size()[3])
        # 计算空间注意力输出
        spatial_out = spatial_attention.cuda() * channel_out
        x = spatial_out

         # 104,104,128 -> 52,52,256=====================================================================================================
        out3 = self.layer3(x)
        maxout=self.max_pool(out3)
        maxout=self.mlp_out3(maxout.view(maxout.size(0),-1))
        maxout = self.sigmoid(maxout)

        avgout=self.avg_pool(out3)
        avgout=self.mlp_out3(avgout.view(avgout.size(0),-1))
        avgout = self.sigmoid(avgout)
        # 将数据都转为1维
        maxout = maxout.reshape(-1)
        avgout = avgout.reshape(-1)
        # 将数据从tensor变为numpy
        maxout = maxout.cpu().detach().numpy()
        avgout = avgout.cpu().detach().numpy()
        # 定义通道注意力的容器
        container_channel = []
        for index in range(avgout.shape[0]):
            # 逐个完成模糊推理
            container_channel.append(fuzzy_channel_inference(maxout[index], avgout[index]))
        # 将list转为numpy，进而转为tensor
        channel_attention = torch.tensor(np.array(container_channel)).float()
        # 将维度还原
        channel_attention = channel_attention.view(out3.size(0), out3.size(1), 1, 1)
        # 计算通道注意力的输出
        channel_out = channel_attention.cuda() * out3

                ##-----模糊空间注意力机制---------
        # 等号左侧max_out和mean_out的shape均为([6, 1, 8, 8])
        max_out,_=torch.max(channel_out,dim=1,keepdim=True)
        mean_out=torch.mean(channel_out,dim=1,keepdim=True)
        # 通道维度拼接max_out和mean_out
        out = torch.cat((max_out, mean_out), dim=1)
        # 等号左侧out3x3、out5x5和out7x7的shape均为([6, 1, 8, 8])
        out3x3 = self.sigmoid(self.conv3x3(out))
        out5x5 = self.sigmoid(self.conv5x5(out))
        out7x7 = self.sigmoid(self.conv7x7(out))
        # 将所有的数据均变换为1维
        out3x3 = out3x3.reshape(-1)
        out5x5 = out5x5.reshape(-1)
        out7x7 = out7x7.reshape(-1)
        # 将数据从tensor变为numpy
        out3x3 = out3x3.cpu().detach().numpy()
        out5x5 = out5x5.cpu().detach().numpy()
        out7x7 = out7x7.cpu().detach().numpy()
        # 定义空间注意力的容器
        container_spatial = []
        for index in range(out3x3.shape[0]):
            # 逐个完成模糊推理
            container_spatial.append(fuzzy_spatial_inference(out3x3[index], out5x5[index], out7x7[index]))
        # 将list转为numpy，进而转为tensor
        spatial_attention = torch.tensor(np.array(container_spatial)).float()
        # 将维度还原
        spatial_attention = spatial_attention.view(out3.size(0), 1, out3.size()[2], out3.size()[3])
        # 计算空间注意力输出
        spatial_out = spatial_attention.cuda() * channel_out
        out3 = spatial_out

         # 52,52,256 -> 26,26,512===============================================================================================
        out4 = self.layer4(out3)
        maxout=self.max_pool(out4)
        maxout=self.mlp_out4(maxout.view(maxout.size(0),-1))
        maxout = self.sigmoid(maxout)

        avgout=self.avg_pool(out4)
        avgout=self.mlp_out4(avgout.view(avgout.size(0),-1))
        avgout = self.sigmoid(avgout)
        # 将数据都转为1维
        maxout = maxout.reshape(-1)
        avgout = avgout.reshape(-1)
        # 将数据从tensor变为numpy
        maxout = maxout.cpu().detach().numpy()
        avgout = avgout.cpu().detach().numpy()
        # 定义通道注意力的容器
        container_channel = []
        for index in range(avgout.shape[0]):
            # 逐个完成模糊推理
            container_channel.append(fuzzy_channel_inference(maxout[index], avgout[index]))
        # 将list转为numpy，进而转为tensor
        channel_attention = torch.tensor(np.array(container_channel)).float()
        # 将维度还原
        channel_attention = channel_attention.view(out4.size(0), out4.size(1), 1, 1)
        # 计算通道注意力的输出
        channel_out = channel_attention.cuda() * out4

                ##-----模糊空间注意力机制---------
        # 等号左侧max_out和mean_out的shape均为([6, 1, 8, 8])
        max_out,_=torch.max(channel_out,dim=1,keepdim=True)
        mean_out=torch.mean(channel_out,dim=1,keepdim=True)
        # 通道维度拼接max_out和mean_out
        out = torch.cat((max_out, mean_out), dim=1)
        # 等号左侧out3x3、out5x5和out7x7的shape均为([6, 1, 8, 8])
        out3x3 = self.sigmoid(self.conv3x3(out))
        out5x5 = self.sigmoid(self.conv5x5(out))
        out7x7 = self.sigmoid(self.conv7x7(out))
        # 将所有的数据均变换为1维
        out3x3 = out3x3.reshape(-1)
        out5x5 = out5x5.reshape(-1)
        out7x7 = out7x7.reshape(-1)
        # 将数据从tensor变为numpy
        out3x3 = out3x3.cpu().detach().numpy()
        out5x5 = out5x5.cpu().detach().numpy()
        out7x7 = out7x7.cpu().detach().numpy()
        # 定义空间注意力的容器
        container_spatial = []
        for index in range(out3x3.shape[0]):
            # 逐个完成模糊推理
            container_spatial.append(fuzzy_spatial_inference(out3x3[index], out5x5[index], out7x7[index]))
        # 将list转为numpy，进而转为tensor
        spatial_attention = torch.tensor(np.array(container_spatial)).float()
        # 将维度还原
        spatial_attention = spatial_attention.view(out4.size(0), 1, out4.size()[2], out4.size()[3])
        # 计算空间注意力输出
        spatial_out = spatial_attention.cuda() * channel_out
        out4 = spatial_out

        # 26,26,512 -> 13,13,1024==============================================================================================
        out5 = self.layer5(out4)
        maxout=self.max_pool(out5)
        maxout=self.mlp_out5(maxout.view(maxout.size(0),-1))
        maxout = self.sigmoid(maxout)

        avgout=self.avg_pool(out5)
        avgout=self.mlp_out5(avgout.view(avgout.size(0),-1))
        avgout = self.sigmoid(avgout)
        # 将数据都转为1维
        maxout = maxout.reshape(-1)
        avgout = avgout.reshape(-1)
        # 将数据从tensor变为numpy
        maxout = maxout.cpu().detach().numpy()
        avgout = avgout.cpu().detach().numpy()
        # 定义通道注意力的容器
        container_channel = []
        for index in range(avgout.shape[0]):
            # 逐个完成模糊推理
            container_channel.append(fuzzy_channel_inference(maxout[index], avgout[index]))
        # 将list转为numpy，进而转为tensor
        channel_attention = torch.tensor(np.array(container_channel)).float()
        # 将维度还原
        channel_attention = channel_attention.view(out5.size(0), out5.size(1), 1, 1)
        # 计算通道注意力的输出
        channel_out = channel_attention.cuda() * out5

                ##-----模糊空间注意力机制---------
        # 等号左侧max_out和mean_out的shape均为([6, 1, 8, 8])
        max_out,_=torch.max(channel_out,dim=1,keepdim=True)
        mean_out=torch.mean(channel_out,dim=1,keepdim=True)
        # 通道维度拼接max_out和mean_out
        out = torch.cat((max_out, mean_out), dim=1)
        # 等号左侧out3x3、out5x5和out7x7的shape均为([6, 1, 8, 8])
        out3x3 = self.sigmoid(self.conv3x3(out))
        out5x5 = self.sigmoid(self.conv5x5(out))
        out7x7 = self.sigmoid(self.conv7x7(out))
        # 将所有的数据均变换为1维
        out3x3 = out3x3.reshape(-1)
        out5x5 = out5x5.reshape(-1)
        out7x7 = out7x7.reshape(-1)
        # 将数据从tensor变为numpy
        out3x3 = out3x3.cpu().detach().numpy()
        out5x5 = out5x5.cpu().detach().numpy()
        out7x7 = out7x7.cpu().detach().numpy()
        # 定义空间注意力的容器
        container_spatial = []
        for index in range(out3x3.shape[0]):
            # 逐个完成模糊推理
            container_spatial.append(fuzzy_spatial_inference(out3x3[index], out5x5[index], out7x7[index]))
        # 将list转为numpy，进而转为tensor
        spatial_attention = torch.tensor(np.array(container_spatial)).float()
        # 将维度还原
        spatial_attention = spatial_attention.view(out5.size(0), 1, out5.size()[2], out5.size()[3])
        # 计算空间注意力输出
        spatial_out = spatial_attention.cuda() * channel_out

        out5 = spatial_out
        return out3, out4, out5

def darknet53(pretrained, **kwargs):
    model = DarkNet([1, 2, 8, 8, 4])
    # 下面加载预训练权重部分，不会被使用，
    if pretrained:
        if isinstance(pretrained, str): # CTK 判定所传入的pretrained参数是不是一个str类型
            model.load_state_dict(torch.load(pretrained))
        else:
            raise Exception("darknet request a pretrained path. got [{}]".format(pretrained))
    return model
