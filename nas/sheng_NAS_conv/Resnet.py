'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import psutil
import os
from torch import random

def memory_info(label=None):
    info = psutil.virtual_memory()
    #print(label)
    #print('memory_used: ', psutil.Process(os.getpid()).memory_info().rss)
    #print('memory_all:', info.total)
    #print('memory_percent: ', info.percent)
    #print('cpu_num: ', psutil.cpu_count())


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes,kernel_size,stride=1):
        super(BasicBlock, self).__init__()
        #list=[3,5,7,9]
        #kernel_size=get_kernel_size(kernel_size)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=kernel_size, stride=stride, padding=int((kernel_size-1)/2), bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size,
                               stride=1, padding=int((kernel_size-1)/2), bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=1, bias=False))
        if stride != 1:# or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d((self.expansion*planes))
            )


    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.relu1(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    expansion = 1
    def __init__(self, in_planes, out_planes,kernel_size,stride=2):
        super(Block, self).__init__()
        self.stride = stride
        # 通过 expansion 增大 feature map 的数量
        planes = out_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride, padding=int((kernel_size-1)/2), groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=kernel_size, stride=stride, padding=int((kernel_size-1)/2), bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        # 步长为 1 时，如果 in 和 out 的 feature map 通道不同，用一个卷积改变通道数
        self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes,
                                                kernel_size=1, stride=1, bias=False))
        if stride != 1:  # or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d((self.expansion * planes))
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        # 步长为1，加 shortcut 操作
        if self.stride == 1:
            return out + self.shortcut(x)
        # 步长为2，直接输出
        else:
            return out

class Block_bais(nn.Module):
    expansion=1
    def __init__(self, in_planes,planes, kernel_size, stride=1):
        super(Block_bais, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size,
                               stride=2, padding=int((kernel_size-1)/2), bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))

        return out




class Net(nn.Module):
    def __init__(self, block1,block2, number_channel,num_blocks,kernel_size,num_classes=5):
        super(Net, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)

        self.layer1 = self._make_layer(BasicBlock, number_channel[0], num_blocks[0],kernel_size,stride=2)
        self.layer2 = self._make_layer(BasicBlock, number_channel[1], num_blocks[1],kernel_size, stride=2)
        self.layer3 = self._make_layer(block1, number_channel[2], num_blocks[2],kernel_size, stride=2)
        self.layer4 = self._make_layer(block2, number_channel[3], num_blocks[3],kernel_size,stride=2)
        self.linear = nn.Linear(number_channel[3]*BasicBlock.expansion, num_classes)


    def _make_layer(self, block, planes, num_blocks,kernel_size, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes,kernel_size, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.relu1(out)
        out = self.pool1(out)
        print("out0:",out.shape)
        out = self.layer1(out)
        print("out1:", out.shape)
        out = self.layer2(out)
        print("out2:", out.shape)
        out = self.layer3(out)
        print("out3:", out.shape)
        out = self.layer4(out)
        print("out4:", out.shape)
        #out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def Res18(kernel_size,num_channel,conv_type,conv_type_second):
    print("kernel_size:::::",kernel_size)
    print("conv_type",conv_type)
    print("conv_type_second", conv_type_second)

    if conv_type ==[1]:
        block1=BasicBlock
    elif conv_type==[2]:
         block1=Block
    elif conv_type==[3]:
        block1=Block_bais
    print("block1",block1)
    if conv_type_second ==[1]:
        block2=BasicBlock
    elif conv_type_second==[2]:
         block2=Block
    elif conv_type_second==[3]:
        block2=Block_bais
    if conv_type!=[1] and conv_type_second!=[1]:
     return Net(block1,block2,num_channel,[2,2,2,2],kernel_size)
    else:
     return Net(Block, Block_bais, num_channel, [2, 2, 2, 2], kernel_size)



'''def test():
    net = ResNet18()

    # net.layer1[0].conv3 = nn.Conv2d(64, 17, kernel_size=(1, 1), stride=(1, 1), bias=False)

    # y = net(torch.randn(1, 3, 32, 32))
    # print(y.size())

    print("net18",net)
    # # print(net.layer1[0].conv3)

#test()'''
